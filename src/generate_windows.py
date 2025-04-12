import os
import pandas as pd
from datetime import timedelta
import multiprocessing as mp
from functools import partial

SHERLOCK_DATASET_PATH = "/scratch/groups/euan/mhc/mhc_dataset"
OUTPUT_PATH = "/scratch/groups/euan/calvinxu/mhc_analysis"
BATCH_SIZE = 500
NUM_PROCESSES = 16


def get_user_ids(base_path):
    return [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]


def load_user_metadata(base_path, user_id):
    metadata_path = os.path.join(base_path, user_id, "metadata.parquet")
    if os.path.exists(metadata_path):
        return pd.read_parquet(metadata_path)
    return None


def find_non_overlapping_7day_windows(dates, window_size=7, min_required_days=5):
    """
    Given a sorted list of datetime objects (dates) on which the user
    has valid data, find all non-overlapping 7-day windows with >= min_required_days.

    Args:
        dates: List of sorted datetime objects
        window_size: Size of the window in days
        min_required_days: Minimum number of days required in a window

    Returns a list of (window_start, window_end) tuples.
    """
    if not dates:
        return []

    valid_windows = []
    idx = 0
    last_date = dates[-1]

    current_start = dates[0]
    while current_start <= last_date - timedelta(days=window_size - 1):
        window_end = current_start + timedelta(days=window_size - 1)

        days_in_window = sum(current_start <= d <= window_end for d in dates)

        if days_in_window >= min_required_days:
            valid_windows.append((current_start, window_end))
            current_start = window_end + timedelta(days=1)
        else:
            current_start += timedelta(days=1)

    return valid_windows


def get_file_uris_for_window(base_path, user_id, window_start, window_end):
    """
    For each day in [window_start, window_end], check if a .npy file exists.
    If it does, record the relative path: "<user_id>/<YYYY-MM-DD>.npy".
    Returns a list of such file URIs.
    """
    file_uris = []
    num_days = (window_end - window_start).days + 1
    for i in range(num_days):
        day = window_start + timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        npy_filename = f"{day_str}.npy"
        full_path = os.path.join(base_path, user_id, npy_filename)
        if os.path.exists(full_path):
            file_uris.append(os.path.join(user_id, npy_filename))
    return file_uris


def meets_coverage_criteria(day_data, min_channel_coverage=None, min_channels_with_data=None):
    """
    Check if a single day's data meets the coverage criteria. This function evaluates whether
    a day's sensor data has sufficient quality based on two possible criteria:
    1. Total data coverage - checks if the sum of data coverage percentages across all channels meets the minimum
    2. Number of channels with data - checks if enough distinct channels have any data at all
    
    The function can apply either or both criteria depending on which parameters are provided.
    
    Args:
        day_data: DataFrame containing metadata for a single day
        min_channel_coverage: Minimum total data coverage required across all channels, if None no coverage check is applied
        min_channels_with_data: Minimum number of channels that must have coverage > 0, if None no channel count check is applied
        
    Returns:
        bool: True if the day meets criteria, False otherwise
    """
    # If both criteria are None, no filtering is applied
    if min_channel_coverage is None and min_channels_with_data is None:
        return True
    
    channels_with_data = sum(day_data['data_coverage'] > 0)
    
    # Check coverage criteria if specified
    if min_channel_coverage is not None:
        # Check if the sum of data coverage meets the minimum threshold
        total_coverage = day_data['data_coverage'].sum()
        
        # If total coverage doesn't meet the minimum, the day doesn't meet criteria
        if total_coverage < min_channel_coverage:
            return False
    
    # Check channel count criteria if specified
    if min_channels_with_data is not None:
        return channels_with_data >= min_channels_with_data
    
    return True


def get_valid_dates(metadata_df, min_channel_coverage=None, min_channels_with_data=None):
    """
    Filter metadata to find dates that meet the coverage criteria.
    
    Args:
        metadata_df: DataFrame containing metadata for all days
        min_channel_coverage: Minimum percentage coverage required per channel
        min_channels_with_data: Minimum number of channels that must have coverage > 0
        
    Returns:
        list: Sorted list of datetime objects for valid dates
    """
    valid_dates = []
    
    for date, day_data in metadata_df.groupby('date'):
        if meets_coverage_criteria(day_data, min_channel_coverage, min_channels_with_data):
            valid_dates.append(pd.to_datetime(date))
    
    return sorted(valid_dates)


def process_user(user_id, base_path, min_channel_coverage=10.0, min_channels_with_data=3, 
                window_size=7, min_required_days=5):
    """
    For a single user, find all valid 7-day windows (non-overlapping) with
    >= min_required_days of data that also meet coverage criteria.
    """
    metadata_df = load_user_metadata(base_path, user_id)
    if metadata_df is None or metadata_df.empty:
        return []
    
    # Get valid dates based on coverage criteria
    valid_dates = get_valid_dates(metadata_df, min_channel_coverage, min_channels_with_data)
    
    # Find valid windows using filtered dates
    valid_windows = find_non_overlapping_7day_windows(valid_dates, window_size, min_required_days)
    
    result_rows = []
    for start_date, end_date in valid_windows:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        time_range = f"{start_str}_{end_str}"

        file_uris = get_file_uris_for_window(base_path, user_id, start_date, end_date)

        row = {"healthCode": user_id, "time_range": time_range, "file_uris": file_uris}
        result_rows.append(row)

    return result_rows


def run_analysis_on_sherlock():
    print(f"Searching for users in {SHERLOCK_DATASET_PATH}...")
    user_ids = get_user_ids(SHERLOCK_DATASET_PATH)
    total_users = len(user_ids)
    print(f"Found {total_users} user directories.")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    final_outfile = os.path.join(OUTPUT_PATH, "valid_7day_windows.csv")

    all_results = []
    batch_count = (total_users - 1) // BATCH_SIZE + 1

    process_func = partial(process_user, base_path=SHERLOCK_DATASET_PATH)

    for batch_idx in range(batch_count):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_users)
        batch_user_ids = user_ids[start_idx:end_idx]

        print(
            f"Processing batch {batch_idx + 1}/{batch_count} "
            f"(users {start_idx} to {end_idx - 1})..."
        )

        with mp.Pool(processes=NUM_PROCESSES) as pool:
            batch_results = pool.map(process_func, batch_user_ids)

        flattened = [row for user_rows in batch_results for row in user_rows]

        if flattened:
            batch_df = pd.DataFrame(flattened)
            all_results.append(batch_df)

            batch_file = os.path.join(
                OUTPUT_PATH, f"valid_7day_windows_batch_{batch_idx + 1}.csv"
            )
            batch_df.to_csv(batch_file, index=False)
            print(f"  ... saved batch CSV to {batch_file}")

        del batch_results

    # Combine all batches
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(final_outfile, index=False)
        print(f"\nCombined all batches into {final_outfile}")
    else:
        print("\nNo valid 7-day windows found across all users.")

    print("Done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate valid 7-day windows from Sherlock dataset.')
    parser.add_argument('--sherlock_path', type=str, required=True,
                        help='Path to the Sherlock dataset directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Directory to save the output CSV files')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of users to process in each batch (default: 100)')
    parser.add_argument('--num_processes', type=int, default=mp.cpu_count(),
                        help=f'Number of parallel processes to use (default: {mp.cpu_count()})')
    parser.add_argument('--min_channel_coverage', type=float, default=10.0,
                        help='Minimum total data coverage required across all channels (default: 10.0)')
    parser.add_argument('--min_channels_with_data', type=int, default=3,
                        help='Minimum number of channels that must have coverage > 0 (default: 3)')
    parser.add_argument('--window_size', type=int, default=7,
                        help='Size of the window in days (default: 7)')
    parser.add_argument('--min_required_days', type=int, default=5,
                        help='Minimum number of days with data required in a window (default: 5)')
    
    args = parser.parse_args()
    
    # Create a partial function with all the parameters
    process_func = partial(
        process_user, 
        base_path=args.sherlock_path,
        min_channel_coverage=args.min_channel_coverage,
        min_channels_with_data=args.min_channels_with_data,
        window_size=args.window_size,
        min_required_days=args.min_required_days
    )
    
    print(f"Searching for users in {args.sherlock_path}...")
    user_ids = get_user_ids(args.sherlock_path)
    total_users = len(user_ids)
    print(f"Found {total_users} user directories.")

    os.makedirs(args.output_path, exist_ok=True)
    final_outfile = os.path.join(args.output_path, "valid_7day_windows.csv")

    all_results = []
    batch_count = (total_users - 1) // args.batch_size + 1

    for batch_idx in range(batch_count):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, total_users)
        batch_user_ids = user_ids[start_idx:end_idx]

        print(
            f"Processing batch {batch_idx + 1}/{batch_count} "
            f"(users {start_idx} to {end_idx - 1})..."
        )

        with mp.Pool(processes=args.num_processes) as pool:
            batch_results = pool.map(process_func, batch_user_ids)

        flattened = [row for user_rows in batch_results for row in user_rows]

        if flattened:
            batch_df = pd.DataFrame(flattened)
            all_results.append(batch_df)

            batch_file = os.path.join(
                args.output_path, f"valid_7day_windows_batch_{batch_idx + 1}.csv"
            )
            batch_df.to_csv(batch_file, index=False)
            print(f"  ... saved batch CSV to {batch_file}")

        del batch_results

    # Combine all batches
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(final_outfile, index=False)
        print(f"\nCombined all batches into {final_outfile}")
        print(f"Total windows found: {len(final_df)}")
    else:
        print("\nNo valid 7-day windows found across all users.")

    print("Done.")
