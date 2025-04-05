import os
import pandas as pd
from datetime import timedelta
import multiprocessing as mp
from functools import partial

SHERLOCK_DATASET_PATH = "/scratch/groups/euan/mhc/mhc_dataset"
OUTPUT_PATH = "/scratch/groups/euan/calvinxu/mhc_analysis"
BATCH_SIZE = 500
NUM_PROCESSES = 16
WINDOW_SIZE = 7
MIN_REQUIRED_DAYS = 5  # at most 2 missing days in a 7-day window


def get_user_ids(base_path):
    return [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]


def load_user_metadata(base_path, user_id):
    metadata_path = os.path.join(base_path, user_id, "metadata.parquet")
    if os.path.exists(metadata_path):
        return pd.read_parquet(metadata_path)
    return None


def find_non_overlapping_7day_windows(dates):
    """
    Given a sorted list of datetime objects (dates) on which the user
    has valid data, find all non-overlapping 7-day windows with >= MIN_REQUIRED_DAYS.

    Returns a list of (window_start, window_end) tuples.
    """
    if not dates:
        return []

    valid_windows = []
    idx = 0
    last_date = dates[-1]

    current_start = dates[0]
    while current_start <= last_date - timedelta(days=WINDOW_SIZE - 1):
        window_end = current_start + timedelta(days=WINDOW_SIZE - 1)

        days_in_window = sum(current_start <= d <= window_end for d in dates)

        if days_in_window >= MIN_REQUIRED_DAYS:
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


def process_user(user_id, base_path):
    """
    For a single user, find all valid 7-day windows (non-overlapping) with
    >= MIN_REQUIRED_DAYS of data. For each valid window, gather a list of
    file URIs. Return a list of dictionaries.
    """
    metadata_df = load_user_metadata(base_path, user_id)
    if metadata_df is None or metadata_df.empty:
        return []

    user_dates = sorted(pd.to_datetime(metadata_df["date"].unique()))

    valid_windows = find_non_overlapping_7day_windows(user_dates)

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
