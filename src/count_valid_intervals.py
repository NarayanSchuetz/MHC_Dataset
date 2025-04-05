import os
import pandas as pd
from datetime import timedelta
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

SHERLOCK_DATASET_PATH = "/scratch/groups/euan/mhc/mhc_dataset"

OUTPUT_PATH = "/scratch/groups/euan/calvinxu/mhc_analysis"

THRESHOLDS = {
    "7day": 2,  # At most 2 missing days in any 7-day window
    "14day": 4,
}


def get_user_ids(base_path):
    """Get all user IDs from the dataset directory."""
    user_dirs = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    return user_dirs


def load_user_metadata(base_path, user_id):
    metadata_path = os.path.join(base_path, user_id, "metadata.parquet")
    if os.path.exists(metadata_path):
        return pd.read_parquet(metadata_path)
    return None


def has_valid_window(dates, window_size, max_missing):
    """
    Check if there's at least one contiguous window of length window_size
    with at most max_missing days of data missing.
    """
    if not dates.size:
        return False

    dates_sorted = sorted(pd.to_datetime(dates))
    start_date, end_date = dates_sorted[0], dates_sorted[-1]

    # Check rolling windows
    current_date = start_date
    while current_date <= end_date - timedelta(days=window_size - 1):
        window_start = current_date
        window_end = current_date + timedelta(days=window_size - 1)

        days_in_window = 0
        for d in dates_sorted:
            if window_start <= d <= window_end:
                days_in_window += 1

        if (window_size - days_in_window) <= max_missing:
            return True

        current_date += timedelta(days=1)

    return False


def analyze_user_data(metadata_df, threshold_7day, threshold_14day):
    """
    For a single user, determine whether they have:
      1) Any 7-day window with at most threshold_7day missing days.
      2) Any 14-day window with at most threshold_14day missing days.
    """
    if metadata_df is None or metadata_df.empty:
        return None

    # Gather all unique dates
    dates = pd.to_datetime(metadata_df["date"].unique())

    # Check 7-day
    has_7day_window = has_valid_window(dates, 7, threshold_7day)

    # Check 14-day
    has_14day_window = has_valid_window(dates, 14, threshold_14day)

    return {
        "7day_window_qualified": has_7day_window,
        "14day_window_qualified": has_14day_window,
    }


def process_user(user_id, base_path, threshold_7day, threshold_14day):
    metadata_df = load_user_metadata(base_path, user_id)
    if metadata_df is not None:
        results = analyze_user_data(metadata_df, threshold_7day, threshold_14day)
        return user_id, results
    return user_id, None


def visualize_results(results, total_users, title="Valid Window Counts"):
    """
    Creates a bar chart from the aggregated results.

    Args:
        results: dict of { metric_name: (count, percentage) }
        total_users: total number of users processed
        title: chart title
    """
    rules = list(results.keys())
    counts = [results[rule][0] for rule in rules]
    percentages = [results[rule][1] for rule in rules]

    rule_labels = {
        "7day_window_qualified": "Has 7-Day Window ≤2 Missing",
        "14day_window_qualified": "Has 14-Day Window ≤4 Missing",
    }
    labels = [rule_labels.get(rule, rule) for rule in rules]

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(rules)), percentages)
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        plt.text(
            i,
            percentage + 1,
            f"{count} ({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title(title, fontsize=16, pad=20)
    plt.ylabel("Percentage of Users (%)", fontsize=12)
    plt.ylim(0, max(percentages) * 1.2 + 5 if percentages else 100)
    plt.xticks(range(len(rules)), labels, rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()


def analyze_dataset_on_sherlock(num_processes=16, batch_size=500):
    """
    Analyze the dataset on Sherlock using parallel processing.

    Args:
        num_processes: Number of parallel processes to use
        batch_size: Number of users to process in each batch

    Returns:
        (aggregated_results, total_users)
          aggregated_results = {
             "7day_window_qualified": (count, percentage),
             "14day_window_qualified": (count, percentage)
          }
    """
    user_ids = get_user_ids(SHERLOCK_DATASET_PATH)
    total_users = len(user_ids)
    print(f"Found {total_users} users in Sherlock dataset")

    metrics = [
        "7day_window_qualified",
        "14day_window_qualified",
    ]
    aggregated_results = {m: 0 for m in metrics}

    user_results_dir = os.path.join(OUTPUT_PATH, "user_results")
    os.makedirs(user_results_dir, exist_ok=True)

    user_results_all = []

    process_func = partial(
        process_user,
        base_path=SHERLOCK_DATASET_PATH,
        threshold_7day=THRESHOLDS["7day"],
        threshold_14day=THRESHOLDS["14day"],
    )

    for i in range(0, total_users, batch_size):
        batch_user_ids = user_ids[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/"
            f"{(total_users - 1) // batch_size + 1} ({len(batch_user_ids)} users)"
        )

        with mp.Pool(processes=num_processes) as pool:
            batch_results = pool.map(process_func, batch_user_ids)

        # Aggregate batch results
        batch_user_data = []
        for user_id, user_res in batch_results:
            if user_res:
                row = {"user_id": user_id}
                for m in metrics:
                    qualified = user_res.get(m, False)
                    row[m] = qualified
                    if qualified:
                        aggregated_results[m] += 1
                batch_user_data.append(row)

        # Save batch results to disk
        if batch_user_data:
            batch_df = pd.DataFrame(batch_user_data)
            batch_file = os.path.join(
                user_results_dir, f"user_results_batch_{i // batch_size + 1}.csv"
            )
            batch_df.to_csv(batch_file, index=False)
            user_results_all.extend(batch_user_data)

        # Release memory
        del batch_results
        del batch_user_data

        print(f"Completed batch {i // batch_size + 1}, current counts:")
        for m in metrics:
            count = aggregated_results[m]
            print(f"  {m}: {count} users ({count / total_users * 100:.1f}%)")

    # Combine all user results
    try:
        all_user_results_df = pd.DataFrame(user_results_all)
        all_user_results_file = os.path.join(OUTPUT_PATH, "all_user_results.csv")
        all_user_results_df.to_csv(all_user_results_file, index=False)
        print(f"Saved all user results to {all_user_results_file}")
    except Exception as e:
        print(f"Warning: Could not save combined user results: {e}")
        print(
            "Individual batch results are still available in the user_results directory"
        )

    formatted_results = {}
    for m in metrics:
        c = aggregated_results[m]
        perc = (c / total_users) * 100 if total_users > 0 else 0
        formatted_results[m] = (c, perc)

    return formatted_results, total_users


def run_analysis_on_sherlock():
    formatted_results, total_users = analyze_dataset_on_sherlock()

    print(f"\nResults for dataset ({total_users} users):")
    print("-" * 50)
    for metric, (count, percentage) in formatted_results.items():
        print(f"{metric}: {count} users ({percentage:.1f}%)")

    visualize_results(
        formatted_results,
        total_users,
        title=(
            f"Valid Window Counts (7-day threshold={THRESHOLDS['7day']}, "
            f"14-day threshold={THRESHOLDS['14day']})"
        ),
    )
    plt.savefig(os.path.join(OUTPUT_PATH, "sherlock_missing_data_analysis.png"))
    plt.close()

    results_df = pd.DataFrame(
        {
            "metric": list(formatted_results.keys()),
            "count": [formatted_results[m][0] for m in formatted_results],
            "percentage": [formatted_results[m][1] for m in formatted_results],
        }
    )
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    results_df.to_csv(os.path.join(OUTPUT_PATH, "sherlock_results.csv"), index=False)
    print(
        f"Saved aggregated results to {os.path.join(OUTPUT_PATH, 'sherlock_results.csv')}"
    )

    return formatted_results, total_users
