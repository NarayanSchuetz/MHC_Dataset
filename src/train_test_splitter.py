import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
from pathlib import Path

def stratify_and_split(df, test_size=0.2, num_bins=5, random_state=42):
    """
    Performs a train-test split stratified by the number of non-NaN label types per healthCode.

    Args:
        df (pd.DataFrame): The input dataframe with 'healthCode' and label columns ending in '_value', as produced by the `labelled_dataset.py` script.
        test_size (float): The proportion of the dataset to include in the test split.
        num_bins (int): The number of bins to create for stratification based on label counts.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: A tuple containing the train DataFrame and the test DataFrame (train_df, test_df).
    """
    print("Starting stratification and split...")

    # 1. Calculate the number of distinct non-NaN label types per healthCode
    label_value_cols = [col for col in df.columns if col.endswith('_value')]
    if not label_value_cols:
        raise ValueError("No label columns found ending with '_value'. Cannot stratify.")

    # Create a boolean dataframe indicating where labels are non-NaN
    label_present_df = df[['healthCode'] + label_value_cols].copy()
    for col in label_value_cols:
        label_present_df[col] = label_present_df[col].notna()

    # Group by healthCode and count how many label *types* have at least one True value
    non_nan_counts = label_present_df.groupby('healthCode')[label_value_cols].any().sum(axis=1)
    non_nan_counts = non_nan_counts.reset_index(name='non_nan_label_count')

    print("Distribution of non-NaN label counts per healthCode:")
    print(non_nan_counts['non_nan_label_count'].value_counts().sort_index())

    # 2. Bin the counts
    # Using qcut ensures roughly equal numbers of healthCodes per bin
    # Handle cases where there might be too few unique counts for the requested number of bins
    try:
        non_nan_counts['count_bin'] = pd.qcut(non_nan_counts['non_nan_label_count'], q=num_bins, labels=False, duplicates='drop')
        actual_bins = non_nan_counts['count_bin'].nunique()
        if actual_bins < num_bins:
             print(f"\\nWarning: Could only create {actual_bins} bins due to data distribution.")
        else:
             print(f"\nBinned counts into {actual_bins} bins.")
    except ValueError:
        # If qcut fails (e.g., all counts are the same), just use the raw count for stratification
        print(f"\\nWarning: Could not create {num_bins} bins for counts. Using raw counts for stratification.")
        non_nan_counts['count_bin'] = non_nan_counts['non_nan_label_count']
        actual_bins = non_nan_counts['count_bin'].nunique()


    print("\nDistribution of binned counts:")
    print(non_nan_counts['count_bin'].value_counts(normalize=True).sort_index())

    # 3. Perform stratified split on the healthCodes based on the bins
    # Ensure we only split healthCodes that were part of the count calculation
    valid_healthcodes = non_nan_counts['healthCode']
    stratify_labels = non_nan_counts['count_bin']

    # Handle cases where stratification might not be possible (e.g., only one bin)
    if actual_bins <= 1:
        print("\\nWarning: Only one stratification bin. Performing regular group split instead.")
        unique_hc = valid_healthcodes.unique()
        train_hc, test_hc = train_test_split(
            unique_hc,
            test_size=test_size,
            random_state=random_state
        )
    else:
         train_hc, test_hc = train_test_split(
            valid_healthcodes,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state
        )


    print(f"\nSplit into {len(train_hc)} train healthCodes and {len(test_hc)} test healthCodes.")

    # 4. Create final train/test DataFrames
    train_df = df[df['healthCode'].isin(train_hc)].copy()
    test_df = df[df['healthCode'].isin(test_hc)].copy()

    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")

    # Verification
    common_hc = set(train_df['healthCode'].unique()) & set(test_df['healthCode'].unique())
    print(f"Common healthCodes between train/test: {len(common_hc)}")
    if len(common_hc) > 0:
        print("WARNING: Overlap detected in healthCodes between train and test sets!")

    print("Stratification and split complete.")
    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stratify and split health data by healthCode based on label completeness.')
    parser.add_argument('--input_parquet', required=True,
                        help='Path to the input denormalized Parquet file.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save the output train and test Parquet files.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split (default: 0.2).')
    parser.add_argument('--num_bins', type=int, default=5,
                        help='Number of bins for stratifying label counts (default: 5).')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42).')

    args = parser.parse_args()

    # Expand user paths and create output directory
    input_path = Path(os.path.expanduser(args.input_parquet)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    train_output_path = output_dir / "train_dataset.parquet"
    test_output_path = output_dir / "test_dataset.parquet"

    # --- Execution ---
    print(f"Loading data from: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df_full = pd.read_parquet(input_path)
    print(f"Loaded dataframe shape: {df_full.shape}")

    # Perform the split
    train_df, test_df = stratify_and_split(
        df_full,
        test_size=args.test_size,
        num_bins=args.num_bins,
        random_state=args.random_state
    )

    # Save the results
    print(f"Saving train dataset to: {train_output_path}")
    train_df.to_parquet(train_output_path, index=False)

    print(f"Saving test dataset to: {test_output_path}")
    test_df.to_parquet(test_output_path, index=False)

    print("Script finished successfully.") 

    """Example usage:
    python src/train_test_splitter.py \
    --input_parquet ~/Downloads/global_records.parquet \
    --output_dir ~/Downloads/ \
    --test_size 0.2 \
    --num_bins 5
    """
