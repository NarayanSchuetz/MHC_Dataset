import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List 
from tqdm.auto import tqdm # Optional: for progress bar


def calculate_standardization_from_files(metadata_filepaths: List[str]) -> pd.DataFrame:
    """
    Calculates the overall mean and standard deviation for each feature 
    by aggregating statistics from a provided list of metadata files.

    The metadata files are expected to have been created by the 
    `create_dataset` function (or similar), containing columns 'n', 'sum', 
    'sum_of_squares', and an index representing the feature identifier 
    (expected to be named 'feature_index').

    Args:
        metadata_filepaths: A list of strings, where each string is the full 
                             path to a metadata file (e.g., a .parquet file).

    Returns:
        A pandas DataFrame indexed by 'feature_index' with columns 'mean' 
        and 'std_dev' representing the calculated standardization parameters 
        across all provided files. Returns an empty DataFrame if the list 
        is empty, no files could be processed, or they lack the required 
        columns/index.
    """
    if not metadata_filepaths:
        print("Warning: Received an empty list of metadata file paths.")
        return pd.DataFrame(columns=['mean', 'std_dev'])

    # Use defaultdict to easily accumulate sums per feature index
    aggregated_stats = defaultdict(lambda: {'total_n': 0.0, 'total_sum': 0.0, 'total_sum_of_squares': 0.0})

    print(f"Processing {len(metadata_filepaths)} metadata files. Aggregating statistics...")
    # Use tqdm if installed for a progress bar, otherwise just iterate
    file_iterator = tqdm(metadata_filepaths) if 'tqdm' in globals() else metadata_filepaths
    
    for file_path in file_iterator:
        try:
            df = pd.read_parquet(file_path)

            # Verify necessary columns and index name
            required_cols = {'n', 'sum', 'sum_of_squares'}
            if not required_cols.issubset(df.columns):
                print(f"Warning: Skipping {file_path}. Missing required columns: {required_cols - set(df.columns)}")
                continue
                
            # Check for 'feature_index' either as index name or column
            if df.index.name != 'feature_index':
                 if 'feature_index' in df.columns:
                     # Promote 'feature_index' column to be the index
                     df = df.set_index('feature_index')
                 else:
                    # Try using the existing index if it's unnamed, hoping it's the feature index
                    if df.index.name is None:
                        print(f"Warning: Index name in {file_path} is not 'feature_index'. Assuming the unnamed index represents features.")
                        df.index.name = 'feature_index' # Assign the expected name
                    else:
                        print(f"Warning: Skipping {file_path}. Index name is '{df.index.name}' (expected 'feature_index') and column not found.")
                        continue

            # Ensure data types are appropriate for summation
            try:
                df[list(required_cols)] = df[list(required_cols)].astype(float)
            except ValueError as e:
                 print(f"Warning: Skipping {file_path}. Could not convert required columns to float: {e}")
                 continue


            # Iterate through features in the current file and add to totals
            for feature_idx, row in df.iterrows():
                 # Check for NaN values in stats before aggregating
                 if pd.isna(row['n']) or pd.isna(row['sum']) or pd.isna(row['sum_of_squares']):
                     # Optionally print a warning about NaN stats, or just skip
                     # print(f"Warning: Found NaN statistics for feature {feature_idx} in {file_path}. Skipping this row.")
                     continue
                 
                 aggregated_stats[feature_idx]['total_n'] += row['n']
                 aggregated_stats[feature_idx]['total_sum'] += row['sum']
                 aggregated_stats[feature_idx]['total_sum_of_squares'] += row['sum_of_squares']

        except FileNotFoundError:
            print(f"Error: File not found {file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}. Skipping.")
            continue # Skip to the next file on error

    print("Aggregation complete. Calculating final standardization parameters...")
    
    results = []
    for feature_idx, stats in aggregated_stats.items():
        total_n = stats['total_n']
        total_sum = stats['total_sum']
        total_sum_of_squares = stats['total_sum_of_squares']

        if total_n > 0:
            mean = total_sum / total_n
            # Ensure variance is non-negative due to potential floating point issues
            # Also handle cases where sum_of_squares might be slightly less than sum^2/n due to precision
            variance_raw = (total_sum_of_squares / total_n) - (mean ** 2)
            variance = max(0, variance_raw) 
            std_dev = np.sqrt(variance)
            
            # Optional: Add a check for extremely small variance close to zero
            # if variance < 1e-10: # Adjust tolerance as needed
            #     std_dev = 0.0 
                
        else:
            # Handle cases with no valid data points for a feature
            mean = np.nan 
            std_dev = np.nan
            print(f"Warning: Feature index {feature_idx} had total_n = 0 across all processed files.")

        results.append({'feature_index': feature_idx, 'mean': mean, 'std_dev': std_dev})

    if not results:
        print("No statistics were aggregated successfully from the provided files.")
        return pd.DataFrame(columns=['mean', 'std_dev'])

    # Create final DataFrame
    final_params_df = pd.DataFrame(results)
    final_params_df = final_params_df.set_index('feature_index')
    final_params_df = final_params_df.sort_index() # Sort by feature index for consistency

    print("Standardization parameters calculated.")
    return final_params_df

def save_standardization_params(params_df, output_path):
    """
    Save standardization parameters to a CSV file.
    
    Args:
        params_df (pd.DataFrame): DataFrame containing standardization parameters
        output_path (str): Path to save the CSV file
    """
    try:
        params_df.to_csv(output_path)
        print(f"Standardization parameters saved to {output_path}")
    except Exception as e:
        print(f"Error saving standardization parameters: {e}")


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Calculate standardization parameters from statistics files.')
    parser.add_argument('--stats_dir', required=True,
                        help='Directory containing the statistics files')
    parser.add_argument('--output_file', required=True,
                        help='Path to save the standardization parameters CSV file')
    parser.add_argument('--pattern', default='*_stats.csv',
                        help='File pattern to match statistics files (default: *_stats.csv)')
    
    args = parser.parse_args()
    
    # Expand user paths
    stats_dir = Path(os.path.expanduser(args.stats_dir)).resolve()
    output_file = Path(os.path.expanduser(args.output_file)).resolve()
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Calculating standardization parameters from files in: {stats_dir}")
    print(f"Using file pattern: {args.pattern}")
    
    # Calculate standardization parameters
    params_df = calculate_standardization_params(stats_dir, args.pattern)
    
    if not params_df.empty:
        # Save parameters to CSV
        save_standardization_params(params_df, output_file)
        print(f"Successfully calculated standardization parameters for {len(params_df)} features.")
    else:
        print("No standardization parameters were calculated. Check input files and pattern.")
