import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
import bisect
import json
import os
import argparse


LABELS = ('sleep_diagnosis1', 'happiness', 'heart_disease', 'feel_worthwhile1', 
          'feel_worthwhile2', 'feel_worthwhile3', 'Diabetes', 'Hypertension')

EXPLODE_COLS = ('heart_disease',)


def create_labelled_dataset(interval_df, label_df, labels=LABELS, explode_cols=EXPLODE_COLS):
    """
    Combine label dataset with MHC interval data by finding the closest label date to each interval. 
    
    Args:
        interval_df: DataFrame with healthCode, time_range, and file_uris columns
        label_df: DataFrame with healthCode, createdOn, and various label columns
        labels: List of label columns to process (if None, will try to infer)
        explode_cols: List of label columns that might contain string representations 
                      of lists and should be exploded. Defaults to ['heart_disease'].
        
    Returns:
        List of record dictionaries with matched labels. Each dictionary contains:
          - Original fields from interval_df (healthCode, time_range, file_uris)
          - For each matched label, a nested dictionary with:
              - label_value: The actual label value (converted to int when possible)
              - label_date: Timestamp when the label was created
          - Labels are matched by finding the closest temporal data point for each healthCode
    """
    # Preprocess label data
    if explode_cols is None:
        explode_cols = ['heart_disease'] # Default to known list-like column
    label_df, labels = _preprocess_label_data(label_df, labels, explode_cols)
    
    # Process health codes
    hcs = list(interval_df.healthCode.unique())
    global_records = []
    
    for hc in hcs:
        i_df = interval_df[interval_df.healthCode == hc].copy()
        if i_df.empty:
            continue
        
        records = i_df.to_dict('records')
        
        for label in labels:
            if label not in label_df.columns:
                continue
                
            l_df = label_df[(label_df.healthCode == hc) & (label_df[label].notna())].copy()
            if l_df.empty:
                continue
                
            dates = l_df.createdOn.dt.date.astype(str)
            matches = _find_closest_dates(dates, i_df.time_range)
            
            for record, match_key in zip(records, matches):
                match_idx = matches[match_key]
                label_dict = {
                    'label_value': l_df.iloc[match_idx][label],
                    'label_date': l_df.iloc[match_idx]['createdOn']
                }

                record[label] = label_dict
                
        global_records.extend(records)
    
    return global_records

def _preprocess_label_data(label_df, labels=None, explode_cols=None):
    """
    Preprocess label data for processing.
    
    Args:
        label_df: DataFrame with label data
        labels: List of label columns to process (if None, will infer)
        explode_cols: List of label columns to attempt parsing as lists and exploding.
        
    Returns:
        Processed label DataFrame and list of label columns
    """
    if explode_cols is None:
        explode_cols = []

    # Ensure createdOn is datetime
    if not pd.api.types.is_datetime64_any_dtype(label_df.createdOn):
        label_df = label_df.copy()
        label_df.createdOn = pd.to_datetime(label_df.createdOn)
    
    # If no labels provided, use all columns except metadata columns
    if labels is None:
        labels = [col for col in label_df.columns 
                 if col not in ['healthCode', 'createdOn', 'file_uris']]
    
    # Process label columns to ensure they're numeric
    for label in labels:
        if label in label_df.columns:
            # Only attempt parsing/exploding for specified columns
            if label_df[label].dtype == 'object' and label in explode_cols:
                try:
                    # Apply parsing first
                    parsed_col = label_df[label].apply(_parse_string_list_safely)
                    # Check if parsing resulted in any actual lists before exploding
                    if any(isinstance(item, list) for item in parsed_col.dropna()):
                        label_df[label] = parsed_col
                        label_df = label_df.explode(label)
                    else: 
                        # If no lists found after parsing, just keep the parsed (likely scalar) results
                        label_df[label] = parsed_col
                except Exception as e:
                    print(f"Warning: Could not parse/explode column {label}. Error: {e}")
                    pass # Keep original data or potentially NaNs from parsing if it failed partially
            
            # Apply type conversion (attempt int, otherwise keep as is)
            # Ensure this happens *after* potential explosion
            if label in label_df.columns: # Check again as explode might change columns if empty
                 label_df[label] = label_df[label].apply(
                    lambda x: np.nan if pd.isna(x) else int(x) if isinstance(x, (int, float)) and not pd.isna(x) else x
                )
            
    return label_df, labels

def _parse_string_list_safely(item):
    if isinstance(item, list):
        return item
    elif pd.isna(item):
        return np.nan
    elif isinstance(item, str):
        item = item.strip()
        if not item:
            return np.nan
        
        try:
            evaluated = ast.literal_eval(item)
            if isinstance(evaluated, list):
                return evaluated
            elif item.startswith('[') and item.endswith(']'):
                return [evaluated]
            else:
                return np.nan
        except (ValueError, SyntaxError, TypeError):
            pass
        
        if item.startswith('[') and item.endswith(']'):
            content = item[1:-1].strip()
            if not content:
                return []
            try:
                parsed_list = [float(num_str) for num_str in re.split(r'\s+', content) if num_str]
                return parsed_list
            except Exception:
                return np.nan
        
        return np.nan
    else:
        return np.nan

def _find_closest_dates(dates_list, intervals_list):
    dates_dt = [datetime.strptime(date, "%Y-%m-%d") for date in dates_list]
    sorted_dates_with_indices = sorted(enumerate(dates_dt), key=lambda x: x[1])
    sorted_dates = [d for _, d in sorted_dates_with_indices]
    original_indices = [i for i, _ in sorted_dates_with_indices]
    
    result = {}
    
    for interval in intervals_list:
        start_str, end_str = interval.split('_')
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        midpoint = start_date + (end_date - start_date) / 2
        
        pos = bisect.bisect_left(sorted_dates, midpoint)
        
        if pos == 0:
            closest_idx = 0
        elif pos == len(sorted_dates):
            closest_idx = len(sorted_dates) - 1
        else:
            if abs(sorted_dates[pos] - midpoint) < abs(sorted_dates[pos-1] - midpoint):
                closest_idx = pos
            else:
                closest_idx = pos - 1
        
        original_idx = original_indices[closest_idx]
        result[interval] = original_idx
    
    return result


def _convert_numpy_to_native(obj):
    """Convert numpy values to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_native(item) for item in obj]
    elif hasattr(obj, 'item'):  # Check if it's a numpy scalar
        return obj.item()
    else:
        return obj

def _create_denormalized_df(records):
    """Create a denormalized pandas dataframe from records"""
    flattened_records = []
    for record in records:
        flat_record = {}
        for key, value in record.items():
            if isinstance(value, dict) and 'label_value' in value:
                # Flatten label dictionaries
                flat_record[f"{key}_value"] = value['label_value']
                flat_record[f"{key}_date"] = value['label_date']
            else:
                flat_record[key] = value
        flattened_records.append(flat_record)
    
    return pd.DataFrame(flattened_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process health records')
    parser.add_argument('--interval_csv', default="~/Downloads/sherlock/valid_7day_windows.csv",
                      help='Path to interval CSV file')
    parser.add_argument('--label_csv', default="~/Downloads/sherlock/combined_mhc_data.csv",
                      help='Path to labels CSV file')
    parser.add_argument('--json_output', default="~/Downloads/global_records_1.json",
                      help='Path for JSON output')
    parser.add_argument('--parquet_output', default="~/Downloads/global_records_1.parquet",
                      help='Path for Parquet output')
    
    args = parser.parse_args()
    
    # Expand user paths (~ to home directory)
    interval_path = os.path.expanduser(args.interval_csv)
    label_path = os.path.expanduser(args.label_csv)
    json_output_path = os.path.expanduser(args.json_output)
    parquet_output_path = os.path.expanduser(args.parquet_output)
    
    print(f"Loading interval data from {interval_path}")
    interval_df = pd.read_csv(interval_path)
    
    print(f"Loading label data from {label_path}")
    label_df = pd.read_csv(label_path)
    label_df.createdOn = pd.to_datetime(label_df.createdOn, format='ISO8601')

    
    print("Processing health records...")
    global_records = create_labelled_dataset(interval_df, label_df, LABELS)
    
    # Convert to JSON-serializable format
    serializable_records = _convert_numpy_to_native(global_records)
    
    # Save to JSON
    print(f"Saving JSON output to {json_output_path}")
    with open(json_output_path, 'w') as f:
        json.dump(serializable_records, f, indent=2, default=str)
    
    # Create and save denormalized dataframe
    print(f"Creating denormalized dataframe")
    denormalized_df = _create_denormalized_df(global_records)
    
    print(f"Saving Parquet output to {parquet_output_path}")
    denormalized_df.to_parquet(parquet_output_path)
    
    print(f"Processing complete. Found {len(global_records)} records.")
