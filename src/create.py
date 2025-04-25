import pandas as pd
import numpy as np
import os
import warnings
from typing import List, Dict

from constants import HKQuantityType, FileType, MotionActivityType, HKWorkoutType
from utils import split_intervals_at_midnight, split_sleep_intervals_at_midnight
from filters import FilterFactory
from metadata import calculate_data_coverage, compute_array_statistics


def _adjust_start_time(row):
    try:
        if pd.isna(row['startTime_timezone_offset']):
            return pd.to_datetime(row['startTime'])
        adjusted_time = pd.to_datetime(row['startTime']) + pd.Timedelta(minutes=row['startTime_timezone_offset'])
        return adjusted_time
    except:
        return pd.to_datetime(row['startTime'])

def _adjust_end_time(row):
    try:
        if pd.isna(row['endTime_timezone_offset']):
            return pd.to_datetime(row['endTime'])
        adjusted_time = pd.to_datetime(row['endTime']) + pd.Timedelta(minutes=row['endTime_timezone_offset'])
        return adjusted_time
    except:
        return pd.to_datetime(row['endTime'])
    

def _adjust_end_time_motion(row):
    try:
        if pd.isna(row['startTime_timezone_offset']):
            return row['endTime']
        adjusted_time = row['endTime'] + pd.Timedelta(minutes=row['startTime_timezone_offset'])
        return pd.to_datetime(adjusted_time)
    except:
        return row['endTime']
    

def _get_seconds_till_midnight(datetime: pd.Timestamp):
    return datetime.hour*60*60 + datetime.minute*60 + datetime.second


def _convert_healthkit_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert HealthKit units to standardized units (SI where applicable).
    Returns a copy of the DataFrame with converted units.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    if df.empty or 'unit' not in df.columns:
        return df
    
    # Convert counts per second to counts per minute
    # mask_counts = df['unit'] == 'count/s'
    # df.loc[mask_counts, 'value'] *= 60
    # df.loc[mask_counts, 'unit'] = 'count/min'
    
    # Convert calories to Calories (kcal)  -- TODO: Clarify when we should convert this
    # mask_cal = df['unit'] == 'cal'
    # df.loc[mask_cal, 'value'] /= 1000
    # df.loc[mask_cal, 'unit'] = 'Cal' # Note: Using 'Cal' to represent kcal for consistency
    
    # Convert calories to Calories (kcal) - ensure consistency
    # mask_kcal = df['unit'] == 'kcal'
    # df.loc[mask_kcal, 'unit'] = 'Cal' # Map kcal also to 'Cal'

    # Convert all length measurements to meters
    mask_feet = df['unit'] == 'ft'
    df.loc[mask_feet, 'value'] *= 0.3048
    df.loc[mask_feet, 'unit'] = 'm'
    
    mask_inches = df['unit'] == 'in'
    df.loc[mask_inches, 'value'] *= 0.0254
    df.loc[mask_inches, 'unit'] = 'm'
    
    mask_cm = df['unit'] == 'cm'
    df.loc[mask_cm, 'value'] /= 100
    df.loc[mask_cm, 'unit'] = 'm'
    
    mask_km = df['unit'] == 'km'
    df.loc[mask_km, 'value'] *= 1000
    df.loc[mask_km, 'unit'] = 'm'

    mask_mi = df['unit'] == 'mi'
    df.loc[mask_mi, 'value'] *= 1609.34
    df.loc[mask_mi, 'unit'] = 'm'

    # Convert all mass measurements to kilograms
    mask_pounds = df['unit'] == 'lb'
    df.loc[mask_pounds, 'value'] *= 0.45359237
    df.loc[mask_pounds, 'unit'] = 'kg'
    
    mask_grams = df['unit'] == 'g'
    df.loc[mask_grams, 'value'] /= 1000
    df.loc[mask_grams, 'unit'] = 'kg'
    
    # Convert speed to meters per second
    mask_mph = df['unit'] == 'mph'
    df.loc[mask_mph, 'value'] *= 0.44704
    df.loc[mask_mph, 'unit'] = 'm/s'
    
    # Convert volume to liters
    mask_floz = df['unit'] == 'fl_oz'
    df.loc[mask_floz, 'value'] *= 0.0295735
    df.loc[mask_floz, 'unit'] = 'L'
    
    return df


def _get_average_values_healthkit(df_healthkit):
    values = []
    
    if df_healthkit.empty: # Early exit if DataFrame is empty
        return np.zeros(24*60*60, dtype=np.float32)

    # No unit check needed here anymore

    for _, df in df_healthkit.iterrows():
        start_time = df.startTime
        end_time = df.endTime
        duration = end_time - start_time
        
        if duration.total_seconds() < 0:  # Skip negative durations
            continue
            
        start_index = _get_seconds_till_midnight(start_time)
        duration_secs = duration.total_seconds()

        value_arr = np.full(24*60*60, np.nan, dtype=np.float32)

        is_heart_rate = df.type == HKQuantityType.HKQuantityTypeIdentifierHeartRate.value

        if duration_secs == 0:  # Handle point estimates (assign value directly) only for heart rate as it can mess up other types.
            if is_heart_rate:
                if 0 <= start_index < 24*60*60: # Ensure index is within bounds
                    value_arr[start_index] = df.value 

        else:  # Handle normal duration measurements
            duration_secs_int = int(duration_secs) # Use integer for slicing
            end_index = min(start_index + duration_secs_int, 24*60*60) # Cap end_index
            if is_heart_rate:  # Heart rate is a rate, so we need to divide by duration all others are counts
                if start_index < 24*60*60 and start_index < end_index:
                    value_arr[start_index:end_index] = df.value
            else:

                if start_index < 24*60*60 and start_index < end_index:
                    value_arr[start_index:end_index] = df.value / duration_secs 

        values.append(value_arr)
    
    if len(values) == 0:
        return np.zeros(24*60*60, dtype=np.float32)

    stacked_values = np.vstack(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        average_values = np.nanmean(stacked_values, axis=0)

    return average_values


def _set_time(df, file_type: FileType):
    try:
        # Always adjust startTime if the column exists
        if 'startTime' in df.columns and 'startTime_timezone_offset' in df.columns:
             df['startTime'] = df.apply(_adjust_start_time, axis=1).dt.tz_localize(None)
             df.index = df['startTime'] # Set index after adjusting startTime
        elif 'startTime' in df.columns:
             df.index = pd.to_datetime(df['startTime']).dt.tz_localize(None) # Ensure index is datetime
        else:
             print(f"Warning: 'startTime' column missing for {file_type.value}. Cannot set index.")
             # Handle cases without startTime appropriately, maybe return or raise error
             return # Or raise an error depending on requirements

        # Adjust endTime based on file type, checking for column existence
        if file_type == FileType.HEALTHKIT:
            if 'endTime' in df.columns and 'endTime_timezone_offset' in df.columns:
                 df['endTime'] = df.apply(_adjust_end_time, axis=1).dt.tz_localize(None)
            elif 'endTime' in df.columns:
                 # Handle case with endTime but no offset
                 df['endTime'] = pd.to_datetime(df['endTime']).dt.tz_localize(None)
            else:
                 print(f"Warning: 'endTime' column missing for {file_type.value}. Cannot adjust endTime.")
                 # Optionally default endTime, e.g., df['endTime'] = df['startTime']

        elif file_type == FileType.MOTION:
            if 'endTime' in df.columns and 'startTime_timezone_offset' in df.columns: # Motion uses startTime offset for endTime
                 df['endTime'] = df.apply(_adjust_end_time_motion, axis=1).dt.tz_localize(None)
            elif 'endTime' in df.columns:
                 df['endTime'] = pd.to_datetime(df['endTime']).dt.tz_localize(None)
            else:
                 print(f"Warning: 'endTime' column missing for {file_type.value}. Cannot adjust endTime.")

        elif file_type == FileType.WORKOUT:
            if 'endTime' in df.columns and 'endTime_timezone_offset' in df.columns:
                 df['endTime'] = df.apply(_adjust_end_time, axis=1).dt.tz_localize(None)
            elif 'endTime' in df.columns:
                 df['endTime'] = pd.to_datetime(df['endTime']).dt.tz_localize(None)
            else:
                 # Handle missing endTime for WORKOUT specifically
                 print(f"Warning: 'endTime' column missing for {file_type.value}. Defaulting endTime to startTime.")
                 df['endTime'] = df['startTime'] # Default endTime to startTime as a fallback

        # No endTime processing needed for SLEEP as it's handled differently later if needed

    except Exception as e:
        print("_"*50)
        print(f"Error processing time for {file_type.value}")
        print("Is empty:", df.empty)
        print("Columns:", df.columns)
        print(df.head())
        raise e


def _generate_minute_level_data_factory(file_type: FileType):
    if file_type == FileType.HEALTHKIT:
        return _generate_healthkit_minute_level_daily_data
    elif file_type == FileType.MOTION:
        return _generate_motion_minute_level_daily_data
    elif file_type == FileType.WORKOUT:
        return _generate_workout_minute_level_daily_data
    elif file_type == FileType.SLEEP:
        return _generate_sleep_minute_level_daily_data


def create_dataset(
    dfs: Dict[FileType, pd.DataFrame],
    output_root_dir: str,
    skip: List[FileType] = [],
    force_recompute: bool = False,
    force_recompute_metadata: bool = False,
):
    """
    Create a minute-level dataset from multiple data sources.

    This function processes data from different file types (HealthKit, Motion, Workout, Sleep) 
    and generates a minute-by-minute dataset. For each day, it creates a numpy array containing
    data from all available sources resampled to per-minute intervals.

    Args:
        dfs (Dict[FileType, pd.DataFrame]): Dictionary mapping FileType to corresponding DataFrame
            containing the raw data for that type.
        output_root_dir (str): Directory path where the processed numpy arrays will be saved.
        skip (List[FileType], optional): List of FileTypes to skip processing. Defaults to empty list.
        force_recompute (bool, optional): If True, recompute and overwrite existing output files.
            If False, skip dates that already have output files. Defaults to False.
        force_recompute_metadata (bool, optional): If True, recompute and overwrite existing metadata file.
            Defaults to False.

    The function:
    1. Creates the output directory if it doesn't exist
    2. Converts units for HealthKit data
    3. Applies filters based on file type before modifying timestamps or splitting intervals
    4. Processes timestamps and timezone offsets for each data type
    5. Splits multi-day intervals at midnight boundaries
    6. Groups data by date
    7. For each date with HealthKit data:
        - Generates minute-level arrays for each data type
        - Combines them into a single numpy array
        - Saves the array to a .npy file named YYYY-MM-DD.npy
    8. Calculates and saves metadata for the processed days.
    """

    # Check if metadata file exists and we're not forcing recompute
    metadata_filepath = os.path.join(output_root_dir, "metadata.parquet")
    if os.path.exists(metadata_filepath) and not force_recompute_metadata and not force_recompute:
        print(f"Skipping dataset and metadata creation for {output_root_dir} because metadata.parquet already exists and force flags are False.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_root_dir, exist_ok=True)
    
    # Convert units ONLY for HealthKit DataFrame before any other processing
    processed_dfs = {}
    for file_type, df in dfs.items():
        if file_type in skip:
            continue
        
        if file_type == FileType.HEALTHKIT:
            processed_dfs[file_type] = _convert_healthkit_units(df)
        else:
            # For other file types, just copy the DataFrame
            processed_dfs[file_type] = df.copy() 
            
    # Process each file type: filter, adjust times, and split intervals at midnight boundaries.
    for file_type in FileType:
        if file_type in skip:
            continue

        # Check if the file_type exists after potential skipping/unit conversion
        if file_type not in processed_dfs:
            print(f"Skipping {file_type} because it is not in the processed dictionary (likely missing input or skipped).")
            # Ensure it's handled gracefully later if needed, e.g., during daily data generation
            continue 

        filter_fn = FilterFactory.create_filter(file_type.value)
        # Apply filtering to the potentially converted DataFrame
        processed_dfs[file_type] = filter_fn(processed_dfs[file_type]) 

        if processed_dfs[file_type].empty:
            print(f"DataFrame for {file_type.value} is empty after filtering.")
            continue # Skip time/interval processing if empty
        
        # Adjust times based on file type (also sets initial index)
        _set_time(processed_dfs[file_type], file_type)
        
        # Split intervals: sleep uses a different split method.
        if file_type == FileType.SLEEP:
            processed_dfs[file_type] = split_sleep_intervals_at_midnight(processed_dfs[file_type])
        else:
            # Ensure endTime exists before splitting
            if 'endTime' in processed_dfs[file_type].columns:
                 processed_dfs[file_type] = split_intervals_at_midnight(processed_dfs[file_type])
            else:
                 print(f"Warning: 'endTime' column missing for {file_type.value}, skipping interval splitting.")
        
        # Reset index AFTER splitting to ensure grouping uses the correct date, 
        # especially for the second part of split intervals.
        if 'startTime' in processed_dfs[file_type].columns and not processed_dfs[file_type].empty:
            processed_dfs[file_type].index = pd.to_datetime(processed_dfs[file_type]['startTime'])

    # Use processed_dfs instead of dfs for the rest of the function
    dfs_dict_daily = {}
    
    for file_type in FileType:
        if file_type in skip:
            continue
        
        # Use the processed DataFrame, default to empty if not present
        df = processed_dfs.get(file_type, pd.DataFrame()) 
        if df.empty:
            dfs_dict_daily[file_type] = {}
            continue
        
        # Group by date (using the adjusted timestamps as index)
        # Ensure index is datetime before grouping
        if not pd.api.types.is_datetime64_any_dtype(df.index):
             print(f"Warning: Index for {file_type.value} is not datetime. Attempting conversion.")
             try:
                 df.index = pd.to_datetime(df.index)
             except Exception as e:
                 print(f"Error converting index for {file_type.value} to datetime: {e}. Skipping grouping.")
                 dfs_dict_daily[file_type] = {}
                 continue

        daily_dfs = {date: group for date, group in df.groupby(df.index.date)}
        dfs_dict_daily[file_type] = daily_dfs

    # Determine the set of all dates present across all data types *after* processing
    all_dates = set()
    for daily_map in dfs_dict_daily.values():
        all_dates.update(daily_map.keys())

    if not all_dates:
        print("No dates found with data after processing. Exiting dataset creation.")
        return

    print(f"Processing data for {len(all_dates)} unique dates.")
    
    metadata_collection = []    
    # Ensure we iterate through all dates that have *any* data
    for date in sorted(list(all_dates)): 
        # Check if HealthKit data exists for the day, as it's used for timezone offset and filename
        if FileType.HEALTHKIT not in skip and FileType.HEALTHKIT in dfs_dict_daily and date in dfs_dict_daily[FileType.HEALTHKIT]:
            df_hk_day = dfs_dict_daily[FileType.HEALTHKIT][date]
            # Use timezone offset from the first record of the day's HealthKit data if available
            original_time_offset = df_hk_day.iloc[0]['startTime_timezone_offset'] if not df_hk_day.empty and 'startTime_timezone_offset' in df_hk_day.columns else 0

        output_filepath = os.path.join(output_root_dir, date.strftime("%Y-%m-%d") + ".npy")

        # Generate data or load existing file
        if os.path.exists(output_filepath) and not force_recompute:
            print(f"Skipping data generation for {date} in {output_root_dir}, file exists.")
            # Load if metadata recomputation is forced, otherwise skip metadata too if file exists
            if force_recompute_metadata:
                 try:
                     daily_minute_level_matrix = np.load(output_filepath)
                 except Exception as e:
                     print(f"Error loading existing file {output_filepath}: {e}. Skipping metadata calculation for this date.")
                     continue # Skip metadata calc for this date if loading fails
            else:
                 continue # Skip metadata calculation as well if not forcing recompute
        else:
            # Check if there's any data for this date across included file types
            has_data_for_date = any(date in dfs_dict_daily.get(ft, {}) for ft in FileType if ft not in skip)
            if not has_data_for_date:
                 print(f"Skipping date {date}: No data found after processing across included file types.")
                 continue # Skip if no data exists for this date at all

            print(f"Generating data for {date}...")
            
            try:
                daily_minute_level_matrix = _generate_daily_data(dfs_dict_daily, date, skip)
                np.save(output_filepath, daily_minute_level_matrix)
            except Exception as e:
                print(f"Error generating or saving data for date {date}: {e}")
                continue # Skip to next date on error

        # Calculate metadata only if data was generated or loaded successfully for metadata recompute
        if 'daily_minute_level_matrix' in locals() and daily_minute_level_matrix is not None:
            try:
                data_coverage = calculate_data_coverage(daily_minute_level_matrix[1]) # Use data channel for coverage
                stats_df = compute_array_statistics(daily_minute_level_matrix)
                # Ensure indices match before concatenating, use stats_df index as reference
                data_coverage = data_coverage.reindex(stats_df.index) 
                metadata_day_df = pd.concat([data_coverage, stats_df], axis=1)
                metadata_day_df.columns = ["data_coverage", "n", "sum", "sum_of_squares"]
                metadata_day_df["date"] = date
                metadata_day_df["original_time_offset"] = original_time_offset
                metadata_collection.append(metadata_day_df)
                del daily_minute_level_matrix # Clean up memory
            except Exception as e:
                 print(f"Error calculating metadata for date {date}: {e}")
                 # Decide if we should continue or stop metadata processing

    # Only update metadata if we processed any files and collected metadata
    if metadata_collection:
        print(f"Saving metadata for {len(metadata_collection)} days to {metadata_filepath}")
        final_metadata_df = pd.concat(metadata_collection)
        # Add index name if it doesn't exist for clarity in parquet
        if final_metadata_df.index.name is None:
             final_metadata_df.index.name = 'feature_index' 
        final_metadata_df.to_parquet(metadata_filepath)
    elif not os.path.exists(metadata_filepath) or force_recompute_metadata:
         print("No metadata collected or generated. Metadata file might be empty or not updated.")


def _generate_healthkit_minute_level_daily_data(df_healthkit_day, date: pd.Timestamp):
    """
    Generates minute-level data for a single day from HealthKit measurements.
    
    Takes a DataFrame of HealthKit records for a single day and converts them into a 
    minute-by-minute array for each HealthKit quantity type (steps, energy, etc).
    
    The function first creates a per-second time series for the full day, then 
    resamples to minute-level averages. For any HealthKit type with no data for
    the day, that row will be filled with NaN values.

    Parameters:
        df_healthkit_day (pd.DataFrame): DataFrame with HealthKit data for one day.
            Expected to have columns: 'type', 'startTime', 'endTime', 'value'.
            The 'type' column should contain values from HKQuantityType enum.
        date (pd.Timestamp): The day to generate data for. Should be normalized to
            midnight of the desired day.

    Returns:
        np.ndarray: Array of shape (n_types, 1440) containing the minute-level data,
            where n_types is the number of HKQuantityType enum values. Each row
            corresponds to a different HealthKit quantity type in the order defined
            by the HKQuantityType enum. Values are float32.

    Raises:
        AssertionError: If input DataFrame is empty or if output shape is incorrect.
    """
    _df = df_healthkit_day.copy()

    assert not _df.empty, "The DataFrame is empty, which should not happen."

    minute_values = []
    # Create a time index at second resolution for multiplication
    time_index = pd.date_range(start=date, periods=24 * 60 * 60, freq='s')

    # Loop through each HealthKit category for the day's data.
    for hk_type in HKQuantityType:
        # Filter the current day's records for the specific HealthKit category.
        df_day_type = _df[_df.type == hk_type.value]
        if df_day_type.empty:
            minute_values.append(np.full(24 * 60, np.nan, dtype=np.float32))
            continue

        average_values = _get_average_values_healthkit(df_day_type)
        average_series = pd.Series(average_values, index=time_index)
        minute_series = average_series.resample('1min').mean()
        minute_series = minute_series.fillna(0)
        minute_values.append(minute_series.values.astype(np.float32))

    # Final matrix consists only of the HealthKit category data.
    minute_values = np.array(minute_values)
    expected_shape = (len(HKQuantityType), 24 * 60)
    assert minute_values.shape == expected_shape, (
        f"Unexpected shape: {minute_values.shape}, expected: {expected_shape}"
    )

    return minute_values


def _generate_sleep_minute_level_daily_data(df_sleep_day, date: pd.Timestamp):
    """
    Generates minute-level sleep data for a single day from HealthKit sleep intervals.
    
    Assumes the DataFrame is for a single day and has been split at midnight.
    Returns a numpy array of shape (2, 1440) where:
      - The first row corresponds to "HKCategoryValueSleepAnalysisAsleep".
      - The second row corresponds to "HKCategoryValueSleepAnalysisInBed".
    
    Each minute covered by a sleep interval is flagged as 1. If multiple records indicate
    sleep for the same minute, an OR operation is performed (i.e. the minute remains 1).
    
    Parameters:
        df_sleep_day (pd.DataFrame): DataFrame with sleep data for the day.
            Expected columns: 'startTime', 'endTime', and 'type'.
        date (pd.Timestamp): The day for which to generate the data; should represent
            the start of the day (will be normalized to midnight).
    
    Returns:
        np.ndarray: An array of shape (2, 1440):
            - Row 0: Sleep Asleep.
            - Row 1: Sleep In-Bed.
    """
    n_minutes = 24 * 60  # Total minutes in a day
    
    # Return array of nans if input DataFrame is empty
    if df_sleep_day.empty:
        return np.full((2, n_minutes), np.nan, dtype=np.float32)
    
    asleep_arr = np.zeros(n_minutes, dtype=np.float32)
    inbed_arr = np.zeros(n_minutes, dtype=np.float32)

    # Normalize the provided date to midnight
    day_start = pd.Timestamp(date).normalize()

    for _, row in df_sleep_day.iterrows():
        if pd.isna(row['startTime']) or pd.isna(row['endTime']):
            continue

        # Compute the minute offsets from day_start:
        # Use floor for the start minute (so partial minutes are included)
        # and ceil for the end minute (ensuring that even partial coverage marks the minute)
        start_offset = (row['startTime'] - day_start).total_seconds()
        end_offset = (row['endTime'] - day_start).total_seconds()
        if end_offset <= 0:
            continue

        start_min = int(start_offset // 60)
        end_min = int(np.ceil(end_offset / 60))
        # In case the end time is exactly midnight next day, cap it.
        end_min = min(end_min, n_minutes)

        if start_min >= end_min:
            continue

        if row['category value'] == "HKCategoryValueSleepAnalysisAsleep":
            asleep_arr[start_min:end_min] = 1.0
        elif row['category value'] == "HKCategoryValueSleepAnalysisInBed":
            inbed_arr[start_min:end_min] = 1.0

    result = np.stack([asleep_arr, inbed_arr])
    assert result.shape == (2, 1440), f"Unexpected shape: {result.shape}, expected: (2, 1440)"
    
    return result


def _generate_motion_minute_level_daily_data(df_motion_day, date: pd.Timestamp):
    """
    Generates minute-level daily data for motion data, one array per motion type.
    
    For each motion type (as defined in constants.MotionActivityType), this function
    creates a minute-level numpy array (shape [1440,]) representing the confidence value 
    over each minute of the day. For each motion record:
      - It fills an array at the second resolution with the record's "confidence" value 
        over the period from startTime to endTime.
      - It then resamples this second-level array to minute resolution by taking the mean.
    
    If there are no records for a motion type (or if df_motion_day is empty), the minute-level
    row for that type is filled with NaNs.
    
    Parameters:
        df_motion_day (pd.DataFrame): DataFrame with motion data for the day. Expected columns:
            'startTime', 'endTime', 'activity', and 'confidence'.
        date (pd.Timestamp): The day for which to generate the data (should be normalized to midnight).
    
    Returns:
        np.ndarray: An array with shape (len(MotionActivityType), 1440), where each row corresponds
                    to a motion type (in order of constants.MotionActivityType).
    """
    n_minutes = 24 * 60
    n_seconds = 24 * 60 * 60

    # If input is empty, return an all-NaN output with the expected shape
    if df_motion_day.empty:
        return np.full((len(MotionActivityType), n_minutes), np.nan, dtype=np.float32)
    
    # Create a time index at second resolution for resampling to minute level
    time_index = pd.date_range(start=date, periods=n_seconds, freq='s')
    motion_minute_data = []

    # Process each motion type separately
    for motion_type in MotionActivityType:
        # Filter rows corresponding to the current motion type (using the 'activity' column)
        df_type = df_motion_day[df_motion_day['activity'] == motion_type.value]
        if df_type.empty:
            # No events: output an array of NaNs for this type
            motion_minute_data.append(np.full(n_minutes, np.nan, dtype=np.float32))
        else:
            # For each row, create a second-level array and fill in the confidence value.
            sec_values_list = []
            for _, row in df_type.iterrows():
                start_time = row['startTime']
                end_time = row['endTime']
                duration = (end_time - start_time).total_seconds()
                # Skip if the duration is not positive.
                if duration <= 0:
                    continue
                # Create an array of NaNs for the day (per second)
                sec_array = np.full(n_seconds, np.nan, dtype=np.float32)
                # Compute the start index: number of seconds since midnight.
                start_index = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
                # Compute the end index, ensuring we don't exceed the day
                end_index = min(start_index + int(duration), n_seconds)
                sec_array[start_index:end_index] = row['confidence']
                sec_values_list.append(sec_array)
            
            if not sec_values_list:
                minute_series_vals = np.full(n_minutes, np.nan, dtype=np.float32)
            else:
                # Stack all the second-level arrays and compute a (per-second) mean.
                stacked = np.vstack(sec_values_list)
                # Ignore warnings if all values are NaN in some seconds.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_seconds = np.nanmean(stacked, axis=0)
                # Safety: use nan_to_num if needed (here NaNs will remain if no value).
                avg_seconds = np.nan_to_num(avg_seconds, nan=np.nan)
                # Create a Series with second-level data and resample into minute bins.
                avg_series = pd.Series(avg_seconds, index=time_index)
                minute_series_vals = avg_series.resample('1min').mean().values.astype(np.float32)
            motion_minute_data.append(minute_series_vals)
    
    motion_minute_data = np.array(motion_minute_data)
    expected_shape = (len(MotionActivityType), n_minutes)
    assert motion_minute_data.shape == expected_shape, (
        f"Unexpected shape: {motion_minute_data.shape}, expected: {expected_shape}"
    )
    return motion_minute_data


def _generate_workout_minute_level_daily_data(df_workout_day, date: pd.Timestamp):
    """
    Generates minute-level workout data for a single day from HealthKit workout records.

    For each workout type as defined in the HKWorkoutType enum, this function generates a minute-level
    binary indicator (1.0 if a workout of that type was active during that minute, 0.0 otherwise).

    If multiple rows cover the same workout event, an OR operation is performed so that the longest
    possible exercise is reported (i.e. the union of the intervals is taken).

    This implementation does not group by recordId.

    If there are no records for a given workout type (or if df_workout_day is empty), the corresponding row will be
    filled with np.nan.

    Parameters:
        df_workout_day (pd.DataFrame): DataFrame with workout data for the day.
            Expected columns: 'startTime', 'endTime', 'workoutType', and optionally 'recordId'.
        date (pd.Timestamp): The day for which to generate the data; should represent the start of the day (midnight).

    Returns:
        np.ndarray: An array of shape (n_types, 1440) where n_types is the number of workout types in HKWorkoutType.
                    Each row corresponds to a workout type as defined in HKWorkoutType.
    """
    n_minutes = 24 * 60

    # If the entire DataFrame is empty, return an array of shape (n_types, n_minutes) filled with NaNs.
    if df_workout_day.empty:
        return np.full((len(HKWorkoutType), n_minutes), np.nan, dtype=np.float32)

    # Normalize the provided date and set day's boundaries.
    day_start = pd.Timestamp(date).normalize()
    day_end = day_start + pd.Timedelta(days=1)

    result = []

    # Process each workout type as specified in the HKWorkoutType enum.
    for workout_type in HKWorkoutType:
        # Filter records matching this workout type.
        df_type = df_workout_day[df_workout_day['workoutType'] == workout_type.value]

        # If no record exists for the current type, append an array of NaNs.
        if df_type.empty:
            result.append(np.full(n_minutes, np.nan, dtype=np.float32))
            continue

        # Initialize a minute-level indicator (all zeros).
        indicator = np.zeros(n_minutes, dtype=np.float32)
        
        # Iterate over each record and update the indicator using an OR operation.
        for _, row in df_type.iterrows():
            start_time = row['startTime']
            end_time = row['endTime']

            # Skip events that do not overlap with the current day.
            if end_time <= day_start or start_time >= day_end:
                continue

            # Clip the interval to the boundaries of the day.
            interval_start = max(start_time, day_start)
            interval_end = min(end_time, day_end)

            # Convert the interval to minute indices relative to day_start.
            start_min = int((interval_start - day_start).total_seconds() // 60)
            end_min = int(np.ceil((interval_end - day_start).total_seconds() / 60))
            end_min = min(end_min, n_minutes)

            if start_min < end_min:
                # Mark these minutes as active; if multiple intervals overlap, OR operation ensures the union.
                indicator[start_min:end_min] = 1.0

        result.append(indicator)

    return np.array(result)


def create_synthetic_dfs() -> Dict[FileType, pd.DataFrame]:
    # HEALTHKIT synthetic data
    start = pd.Timestamp("2022-01-01 00:00:00")
    end = start + pd.Timedelta(hours=1)  # one-hour event
    hk_df = pd.DataFrame({
        "startTime": [start],
        "endTime": [end],
        "startTime_timezone_offset": [0],
        "endTime_timezone_offset": [0],
        "value": [3600.0],
        "type": [HKQuantityType.HKQuantityTypeIdentifierStepCount.value],
    })
    hk_df.index = pd.to_datetime(hk_df["startTime"])

    # MOTION synthetic data
    start_motion = pd.Timestamp("2022-01-01 00:10:00")
    end_motion = pd.Timestamp("2022-01-01 00:20:00")
    motion_df = pd.DataFrame({
        "startTime": [start_motion],
        "endTime": [end_motion],
        "activity": [MotionActivityType.STATIONARY.value],
        "confidence": [5.0],
        "startTime_timezone_offset": [0],
        "source": ["SyntheticMotion"],
        "appVersion": ["1.0"],
        "recordId": ["motion_dummy"],
        "healthCode": ["synthetic_motion"],
    })
    motion_df.index = pd.to_datetime(motion_df["startTime"])

    # SLEEP synthetic data
    start_sleep = pd.Timestamp("2022-01-01 22:00:00")
    end_sleep = pd.Timestamp("2022-01-01 23:00:00")
    sleep_df = pd.DataFrame({
        "startTime": [start_sleep],
        "endTime": [end_sleep],
        "type": ["HKCategoryTypeIdentifierSleepAnalysis"],
        "category value": ["HKCategoryValueSleepAnalysisAsleep"],
        "value": [3600.0],
        "unit": ["s"],
        "source": ["Synthetic"],
        "sourceIdentifier": ["dummy"],
        "appVersion": ["1.0"],
        "startTime_timezone_offset": [0],
        "healthCode": ["synthetic"],
        "recordId": ["dummy"],
    })
    sleep_df.index = pd.to_datetime(sleep_df["startTime"])

    # WORKOUT synthetic data
    start_workout = pd.Timestamp("2022-01-01 03:00:00")
    end_workout = pd.Timestamp("2022-01-01 03:30:00")
    workout_df = pd.DataFrame({
        "startTime": [start_workout],
        "endTime": [end_workout],
        "workoutType": [HKWorkoutType.HKWorkoutActivityTypeRunning.value],
        "recordId": ["dummy"],
    })
    workout_df.index = pd.to_datetime(workout_df["startTime"])

    return {
        FileType.HEALTHKIT: hk_df,
        FileType.MOTION: motion_df,
        FileType.SLEEP: sleep_df,
        FileType.WORKOUT: workout_df,
    }


def _generate_daily_data(dfs_dict_daily, date, skip):
    """Generate and process daily data matrix with mask concatenation"""
    matrix_components = []
    
    for file_type in FileType:
        if file_type in skip:
            continue
        df = dfs_dict_daily[file_type].get(date, pd.DataFrame())
        minute_data = _generate_minute_level_data_factory(file_type)(df, date)
        matrix_components.append(minute_data)
    
    stacked_matrix = np.vstack(matrix_components)
    
    # Create data mask and combine with actual data
    data_mask = np.ones_like(stacked_matrix)
    data_mask[(stacked_matrix == 0) | np.isnan(stacked_matrix)] = 0
    
    return np.concatenate([
        data_mask[np.newaxis, ...],  # Mask channel
        stacked_matrix[np.newaxis, ...]  # Data channel
    ], axis=0)
