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
            return row['startTime']
        adjusted_time = row['startTime'] + pd.Timedelta(minutes=row['startTime_timezone_offset'])
        return pd.to_datetime(adjusted_time)
    except:
        return row['startTime']


def _adjust_end_time(row):
    try:
        if pd.isna(row['endTime_timezone_offset']):
            return row['endTime']
        adjusted_time = row['endTime'] + pd.Timedelta(minutes=row['endTime_timezone_offset'])
        return pd.to_datetime(adjusted_time)
    except:
        return row['endTime']
    

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


def _get_average_values_healthkit(df_healthkit):
    values = []
    for _, df in df_healthkit.iterrows():
        start_time = df.startTime
        end_time = df.endTime
        duration = end_time - start_time
        if duration.total_seconds() <= 0:
            continue

        start_index = _get_seconds_till_midnight(start_time)
        end_index = start_index + int(duration.total_seconds())
        value_arr = np.full(24*60*60, np.nan, dtype=np.float32)
        value_arr[start_index:end_index] = df.value / duration.total_seconds()
        values.append(value_arr)
    
    if len(values) == 0:
        return np.zeros(24*60*60, dtype=np.float32)

    stacked_values = np.vstack(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        average_values = np.nanmean(stacked_values, axis=0)
    average_values = np.nan_to_num(average_values, 0)
    return average_values


def _set_time(df, file_type: FileType):
    if file_type == FileType.HEALTHKIT:
        df['startTime'] = df.apply(_adjust_start_time, axis=1).dt.tz_localize(None)
        df['endTime'] = df.apply(_adjust_end_time, axis=1).dt.tz_localize(None)
        df.index = df['startTime']

    elif file_type == FileType.MOTION:
        df['startTime'] = df.apply(_adjust_start_time, axis=1).dt.tz_localize(None)
        df['endTime'] = df.apply(_adjust_end_time_motion, axis=1).dt.tz_localize(None)
        df.index = df['startTime']

    elif file_type == FileType.WORKOUT:
        df['startTime'] = df.apply(_adjust_start_time, axis=1).dt.tz_localize(None)
        df['endTime'] = df.apply(_adjust_end_time, axis=1).dt.tz_localize(None)
        df.index = df['startTime']

    elif file_type == FileType.SLEEP:
        df['startTime'] = df.apply(_adjust_start_time, axis=1).dt.tz_localize(None)
        df.index = df['startTime']


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

    The function:
    1. Creates the output directory if it doesn't exist
    2. Applies filters based on file type before modifying timestamps or splitting intervals
    3. Processes timestamps and timezone offsets for each data type
    4. Splits multi-day intervals at midnight boundaries
    5. Groups data by date
    6. For each date with HealthKit data:
        - Generates minute-level arrays for each data type
        - Combines them into a single numpy array
        - Saves the array to a .npy file named YYYY-MM-DD.npy
    """
    # Check if metadata file exists and we're not forcing recompute
    metadata_filepath = os.path.join(output_root_dir, "metadata.parquet")
    if os.path.exists(metadata_filepath) and not force_recompute:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_root_dir, exist_ok=True)
    
    # Process each file type: adjust times and split intervals at midnight boundaries.
    for file_type in FileType:
        if file_type in skip:
            continue

        filter_fn = FilterFactory.create_filter(file_type.value)
        if file_type not in dfs:
            print(f"Skipping {file_type} because it is not in the input dictionary.")
            print(f"Input dictionary keys: {dfs.keys()}")
            raise ValueError(f"Missing synthetic data for {file_type}")

        dfs[file_type] = filter_fn(dfs[file_type])
        
        # Adjust times based on file type
        _set_time(dfs[file_type], file_type)
        
        # Split intervals: sleep uses a different split method.
        if file_type == FileType.SLEEP:
            dfs[file_type] = split_sleep_intervals_at_midnight(dfs[file_type])
        else:
            dfs[file_type] = split_intervals_at_midnight(dfs[file_type])
    
    # Create a dictionary to store daily resampled DataFrames for each file type
    dfs_dict_daily = {}
    
    for file_type in FileType:
        if file_type in skip:
            continue
        
        df = dfs[file_type]
        if df.empty:
            dfs_dict_daily[file_type] = {}
            continue
        
        # Group by date (using the adjusted timestamps as index)
        daily_dfs = {date: group for date, group in df.groupby(df.index.date)}
        dfs_dict_daily[file_type] = daily_dfs

    metadata_collection = []    
    for date, df_hk in dfs_dict_daily[FileType.HEALTHKIT].items():
        original_time_offset = df_hk.iloc[0]['startTime_timezone_offset']
        output_filepath = os.path.join(output_root_dir, date.strftime("%Y-%m-%d") + ".npy")
        if os.path.exists(output_filepath) and not force_recompute:
            print(f"Skipping {date} for output directory {output_root_dir} because it already exists.")
            continue

        if df_hk.empty:
            # we don't want to generate data where even the HealthKit data is missing (by far the most common data type)
            continue

        daily_minute_level_matrix = []

        # generate the minute-level data for all other file types
        for file_type in FileType:
            if file_type in skip:
                continue
            
            df = dfs_dict_daily[file_type].get(date, pd.DataFrame())
            # generate the minute-level data (each is a 2D array)
            minute_data = _generate_minute_level_data_factory(file_type)(df, date)
            daily_minute_level_matrix.append(minute_data)

        # Vertically stack the 2D arrays into a single 2D array.
        daily_minute_level_matrix = np.vstack(daily_minute_level_matrix)

        # Calculate     
        data_coverage = calculate_data_coverage(daily_minute_level_matrix)

        # Calculate a mask of same dimension as daily_minute_level_matrix where each values is 1 if the corresponding value in daily_minute_level_matrix is not 0 or nan.
        data_mask = np.ones_like(daily_minute_level_matrix)
        data_mask[daily_minute_level_matrix == 0] = 0
        data_mask[np.isnan(daily_minute_level_matrix)] = 0

        # Reshape both arrays to have matching dimensions (2, C, 1440)
        data_mask = data_mask[np.newaxis, ...]  # Shape: (1, C, 1440)
        daily_minute_level_matrix = daily_minute_level_matrix[np.newaxis, ...]  # Shape: (1, C, 1440)
        
        # Concatenate along first dimension to get shape (2, C, 1440)
        daily_minute_level_matrix = np.concatenate([data_mask, daily_minute_level_matrix], axis=0)

        stats_df = compute_array_statistics(daily_minute_level_matrix)
        metadata_df = pd.concat([data_coverage, stats_df], axis=1)
        metadata_df.columns = ["data_coverage", "n", "sum", "sum_of_squares"]
        metadata_df["date"] = date
        metadata_df["original_time_offset"] = original_time_offset
        metadata_collection.append(metadata_df)

        np.save(output_filepath, daily_minute_level_matrix)
    
    # Only update metadata if we processed any files
    if metadata_collection:
        metadata_df = pd.concat(metadata_collection)
        metadata_df.to_parquet(os.path.join(output_root_dir, "metadata.parquet"))


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
