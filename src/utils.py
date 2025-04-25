import pandas as pd
from constants import HKQuantityType

def convert_to_local_time(df, offset_column="startTime_timezone_offset"):
    """
    Converts the timezone of a DataFrame to local time based on the timezone offset.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    df = df.copy()
    df.index = df.index + pd.to_timedelta(df[offset_column], unit='m')
    return df


def split_intervals_at_midnight(df, start_event_col='startTime', end_event_col='endTime'):
    """
    Splits events that cross midnight into two separate events.
    If the event has a 'value' column and the 'type' is not HeartRate,
    the value is split proportionally based on the duration of the new intervals.
    
    Args:
        df: DataFrame with datetime index
        start_event_col: Column name for event start time (default: 'startTime') 
        end_event_col: Column name for event end time (default: 'endTime')
        
    Returns:
        DataFrame with midnight-crossing events split into separate rows
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    
    if start_event_col not in df.columns:
        raise ValueError(f"Column '{start_event_col}' not found in DataFrame")
    if end_event_col not in df.columns:
        raise ValueError(f"Column '{end_event_col}' not found in DataFrame")
    
    if df.empty:
        return df
        
    df = df.copy()
    
    # Find events that cross midnight
    mask = (df[start_event_col].dt.date != df[end_event_col].dt.date)
    crossing_events = df[mask].copy()
    non_crossing_events = df[~mask].copy()
    
    split_events = []
    for _, event in crossing_events.iterrows():
        # Calculate original duration
        original_start_time = event[start_event_col]
        original_end_time = event[end_event_col]
        original_duration = original_end_time - original_start_time
        original_duration_secs = original_duration.total_seconds()

        # Define split points
        midnight = pd.Timestamp(original_end_time.date()) 
        first_part_end_time = midnight - pd.Timedelta(seconds=2)
        second_part_start_time = midnight

        # Create first part
        first_part = event.copy()
        first_part[end_event_col] = first_part_end_time
        
        # Create second part
        second_part = event.copy()
        second_part[start_event_col] = second_part_start_time

        # Proportionally split value if applicable (not HeartRate and duration > 0)
        is_heart_rate = 'type' in event and event['type'] == HKQuantityType.HKQuantityTypeIdentifierHeartRate.value
        has_value = 'value' in event

        if has_value and not is_heart_rate and original_duration_secs > 0:
            first_part_duration = first_part[end_event_col] - first_part[start_event_col]
            second_part_duration = second_part[end_event_col] - second_part[start_event_col]

            proportion_first = first_part_duration.total_seconds() / original_duration_secs
            proportion_second = second_part_duration.total_seconds() / original_duration_secs
            
            # Clamp proportions to avoid potential floating point issues leading to > 1 sum
            proportion_first = max(0.0, min(1.0, proportion_first))
            proportion_second = max(0.0, min(1.0, proportion_second))

            original_value = event['value']
            first_part['value'] = original_value * proportion_first
            second_part['value'] = original_value * proportion_second
        elif has_value and not is_heart_rate and original_duration_secs <= 0:
             # Handle zero/negative duration case - assign zero value? Or keep original? Let's assign 0.
             first_part['value'] = 0
             second_part['value'] = 0
             print(f"Warning: Original duration <= 0 for event index {event.name} during split. Assigning 0 value to split parts.")
        
        # else: Value is kept as original (e.g., for HeartRate or if no 'value' column)

        split_events.extend([first_part, second_part])
        
    if split_events:
        split_df = pd.DataFrame(split_events)
        # Combine with non-crossing events and sort by start time
        result = pd.concat([non_crossing_events, split_df])
        return result.sort_values(start_event_col)
    
    return df


def split_sleep_intervals_at_midnight(df, start_col='startTime', duration_col='value'):
    """
    Splits sleep events that cross midnight into separate rows by reusing the 
    existing split_intervals_at_midnight function. This function assumes that
    the sleep DataFrame has a start time (default 'startTime') and a duration 
    column in seconds (default 'value').
    
    It first converts the DataFrame to the same format as expected by the existing
    function by computing an 'endTime = startTime + duration', then delegates the 
    splitting logic.
    
    After splitting, it recalculates the duration for each resulting interval.
    
    Args:
        df (pd.DataFrame): DataFrame containing sleep events.
        start_col (str): Name of the column with the start time. Default is 'startTime'.
        duration_col (str): Name of the column with the duration in seconds. Default is 'value'.
        
    Returns:
        pd.DataFrame: Modified DataFrame where sleep events crossing midnight are split 
                      into separate rows.
    """
    df = df.copy()
    
    # Ensure start time is a datetime type.
    if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
        df[start_col] = pd.to_datetime(df[start_col])
    
    # Compute endTime column as startTime + duration (converted from seconds)
    df['endTime'] = df[start_col] + pd.to_timedelta(df[duration_col], unit='s')
    
    # Ensure that the df index is a DatetimeIndex. If not, set it to the start time column.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = df[start_col]
    
    # Reuse the existing function to split events that cross midnight.
    split_df = split_intervals_at_midnight(df, start_event_col=start_col, end_event_col='endTime')
    
    # Recalculate the duration of each split event.
    split_df[duration_col] = (split_df['endTime'] - split_df[start_col]).dt.total_seconds()
    
    return split_df


def load_data(base_path: str, user_id: str, file_type: str) -> pd.DataFrame:
    """
    Load HealthKit data from parquet files. If the file is not found, return an empty DataFrame.
    """
    file_paths = {
        'healthkit': f"{base_path}/healthkit/private/{user_id}.parquet",
        'workout': f"{base_path}/healthkit_workout/private/healthkit_workout.parquet",
        'sleep': f"{base_path}/healthkit_sleep/private/healthkit_sleep.parquet",
        'motion': f"{base_path}/motion_collector_preprocessed/{user_id}.parquet"
    }

    file_path = file_paths.get(file_type)
    if file_path is None:
        raise ValueError(f"Invalid file_type '{file_type}'. Must be one of {list(file_paths.keys())}")

    try:
        df = pd.read_parquet(file_path)
        if file_type in ['workout', 'sleep']:
            df = df[df.healthCode == user_id]
        df.index = df.startTime
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading file at {file_path}: {e}")
        return pd.DataFrame(columns=['source', 'healthCode'], index=pd.DatetimeIndex([]))