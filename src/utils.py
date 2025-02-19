import pandas as pd

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
    
    Args:
        df: DataFrame with datetime index
        start_event: Column name for event start time (default: 'startTime') 
        end_event: Column name for event end time (default: 'endTime')
        
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
        # Create first part - original start to midnight
        midnight = pd.Timestamp(event[end_event_col].date()) 
        first_part = event.copy()
        first_part[end_event_col] = midnight - pd.Timedelta(microseconds=1)
        
        # Create second part - midnight to original end
        second_part = event.copy()
        second_part[start_event_col] = midnight
        
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