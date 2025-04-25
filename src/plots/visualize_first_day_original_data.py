import os
import pandas as pd
from collections import defaultdict
from plots.original_data_plots import (
    find_iphones_and_apple_watches,
    convert_to_local_time,
    create_plots
)

def visualize_first_day_with_data(
    user_id,
    healthkit_path,
    workout_path,
    sleep_path,
    motion_path=None,
    output_dir=None,
    output_size=(1200, 1200),
    dpi=100,
    show_output=True,
    resample_by_minute=False,
    labels=False
):
    """
    Visualizes the first day where healthkit, sleep, and workout data are all available.
    
    Parameters:
    -----------
    user_id : str
        The user identifier
    healthkit_path : str
        Full path to the healthkit parquet file
    workout_path : str
        Full path to the workout parquet file
    sleep_path : str
        Full path to the sleep parquet file
    motion_path : str, optional
        Full path to the motion parquet file
    output_dir : str, optional
        Directory where the output should be saved. If None, the plot is only displayed.
    output_size : tuple, optional
        Size of the output image in pixels (width, height)
    dpi : int, optional
        DPI of the output image
    show_output : bool, optional
        Whether to display the plot
    resample_by_minute : bool, optional
        Whether to resample data to minute intervals using mean aggregation
    labels : bool, optional
        Whether to enable labels for better visibility
        
    Returns:
    --------
    str or None:
        Path to the saved file if output_dir is provided, otherwise None
    """
    print(f"Processing data for user: {user_id}")
    
    # Load data files
    try:
        df_healthkit = pd.read_parquet(healthkit_path)
        df_healthkit.index = df_healthkit.startTime
        df_healthkit.sort_index(inplace=True)
    except Exception as e:
        print(f"Error loading healthkit data: {e}")
        return None
    
    try:
        df_workout = pd.read_parquet(workout_path)
        df_workout = df_workout[df_workout.healthCode == user_id]
        df_workout.index = df_workout.startTime
        df_workout.sort_index(inplace=True)
    except Exception as e:
        print(f"Error loading workout data: {e}")
        df_workout = pd.DataFrame(columns=['source', 'healthCode'], index=pd.DatetimeIndex([]))
    
    try:
        df_sleep = pd.read_parquet(sleep_path)
        df_sleep = df_sleep[df_sleep.healthCode == user_id]
        df_sleep.index = df_sleep.startTime
        df_sleep.sort_index(inplace=True)
    except Exception as e:
        print(f"Error loading sleep data: {e}")
        df_sleep = pd.DataFrame(columns=['source', 'healthCode'], index=pd.DatetimeIndex([]))
    
    # Motion data is optional
    if motion_path:
        try:
            df_motion = pd.read_parquet(motion_path)
            df_motion.index = df_motion.startTime
            df_motion.sort_index(inplace=True)
        except Exception as e:
            print(f"Error loading motion data: {e}")
            df_motion = pd.DataFrame(columns=['source'], index=pd.DatetimeIndex([]))
    else:
        df_motion = pd.DataFrame(columns=['source'], index=pd.DatetimeIndex([]))
    
    if df_healthkit.empty:
        print("No healthkit data available")
        return None
    
    # Identify devices
    apple_watches, iphones = find_iphones_and_apple_watches(df_healthkit)
    
    # Function to categorize device
    def _to_category(x):
        if x in apple_watches:
            return "AppleWatch"
        elif x in iphones:
            return "iPhone"
        else:
            return "Unknown"
    
    # Apply device categorization
    df_healthkit["device"] = df_healthkit["source"].apply(_to_category)
    if not df_workout.empty:
        df_workout["device"] = df_workout["source"].apply(_to_category)
    if not df_sleep.empty:
        df_sleep["device"] = df_sleep["source"].apply(_to_category)
    if not df_motion.empty:
        df_motion["device"] = df_motion["source"].apply(_to_category)
    
    # Apply timezone conversion
    try:
        df_healthkit = convert_to_local_time(df_healthkit, "startTime_timezone_offset")
        if not df_workout.empty:
            df_workout = convert_to_local_time(df_workout, "startTime_timezone_offset")
        if not df_sleep.empty:
            df_sleep = convert_to_local_time(df_sleep, "startTime_timezone_offset")
        if not df_motion.empty:
            df_motion = convert_to_local_time(df_motion, "startTime_timezone_offset")
    except Exception as e:
        print(f"Error in timezone conversion: {e}")
    
    # Find the earliest date with data in all datasets
    dates_with_data = set()
    
    # Add dates from healthkit
    healthkit_dates = set(df_healthkit.index.date)
    dates_with_data.update(healthkit_dates)
    
    # Filter for dates with workout data if available
    if not df_workout.empty:
        workout_dates = set(df_workout.index.date)
        dates_with_data = dates_with_data.intersection(workout_dates) if dates_with_data else workout_dates
    
    # Filter for dates with sleep data if available
    if not df_sleep.empty:
        sleep_dates = set(df_sleep.index.date)
        dates_with_data = dates_with_data.intersection(sleep_dates) if dates_with_data else sleep_dates
    
    # If no dates have all three data types, use the first date with healthkit data
    if not dates_with_data:
        if healthkit_dates:
            first_date = min(healthkit_dates)
            print(f"No dates with all data types. Using first date with healthkit data: {first_date}")
        else:
            print("No data available")
            return None
    else:
        first_date = min(dates_with_data)
        print(f"First date with all available data: {first_date}")
    
    # Filter data for the first date
    first_date_start = pd.Timestamp(first_date)
    first_date_end = pd.Timestamp(first_date) + pd.Timedelta(days=1)
    
    # Use a more robust filtering approach
    df_healthkit_day = df_healthkit[df_healthkit.index.date == first_date]
    
    if not df_workout.empty:
        df_workout_day = df_workout[df_workout.index.date == first_date]
    else:
        df_workout_day = df_workout
        
    if not df_sleep.empty:
        df_sleep_day = df_sleep[df_sleep.index.date == first_date]
    else:
        df_sleep_day = df_sleep
        
    if not df_motion.empty:
        df_motion_day = df_motion[df_motion.index.date == first_date]
    else:
        df_motion_day = df_motion
    
    # Create output filepath if directory is provided
    filepath = None
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, f"{user_id}_{first_date}_plots.png")
    
    # Create the visualization
    create_plots(
        df_healthkit=df_healthkit_day,
        df_sleep=df_sleep_day,
        df_workout=df_workout_day,
        df_motion=df_motion_day,
        filepath=filepath,
        dpi=dpi,
        output_size=output_size,
        show_output=show_output,
        resample_by_minute=resample_by_minute,
        labels=labels
    )
    
    print(f"Successfully processed data for user: {user_id}")
    return filepath


if __name__ == "__main__":
    # Example usage
    user_id = "Wmzvhl88mFNfAXCDqy-dUR9Y"
    healthkit_path = "/Users/narayanschuetz/tmp_data/MHC_healthkit/Wmzvhl88mFNfAXCDqy-dUR9Y.parquet"
    workout_path = "/Users/narayanschuetz/tmp_data/healthkit_workout.parquet"
    sleep_path = "/Users/narayanschuetz/tmp_data/healthkit_sleep.parquet"
    motion_path = None  # Optional, set to None if not available
    output_dir = "/Users/narayanschuetz/tmp_data/"
    
    visualize_first_day_with_data(
        user_id=user_id,
        healthkit_path=healthkit_path,
        workout_path=workout_path,
        sleep_path=sleep_path,
        motion_path=motion_path,
        output_dir=output_dir,
        show_output=True,
        labels=True,  # Enable labels for better visibility
        resample_by_minute=False  # Enable minute-level resampling
    ) 