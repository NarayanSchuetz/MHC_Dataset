from typing import Callable
from constants import FileType, HKQuantityType

import pandas as pd
import numpy as np  # ensure numpy is imported


class FilterFactory:
    """Factory class for creating filters."""

    @staticmethod
    def create_filter(filter_type: str) -> Callable:
        """Create a filter based on the filter type.
        
        Args:
            filter_type: The type of filter to create, matching FileType enum values
            
        Returns:
            A callable filter function
            
        Raises:
            ValueError: If the filter type is unknown
            NotImplementedError: For all current filter types as they are not yet implemented
        """
        if filter_type == FileType.HEALTHKIT.value:
            def healthkit_filter(df: pd.DataFrame) -> pd.DataFrame:
                df = unequal_timezone_offsets(df)
                df = negative_duration(df)
                df = long_events(df)
                df = step_filter(df)
                df = active_energy_burned_filter(df)
                df = distance_walking_running_filter(df)
                df = distance_cycling_filter(df)
                df = apple_stand_time_filter(df)
                df = heart_rate_filter(df)
                return df
            return healthkit_filter
        
        elif filter_type == FileType.WORKOUT.value:
            def workout_filter(df: pd.DataFrame) -> pd.DataFrame:
                df = unequal_timezone_offsets(df)
                df = negative_duration(df)
                df = long_events(df)
                return df
            return workout_filter
        
        elif filter_type == FileType.SLEEP.value:
            def sleep_filter(df: pd.DataFrame) -> pd.DataFrame:
                df = unequal_timezone_offsets(df)
                df = negative_duration(df)
                df = long_events(df)
                return df
            return sleep_filter

        elif filter_type == FileType.MOTION.value:
            def motion_filter(df: pd.DataFrame) -> pd.DataFrame:
                df = negative_duration(df)
                df = long_events(df)
                return df
            return motion_filter
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        

def unequal_timezone_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows where start and end timezone offsets don't match within the same row.
    
    Args:
        df: The input DataFrame
        
    Returns:
        A filtered DataFrame with only rows where timezone offsets match
    """
    if df.empty:
        return df
    
    # Check if both start and end timezone offsets exist
    has_end_offset = 'endTime_timezone_offset' in df.columns
    
    if has_end_offset:
        # Keep only rows where start and end offsets match
        return df[df['startTime_timezone_offset'] == df['endTime_timezone_offset']]
    
    # If only start offset exists, return original DataFrame
    return df


def negative_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows where the duration is negative.
    
    Args:
        df: The input DataFrame
        
    Returns:
        A filtered DataFrame with only rows where duration is positive
    """
    if df.empty:
        return df
    
    # Check if both start and end times exist
    has_end_time = 'endTime' in df.columns
    
    if has_end_time:
        # Calculate duration and keep only positive durations
        return df[df['endTime'] >= df['startTime']]
    
    # If no end time exists, return original DataFrame
    return df


def long_events(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows where the duration is too long.
    
    Args:
        df: The input DataFrame
        
    Returns:
        A filtered DataFrame with only rows where duration is less than or equal to 24 hours
    """
    __MAX_DURATION_HOURS = 24

    if df.empty:
        return df
    
    # Check if both start and end times exist
    has_end_time = 'endTime' in df.columns
    
    if has_end_time:
        # Calculate duration in hours and keep only events <= 24 hours
        duration_hours = (pd.to_datetime(df['endTime']) - 
                        pd.to_datetime(df['startTime'])).dt.total_seconds() / 3600
        return df[duration_hours <= __MAX_DURATION_HOURS]
    
    # If no end time exists, return original DataFrame
    return df


def step_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and process step count data from HealthKit while handling non‐unique indexes.

    This version avoids Pandas' automatic alignment by performing all computations
    on the underlying NumPy arrays. Duplicate index labels are preserved and the per‐row
    decisions are based solely on row order.

    Args:
        df: The input DataFrame containing HealthKit data
        
    Returns:
        The DataFrame with invalid step count records removed.
    """
    __MAX_STEPS_PER_SECOND = 5

    if df.empty:
        return df

    # Determine which rows are step count records.
    # Using .to_numpy() bypasses alignment by index.
    step_mask = (df["type"] == HKQuantityType.HKQuantityTypeIdentifierStepCount.value).to_numpy()
    if not step_mask.any():
        return df

    # Assume that all rows start as valid.
    valid = np.ones(len(df), dtype=bool)

    # Work for step records: use integer positional indexing (via .iloc) so that duplicate
    # indexes do not cause alignment issues.
    step_indices = np.flatnonzero(step_mask)

    # Compute durations (in seconds) using the positional rows
    start_times = pd.to_datetime(df.iloc[step_indices]["startTime"]).to_numpy().astype("datetime64[s]")
    end_times = pd.to_datetime(df.iloc[step_indices]["endTime"]).to_numpy().astype("datetime64[s]")
    durations = (end_times - start_times).astype("timedelta64[s]").astype(float)

    # Compute the steps per second rate for the step records.
    step_values = df.iloc[step_indices]["value"].to_numpy(dtype=float)
    steps_per_second = step_values / durations

    # Apply the condition: value must be non-negative and rate must be below the threshold.
    valid[step_indices] = (step_values >= 0) & (steps_per_second <= __MAX_STEPS_PER_SECOND)

    # Finally, use .iloc to select the rows by position.
    return df.iloc[valid]


def distance_walking_running_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and process distance data from HealthKit.

    Args:
        df: The input DataFrame containing HealthKit data

    Returns:
        The full DataFrame with invalid distance records removed.
        Non-distance records are returned unchanged.
    """
    __MAX_DISTANCE_PER_SECOND = 10  # meters/second

    if df.empty:
        return df

    # Convert to NumPy array to bypass index alignment issues.
    distance_mask = (df['type'] == HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value).to_numpy()
    if not distance_mask.any():
        return df

    valid = np.ones(len(df), dtype=bool)
    distance_indices = np.flatnonzero(distance_mask)

    # Calculate durations using positional indexing.
    start_times = pd.to_datetime(df.iloc[distance_indices]['startTime']).to_numpy().astype('datetime64[s]')
    end_times = pd.to_datetime(df.iloc[distance_indices]['endTime']).to_numpy().astype('datetime64[s]')
    durations = (end_times - start_times).astype('timedelta64[s]').astype(float)
    
    # Get values and compute the distance per second rate.
    distance_values = df.iloc[distance_indices]['value'].to_numpy(dtype=float)
    distance_per_second = distance_values / durations

    # Update valid mask: valid if non-negative and rate is realistic.
    valid[distance_indices] = (distance_values >= 0) & (distance_per_second <= __MAX_DISTANCE_PER_SECOND)

    return df.iloc[valid]


def active_energy_burned_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and process Active Energy Burned data from HealthKit.

    Args:
        df: The input DataFrame containing HealthKit data.
        
    Returns:
        The DataFrame with invalid Active Energy Burned records removed.
    """
    __MAX_ENERGY_PER_SECOND = 250

    if df.empty:
        return df

    energy_mask = (df['type'] == HKQuantityType.HKQuantityTypeIdentifierActiveEnergyBurned.value).to_numpy()
    if not energy_mask.any():
        return df

    valid = np.ones(len(df), dtype=bool)
    energy_indices = np.flatnonzero(energy_mask)

    # Check if essential time columns are missing for rate calculation
    if 'endTime' not in df.columns or 'startTime' not in df.columns:
        print("Warning: 'startTime' or 'endTime' column missing. Filtering out ALL Active Energy records as rate cannot be calculated.")
        valid[energy_indices] = False
        return df.iloc[valid] # Return early as no further processing needed for energy records

    # Proceed with rate calculation only if both columns exist
    start_times = pd.to_datetime(df.iloc[energy_indices]['startTime']).to_numpy().astype('datetime64[s]')
    end_times = pd.to_datetime(df.iloc[energy_indices]['endTime']).to_numpy().astype('datetime64[s]')
    durations = (end_times - start_times).astype('timedelta64[s]').astype(float)
    
    energy_values = df.iloc[energy_indices]['value'].to_numpy(dtype=float)
    
    # Handle potential division by zero or negative duration issues before calculating rate
    # Mark records with non-positive duration as invalid
    non_positive_duration_mask = (durations <= 0)
    valid[energy_indices[non_positive_duration_mask]] = False

    # Create a mask for records that are still potentially valid (positive duration)
    potentially_valid_indices = energy_indices[durations > 0]
    
    if len(potentially_valid_indices) > 0:
        # Calculate rate only for records with positive duration
        potentially_valid_durations = durations[durations > 0]
        potentially_valid_values = energy_values[durations > 0]
        energy_rate = potentially_valid_values / potentially_valid_durations
        
        # Apply the rate limit and non-negative value check to potentially valid records
        rate_check_passed = (potentially_valid_values >= 0) & (energy_rate <= __MAX_ENERGY_PER_SECOND)
        
        # Update the main valid array only for those checked
        valid[potentially_valid_indices] = rate_check_passed

    return df.iloc[valid]


def distance_cycling_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and process distance data for cycling from HealthKit.

    Args:
        df: The input DataFrame containing HealthKit data.
        
    Returns:
        The DataFrame with invalid cycling distance records removed.
    """
    __MAX_DISTANCE_PER_SECOND = 25

    if df.empty:
        return df

    cycling_mask = (df['type'] == HKQuantityType.HKQuantityTypeIdentifierDistanceCycling.value).to_numpy()
    if not cycling_mask.any():
        return df

    valid = np.ones(len(df), dtype=bool)
    cycling_indices = np.flatnonzero(cycling_mask)

    if 'endTime' in df.columns:
        start_times = pd.to_datetime(df.iloc[cycling_indices]['startTime']).to_numpy().astype('datetime64[s]')
        end_times = pd.to_datetime(df.iloc[cycling_indices]['endTime']).to_numpy().astype('datetime64[s]')
        durations = (end_times - start_times).astype('timedelta64[s]').astype(float)

        distance_values = df.iloc[cycling_indices]['value'].to_numpy(dtype=float)
        distance_rate = distance_values / durations

        valid[cycling_indices] = (distance_values >= 0) & (distance_rate <= __MAX_DISTANCE_PER_SECOND)
    else:
        valid[cycling_indices] = df.iloc[cycling_indices]['value'].to_numpy(dtype=float) >= 0

    return df.iloc[valid]


def apple_stand_time_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and process Apple Stand Time data from HealthKit.

    Args:
        df: The input DataFrame containing HealthKit data.
        
    Returns:
        The DataFrame with invalid Apple Stand Time records removed.
    """
    __MAX_STAND_RATE = 1  # cannot have more than 1 minute of stand time per minute :)
    __MIN_STAND_RATE = 0  # cannot have negative stand time

    if df.empty:
        return df

    stand_mask = (df['type'] == HKQuantityType.HKQuantityTypeIdentifierAppleStandTime.value).to_numpy()
    if not stand_mask.any():
        return df

    valid = np.ones(len(df), dtype=bool)
    stand_indices = np.flatnonzero(stand_mask)

    if 'endTime' in df.columns:
        start_times = pd.to_datetime(df.iloc[stand_indices]['startTime']).to_numpy().astype('datetime64[s]')
        end_times = pd.to_datetime(df.iloc[stand_indices]['endTime']).to_numpy().astype('datetime64[s]')
        durations = (end_times - start_times).astype('timedelta64[s]').astype(float)
        
        # Convert stand time values (in minutes) to seconds.
        stand_values = df.iloc[stand_indices]['value'].to_numpy(dtype=float) * 60
        stand_rate = stand_values / durations
        
        valid[stand_indices] = (stand_values >= __MIN_STAND_RATE) & (stand_values <= __MAX_STAND_RATE) & (stand_rate <= __MAX_STAND_RATE)
    else:
        valid[stand_indices] = (df.iloc[stand_indices]['value'].to_numpy(dtype=float) >= __MIN_STAND_RATE) & (df.iloc[stand_indices]['value'].to_numpy(dtype=float) <= __MAX_STAND_RATE)

    return df.iloc[valid]


def heart_rate_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and process heart rate data from HealthKit.

    Args:
        df: The input DataFrame containing HealthKit data.
        
    Returns:
        The DataFrame with invalid heart rate records removed.
    """
    MIN_HEART_RATE = 40/60  # 40 beats per minute = 0.67 beats per second
    MAX_HEART_RATE = 200/60  # 200 beats per minute = 3.33 beats per second

    if df.empty:
        return df

    hr_mask = (df['type'] == HKQuantityType.HKQuantityTypeIdentifierHeartRate.value).to_numpy()
    if not hr_mask.any():
        return df

    valid = np.ones(len(df), dtype=bool)
    hr_indices = np.flatnonzero(hr_mask)

    hr_values = df.iloc[hr_indices]['value'].to_numpy(dtype=float)
    valid[hr_indices] = (hr_values >= MIN_HEART_RATE) & (hr_values <= MAX_HEART_RATE)

    return df.iloc[valid]