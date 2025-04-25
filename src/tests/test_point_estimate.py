import pandas as pd
import numpy as np
import pytest
from create import _get_average_values_healthkit


def test_point_estimate_single_second():
    """Test that point estimates only affect a single second, not an entire minute."""
    # Create a sample dataframe with a point estimate
    # 8:30:15 AM - a specific second
    start_time = pd.Timestamp('2023-01-01 08:30:15')
    end_time = start_time  # Same time for point estimate
    heart_rate = 80.0  # 80 bpm
    
    df = pd.DataFrame({
        'startTime': [start_time],
        'endTime': [end_time],
        'value': [heart_rate],
        'type': ['HKQuantityTypeIdentifierHeartRate']  # Add type column for heart rate
    })
    
    # Get second-level values
    second_level_values = _get_average_values_healthkit(df)
    
    # Calculate the expected index for 8:30:15
    # 8 hours * 3600 sec/hour + 30 min * 60 sec/min + 15 sec = 30615
    expected_index = 8*3600 + 30*60 + 15
    
    # The second-level array should have the heart rate value ONLY at that specific second
    non_zero_indices = np.where(second_level_values > 0)[0]
    
    # Should only have 1 non-zero element
    assert len(non_zero_indices) == 1
    # That element should be at the expected index
    assert non_zero_indices[0] == expected_index
    # The value at that index should be the heart rate
    assert second_level_values[expected_index] == heart_rate
    
    # All other values should be 0 or NaN
    for i in range(24*60*60):
        if i != expected_index:
            assert np.isnan(second_level_values[i]) or second_level_values[i] == 0


def test_multiple_point_estimates():
    """Test that multiple point estimates each affect only their specific seconds."""
    # Create a dataframe with two point estimates, 5 seconds apart
    start_time1 = pd.Timestamp('2023-01-01 08:30:15')
    end_time1 = start_time1
    heart_rate1 = 80.0
    
    start_time2 = pd.Timestamp('2023-01-01 08:30:20')
    end_time2 = start_time2
    heart_rate2 = 85.0
    
    df = pd.DataFrame({
        'startTime': [start_time1, start_time2],
        'endTime': [end_time1, end_time2],
        'value': [heart_rate1, heart_rate2],
        'type': ['HKQuantityTypeIdentifierHeartRate', 'HKQuantityTypeIdentifierHeartRate']  # Add type column for both rows
    })
    
    # Get second-level values
    second_level_values = _get_average_values_healthkit(df)
    
    # Calculate expected indices
    expected_index1 = 8*3600 + 30*60 + 15
    expected_index2 = 8*3600 + 30*60 + 20
    
    # Should only have 2 non-zero/non-nan elements
    non_zero_indices = np.where(~np.isnan(second_level_values) & (second_level_values > 0))[0]
    assert len(non_zero_indices) == 2
    
    # Values at those indices should match heart rates
    assert second_level_values[expected_index1] == heart_rate1
    assert second_level_values[expected_index2] == heart_rate2 