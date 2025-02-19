import pandas as pd
import numpy as np
import pytest
from datetime import timedelta
from constants import HKQuantityType
from src.filters import distance_walking_running_filter

# NOTE: These tests assume that a realistic walking/running speed is less than ~7 m/s.
# That is, if (value / duration) > 7, the record is considered spurious and should be dropped.

class TestDistanceWalkingRunningFilter:
    
    def test_empty_dataframe(self):
        """Test that an empty DataFrame returns an empty DataFrame."""
        # Including the required columns.
        columns = ['startTime', 'endTime', 'value', 'type']
        df = pd.DataFrame(columns=columns)
        filtered_df = distance_walking_running_filter(df)
        assert filtered_df.empty, "Filtered DataFrame should be empty if input is empty."

    def test_valid_entry(self):
        """
        Create a valid walking/running record.
        Duration: 10 seconds, distance: 15 m  => speed = 1.5 m/s (valid).
        """
        start_time = pd.Timestamp("2022-01-01 10:00:00")
        end_time = start_time + pd.Timedelta(seconds=10)
        df = pd.DataFrame([{
            'startTime': start_time,
            'endTime': end_time,
            'value': 15,
            'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value,
        }])
        filtered_df = distance_walking_running_filter(df)
        # The valid row should be kept.
        assert len(filtered_df) == 1, "Valid entry should be retained."
        assert filtered_df.iloc[0]['value'] == 15

    def test_invalid_entry_high_speed(self):
        """
        Create an entry with unrealistic high speed.
        Duration: 10 seconds, distance: 100 m  => speed = 10 m/s (above threshold).
        This record should be filtered out.
        """
        start_time = pd.Timestamp("2022-01-01 09:00:00")
        end_time = start_time + pd.Timedelta(seconds=10)
        df = pd.DataFrame([{
            'startTime': start_time,
            'endTime': end_time,
            'value': 1010,
            'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value,
        }])
        filtered_df = distance_walking_running_filter(df)
        assert filtered_df.empty, "High-speed entry should be filtered out."

    def test_invalid_entry_zero_duration(self):
        """
        Create an entry with zero duration.
        Start and end times are identical.
        Such an entry cannot be used for a valid rate computation and should be dropped.
        """
        start_time = pd.Timestamp("2022-01-01 10:00:00")
        # Zero duration because startTime equals endTime.
        df = pd.DataFrame([{
            'startTime': start_time,
            'endTime': start_time,
            'value': 10,
            'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value,
        }])
        filtered_df = distance_walking_running_filter(df)
        assert filtered_df.empty, "Entry with zero duration should be filtered out."

    def test_multiple_entries(self):
        """
        Create a mix of valid and invalid entries.
          - First entry: valid (1.5 m/s)
          - Second entry: invalid (11 m/s)
          - Third entry: valid (approx. 1.2 m/s)
        """
        st1 = pd.Timestamp("2022-01-01 08:00:00")
        et1 = st1 + pd.Timedelta(seconds=10)  # valid: 15 m in 10s -> 1.5 m/s
        st2 = pd.Timestamp("2022-01-01 08:10:00")
        et2 = st2 + pd.Timedelta(seconds=10)  # invalid: 100 m in 10s -> 10 m/s
        st3 = pd.Timestamp("2022-01-01 08:20:00")
        et3 = st3 + pd.Timedelta(seconds=25)  # valid: 30 m in 25s -> 1.2 m/s

        data = [
            {'startTime': st1, 'endTime': et1, 'value': 15, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
            {'startTime': st2, 'endTime': et2, 'value': 110, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
            {'startTime': st3, 'endTime': et3, 'value': 30, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
        ]
        df = pd.DataFrame(data)
        filtered_df = distance_walking_running_filter(df)
        # Expect only the valid rows to remain (first and third).
        assert len(filtered_df) == 2, "Only valid entries should be retained."
        # Check that the retained values are 15 and 30.
        assert set(filtered_df['value']) == {15, 30}

    def test_multiple_entries_mixed_types(self):
        """
        Create a mix of walking/running and other types of entries.
        Only walking/running entries should be filtered, others should remain unchanged.
        """
        st1 = pd.Timestamp("2022-01-01 08:00:00")
        et1 = st1 + pd.Timedelta(seconds=10)  # valid: 15 m in 10s -> 1.5 m/s
        st2 = pd.Timestamp("2022-01-01 08:10:00")
        et2 = st2 + pd.Timedelta(seconds=10)  # invalid: 110 m in 10s -> 11 m/s
        st3 = pd.Timestamp("2022-01-01 08:20:00")
        et3 = st3 + pd.Timedelta(seconds=25)  # valid: 30 m in 25s -> 1.2 m/s

        data = [
            {'startTime': st1, 'endTime': et1, 'value': 15, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
            {'startTime': st2, 'endTime': et2, 'value': 110, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
            {'startTime': st3, 'endTime': et3, 'value': 30, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
            {'startTime': st1, 'endTime': et1, 'value': 75, 'type': 'SomeOtherType'},
            {'startTime': st2, 'endTime': et2, 'value': 200, 'type': 'AnotherType'},
        ]
        df = pd.DataFrame(data)
        filtered_df = distance_walking_running_filter(df)
        
        # Expect valid walking/running entries (first and third) plus the two other type entries
        assert len(filtered_df) == 4, "Should retain valid walking/running entries and all other types"
        walking_running_entries = filtered_df[filtered_df['type'] == HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value]
        assert set(walking_running_entries['value']) == {15, 30}
        other_type_entries = filtered_df[filtered_df['type'] != HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value]
        assert set(other_type_entries['value']) == {75, 200}

    def test_duplicate_start_times(self):
        """
        Create entries with duplicate start times.
        All valid entries should be retained regardless of duplicate timestamps.
        """
        st1 = pd.Timestamp("2022-01-01 08:00:00")
        et1 = st1 + pd.Timedelta(seconds=10)  # valid: 15 m in 10s -> 1.5 m/s
        et2 = st1 + pd.Timedelta(seconds=20)  # valid: 35 m in 20s -> 1.75 m/s
        st3 = pd.Timestamp("2022-01-01 08:20:00")
        et3 = st3 + pd.Timedelta(seconds=25)  # valid: 30 m in 25s -> 1.2 m/s

        data = [
            {'startTime': st1, 'endTime': et1, 'value': 15, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
            {'startTime': st1, 'endTime': et2, 'value': 35, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
            {'startTime': st3, 'endTime': et3, 'value': 30, 'type': HKQuantityType.HKQuantityTypeIdentifierDistanceWalkingRunning.value},
        ]
        df = pd.DataFrame(data)
        filtered_df = distance_walking_running_filter(df)
        
        # All entries are valid (speeds < 7 m/s), so all should be retained
        assert len(filtered_df) == 3, "All valid entries should be retained, even with duplicate start times"
        assert set(filtered_df['value']) == {15, 35, 30}
        # Verify we have two entries with the same start time
        start_time_counts = filtered_df['startTime'].value_counts()
        assert start_time_counts[st1] == 2, "Should have two entries with the same start time"
