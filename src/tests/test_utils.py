import pandas as pd
import pytest
from src.utils import split_intervals_at_midnight, split_sleep_intervals_at_midnight


class Test_split_intevals_at_midnight:

    def test_split_event_no_midnight(self):
        """
        Test that an event that does not cross midnight is not split.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 10:00:00")],
            "endTime":   [pd.Timestamp("2023-10-09 11:00:00")]
        })
        # The DataFrame index must be a DatetimeIndex.
        df.index = pd.to_datetime(["2023-10-09 10:00:00"])
        
        result = split_intervals_at_midnight(df)
        # Expect one row as no event crosses midnight.
        assert result.shape[0] == 1
        row = result.iloc[0]
        assert row["startTime"] == pd.Timestamp("2023-10-09 10:00:00")
        assert row["endTime"] == pd.Timestamp("2023-10-09 11:00:00")

    def test_split_event_cross_midnight(self):
        """
        Test that an event crossing midnight is split into two events with proper boundaries.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 23:30:00")],
            "endTime":   [pd.Timestamp("2023-10-10 00:30:00")]
        })
        # Set the index to a DatetimeIndex.
        df.index = pd.to_datetime(["2023-10-09 23:30:00"])
        
        result = split_intervals_at_midnight(df)
        # Expect two rows after splitting.
        assert result.shape[0] == 2

        expected_midnight = pd.Timestamp("2023-10-10")
        first_expected_end = expected_midnight - pd.Timedelta(microseconds=1)
        
        # The first event should start at the original start time.
        first_event = result[result["startTime"] == pd.Timestamp("2023-10-09 23:30:00")].iloc[0]
        # The second event should start at midnight.
        second_event = result[result["startTime"] == expected_midnight].iloc[0]
        
        # First part
        assert first_event["startTime"] == pd.Timestamp("2023-10-09 23:30:00")
        assert first_event["endTime"] == first_expected_end

        # Second part
        assert second_event["startTime"] == expected_midnight
        assert second_event["endTime"] == pd.Timestamp("2023-10-10 00:30:00")

    def test_split_event_with_resample(self):
        """
        Test splitting a crossing-midnight event and then resampling the result by day 
        (using the startTime column) to ensure that the split parts fall on the correct days.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 23:30:00")],
            "endTime":   [pd.Timestamp("2023-10-10 00:30:00")]
        })
        df.index = pd.to_datetime(["2023-10-09 23:30:00"])
        
        split_df = split_intervals_at_midnight(df)
        
        split_df = split_df.copy()
        split_df.set_index("startTime", inplace=True)
        daily_counts = split_df.resample("D").size()

        # One event should begin on 2023-10-09 and one on 2023-10-10.
        assert daily_counts.loc[pd.Timestamp("2023-10-09")] == 1
        assert daily_counts.loc[pd.Timestamp("2023-10-10")] == 1


class Test_split_sleep_intervals_at_midnight:

    def test_sleep_event_no_midnight(self):
        """
        Test a sleep event that does not cross midnight remains unmodified.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 10:00:00")],
            "value": [3600.0]  # 1 hour in seconds
        })
        # Note: The sleep splitting wrapper sets the index if necessary.
        result = split_sleep_intervals_at_midnight(df)
        # Expect one row as the event does not cross midnight.
        assert result.shape[0] == 1
        row = result.iloc[0]
        expected_end = pd.Timestamp("2023-10-09 11:00:00")
        assert row["endTime"] == expected_end
        # Duration should be unchanged.
        assert pytest.approx(row["value"], rel=1e-6) == 3600.0

    def test_sleep_event_cross_midnight(self):
        """
        Test that a sleep event crossing midnight is split properly into two events.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 23:30:00")],
            "value": [3600.0]  # 1 hour in seconds yields an endTime of 00:30:00
        })
        result = split_sleep_intervals_at_midnight(df)
        # Expect two split events.
        assert result.shape[0] == 2
        expected_midnight = pd.Timestamp("2023-10-10")
        first_expected_end = expected_midnight - pd.Timedelta(microseconds=1)
        
        # Identify the two events based on their startTime.
        first_event = result[result["startTime"] == pd.Timestamp("2023-10-09 23:30:00")].iloc[0]
        second_event = result[result["startTime"] == expected_midnight].iloc[0]
        
        # Check boundaries and durations.
        assert first_event["endTime"] == first_expected_end
        # The duration of the first event is approximately 1800 seconds.
        assert pytest.approx(first_event["value"], rel=1e-6) == 1800.0

        assert second_event["endTime"] == pd.Timestamp("2023-10-10 00:30:00")
        # The duration of the second event is 1800 seconds.
        assert pytest.approx(second_event["value"], rel=1e-6) == 1800.0

    def test_sleep_event_with_resample(self):
        """
        Test that after splitting a sleep event crossing midnight and resampling by day,
        each day's event has the correct duration.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 22:45:00")],
            "value": [5400.0]  # 1.5 hours (90 minutes): ends at 2023-10-10 00:15:00
        })
        result = split_sleep_intervals_at_midnight(df)
        result = result.copy()
        result.set_index("startTime", inplace=True)
        daily_counts = result.resample("d").size()

        # Expect one event on 2023-10-09 and one on 2023-10-10.
        assert daily_counts.loc[pd.Timestamp("2023-10-09")] == 1
        assert daily_counts.loc[pd.Timestamp("2023-10-10")] == 1

        # For the first event:
        # It should run from 22:45:00 to midnight minus 1 microsecond, i.e. about 4500 seconds.
        first_event = result.loc[pd.Timestamp("2023-10-09 22:45:00")]
        # For the second event:
        # It should run from midnight to 00:15:00, i.e. 900 seconds.
        second_event = result.loc[pd.Timestamp("2023-10-10 00:00:00")]
        assert pytest.approx(first_event["value"], rel=1e-6) == 4500.0
        assert pytest.approx(second_event["value"], rel=1e-6) == 900.0
