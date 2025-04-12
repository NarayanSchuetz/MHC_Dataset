import os
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch
from src.generate_windows import (
    get_user_ids,
    load_user_metadata,
    find_non_overlapping_7day_windows,
    get_file_uris_for_window,
    process_user,
    WINDOW_SIZE,
    MIN_REQUIRED_DAYS,
    meets_coverage_criteria,
    get_valid_dates,
)


class TestGenerateWindows:
    def test_get_user_ids(self):
        """Test that get_user_ids correctly filters directories."""
        with patch("os.listdir") as mock_listdir, patch("os.path.isdir") as mock_isdir:
            mock_listdir.return_value = ["user1", "user2", "not_a_dir", "user3"]
            # Only user1, user2, and user3 are directories
            mock_isdir.side_effect = lambda path: "not_a_dir" not in path

            base_path = "/fake/path"
            result = get_user_ids(base_path)

            mock_listdir.assert_called_once_with(base_path)
            assert sorted(result) == sorted(
                ["user1", "user2", "user3"]
            ), "Should return only directory names"

    def test_load_user_metadata_exists(self):
        """Test loading user metadata when the file exists."""
        with patch("os.path.exists") as mock_exists, patch(
            "pandas.read_parquet"
        ) as mock_read_parquet:
            mock_exists.return_value = True
            expected_df = pd.DataFrame({"date": [datetime(2022, 1, 1)]})
            mock_read_parquet.return_value = expected_df

            base_path = "/fake/path"
            user_id = "test_user"
            result = load_user_metadata(base_path, user_id)

            expected_path = os.path.join(base_path, user_id, "metadata.parquet")
            mock_exists.assert_called_once_with(expected_path)
            mock_read_parquet.assert_called_once_with(expected_path)
            assert result.equals(expected_df), "Should return DataFrame from read_parquet"

    def test_load_user_metadata_not_exists(self):
        """Test loading user metadata when the file doesn't exist."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            base_path = "/fake/path"
            user_id = "test_user"
            result = load_user_metadata(base_path, user_id)

            expected_path = os.path.join(base_path, user_id, "metadata.parquet")
            mock_exists.assert_called_once_with(expected_path)
            assert result is None, "Should return None when file doesn't exist"

    def test_find_non_overlapping_7day_windows_empty(self):
        """Test finding windows with empty date list."""
        dates = []
        result = find_non_overlapping_7day_windows(dates)
        assert result == [], "Empty date list should return empty result"

    def test_find_non_overlapping_7day_windows_insufficient_days(self):
        """Test finding windows when no window has enough days."""
        # Create a date list with gaps, so no 7-day window has >= MIN_REQUIRED_DAYS
        start_date = datetime(2022, 1, 1)
        dates = [
            start_date,
            start_date + timedelta(days=3),
            start_date + timedelta(days=6),
            start_date + timedelta(days=9),
        ]
        result = find_non_overlapping_7day_windows(dates)
        assert result == [], "Should return empty list when no window has enough days"

    def test_find_non_overlapping_7day_windows_single_window(self):
        """Test finding windows with a single valid window."""
        start_date = datetime(2022, 1, 1)
        
        # Need to make sure we have enough days spread across the 7-day window
        # Make them explicit to ensure they're actually recognized as a valid window
        dates = [
            start_date,  # Day 1
            start_date + timedelta(days=1),  # Day 2
            start_date + timedelta(days=2),  # Day 3
            start_date + timedelta(days=3),  # Day 4
            start_date + timedelta(days=4),  # Day 5
            start_date + timedelta(days=5),  # Day 6
            start_date + timedelta(days=6),  # Day 7
        ]
        
        # The implementation looks for windows with adequate coverage
        result = find_non_overlapping_7day_windows(dates)
        
        # Check the structure of the result rather than exact values
        assert len(result) == 1, "Should find one valid window"
        assert isinstance(result[0], tuple), "Result should be a tuple of (start, end)"
        assert len(result[0]) == 2, "Result tuple should have two elements"
        
        window_start, window_end = result[0]
        assert (window_end - window_start).days == (WINDOW_SIZE - 1), "Window should be WINDOW_SIZE days"

    def test_find_non_overlapping_7day_windows_multiple_windows(self):
        """Test finding multiple non-overlapping windows."""
        start_date = datetime(2022, 1, 1)
        # Create two sets of dates with a gap in between
        window1_dates = [start_date + timedelta(days=i) for i in range(MIN_REQUIRED_DAYS)]
        window2_start = start_date + timedelta(days=WINDOW_SIZE + 3)  # After first window with a gap
        window2_dates = [window2_start + timedelta(days=i) for i in range(MIN_REQUIRED_DAYS)]
        dates = sorted(window1_dates + window2_dates)

        result = find_non_overlapping_7day_windows(dates)
        
        # Verify we found two windows
        assert len(result) == 2, "Should find two valid non-overlapping windows"
        
        # Verify structure of each window
        for window in result:
            assert isinstance(window, tuple), "Each result should be a tuple of (start, end)"
            assert len(window) == 2, "Each result tuple should have two elements"
            window_start, window_end = window
            assert (window_end - window_start).days == (WINDOW_SIZE - 1), "Each window should be WINDOW_SIZE days"
        
        # Verify windows don't overlap
        window1_end = result[0][1]
        window2_start = result[1][0]
        assert window2_start > window1_end, "Second window should start after first window ends"

    def test_get_file_uris_for_window(self):
        """Test getting file URIs for a given window."""
        with patch("os.path.exists") as mock_exists:
            # Set up mock to indicate files exist for specific days
            mock_exists.side_effect = lambda path: (
                "2022-01-01" in path
                or "2022-01-03" in path
                or "2022-01-05" in path
                or "2022-01-07" in path
            )

            base_path = "/fake/path"
            user_id = "test_user"
            window_start = datetime(2022, 1, 1)
            window_end = window_start + timedelta(days=WINDOW_SIZE - 1)

            result = get_file_uris_for_window(base_path, user_id, window_start, window_end)
            expected = [
                "test_user/2022-01-01.npy",
                "test_user/2022-01-03.npy",
                "test_user/2022-01-05.npy",
                "test_user/2022-01-07.npy",
            ]
            assert result == expected, "Should return URIs for existing files only"

    @patch("src.generate_windows.load_user_metadata")
    @patch("src.generate_windows.find_non_overlapping_7day_windows")
    @patch("src.generate_windows.get_file_uris_for_window")
    @patch("src.generate_windows.get_valid_dates")
    def test_process_user_no_valid_windows(
        self, mock_get_valid_dates, mock_get_uris, mock_find_windows, mock_load_metadata
    ):
        """Test processing a user with metadata but no valid windows."""
        mock_df = pd.DataFrame({"date": [datetime(2022, 1, 1)], "data_coverage": [15.0]})
        mock_load_metadata.return_value = mock_df
        mock_get_valid_dates.return_value = [datetime(2022, 1, 1)]
        mock_find_windows.return_value = []  # No valid windows

        base_path = "/fake/path"
        user_id = "test_user"
        result = process_user(user_id, base_path)

        mock_load_metadata.assert_called_once_with(base_path, user_id)
        mock_get_valid_dates.assert_called_once_with(mock_df, 10.0, 3)
        mock_find_windows.assert_called_once_with([datetime(2022, 1, 1)])
        mock_get_uris.assert_not_called()
        assert result == [], "Should return empty list when no valid windows found"

    @patch("src.generate_windows.load_user_metadata")
    @patch("src.generate_windows.find_non_overlapping_7day_windows")
    @patch("src.generate_windows.get_file_uris_for_window")
    @patch("src.generate_windows.get_valid_dates")
    def test_process_user_with_valid_windows(
        self, mock_get_valid_dates, mock_get_uris, mock_find_windows, mock_load_metadata
    ):
        """Test processing a user with valid windows."""
        # Setup mock metadata
        mock_df = pd.DataFrame({
            "date": [datetime(2022, 1, i) for i in range(1, 8)],
            "data_coverage": [15.0 for _ in range(7)]
        })
        mock_load_metadata.return_value = mock_df

        # Setup valid dates after coverage filtering
        valid_dates = [datetime(2022, 1, i) for i in range(1, 8)]
        mock_get_valid_dates.return_value = valid_dates

        # Setup mock windows
        window_start = datetime(2022, 1, 1)
        window_end = window_start + timedelta(days=WINDOW_SIZE - 1)
        mock_find_windows.return_value = [(window_start, window_end)]

        # Setup mock file URIs
        file_uris = ["test_user/2022-01-01.npy", "test_user/2022-01-02.npy"]
        mock_get_uris.return_value = file_uris

        base_path = "/fake/path"
        user_id = "test_user"
        min_channel_coverage = 10.0
        min_channels_with_data = 3
        
        result = process_user(user_id, base_path, min_channel_coverage, min_channels_with_data)

        expected = [{
            "healthCode": user_id,
            "time_range": "2022-01-01_2022-01-07",
            "file_uris": file_uris
        }]

        mock_load_metadata.assert_called_once_with(base_path, user_id)
        mock_get_valid_dates.assert_called_once_with(mock_df, min_channel_coverage, min_channels_with_data)
        mock_find_windows.assert_called_once_with(valid_dates)
        mock_get_uris.assert_called_once_with(base_path, user_id, window_start, window_end)
        assert result == expected, "Should return expected results for valid windows"

    @patch("src.generate_windows.load_user_metadata")
    @patch("src.generate_windows.find_non_overlapping_7day_windows")
    @patch("src.generate_windows.get_file_uris_for_window")
    def test_process_user_no_metadata(
        self, mock_get_uris, mock_find_windows, mock_load_metadata
    ):
        """Test processing a user with no metadata."""
        mock_load_metadata.return_value = None

        base_path = "/fake/path"
        user_id = "test_user"
        result = process_user(user_id, base_path)

        mock_load_metadata.assert_called_once_with(base_path, user_id)
        mock_find_windows.assert_not_called()
        mock_get_uris.assert_not_called()
        assert result == [], "Should return empty list when no metadata found"

    @patch("src.generate_windows.load_user_metadata")
    @patch("src.generate_windows.find_non_overlapping_7day_windows")
    @patch("src.generate_windows.get_file_uris_for_window")
    def test_process_user_empty_metadata(
        self, mock_get_uris, mock_find_windows, mock_load_metadata
    ):
        """Test processing a user with empty metadata."""
        mock_load_metadata.return_value = pd.DataFrame(columns=["date"])

        base_path = "/fake/path"
        user_id = "test_user"
        result = process_user(user_id, base_path)

        mock_load_metadata.assert_called_once_with(base_path, user_id)
        mock_find_windows.assert_not_called()
        mock_get_uris.assert_not_called()
        assert result == [], "Should return empty list when metadata is empty"

    def test_meets_coverage_criteria_no_criteria(self):
        """Test meets_coverage_criteria when no criteria are specified."""
        day_data = pd.DataFrame({"data_coverage": [5.0, 15.0, 0.0]})
        result = meets_coverage_criteria(day_data)
        assert result is True, "Should return True when no criteria are specified"

    def test_meets_coverage_criteria_min_coverage_met(self):
        """Test meets_coverage_criteria when min_channel_coverage is met."""
        day_data = pd.DataFrame({"data_coverage": [5.0, 15.0, 0.0]})
        # Sum is 20.0, so threshold of 19.0 should pass
        result = meets_coverage_criteria(day_data, min_channel_coverage=19.0)
        assert result is True, "Should return True when total data_coverage sum meets the minimum"

    def test_meets_coverage_criteria_min_coverage_not_met(self):
        """Test meets_coverage_criteria when min_channel_coverage is not met."""
        day_data = pd.DataFrame({"data_coverage": [5.0, 8.0, 0.0]})
        # Sum is 13.0, so threshold of 14.0 should fail
        result = meets_coverage_criteria(day_data, min_channel_coverage=14.0)
        assert result is False, "Should return False when total data_coverage sum doesn't meet the minimum"

    def test_meets_coverage_criteria_min_channels_met(self):
        """Test meets_coverage_criteria when min_channels_with_data is met."""
        day_data = pd.DataFrame({"data_coverage": [5.0, 15.0, 0.0]})
        result = meets_coverage_criteria(day_data, min_channels_with_data=2)
        assert result is True, "Should return True when min_channels_with_data is met"

    def test_meets_coverage_criteria_min_channels_not_met(self):
        """Test meets_coverage_criteria when min_channels_with_data is not met."""
        day_data = pd.DataFrame({"data_coverage": [5.0, 0.0, 0.0]})
        result = meets_coverage_criteria(day_data, min_channels_with_data=2)
        assert result is False, "Should return False when min_channels_with_data is not met"

    def test_meets_coverage_criteria_both_criteria_met(self):
        """Test meets_coverage_criteria when both criteria are met."""
        day_data = pd.DataFrame({"data_coverage": [5.0, 15.0, 0.0]})
        # Sum is 20.0 and 2 channels have data
        result = meets_coverage_criteria(day_data, min_channel_coverage=20.0, min_channels_with_data=2)
        assert result is True, "Should return True when both total coverage and channel count criteria are met"

    def test_meets_coverage_criteria_one_criterion_not_met(self):
        """Test meets_coverage_criteria when one criterion is not met."""
        # Coverage criterion met, but channels criterion not met
        day_data = pd.DataFrame({"data_coverage": [15.0, 0.0, 0.0]})
        # Sum is 15.0 but only 1 channel has data
        result = meets_coverage_criteria(day_data, min_channel_coverage=15.0, min_channels_with_data=2)
        assert result is False, "Should return False when min_channels_with_data is not met despite total coverage being met"
        
        # Channels criterion met, but coverage criterion not met
        day_data = pd.DataFrame({"data_coverage": [5.0, 8.0, 0.0]})
        # Sum is 13.0 and 2 channels have data
        result = meets_coverage_criteria(day_data, min_channel_coverage=15.0, min_channels_with_data=2)
        assert result is False, "Should return False when total coverage is not met despite min_channels_with_data being met"

    def test_meets_coverage_criteria_empty_data(self):
        """Test meets_coverage_criteria with empty data."""
        day_data = pd.DataFrame({"data_coverage": []})
        # With no criteria
        result = meets_coverage_criteria(day_data)
        assert result is True, "Should return True with empty data and no criteria"
        
        # With min_channel_coverage
        result = meets_coverage_criteria(day_data, min_channel_coverage=10.0)
        assert result is False, "Should return False with empty data and min_channel_coverage"
        
        # With min_channels_with_data
        result = meets_coverage_criteria(day_data, min_channels_with_data=1)
        assert result is False, "Should return False with empty data and min_channels_with_data"

    @patch("src.generate_windows.meets_coverage_criteria")
    def test_get_valid_dates(self, mock_meets_criteria):
        """Test get_valid_dates filters dates correctly based on coverage criteria."""
        from src.generate_windows import get_valid_dates
        
        # Setup dates and mock criteria responses
        date1 = pd.Timestamp("2022-01-01")
        date2 = pd.Timestamp("2022-01-02")
        date3 = pd.Timestamp("2022-01-03")
        
        metadata_df = pd.DataFrame({
            "date": [date1, date1, date1, date2, date2, date3, date3],
            "data_coverage": [5.0, 15.0, 0.0, 5.0, 8.0, 12.0, 20.0]
        })
        
        # Configure mock to return True for date1 and date3, False for date2
        def side_effect(day_data, min_channel_coverage, min_channels_with_data):
            date = day_data.iloc[0]['date']
            return date in [date1, date3]
        
        mock_meets_criteria.side_effect = side_effect
        
        # Call the function
        min_channel_coverage = 10.0
        min_channels_with_data = 2
        result = get_valid_dates(metadata_df, min_channel_coverage, min_channels_with_data)
        
        # Verify correct dates were returned
        assert sorted(result) == sorted([date1, date3]), "Should return only dates that meet criteria"
        
        # Verify the mock was called correctly
        assert mock_meets_criteria.call_count == 3, "Should call meets_coverage_criteria once per unique date"
        
    def test_get_valid_dates_integration(self):
        """Integration test for get_valid_dates with actual meets_coverage_criteria function."""
        from src.generate_windows import get_valid_dates
        
        date1 = pd.Timestamp("2022-01-01")
        date2 = pd.Timestamp("2022-01-02")
        date3 = pd.Timestamp("2022-01-03")
        
        # Create metadata with varying coverage values
        metadata_df = pd.DataFrame({
            "date": [date1, date1, date1, date2, date2, date3, date3],
            "data_coverage": [5.0, 15.0, 0.0, 3.0, 4.0, 12.0, 20.0]
        })
        
        # Test with different criteria
        # Case 1: Only channel coverage criterion 
        # date1 sum=20.0, date2 sum=7.0, date3 sum=32.0
        result1 = get_valid_dates(metadata_df, min_channel_coverage=15.0, min_channels_with_data=None)
        assert sorted(result1) == sorted([date1, date3]), "Only dates 1 and 3 have total coverage ≥15.0"
        
        # Case 2: Only channels with data criterion (all dates should pass with threshold 2)
        result2 = get_valid_dates(metadata_df, min_channel_coverage=None, min_channels_with_data=2)
        assert sorted(result2) == sorted([date1, date2, date3]), "All dates have at least 2 channels with data"
        
        # Case 3: Both criteria
        # date1: sum=20.0, 2 channels; date2: sum=7.0, 2 channels; date3: sum=32.0, 2 channels
        result3 = get_valid_dates(metadata_df, min_channel_coverage=15.0, min_channels_with_data=2)
        assert sorted(result3) == sorted([date1, date3]), "Only dates 1 and 3 meet both criteria"
        
        # Case 4: No criteria (all dates should pass)
        result4 = get_valid_dates(metadata_df, min_channel_coverage=None, min_channels_with_data=None)
        assert sorted(result4) == sorted([date1, date2, date3]), "All dates should pass with no criteria"
