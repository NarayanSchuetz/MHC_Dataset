import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
from src.count_valid_intervals import (
    has_valid_window,
    analyze_user_data,
    process_user,
    get_user_ids,
    load_user_metadata,
)


class TestCountValidIntervals:
    def test_has_valid_window_empty_dates(self):
        """Test that an empty dates array returns False."""
        dates = np.array([])
        window_size = 7
        max_missing = 2

        result = has_valid_window(dates, window_size, max_missing)
        assert result is False, "Empty dates should return False"

    def test_has_valid_window_continuous_dates(self):
        """
        Test with a continuous sequence of dates that meets the criteria.
        7-day window with 0 missing days (well below max_missing=2).
        """
        start_date = datetime(2022, 1, 1)
        dates = pd.Series([start_date + timedelta(days=i) for i in range(7)])
        window_size = 7
        max_missing = 2

        result = has_valid_window(dates, window_size, max_missing)
        assert result is True, "Continuous 7-day window should be valid"

    def test_has_valid_window_with_allowed_missing(self):
        """
        Test with a 7-day window with exactly the allowed number of missing days (2).
        """
        # Create dates with 2 missing days in a 7-day window
        start_date = datetime(2022, 1, 1)
        dates = pd.Series(
            [
                start_date,  # Jan 1
                start_date + timedelta(days=1),  # Jan 2
                # Jan 3 missing
                start_date + timedelta(days=3),  # Jan 4
                start_date + timedelta(days=4),  # Jan 5
                # Jan 6 missing
                start_date + timedelta(days=6),  # Jan 7
            ]
        )
        window_size = 7
        max_missing = 2

        result = has_valid_window(dates, window_size, max_missing)
        assert result is True, "Window with exactly max_missing days should be valid"

    def test_has_valid_window_with_too_many_missing(self):
        """
        Test with a 7-day window with more than the allowed missing days (3 > max_missing=2).
        """
        # Create dates with 3 missing days in a 7-day window
        start_date = datetime(2022, 1, 1)
        dates = pd.Series(
            [
                start_date,  # Jan 1
                # Jan 2 missing
                start_date + timedelta(days=2),  # Jan 3
                # Jan 4 missing
                start_date + timedelta(days=4),  # Jan 5
                # Jan 6 missing
                start_date + timedelta(days=6),  # Jan 7
            ]
        )
        window_size = 7
        max_missing = 2

        result = has_valid_window(dates, window_size, max_missing)
        assert result is False, (
            "Window with more than max_missing days should be invalid"
        )

    def test_has_valid_window_multiple_windows(self):
        """
        Test with multiple windows where one satisfies the criteria.
        """
        # First window doesn't satisfy (3 missing days)
        # Second window does satisfy (2 missing days)
        start_date = datetime(2022, 1, 1)
        dates = pd.Series(
            [
                start_date,  # Jan 1
                # Jan 2 missing
                # Jan 3 missing
                # Jan 4 missing
                start_date + timedelta(days=4),  # Jan 5
                start_date + timedelta(days=5),  # Jan 6
                start_date + timedelta(days=6),  # Jan 7
                start_date + timedelta(days=7),  # Jan 8
                # Jan 9 missing
                start_date + timedelta(days=9),  # Jan 10
                start_date + timedelta(days=10),  # Jan 11
                start_date + timedelta(days=11),  # Jan 12
                # Jan 13 missing
                start_date + timedelta(days=13),  # Jan 14
            ]
        )
        window_size = 7
        max_missing = 2

        result = has_valid_window(dates, window_size, max_missing)
        assert result is True, "Should find at least one valid window"

    def test_analyze_user_data_none_dataframe(self):
        """Test with None metadata DataFrame."""
        result = analyze_user_data(None, 2, 4)
        assert result is None, "None DataFrame should return None"

    def test_analyze_user_data_empty_dataframe(self):
        """Test with empty metadata DataFrame."""
        df = pd.DataFrame(columns=["date"])
        result = analyze_user_data(df, 2, 4)
        assert result is None, "Empty DataFrame should return None"

    def test_analyze_user_data_valid_windows(self):
        """
        Test with data that qualifies for both 7-day and 14-day windows.
        """
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(14)]
        # Skip only 4 days in the 14-day window (matching our max_missing threshold)
        skip_days = [2, 5, 9, 12]
        dates = [d for i, d in enumerate(dates) if i not in skip_days]

        df = pd.DataFrame({"date": dates, "some_data": list(range(len(dates)))})

        result = analyze_user_data(df, 2, 4)
        expected = {
            "7day_window_qualified": True,
            "14day_window_qualified": True,
        }
        assert result == expected, "Should qualify for both 7-day and 14-day windows"

    def test_analyze_user_data_only_7day_valid(self):
        """
        Test with data that qualifies for 7-day window but not 14-day window.
        """
        start_date = datetime(2022, 1, 1)
        # Create 14 days worth of dates
        all_dates = [start_date + timedelta(days=i) for i in range(14)]

        # Skip 2 days in the first 7 days (within threshold)
        seven_day_skips = [2, 4]
        # Skip 6 days in the 14-day period (exceeds threshold of 4)
        fourteen_day_skips = seven_day_skips + [8, 9, 10, 12]

        dates = [d for i, d in enumerate(all_dates) if i not in fourteen_day_skips]

        df = pd.DataFrame({"date": dates, "some_data": list(range(len(dates)))})

        result = analyze_user_data(df, 2, 4)
        expected = {
            "7day_window_qualified": True,
            "14day_window_qualified": False,
        }
        assert result == expected, "Should qualify for 7-day window only"

    @patch("src.count_valid_intervals.load_user_metadata")
    def test_process_user_no_metadata(self, mock_load_metadata):
        """Test process_user when no metadata is found."""
        mock_load_metadata.return_value = None

        user_id = "test_user"
        base_path = "/fake/path"
        threshold_7day = 2
        threshold_14day = 4

        result = process_user(user_id, base_path, threshold_7day, threshold_14day)
        expected = (user_id, None)

        mock_load_metadata.assert_called_once_with(base_path, user_id)
        assert result == expected, (
            "Should return user_id and None when no metadata found"
        )

    @patch("src.count_valid_intervals.load_user_metadata")
    @patch("src.count_valid_intervals.analyze_user_data")
    def test_process_user_with_valid_metadata(self, mock_analyze, mock_load_metadata):
        """Test process_user with valid metadata."""
        # Create a mock DataFrame
        mock_df = pd.DataFrame({"date": [datetime(2022, 1, 1)]})
        mock_load_metadata.return_value = mock_df

        # Mock the analysis result
        expected_analysis = {
            "7day_window_qualified": True,
            "14day_window_qualified": False,
        }
        mock_analyze.return_value = expected_analysis

        user_id = "test_user"
        base_path = "/fake/path"
        threshold_7day = 2
        threshold_14day = 4

        result = process_user(user_id, base_path, threshold_7day, threshold_14day)
        expected = (user_id, expected_analysis)

        mock_load_metadata.assert_called_once_with(base_path, user_id)
        mock_analyze.assert_called_once_with(mock_df, threshold_7day, threshold_14day)
        assert result == expected, "Should return user_id and analysis results"

    @patch("os.path.exists")
    @patch("pandas.read_parquet")
    def test_load_user_metadata(self, mock_read_parquet, mock_exists):
        """Test load_user_metadata function."""
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

    @patch("os.path.exists")
    def test_load_user_metadata_not_exists(self, mock_exists):
        """Test load_user_metadata when file doesn't exist."""
        mock_exists.return_value = False

        base_path = "/fake/path"
        user_id = "test_user"

        result = load_user_metadata(base_path, user_id)

        expected_path = os.path.join(base_path, user_id, "metadata.parquet")
        mock_exists.assert_called_once_with(expected_path)
        assert result is None, "Should return None when file doesn't exist"

    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_get_user_ids(self, mock_isdir, mock_listdir):
        """Test get_user_ids function."""
        # Setup mock directories
        mock_dirs = ["user1", "user2", "not_a_dir"]
        mock_listdir.return_value = mock_dirs

        # Only user1 and user2 are directories
        mock_isdir.side_effect = lambda path: path.endswith("user1") or path.endswith(
            "user2"
        )

        base_path = "/fake/path"
        result = get_user_ids(base_path)

        mock_listdir.assert_called_once_with(base_path)
        assert result == ["user1", "user2"], "Should return only directory names"
