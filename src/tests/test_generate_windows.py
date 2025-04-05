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

    @patch("src.generate_windows.load_user_metadata")
    @patch("src.generate_windows.find_non_overlapping_7day_windows")
    @patch("src.generate_windows.get_file_uris_for_window")
    def test_process_user_no_valid_windows(
        self, mock_get_uris, mock_find_windows, mock_load_metadata
    ):
        """Test processing a user with metadata but no valid windows."""
        mock_df = pd.DataFrame({"date": [datetime(2022, 1, 1)]})
        mock_load_metadata.return_value = mock_df
        mock_find_windows.return_value = []  # No valid windows

        base_path = "/fake/path"
        user_id = "test_user"
        result = process_user(user_id, base_path)

        mock_load_metadata.assert_called_once_with(base_path, user_id)
        mock_find_windows.assert_called_once()
        mock_get_uris.assert_not_called()
        assert result == [], "Should return empty list when no valid windows found"

    @patch("src.generate_windows.load_user_metadata")
    @patch("src.generate_windows.find_non_overlapping_7day_windows")
    @patch("src.generate_windows.get_file_uris_for_window")
    def test_process_user_with_valid_windows(
        self, mock_get_uris, mock_find_windows, mock_load_metadata
    ):
        """Test processing a user with valid windows."""
        # Setup mock metadata
        dates = [datetime(2022, 1, i) for i in range(1, 8)]
        mock_df = pd.DataFrame({"date": dates})
        mock_load_metadata.return_value = mock_df

        # Setup mock windows
        window_start = datetime(2022, 1, 1)
        window_end = window_start + timedelta(days=WINDOW_SIZE - 1)
        mock_find_windows.return_value = [(window_start, window_end)]

        # Setup mock file URIs
        file_uris = ["test_user/2022-01-01.npy", "test_user/2022-01-02.npy"]
        mock_get_uris.return_value = file_uris

        base_path = "/fake/path"
        user_id = "test_user"
        result = process_user(user_id, base_path)

        expected = [{
            "healthCode": user_id,
            "time_range": "2022-01-01_2022-01-07",
            "file_uris": file_uris
        }]

        mock_load_metadata.assert_called_once_with(base_path, user_id)
        mock_find_windows.assert_called_once()
        mock_get_uris.assert_called_once_with(base_path, user_id, window_start, window_end)
        assert result == expected, "Should return expected results for valid windows"
