import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from labelled_dataset import (
    _parse_string_list_safely, 
    _find_closest_dates,
    create_labelled_dataset,
    _convert_numpy_to_native,
    _create_denormalized_df,
    LABELS
)


class TestParseStringListSafely:
    
    def test_nan_input(self):
        """Test that NaN input returns NaN."""
        result = _parse_string_list_safely(np.nan)
        assert pd.isna(result)
    
    def test_actual_list_input(self):
        """Test that an actual list is returned unchanged."""
        input_list = [1, 2, 3]
        result = _parse_string_list_safely(input_list)
        assert result == input_list
    
    def test_empty_string(self):
        """Test that an empty string returns NaN."""
        result = _parse_string_list_safely("")
        assert pd.isna(result)
    
    def test_valid_list_string(self):
        """Test parsing of a valid string representation of a list."""
        result = _parse_string_list_safely("[1, 2, 3]")
        assert result == [1, 2, 3]
    
    def test_nested_list_string(self):
        """Test parsing of a nested list string."""
        result = _parse_string_list_safely("[[1, 2], [3, 4]]")
        assert result == [[1, 2], [3, 4]]
    
    def test_simple_bracket_format(self):
        """Test parsing of a simple bracket format."""
        result = _parse_string_list_safely("[1 2 3]")
        assert result == [1.0, 2.0, 3.0]
    
    def test_invalid_string(self):
        """Test that an invalid string returns NaN."""
        result = _parse_string_list_safely("not a list")
        assert pd.isna(result)
    
    def test_non_string_non_list(self):
        """Test that other types return NaN."""
        result = _parse_string_list_safely(42)
        assert pd.isna(result)


class TestFindClosestDates:
    
    def test_exact_match(self):
        """Test when there's an exact match between interval midpoint and a date."""
        dates_list = ["2023-01-01", "2023-01-15", "2023-01-30"]
        intervals_list = ["2023-01-10_2023-01-20"] # midpoint is 2023-01-15
        
        result = _find_closest_dates(dates_list, intervals_list)
        assert result["2023-01-10_2023-01-20"] == 1  # Should match the index of "2023-01-15"
    
    def test_closest_before(self):
        """Test finding the closest date before the midpoint."""
        dates_list = ["2023-01-01", "2023-01-10", "2023-01-30"]
        intervals_list = ["2023-01-12_2023-01-18"] # midpoint is 2023-01-15
        
        result = _find_closest_dates(dates_list, intervals_list)
        assert result["2023-01-12_2023-01-18"] == 1  # Should match the index of "2023-01-10"
    
    def test_closest_after(self):
        """Test finding the closest date after the midpoint."""
        dates_list = ["2023-01-01", "2023-01-20", "2023-01-30"]
        intervals_list = ["2023-01-12_2023-01-18"] # midpoint is 2023-01-15
        
        result = _find_closest_dates(dates_list, intervals_list)
        assert result["2023-01-12_2023-01-18"] == 1  # Should match the index of "2023-01-20"
    
    def test_multiple_intervals(self):
        """Test matching multiple intervals."""
        dates_list = ["2023-01-01", "2023-01-15", "2023-01-30"]
        intervals_list = ["2023-01-10_2023-01-20", "2023-01-25_2023-02-05"]
        
        result = _find_closest_dates(dates_list, intervals_list)
        assert result["2023-01-10_2023-01-20"] == 1  # Should match the index of "2023-01-15"
        assert result["2023-01-25_2023-02-05"] == 2  # Should match the index of "2023-01-30"
    
    def test_empty_dates(self):
        """Test behavior with empty dates list."""
        with pytest.raises(Exception):  # This should raise an exception of some kind
            _find_closest_dates([], ["2023-01-10_2023-01-20"])
    
    def test_unsorted_dates(self):
        """Test with unsorted date list."""
        dates_list = ["2023-01-30", "2023-01-01", "2023-01-15"]
        intervals_list = ["2023-01-10_2023-01-20"]
        
        result = _find_closest_dates(dates_list, intervals_list)
        # Since the function sorts the dates, it should still match the correct date
        assert result["2023-01-10_2023-01-20"] == 2  # Should match the original index of "2023-01-15"


class TestCreateLabelledDataset:
    
    @pytest.fixture
    def sample_interval_df(self):
        """Create a sample interval DataFrame for testing."""
        return pd.DataFrame({
            "healthCode": ["user1", "user1", "user2"],
            "time_range": ["2023-01-01_2023-01-07", "2023-01-08_2023-01-14", "2023-01-01_2023-01-07"],
            "file_uris": ["uri1", "uri2", "uri3"]
        })
    
    @pytest.fixture
    def sample_label_df(self):
        """Create a sample label DataFrame for testing."""
        return pd.DataFrame({
            "healthCode": ["user1", "user1", "user2"],
            "createdOn": pd.to_datetime(["2023-01-04", "2023-01-11", "2023-01-03"]),
            "happiness": [3, 4, 5],
            "sleep_diagnosis1": [1, 2, 0]
        })
    
    def test_basic_matching(self, sample_interval_df, sample_label_df):
        """Test basic matching of intervals with labels."""
        result = create_labelled_dataset(sample_interval_df, sample_label_df, ["happiness"])
        
        assert len(result) == 3
        
        # Check user1's first interval
        assert result[0]["healthCode"] == "user1"
        assert result[0]["time_range"] == "2023-01-01_2023-01-07"
        assert "happiness" in result[0]
        assert result[0]["happiness"]["label_value"] == 3
        
        # Check user1's second interval
        assert result[1]["healthCode"] == "user1"
        assert result[1]["time_range"] == "2023-01-08_2023-01-14"
        assert "happiness" in result[1]
        assert result[1]["happiness"]["label_value"] == 4
        
        # Check user2's interval
        assert result[2]["healthCode"] == "user2"
        assert result[2]["time_range"] == "2023-01-01_2023-01-07"
        assert "happiness" in result[2]
        assert result[2]["happiness"]["label_value"] == 5
    
    def test_multiple_labels(self, sample_interval_df, sample_label_df):
        """Test matching with multiple label columns."""
        result = create_labelled_dataset(sample_interval_df, sample_label_df, ["happiness", "sleep_diagnosis1"])
        
        assert len(result) == 3
        
        # Check first record has both labels
        assert "happiness" in result[0]
        assert "sleep_diagnosis1" in result[0]
        assert result[0]["happiness"]["label_value"] == 3
        assert result[0]["sleep_diagnosis1"]["label_value"] == 1
    
    def test_missing_label(self, sample_interval_df, sample_label_df):
        """Test behavior when a label column doesn't exist."""
        result = create_labelled_dataset(sample_interval_df, sample_label_df, ["nonexistent_label"])
        
        assert len(result) == 3
        # The nonexistent label should not be in the records
        assert "nonexistent_label" not in result[0]
    
    def test_empty_dataframes(self):
        """Test behavior with empty DataFrames."""
        empty_interval_df = pd.DataFrame(columns=["healthCode", "time_range", "file_uris"])
        empty_label_df = pd.DataFrame(columns=["healthCode", "createdOn", "happiness"])
        
        result = create_labelled_dataset(empty_interval_df, empty_label_df, ["happiness"])
        assert len(result) == 0


class TestConvertNumpyToNative:
    
    def test_convert_numpy_scalar(self):
        """Test converting a numpy scalar to Python native type."""
        input_obj = np.int64(42)
        result = _convert_numpy_to_native(input_obj)
        assert isinstance(result, int)
        assert result == 42
    
    def test_convert_numpy_array(self):
        """Test converting numpy array elements in a list."""
        input_obj = [np.int64(1), np.float64(2.5), 3]
        result = _convert_numpy_to_native(input_obj)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], int)
        assert result == [1, 2.5, 3]
    
    def test_convert_dict_with_numpy(self):
        """Test converting a dictionary with numpy values."""
        input_obj = {"a": np.int64(1), "b": {"c": np.float64(2.5)}}
        result = _convert_numpy_to_native(input_obj)
        assert isinstance(result["a"], int)
        assert isinstance(result["b"]["c"], float)
        assert result == {"a": 1, "b": {"c": 2.5}}
    
    def test_non_numpy_unchanged(self):
        """Test that non-numpy values are left unchanged."""
        input_obj = {"a": 1, "b": "string", "c": [1, 2, 3]}
        result = _convert_numpy_to_native(input_obj)
        assert result == input_obj


class TestCreateDenormalizedDF:
    
    def test_basic_denormalization(self):
        """Test basic denormalization of records."""
        records = [
            {
                "healthCode": "user1",
                "time_range": "2023-01-01_2023-01-07",
                "happiness": {
                    "label_value": 3,
                    "label_date": pd.Timestamp("2023-01-04")
                }
            }
        ]
        
        result = _create_denormalized_df(records)
        
        assert isinstance(result, pd.DataFrame)
        assert "healthCode" in result.columns
        assert "time_range" in result.columns
        assert "happiness_value" in result.columns
        assert "happiness_date" in result.columns
        
        assert result.iloc[0]["healthCode"] == "user1"
        assert result.iloc[0]["happiness_value"] == 3
        assert result.iloc[0]["happiness_date"] == pd.Timestamp("2023-01-04")
    
    def test_multiple_records(self):
        """Test denormalization of multiple records."""
        records = [
            {
                "healthCode": "user1",
                "time_range": "2023-01-01_2023-01-07",
                "happiness": {
                    "label_value": 3,
                    "label_date": pd.Timestamp("2023-01-04")
                }
            },
            {
                "healthCode": "user2",
                "time_range": "2023-01-01_2023-01-07",
                "happiness": {
                    "label_value": 5,
                    "label_date": pd.Timestamp("2023-01-03")
                }
            }
        ]
        
        result = _create_denormalized_df(records)
        
        assert len(result) == 2
        assert result.iloc[1]["healthCode"] == "user2"
        assert result.iloc[1]["happiness_value"] == 5
    
    def test_mixed_label_presence(self):
        """Test denormalization when records have different labels present."""
        records = [
            {
                "healthCode": "user1",
                "time_range": "2023-01-01_2023-01-07",
                "happiness": {
                    "label_value": 3,
                    "label_date": pd.Timestamp("2023-01-04")
                }
            },
            {
                "healthCode": "user2",
                "time_range": "2023-01-01_2023-01-07",
                "sleep_diagnosis1": {
                    "label_value": 1,
                    "label_date": pd.Timestamp("2023-01-03")
                }
            }
        ]
        
        result = _create_denormalized_df(records)
        
        assert len(result) == 2
        assert "happiness_value" in result.columns
        assert "sleep_diagnosis1_value" in result.columns
        
        # Check that missing values are NaN
        assert pd.isna(result.iloc[1]["happiness_value"])
        assert pd.isna(result.iloc[0]["sleep_diagnosis1_value"])
    
    def test_non_label_dict_fields(self):
        """Test that dictionary fields not representing labels are preserved as is."""
        records = [
            {
                "healthCode": "user1",
                "time_range": "2023-01-01_2023-01-07",
                "metadata": {"source": "device1", "version": "1.0"},
                "happiness": {
                    "label_value": 3,
                    "label_date": pd.Timestamp("2023-01-04")
                }
            }
        ]
        
        result = _create_denormalized_df(records)
        
        assert "metadata" in result.columns
        assert result.iloc[0]["metadata"] == {"source": "device1", "version": "1.0"} 