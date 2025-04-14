import unittest
import pandas as pd
import numpy as np
import torch
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import logging # Import logging

# Adjust import path based on your project structure
from src.torch_dataset import BaseMhcDataset, FilteredMhcDataset

class TestMhcDatasets(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory with dummy data for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)

        # Create dummy participant directories
        self.p1_dir = self.root_dir / "healthCode1"
        self.p2_dir = self.root_dir / "healthCode2"
        self.p1_dir.mkdir()
        self.p2_dir.mkdir()

        # Create dummy .npy files for participant 1
        # Correct shape: (Channels=3, 24, 1440), sliced to (24, 1440)
        self.day1_p1_data = np.random.rand(3, 24, 1440).astype(np.float32)
        self.day1_p1_path = self.p1_dir / "data/2023-01-15.npy" # Nested path
        self.day1_p1_path.parent.mkdir()
        np.save(self.day1_p1_path, self.day1_p1_data)

        self.day2_p1_data = np.random.rand(3, 24, 1440).astype(np.float32)
        self.day2_p1_path = self.p1_dir / "2023-01-16.npy" # Direct path
        np.save(self.day2_p1_path, self.day2_p1_data)

        # File with wrong shape after slicing
        self.day3_p1_wrong_shape_data = np.random.rand(3, 20, 1000).astype(np.float32)
        self.day3_p1_wrong_shape_path = self.p1_dir / "2023-01-17_wrong_shape.npy"
        np.save(self.day3_p1_wrong_shape_path, self.day3_p1_wrong_shape_data)

        # File with insufficient dimensions for slicing
        self.day4_p1_wrong_dims_data = np.random.rand(24, 1440).astype(np.float32)
        self.day4_p1_wrong_dims_path = self.p1_dir / "2023-01-18_wrong_dims.npy"
        np.save(self.day4_p1_wrong_dims_path, self.day4_p1_wrong_dims_data)

        # Create dummy .npy file for participant 2
        self.day1_p2_data = np.random.rand(5, 24, 1440).astype(np.float32) # Different channel count
        self.day1_p2_path = self.p2_dir / "2023-02-01.npy"
        np.save(self.day1_p2_path, self.day1_p2_data)


        # Create sample DataFrame
        data = {
            'healthCode': ["healthCode1", "healthCode1", "healthCode1", "healthCode2", "healthCode1", "healthCode1", "healthCode2"],
            'time_range': [
                "2023-01-15_2023-01-16", # P1: Day 1 (nested), Day 2 (direct) present
                "2023-01-15_2023-01-17", # P1: Day 1, Day 2 present, Day 3 missing
                "2023-01-17_2023-01-17", # P1: Includes file with wrong shape
                "2023-02-01_2023-02-02", # P2: Day 1 present, Day 2 missing
                "2023-01-18_2023-01-18", # P1: Includes file with wrong dims
                "2023-01-15_2023-01-16", # P1: file_uris is empty list
                "2023-02-01_2023-02-01", # P2: file_uris is NaN
            ],
            'file_uris': [
                ["healthCode1/data/2023-01-15.npy", "healthCode1/2023-01-16.npy"], # Corresponds to time_range 0
                ["healthCode1/data/2023-01-15.npy", "healthCode1/2023-01-16.npy"], # Corresponds to time_range 1 (day 17 not listed)
                ["healthCode1/2023-01-17_wrong_shape.npy"],          # Corresponds to time_range 2
                ["healthCode2/2023-02-01.npy", "healthCode2/nonexistent/2023-02-02.npy"], # Corresponds to time_range 3 (one exists, one doesn't)
                ["healthCode1/2023-01-18_wrong_dims.npy"],           # Corresponds to time_range 4
                [],                                      # Corresponds to time_range 5
                np.nan,                                  # Corresponds to time_range 6
            ],
            'labelA_value': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0],
            'labelB_value': [0.5, np.nan, 0.7, 0.8, 0.9, 1.0, np.nan],
            'other_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        }
        self.df = pd.DataFrame(data)

        # DataFrame with string list for file_uris to test warning
        data_str_uris = data.copy()
        data_str_uris['file_uris'] = [str(x) if isinstance(x, list) else x for x in data_str_uris['file_uris']]
        self.df_str_uris = pd.DataFrame(data_str_uris)


    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    # --- Test BaseMhcDataset ---

    def test_base_init_success(self):
        """Test successful initialization of BaseMhcDataset."""
        dataset = BaseMhcDataset(self.df, self.root_dir)
        self.assertEqual(len(dataset), len(self.df))
        self.assertEqual(dataset.label_cols, ['labelA_value', 'labelB_value'])
        # Resolve both paths to handle potential symlinks (e.g., /var vs /private/var)
        self.assertEqual(dataset.root_dir.resolve(), self.root_dir.resolve())

    def test_base_init_missing_column(self):
        """Test initialization fails if required columns are missing."""
        df_missing = self.df.drop(columns=['healthCode'])
        with self.assertRaisesRegex(ValueError, "DataFrame must contain column 'healthCode'"):
            BaseMhcDataset(df_missing, self.root_dir)

    def test_base_init_wrong_dataframe_type(self):
        """Test initialization fails if not given a DataFrame."""
        with self.assertRaisesRegex(TypeError, "dataframe must be a pandas DataFrame"):
            BaseMhcDataset([1, 2, 3], self.root_dir)

    def test_base_init_warns_on_string_list_uris(self):
         """Test initialization warns if file_uris contains string representations of lists."""
         # Make a version of the dataset with string representations of the updated URIs
         data_str_uris = self.df.copy()
         data_str_uris['file_uris'] = [str(x) if isinstance(x, list) else x for x in data_str_uris['file_uris']]
         self.df_str_uris = pd.DataFrame(data_str_uris)
         
         with self.assertWarnsRegex(UserWarning, "Consider pre-processing"): # Check if the warning is raised
             dataset = BaseMhcDataset(self.df_str_uris, self.root_dir)
             # Also check if it still processes correctly
             self.assertEqual(len(dataset), len(self.df_str_uris))
             # Check a sample item loads correctly despite the string format initially
             sample = dataset[0]
             self.assertIsInstance(sample['data'], torch.Tensor)


    def test_base_len(self):
        """Test __len__ method."""
        dataset = BaseMhcDataset(self.df, self.root_dir)
        self.assertEqual(len(dataset), 7)

    def test_base_getitem_success_full(self):
        """Test __getitem__ for a sample with all files present."""
        dataset = BaseMhcDataset(self.df, self.root_dir)
        idx = 0
        sample = dataset[idx]

        self.assertIsInstance(sample, dict)
        self.assertIn('data', sample)
        self.assertIn('labels', sample)
        self.assertIn('metadata', sample)

        # Check data shape and type
        self.assertIsInstance(sample['data'], torch.Tensor)
        self.assertEqual(sample['data'].shape, (2, 24, 1440)) # 2 days
        self.assertEqual(sample['data'].dtype, torch.float32)

        # Check data content (compare sliced data)
        expected_day1 = torch.from_numpy(self.day1_p1_data[1, :, :])
        expected_day2 = torch.from_numpy(self.day2_p1_data[1, :, :])
        torch.testing.assert_close(sample['data'][0], expected_day1)
        torch.testing.assert_close(sample['data'][1], expected_day2)

        # Check labels
        self.assertEqual(sample['labels']['labelA'], 1.0)
        self.assertEqual(sample['labels']['labelB'], 0.5)

        # Check metadata
        self.assertEqual(sample['metadata']['healthCode'], "healthCode1")
        self.assertEqual(sample['metadata']['time_range'], "2023-01-15_2023-01-16")

    def test_base_getitem_missing_day_in_range(self):
        """Test __getitem__ handles missing days within the time range."""
        dataset = BaseMhcDataset(self.df, self.root_dir)
        idx = 1
        sample = dataset[idx] # time_range: 2023-01-15_2023-01-17

        self.assertEqual(sample['data'].shape, (3, 24, 1440)) # 3 days expected

        # Check day 1 and 2 are loaded
        expected_day1 = torch.from_numpy(self.day1_p1_data[1, :, :])
        expected_day2 = torch.from_numpy(self.day2_p1_data[1, :, :])
        torch.testing.assert_close(sample['data'][0], expected_day1)
        torch.testing.assert_close(sample['data'][1], expected_day2)

        # Check day 3 (missing) is NaN placeholder
        self.assertTrue(torch.all(torch.isnan(sample['data'][2])))

        # Check labels
        self.assertEqual(sample['labels']['labelA'], 2.0)
        self.assertTrue(np.isnan(sample['labels']['labelB'])) # LabelB is NaN

    def test_base_getitem_file_listed_but_not_found(self):
        """Test __getitem__ handles files in file_uris but not found on disk."""
        dataset = BaseMhcDataset(self.df, self.root_dir)
        idx = 3 # P2: ["healthCode2/2023-02-01.npy", "healthCode2/nonexistent/2023-02-02.npy"]
        sample = dataset[idx]

        self.assertEqual(sample['data'].shape, (2, 24, 1440)) # 2 days

        # Check day 1 (present)
        expected_day1_p2 = torch.from_numpy(self.day1_p2_data[1, :, :])
        torch.testing.assert_close(sample['data'][0], expected_day1_p2)

        # Check day 2 (listed but not found) is NaN placeholder
        self.assertTrue(torch.all(torch.isnan(sample['data'][1])))

        # Check labels and metadata
        self.assertEqual(sample['labels']['labelA'], 4.0)
        self.assertEqual(sample['labels']['labelB'], 0.8)
        self.assertEqual(sample['metadata']['healthCode'], "healthCode2")


    def test_base_getitem_empty_file_uris(self):
        """Test __getitem__ with an empty file_uris list."""
        dataset = BaseMhcDataset(self.df, self.root_dir)
        idx = 5
        sample = dataset[idx]

        # Expect placeholders for all days in the time range
        self.assertEqual(sample['data'].shape, (2, 24, 1440)) # 2 days in range
        self.assertTrue(torch.all(torch.isnan(sample['data'])))

    def test_base_getitem_nan_file_uris(self):
         """Test __getitem__ with NaN file_uris."""
         dataset = BaseMhcDataset(self.df, self.root_dir)
         idx = 6
         sample = dataset[idx]

         # Expect placeholders for all days in the time range
         self.assertEqual(sample['data'].shape, (1, 24, 1440)) # 1 day in range
         self.assertTrue(torch.all(torch.isnan(sample['data'])))


    def test_base_getitem_invalid_time_range(self):
        """Test __getitem__ raises error for invalid time_range format."""
        df_invalid = self.df.copy()
        df_invalid.loc[0, 'time_range'] = "2023/01/15-2023/01/16"
        dataset = BaseMhcDataset(df_invalid, self.root_dir)
        with self.assertRaisesRegex(ValueError, "Invalid time_range format"):
            dataset[0]

    def test_base_getitem_index_out_of_bounds(self):
        """Test __getitem__ raises IndexError for invalid index."""
        dataset = BaseMhcDataset(self.df, self.root_dir)
        with self.assertRaises(IndexError):
            dataset[len(self.df)]
        with self.assertRaises(IndexError):
            dataset[-1]

    def test_generate_date_range(self):
        """Test the static _generate_date_range method."""
        dates = BaseMhcDataset._generate_date_range("2023-12-30", "2024-01-02")
        self.assertEqual(dates, ["2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02"])
        dates_single = BaseMhcDataset._generate_date_range("2023-05-10", "2023-05-10")
        self.assertEqual(dates_single, ["2023-05-10"])

    def test_base_init_feature_stats_validation(self):
        """Test validation of feature_stats parameter."""
        # Valid feature_stats
        valid_stats = {0: (0.5, 1.0), 1: (0.0, 2.0)}
        dataset = BaseMhcDataset(self.df, self.root_dir, feature_stats=valid_stats)
        self.assertEqual(dataset.feature_stats, valid_stats)

        # Invalid type for feature_stats
        with self.assertRaisesRegex(TypeError, "feature_stats must be a dictionary"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats=[(0, (0.5, 1.0))])

        # Invalid format for stats (not a tuple)
        with self.assertRaisesRegex(ValueError, "Feature stats for index 0 must be a tuple"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats={0: [0.5, 1.0]})

        # Invalid length of stats tuple
        with self.assertRaisesRegex(ValueError, "Feature stats for index 0 must be a tuple"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats={0: (0.5, 1.0, 2.0)})

        # Non-numeric values in stats
        with self.assertRaisesRegex(ValueError, "Mean and std for feature 0 must be numeric"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats={0: ('0.5', 1.0)})

    def test_feature_standardization(self):
        """Test that feature standardization is applied correctly."""
        # Create feature stats for standardization
        # Applying to first two features (indices 0 and 1)
        feature_stats = {
            0: (0.5, 2.0),  # mean=0.5, std=2.0
            1: (0.0, 1.0)   # mean=0.0, std=1.0
        }
        
        # Initialize dataset with feature standardization
        dataset = BaseMhcDataset(self.df, self.root_dir, feature_stats=feature_stats)
        
        # Get a sample that has valid data
        sample = dataset[0]  # This should have two days of valid data
        
        # Extract the original (non-standardized) data for comparison
        expected_day1 = self.day1_p1_data[1, :, :].astype(np.float32)
        expected_day2 = self.day2_p1_data[1, :, :].astype(np.float32)
        
        # Manually apply standardization to expected data
        # Feature 0
        expected_day1[0, :] = (expected_day1[0, :] - 0.5) / 2.0
        expected_day2[0, :] = (expected_day2[0, :] - 0.5) / 2.0
        # Feature 1
        expected_day1[1, :] = (expected_day1[1, :] - 0.0) / 1.0
        expected_day2[1, :] = (expected_day2[1, :] - 0.0) / 1.0
        
        # Stack expected results to match output tensor shape
        expected_data = np.stack([expected_day1, expected_day2], axis=0)
        expected_tensor = torch.from_numpy(expected_data)
        
        # Compare the standardized features
        # Feature 0 should be standardized
        torch.testing.assert_close(sample['data'][:, 0, :], expected_tensor[:, 0, :])
        # Feature 1 should be standardized
        torch.testing.assert_close(sample['data'][:, 1, :], expected_tensor[:, 1, :])
        
        # Features 2+ should not be standardized
        # Compare with original data slices for features not in feature_stats
        if expected_data.shape[1] > 2:  # If we have more than 2 features
            for feature_idx in range(2, expected_data.shape[1]):
                original_day1 = torch.from_numpy(self.day1_p1_data[1, feature_idx, :])
                original_day2 = torch.from_numpy(self.day2_p1_data[1, feature_idx, :])
                torch.testing.assert_close(sample['data'][0, feature_idx, :], original_day1)
                torch.testing.assert_close(sample['data'][1, feature_idx, :], original_day2)

    def test_feature_standardization_out_of_bounds(self):
        """Test that out-of-bounds feature indices in feature_stats are handled properly."""
        # Set up a feature_stats with an out-of-bounds index
        feature_stats = {
            999: (0.5, 1.0)  # This index should be out of bounds
        }
        
        # Create dataset and capture log messages
        with self.assertLogs(level=logging.WARNING) as log:
            dataset = BaseMhcDataset(self.df, self.root_dir, feature_stats=feature_stats)
            sample = dataset[0]  # This should log a warning
            
            # Verify that a warning was logged
            self.assertTrue(any("Feature index 999 out of bounds" in message for message in log.output))
        
        # Verify data was not modified (compared to non-standardized version)
        regular_dataset = BaseMhcDataset(self.df, self.root_dir)
        regular_sample = regular_dataset[0]
        
        # Data should be identical since the standardization was skipped
        torch.testing.assert_close(sample['data'], regular_sample['data'])

    def test_feature_standardization_with_mask(self):
        """Test that feature standardization works correctly when include_mask=True."""
        feature_stats = {
            0: (0.5, 2.0)  # standardize just the first feature
        }
        
        # Initialize dataset with both feature standardization and mask
        dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True, feature_stats=feature_stats)
        
        # Get a sample
        sample = dataset[0]
        
        # Verify sample has both data and mask
        self.assertIn('data', sample)
        self.assertIn('mask', sample)
        
        # Verify mask shape matches data shape
        self.assertEqual(sample['data'].shape, sample['mask'].shape)
        
        # Verify standardization was applied to data but not to mask
        # Extract original data and mask
        expected_data_day1 = self.day1_p1_data[1, :, :].copy()
        expected_mask_day1 = self.day1_p1_data[0, :, :].copy()
        
        # Apply standardization to feature 0 of expected data
        expected_data_day1[0, :] = (expected_data_day1[0, :] - 0.5) / 2.0
        
        # Check feature 0 (standardized in data)
        self.assertFalse(torch.allclose(
            sample['data'][0, 0, :], 
            torch.from_numpy(self.day1_p1_data[1, 0, :])
        ))
        
        # Check mask (should be unchanged)
        torch.testing.assert_close(
            sample['mask'][0, 0, :], 
            torch.from_numpy(self.day1_p1_data[0, 0, :])
        )

    # --- Test FilteredMhcDataset ---

    def test_filtered_init_success(self):
        """Test successful initialization and filtering."""
        # Filter by labelA: indices 0, 1, 3, 4, 5, 6 should remain (6 samples)
        dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelA')
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.label_of_interest, 'labelA')
        # Check that the internal df is filtered (target label has no NaNs)
        self.assertFalse(dataset.df['labelA_value'].isna().any())
        # The index is reset by BaseMhcDataset.__init__, so we don't check original indices here.

        # Filter by labelB: indices 0, 2, 3, 4, 5 should remain (5 samples)
        dataset_b = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelB')
        self.assertEqual(len(dataset_b), 5)
        self.assertEqual(dataset_b.label_of_interest, 'labelB')
        # Check that the internal df is filtered (target label has no NaNs)
        self.assertFalse(dataset_b.df['labelB_value'].isna().any())
        # The index is reset by BaseMhcDataset.__init__, so we don't check original indices here.


    def test_filtered_init_label_with_suffix(self):
         """Test initialization works if label_of_interest already has _value suffix."""
         dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelA_value')
         self.assertEqual(len(dataset), 6)
         self.assertEqual(dataset.label_of_interest, 'labelA_value') # Keeps the provided name


    def test_filtered_init_nonexistent_label(self):
        """Test initialization fails if the filter label doesn't exist."""
        with self.assertRaisesRegex(ValueError, "Label column 'nonExistentLabel_value' not found"):
            FilteredMhcDataset(self.df, self.root_dir, label_of_interest='nonExistentLabel')

    def test_filtered_init_no_samples_left(self):
        """Test initialization fails if filtering leaves no samples."""
        df_all_nan = self.df.copy()
        df_all_nan['labelA_value'] = np.nan
        with self.assertRaisesRegex(ValueError, "No samples found with non-NaN values for label 'labelA'"):
             FilteredMhcDataset(df_all_nan, self.root_dir, label_of_interest='labelA')


    def test_filtered_len(self):
        """Test __len__ returns the filtered length."""
        dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelA')
        self.assertEqual(len(dataset), 6)

        dataset_b = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelB')
        self.assertEqual(len(dataset_b), 5)


    def test_filtered_getitem(self):
        """Test __getitem__ returns correctly filtered samples."""
        # Filter by labelB (indices 0, 2, 3, 4, 5 remain)
        dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelB')

        # Get the first sample from the *filtered* dataset (original index 0)
        sample0 = dataset[0]
        self.assertEqual(sample0['metadata']['healthCode'], "healthCode1")
        self.assertEqual(sample0['labels']['labelB'], 0.5) # Check the filtered label value

        # Get the second sample from the *filtered* dataset (original index 2)
        sample1 = dataset[1]
        self.assertEqual(sample1['metadata']['healthCode'], "healthCode1")
        self.assertEqual(sample1['labels']['labelB'], 0.7)
        # Ensure data loading still works (this sample had a load error for its file)
        self.assertTrue(torch.all(torch.isnan(sample1['data'])))

        # Get the third sample (original index 3)
        sample2 = dataset[2]
        self.assertEqual(sample2['metadata']['healthCode'], "healthCode2")
        self.assertEqual(sample2['labels']['labelB'], 0.8)


    def test_filtered_getitem_index_out_of_bounds(self):
         """Test __getitem__ raises IndexError for invalid index on filtered dataset."""
         dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelB') # 5 samples
         with self.assertRaises(IndexError):
             dataset[5] # Index 5 is out of bounds for the filtered length
         with self.assertRaises(IndexError):
             dataset[-1]


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)



