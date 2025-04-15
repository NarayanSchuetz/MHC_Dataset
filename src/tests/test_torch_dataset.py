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
from src.torch_dataset import BaseMhcDataset, FilteredMhcDataset, _EXPECTED_RAW_FEATURES, _EXPECTED_TIME_POINTS

class TestMhcDatasets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up logging for the test class."""
        # Ensure the specific logger exists and set its level
        logger_to_test = logging.getLogger('src.torch_dataset')
        logger_to_test.setLevel(logging.INFO) # Ensure it captures INFO, WARNING, ERROR

        # Configure the root logger handler if necessary (might already be done by test runner)
        # Avoid calling basicConfig multiple times if possible
        if not logging.root.handlers:
            # No handlers configured yet, add a basic one
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
        else:
            # Handlers exist, ensure root logger level is appropriate
            # Note: This might not override levels set on specific handlers
            logging.getLogger().setLevel(logging.INFO)

        # Optionally, quiet other loggers if they are too noisy
        # logging.getLogger('some_other_library').setLevel(logging.WARNING)

    def setUp(self):
        """Set up a temporary directory with dummy data for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.raw_features = _EXPECTED_RAW_FEATURES # 24
        self.time_points = _EXPECTED_TIME_POINTS # 1440

        # Create dummy participant directories
        self.p1_dir = self.root_dir / "healthCode1"
        self.p2_dir = self.root_dir / "healthCode2"
        self.p1_dir.mkdir()
        self.p2_dir.mkdir()

        # Create dummy .npy files for participant 1
        # Shape: (Mask+Data=2+, Features, Time)
        self.day1_p1_data = np.random.rand(2, self.raw_features, self.time_points).astype(np.float32)
        # Make mask binary for clarity
        self.day1_p1_data[0, :, :] = np.random.randint(0, 2, size=(self.raw_features, self.time_points)).astype(np.float32)
        self.day1_p1_path = self.p1_dir / "data/2023-01-15.npy" # Nested path
        self.day1_p1_path.parent.mkdir()
        np.save(self.day1_p1_path, self.day1_p1_data)

        self.day2_p1_data = np.random.rand(2, self.raw_features, self.time_points).astype(np.float32)
        self.day2_p1_data[0, :, :] = np.random.randint(0, 2, size=(self.raw_features, self.time_points)).astype(np.float32)
        self.day2_p1_path = self.p1_dir / "2023-01-16.npy" # Direct path
        np.save(self.day2_p1_path, self.day2_p1_data)

        # File with wrong feature/time shape after slicing (data slice index 1)
        self.day3_p1_wrong_shape_data = np.random.rand(2, 20, 1000).astype(np.float32)
        self.day3_p1_wrong_shape_path = self.p1_dir / "2023-01-17_wrong_shape.npy"
        np.save(self.day3_p1_wrong_shape_path, self.day3_p1_wrong_shape_data)

        # File with insufficient dimensions for slicing (needs at least 3)
        self.day4_p1_wrong_dims_data = np.random.rand(self.raw_features, self.time_points).astype(np.float32)
        self.day4_p1_wrong_dims_path = self.p1_dir / "2023-01-18_wrong_dims.npy"
        np.save(self.day4_p1_wrong_dims_path, self.day4_p1_wrong_dims_data)
        
        # File with insufficient first dimension (needs at least 2)
        self.day5_p1_wrong_first_dim_data = np.random.rand(1, self.raw_features, self.time_points).astype(np.float32)
        self.day5_p1_wrong_first_dim_path = self.p1_dir / "2023-01-19_wrong_first_dim.npy"
        np.save(self.day5_p1_wrong_first_dim_path, self.day5_p1_wrong_first_dim_data)

        # Create dummy .npy file for participant 2
        self.day1_p2_data = np.random.rand(2, self.raw_features, self.time_points).astype(np.float32) # Use 2 in first dim for consistency
        self.day1_p2_path = self.p2_dir / "2023-02-01.npy"
        np.save(self.day1_p2_path, self.day1_p2_data)


        # Create sample DataFrame
        data = {
            'healthCode': ["healthCode1", "healthCode1", "healthCode1", "healthCode2", "healthCode1", "healthCode1", "healthCode2", "healthCode1", "healthCode1"],
            'time_range': [
                "2023-01-15_2023-01-16", # P1: Day 1 (nested), Day 2 (direct) present
                "2023-01-15_2023-01-17", # P1: Day 1, Day 2 present, Day 3 missing
                "2023-01-17_2023-01-17", # P1: Day 3 - wrong shape file listed
                "2023-02-01_2023-02-02", # P2: Day 1 present, Day 2 missing (nonexistent file listed)
                "2023-01-18_2023-01-18", # P1: Day 4 - wrong dims file listed
                "2023-01-15_2023-01-16", # P1: file_uris is empty list
                "2023-02-01_2023-02-01", # P2: file_uris is NaN
                "2023-01-19_2023-01-19", # P1: Day 5 - wrong first dim file listed
                "2023-01-15_2023-01-16", # P1: Copy of first sample for testing FilteredDataset interactions
            ],
            'file_uris': [
                ["healthCode1/data/2023-01-15.npy", "healthCode1/2023-01-16.npy"], # Corresponds to time_range 0
                ["healthCode1/data/2023-01-15.npy", "healthCode1/2023-01-16.npy"], # Corresponds to time_range 1 (day 17 not listed)
                ["healthCode1/2023-01-17_wrong_shape.npy"],          # Corresponds to time_range 2
                ["healthCode2/2023-02-01.npy", "healthCode2/nonexistent/2023-02-02.npy"], # Corresponds to time_range 3 (one exists, one doesn't)
                ["healthCode1/2023-01-18_wrong_dims.npy"],           # Corresponds to time_range 4
                [],                                      # Corresponds to time_range 5
                np.nan,                                  # Corresponds to time_range 6
                ["healthCode1/2023-01-19_wrong_first_dim.npy"],      # Corresponds to time_range 7
                ["healthCode1/data/2023-01-15.npy", "healthCode1/2023-01-16.npy"], # Corresponds to time_range 8
            ],
            'labelA_value': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            'labelB_value': [0.5, np.nan, 0.7, 0.8, 0.9, 1.0, np.nan, 1.1, 1.2],
            'other_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
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
        self.assertIsNone(dataset.feature_indices) # Default is None
        self.assertEqual(dataset.num_features, self.raw_features) # Default is all features

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
        self.assertEqual(len(dataset), 9)

    def test_base_getitem_success_full(self):
        """Test __getitem__ for a sample with all files present."""
        dataset = BaseMhcDataset(self.df, self.root_dir)
        idx = 0
        sample = dataset[idx]

        self.assertIsInstance(sample, dict)
        self.assertIn('data', sample)
        self.assertIn('labels', sample)
        self.assertIn('metadata', sample)
        self.assertNotIn('mask', sample) # Default is no mask

        # Check data shape and type
        self.assertIsInstance(sample['data'], torch.Tensor)
        self.assertEqual(sample['data'].shape, (2, self.raw_features, self.time_points)) # 2 days, all features
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

    def test_base_getitem_with_mask(self):
        """Test __getitem__ with include_mask=True."""
        dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True)
        idx = 0
        sample = dataset[idx]

        self.assertIn('mask', sample)
        self.assertIsInstance(sample['mask'], torch.Tensor)
        self.assertEqual(sample['mask'].shape, (2, self.raw_features, self.time_points)) # Same shape as data
        self.assertEqual(sample['mask'].dtype, torch.float32)

        # Check mask content
        expected_mask1 = torch.from_numpy(self.day1_p1_data[0, :, :])
        expected_mask2 = torch.from_numpy(self.day2_p1_data[0, :, :])
        torch.testing.assert_close(sample['mask'][0], expected_mask1)
        torch.testing.assert_close(sample['mask'][1], expected_mask2)

        # Check data is still correct
        expected_day1 = torch.from_numpy(self.day1_p1_data[1, :, :])
        expected_day2 = torch.from_numpy(self.day2_p1_data[1, :, :])
        torch.testing.assert_close(sample['data'][0], expected_day1)
        torch.testing.assert_close(sample['data'][1], expected_day2)

    def test_base_getitem_missing_day_in_range(self):
        """Test __getitem__ handles missing days within the time range."""
        dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True) # Also test mask placeholder
        idx = 1
        sample = dataset[idx] # time_range: 2023-01-15_2023-01-17

        self.assertEqual(sample['data'].shape, (3, self.raw_features, self.time_points)) # 3 days expected
        self.assertEqual(sample['mask'].shape, (3, self.raw_features, self.time_points))

        # Check day 1 and 2 are loaded
        expected_day1 = torch.from_numpy(self.day1_p1_data[1, :, :])
        expected_day2 = torch.from_numpy(self.day2_p1_data[1, :, :])
        expected_mask1 = torch.from_numpy(self.day1_p1_data[0, :, :])
        expected_mask2 = torch.from_numpy(self.day2_p1_data[0, :, :])
        torch.testing.assert_close(sample['data'][0], expected_day1)
        torch.testing.assert_close(sample['data'][1], expected_day2)
        torch.testing.assert_close(sample['mask'][0], expected_mask1)
        torch.testing.assert_close(sample['mask'][1], expected_mask2)

        # Check day 3 (missing) is NaN placeholder for data, zero placeholder for mask
        self.assertTrue(torch.all(torch.isnan(sample['data'][2])))
        self.assertTrue(torch.all(sample['mask'][2] == 0))

        # Check labels
        self.assertEqual(sample['labels']['labelA'], 2.0)
        self.assertTrue(np.isnan(sample['labels']['labelB'])) # LabelB is NaN

    def test_base_getitem_file_listed_but_not_found(self):
        """Test __getitem__ handles files in file_uris but not found on disk."""
        dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True)
        idx = 3 # P2: ["healthCode2/2023-02-01.npy", "healthCode2/nonexistent/2023-02-02.npy"]
        sample = dataset[idx]

        self.assertEqual(sample['data'].shape, (2, self.raw_features, self.time_points)) # 2 days
        self.assertEqual(sample['mask'].shape, (2, self.raw_features, self.time_points))

        # Check day 1 (present)
        expected_day1_p2 = torch.from_numpy(self.day1_p2_data[1, :, :])
        expected_mask1_p2 = torch.from_numpy(self.day1_p2_data[0, :, :]) # Assume mask at index 0 for P2 as well
        torch.testing.assert_close(sample['data'][0], expected_day1_p2)
        torch.testing.assert_close(sample['mask'][0], expected_mask1_p2)


        # Check day 2 (listed but not found) is placeholder
        self.assertTrue(torch.all(torch.isnan(sample['data'][1])))
        self.assertTrue(torch.all(sample['mask'][1] == 0))

        # Check labels and metadata
        self.assertEqual(sample['labels']['labelA'], 4.0)
        self.assertEqual(sample['labels']['labelB'], 0.8)
        self.assertEqual(sample['metadata']['healthCode'], "healthCode2")


    def test_base_getitem_empty_file_uris(self):
        """Test __getitem__ with an empty file_uris list."""
        dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True)
        idx = 5
        sample = dataset[idx]

        # Expect placeholders for all days in the time range
        self.assertEqual(sample['data'].shape, (2, self.raw_features, self.time_points)) # 2 days in range
        self.assertEqual(sample['mask'].shape, (2, self.raw_features, self.time_points))
        self.assertTrue(torch.all(torch.isnan(sample['data'])))
        self.assertTrue(torch.all(sample['mask'] == 0))


    def test_base_getitem_nan_file_uris(self):
         """Test __getitem__ with NaN file_uris."""
         dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True)
         idx = 6
         sample = dataset[idx]

         # Expect placeholders for all days in the time range
         self.assertEqual(sample['data'].shape, (1, self.raw_features, self.time_points)) # 1 day in range
         self.assertEqual(sample['mask'].shape, (1, self.raw_features, self.time_points))
         self.assertTrue(torch.all(torch.isnan(sample['data'])))
         self.assertTrue(torch.all(sample['mask'] == 0))

    def test_base_getitem_file_load_error_wrong_shape(self):
        """Test __getitem__ handles error during file load (wrong shape)."""
        dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True)
        idx = 2 # File '2023-01-17_wrong_shape.npy' has shape (2, 20, 1000)
        
        # Should return placeholder when encountering wrong shape
        sample = dataset[idx]
        
        # Verify a placeholder was returned (NaN for data, zeros for mask)
        self.assertEqual(sample['data'].shape, (1, self.raw_features, self.time_points))
        self.assertEqual(sample['mask'].shape, (1, self.raw_features, self.time_points))
        self.assertTrue(torch.all(torch.isnan(sample['data'])), "Data should be all NaN")
        self.assertTrue(torch.all(sample['mask'] == 0), "Mask should be all zeros")

    def test_base_getitem_file_load_error_wrong_dims(self):
        """Test __getitem__ handles error during file load (wrong dimensions)."""
        dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True)
        idx = 4 # File '2023-01-18_wrong_dims.npy' has shape (24, 1440) -> ndim=2
        
        # Should return placeholder when encountering wrong dimensions
        sample = dataset[idx]
        
        # Verify a placeholder was returned (NaN for data, zeros for mask)
        self.assertEqual(sample['data'].shape, (1, self.raw_features, self.time_points))
        self.assertEqual(sample['mask'].shape, (1, self.raw_features, self.time_points))
        self.assertTrue(torch.all(torch.isnan(sample['data'])), "Data should be all NaN")
        self.assertTrue(torch.all(sample['mask'] == 0), "Mask should be all zeros")


    def test_base_getitem_file_load_error_wrong_first_dim(self):
        """Test __getitem__ handles error during file load (wrong first dim)."""
        dataset = BaseMhcDataset(self.df, self.root_dir, include_mask=True)
        idx = 7 # File '2023-01-19_wrong_first_dim.npy' has shape (1, 24, 1440)
        
        # Should return placeholder when encountering wrong first dimension
        sample = dataset[idx]
        
        # Verify a placeholder was returned (NaN for data, zeros for mask)
        self.assertEqual(sample['data'].shape, (1, self.raw_features, self.time_points))
        self.assertEqual(sample['mask'].shape, (1, self.raw_features, self.time_points))
        self.assertTrue(torch.all(torch.isnan(sample['data'])), "Data should be all NaN")
        self.assertTrue(torch.all(sample['mask'] == 0), "Mask should be all zeros")


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

    # --- Feature Indices Tests ---
    def test_base_init_feature_indices_valid(self):
        """Test initialization with valid feature_indices."""
        indices = [5, 0, 10, 0] # Test duplicates and unsorted
        dataset = BaseMhcDataset(self.df, self.root_dir, feature_indices=indices)
        self.assertEqual(dataset.feature_indices, [0, 5, 10]) # Should be sorted and unique
        self.assertEqual(dataset.num_features, 3)

        indices_all = list(range(self.raw_features))
        dataset_all = BaseMhcDataset(self.df, self.root_dir, feature_indices=indices_all)
        self.assertEqual(dataset_all.feature_indices, indices_all)
        self.assertEqual(dataset_all.num_features, self.raw_features)

    def test_base_init_feature_indices_invalid(self):
        """Test initialization fails with invalid feature_indices."""
        with self.assertRaisesRegex(ValueError, "feature_indices cannot be an empty list"):
            BaseMhcDataset(self.df, self.root_dir, feature_indices=[])
        with self.assertRaisesRegex(TypeError, "feature_indices must be a list of integers"):
            BaseMhcDataset(self.df, self.root_dir, feature_indices=[0, 5.5])
        with self.assertRaisesRegex(TypeError, "feature_indices must be a list of integers"):
            BaseMhcDataset(self.df, self.root_dir, feature_indices="[0, 5]")
        with self.assertRaisesRegex(ValueError, f"All feature_indices must be between 0 and {self.raw_features - 1}"):
            BaseMhcDataset(self.df, self.root_dir, feature_indices=[0, self.raw_features])
        with self.assertRaisesRegex(ValueError, f"All feature_indices must be between 0 and {self.raw_features - 1}"):
            BaseMhcDataset(self.df, self.root_dir, feature_indices=[-1, 5])

    def test_base_getitem_feature_selection(self):
        """Test __getitem__ selects the correct features."""
        indices = [5, 0] # Select features 5 and 0
        dataset = BaseMhcDataset(self.df, self.root_dir, feature_indices=indices)
        idx = 0
        sample = dataset[idx]

        self.assertEqual(sample['data'].shape, (2, 2, self.time_points)) # 2 days, 2 features

        # Check selected features content
        # Output feature 0 should be original feature 0
        # Output feature 1 should be original feature 5
        expected_day1_f0 = torch.from_numpy(self.day1_p1_data[1, 0, :])
        expected_day1_f5 = torch.from_numpy(self.day1_p1_data[1, 5, :])
        expected_day2_f0 = torch.from_numpy(self.day2_p1_data[1, 0, :])
        expected_day2_f5 = torch.from_numpy(self.day2_p1_data[1, 5, :])

        torch.testing.assert_close(sample['data'][0, 0, :], expected_day1_f0)
        torch.testing.assert_close(sample['data'][0, 1, :], expected_day1_f5)
        torch.testing.assert_close(sample['data'][1, 0, :], expected_day2_f0)
        torch.testing.assert_close(sample['data'][1, 1, :], expected_day2_f5)

    def test_base_getitem_feature_selection_with_mask(self):
        """Test feature selection works correctly with include_mask=True."""
        indices = [10, 20]
        dataset = BaseMhcDataset(self.df, self.root_dir, feature_indices=indices, include_mask=True)
        idx = 0
        sample = dataset[idx]

        self.assertEqual(sample['data'].shape, (2, 2, self.time_points))
        self.assertEqual(sample['mask'].shape, (2, 2, self.time_points))

        # Check selected mask features
        expected_mask1_f10 = torch.from_numpy(self.day1_p1_data[0, 10, :])
        expected_mask1_f20 = torch.from_numpy(self.day1_p1_data[0, 20, :])
        expected_mask2_f10 = torch.from_numpy(self.day2_p1_data[0, 10, :])
        expected_mask2_f20 = torch.from_numpy(self.day2_p1_data[0, 20, :])

        torch.testing.assert_close(sample['mask'][0, 0, :], expected_mask1_f10)
        torch.testing.assert_close(sample['mask'][0, 1, :], expected_mask1_f20)
        torch.testing.assert_close(sample['mask'][1, 0, :], expected_mask2_f10)
        torch.testing.assert_close(sample['mask'][1, 1, :], expected_mask2_f20)


    # --- Feature Stats Tests (interactions) ---

    def test_base_init_feature_stats_validation(self):
        """Test validation of feature_stats parameter (original test adapted)."""
        # Valid feature_stats
        valid_stats = {0: (0.5, 1.0), 1: (0.0, 2.0)}
        dataset = BaseMhcDataset(self.df, self.root_dir, feature_stats=valid_stats)
        self.assertEqual(dataset.feature_stats, valid_stats)
        self.assertIsNotNone(dataset._remapped_feature_stats)
        self.assertEqual(dataset._remapped_feature_stats, valid_stats) # No remapping needed here

        # Invalid type for feature_stats
        with self.assertRaisesRegex(TypeError, "feature_stats must be a dictionary"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats=[(0, (0.5, 1.0))])

        # Invalid key type
        with self.assertRaisesRegex(TypeError, "Keys in feature_stats must be integer indices"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats={'0': (0.5, 1.0)})

        # Invalid format for stats (not a tuple)
        with self.assertRaisesRegex(ValueError, "Feature stats for index 0 must be a tuple"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats={0: [0.5, 1.0]})

        # Invalid length of stats tuple
        with self.assertRaisesRegex(ValueError, "Feature stats for index 0 must be a tuple"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats={0: (0.5, 1.0, 2.0)})

        # Non-numeric values in stats
        with self.assertRaisesRegex(ValueError, "Mean and std for feature 0 must be numeric"):
            BaseMhcDataset(self.df, self.root_dir, feature_stats={0: ('0.5', 1.0)})

        # Out-of-bounds index (should log warning, but not raise error during init)
        with self.assertLogs('src.torch_dataset', level='WARNING') as log:
            dataset_out_of_bounds = BaseMhcDataset(self.df, self.root_dir, feature_stats={self.raw_features: (0.1, 0.2)})
            self.assertTrue(any(f"Feature index {self.raw_features} in feature_stats is out of the expected range" in msg for msg in log.output))
            self.assertIsNone(dataset_out_of_bounds._remapped_feature_stats) # No valid stats remain


    def test_feature_standardization(self):
        """Test that feature standardization is applied correctly (original test adapted)."""
        # Applying to first two features (indices 0 and 1)
        mean0, std0 = 0.5, 2.0
        mean1, std1 = 0.0, 1.0
        feature_stats = { 0: (mean0, std0), 1: (mean1, std1) }
        dataset = BaseMhcDataset(self.df, self.root_dir, feature_stats=feature_stats)
        sample = dataset[0] # 2 days of data

        # Extract the original (non-standardized) data
        original_day1 = self.day1_p1_data[1, :, :].astype(np.float32)
        original_day2 = self.day2_p1_data[1, :, :].astype(np.float32)

        # Manually apply standardization
        expected_day1 = original_day1.copy()
        expected_day2 = original_day2.copy()
        expected_day1[0, :] = (expected_day1[0, :] - mean0) / std0
        expected_day2[0, :] = (expected_day2[0, :] - mean0) / std0
        expected_day1[1, :] = (expected_day1[1, :] - mean1) / std1
        expected_day2[1, :] = (expected_day2[1, :] - mean1) / std1

        expected_tensor = torch.from_numpy(np.stack([expected_day1, expected_day2], axis=0))

        # Compare standardized features
        torch.testing.assert_close(sample['data'][:, 0, :], expected_tensor[:, 0, :])
        torch.testing.assert_close(sample['data'][:, 1, :], expected_tensor[:, 1, :])

        # Compare non-standardized features
        if self.raw_features > 2:
             torch.testing.assert_close(sample['data'][:, 2:, :], expected_tensor[:, 2:, :]) # Should be same as original beyond feature 1

    def test_feature_selection_and_standardization(self):
        """Test standardization works correctly with selected features."""
        indices = [5, 0, 10] # Select features 5, 0, 10 -> output indices 0, 1, 2 (remapped)
        mean0, std0 = 0.1, 1.1
        mean5, std5 = 0.5, 2.5
        mean10, std10 = 1.0, 0.5
        mean_ignored = 0.0, 1.0 # Stats for feature 2 (not selected)
        feature_stats = {
            0: (mean0, std0),
            5: (mean5, std5),
            10: (mean10, std10),
            2: mean_ignored # Should be ignored
        }

        with self.assertLogs('src.torch_dataset', level='WARNING') as log: # Expect warning for feature 2
            dataset = BaseMhcDataset(self.df, self.root_dir, feature_indices=indices, feature_stats=feature_stats)
            self.assertTrue(any("Feature index 2 in feature_stats is not in the selected feature_indices" in msg for msg in log.output))

        sample = dataset[0] # 2 days
        self.assertEqual(sample['data'].shape, (2, 3, self.time_points)) # 2 days, 3 features

        # Original data slices needed
        orig_day1_f0 = self.day1_p1_data[1, 0, :]
        orig_day1_f5 = self.day1_p1_data[1, 5, :]
        orig_day1_f10 = self.day1_p1_data[1, 10, :]
        orig_day2_f0 = self.day2_p1_data[1, 0, :]
        orig_day2_f5 = self.day2_p1_data[1, 5, :]
        orig_day2_f10 = self.day2_p1_data[1, 10, :]

        # Calculate expected standardized values
        exp_day1_f0 = (orig_day1_f0 - mean0) / std0
        exp_day1_f5 = (orig_day1_f5 - mean5) / std5
        exp_day1_f10 = (orig_day1_f10 - mean10) / std10
        exp_day2_f0 = (orig_day2_f0 - mean0) / std0
        exp_day2_f5 = (orig_day2_f5 - mean5) / std5
        exp_day2_f10 = (orig_day2_f10 - mean10) / std10

        # Check against output (remembering selected order [0, 5, 10] maps to output [0, 1, 2])
        torch.testing.assert_close(sample['data'][0, 0, :], torch.from_numpy(exp_day1_f0)) # Output 0 <- Original 0
        torch.testing.assert_close(sample['data'][0, 1, :], torch.from_numpy(exp_day1_f5)) # Output 1 <- Original 5
        torch.testing.assert_close(sample['data'][0, 2, :], torch.from_numpy(exp_day1_f10))# Output 2 <- Original 10
        torch.testing.assert_close(sample['data'][1, 0, :], torch.from_numpy(exp_day2_f0))
        torch.testing.assert_close(sample['data'][1, 1, :], torch.from_numpy(exp_day2_f5))
        torch.testing.assert_close(sample['data'][1, 2, :], torch.from_numpy(exp_day2_f10))

    def test_standardization_zero_std_dev(self):
        """Test standardization handles zero std dev by skipping and warning."""
        indices = [0, 1]
        feature_stats = {
            0: (0.5, 0.0), # Zero std dev
            1: (0.2, 1.0)
        }
        dataset = BaseMhcDataset(self.df, self.root_dir, feature_indices=indices, feature_stats=feature_stats)

        with self.assertLogs('src.torch_dataset', level='WARNING') as log:
            sample = dataset[0]
            self.assertTrue(any("Standard deviation for feature index 0 is zero" in msg for msg in log.output))

        # Check feature 0 was NOT standardized (should be original value)
        orig_day1_f0 = self.day1_p1_data[1, 0, :]
        orig_day2_f0 = self.day2_p1_data[1, 0, :]
        torch.testing.assert_close(sample['data'][0, 0, :], torch.from_numpy(orig_day1_f0))
        torch.testing.assert_close(sample['data'][1, 0, :], torch.from_numpy(orig_day2_f0))

        # Check feature 1 WAS standardized
        orig_day1_f1 = self.day1_p1_data[1, 1, :]
        orig_day2_f1 = self.day2_p1_data[1, 1, :]
        exp_day1_f1 = (orig_day1_f1 - 0.2) / 1.0
        exp_day2_f1 = (orig_day2_f1 - 0.2) / 1.0
        torch.testing.assert_close(sample['data'][0, 1, :], torch.from_numpy(exp_day1_f1))
        torch.testing.assert_close(sample['data'][1, 1, :], torch.from_numpy(exp_day2_f1))



    # --- Test FilteredMhcDataset ---

    def test_filtered_init_success(self):
        """Test successful initialization and filtering."""
        # Filter by labelA: indices 0, 1, 3, 4, 5, 6, 7, 8 should remain (8 samples)
        dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelA')
        self.assertEqual(len(dataset), 8)
        self.assertEqual(dataset.label_of_interest, 'labelA')
        # Check that the internal df is filtered (target label has no NaNs)
        self.assertFalse(dataset.df['labelA_value'].isna().any())
        self.assertIsNone(dataset.feature_indices) # Check defaults passed correctly
        self.assertIsNone(dataset.feature_stats)


        # Filter by labelB: indices 0, 2, 3, 4, 5, 7, 8 should remain (7 samples)
        dataset_b = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelB')
        self.assertEqual(len(dataset_b), 7)
        self.assertEqual(dataset_b.label_of_interest, 'labelB')
        # Check that the internal df is filtered (target label has no NaNs)
        self.assertFalse(dataset_b.df['labelB_value'].isna().any())


    def test_filtered_init_label_with_suffix(self):
         """Test initialization works if label_of_interest already has _value suffix."""
         dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelA_value')
         self.assertEqual(len(dataset), 8)
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
        self.assertEqual(len(dataset), 8)

        dataset_b = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelB')
        self.assertEqual(len(dataset_b), 7)


    def test_filtered_getitem(self):
        """Test __getitem__ returns correctly filtered samples."""
        # Filter by labelB (original indices 0, 2, 3, 4, 5, 7, 8 remain -> filtered indices 0..6)
        dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelB')

        # Get the first sample from the *filtered* dataset (original index 0)
        sample0 = dataset[0]
        self.assertEqual(sample0['metadata']['healthCode'], "healthCode1")
        self.assertEqual(sample0['labels']['labelB'], 0.5) # Check the filtered label value
        self.assertEqual(sample0['data'].shape, (2, self.raw_features, self.time_points)) # Check shape

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

        # Get the last sample (original index 8)
        sample_last = dataset[6]
        self.assertEqual(sample_last['metadata']['healthCode'], "healthCode1")
        self.assertEqual(sample_last['labels']['labelB'], 1.2)
        # Check data loaded correctly for this sample
        expected_day1 = torch.from_numpy(self.day1_p1_data[1, :, :])
        expected_day2 = torch.from_numpy(self.day2_p1_data[1, :, :])
        torch.testing.assert_close(sample_last['data'][0], expected_day1)
        torch.testing.assert_close(sample_last['data'][1], expected_day2)


    def test_filtered_init_with_features(self):
        """Test FilteredMhcDataset initialization passes feature args correctly."""
        indices = [1, 3]
        stats = {1: (0.1, 1.1), 3: (0.3, 1.3)}
        dataset = FilteredMhcDataset(
            self.df,
            self.root_dir,
            label_of_interest='labelA', # 8 samples pass
            feature_indices=indices,
            feature_stats=stats
        )
        self.assertEqual(len(dataset), 8)
        self.assertEqual(dataset.feature_indices, [1, 3]) # Sorted, unique
        self.assertEqual(dataset.num_features, 2)
        self.assertEqual(dataset.feature_stats, stats)
        # Check remapped stats (output index 0 maps to original 1, output 1 maps to original 3)
        self.assertEqual(dataset._remapped_feature_stats, {0: (0.1, 1.1), 1: (0.3, 1.3)})

    def test_filtered_getitem_with_features(self):
         """Test __getitem__ works with both filtering and feature selection/standardization."""
         indices = [5, 0]
         mean0, std0 = 0.1, 1.1
         mean5, std5 = 0.5, 2.5
         stats = {0: (mean0, std0), 5: (mean5, std5)}
         dataset = FilteredMhcDataset(
             self.df,
             self.root_dir,
             label_of_interest='labelA', # 8 samples pass
             feature_indices=indices,
             feature_stats=stats,
             include_mask=True
         )
         self.assertEqual(len(dataset), 8)

         # Get the first sample of filtered dataset (original index 0)
         sample0 = dataset[0]
         self.assertEqual(sample0['metadata']['healthCode'], "healthCode1")
         self.assertEqual(sample0['labels']['labelA'], 1.0)
         self.assertEqual(sample0['data'].shape, (2, 2, self.time_points)) # 2 days, 2 features
         self.assertEqual(sample0['mask'].shape, (2, 2, self.time_points))

         # Original data slices needed for feature 0 and 5
         orig_day1_f0 = self.day1_p1_data[1, 0, :]
         orig_day1_f5 = self.day1_p1_data[1, 5, :]
         orig_day2_f0 = self.day2_p1_data[1, 0, :]
         orig_day2_f5 = self.day2_p1_data[1, 5, :]
         # Original mask slices
         orig_mask1_f0 = self.day1_p1_data[0, 0, :]
         orig_mask1_f5 = self.day1_p1_data[0, 5, :]
         orig_mask2_f0 = self.day2_p1_data[0, 0, :]
         orig_mask2_f5 = self.day2_p1_data[0, 5, :]


         # Calculate expected standardized values
         exp_day1_f0 = (orig_day1_f0 - mean0) / std0
         exp_day1_f5 = (orig_day1_f5 - mean5) / std5
         exp_day2_f0 = (orig_day2_f0 - mean0) / std0
         exp_day2_f5 = (orig_day2_f5 - mean5) / std5

         # Check data (selected order [0, 5] maps to output [0, 1])
         torch.testing.assert_close(sample0['data'][0, 0, :], torch.from_numpy(exp_day1_f0)) # Output 0 <- Original 0
         torch.testing.assert_close(sample0['data'][0, 1, :], torch.from_numpy(exp_day1_f5)) # Output 1 <- Original 5
         torch.testing.assert_close(sample0['data'][1, 0, :], torch.from_numpy(exp_day2_f0))
         torch.testing.assert_close(sample0['data'][1, 1, :], torch.from_numpy(exp_day2_f5))

         # Check mask (should be selected but not standardized)
         torch.testing.assert_close(sample0['mask'][0, 0, :], torch.from_numpy(orig_mask1_f0))
         torch.testing.assert_close(sample0['mask'][0, 1, :], torch.from_numpy(orig_mask1_f5))
         torch.testing.assert_close(sample0['mask'][1, 0, :], torch.from_numpy(orig_mask2_f0))
         torch.testing.assert_close(sample0['mask'][1, 1, :], torch.from_numpy(orig_mask2_f5))

         # Get another sample from filtered dataset (original index 4 -> filtered index 3)
         sample3 = dataset[3]
         self.assertEqual(sample3['metadata']['healthCode'], "healthCode1")
         self.assertEqual(sample3['labels']['labelA'], 5.0)
         # This sample had a load error (wrong dims)
         self.assertEqual(sample3['data'].shape, (1, 2, self.time_points)) # 1 day, 2 features
         self.assertTrue(torch.all(torch.isnan(sample3['data'])))
         self.assertTrue(torch.all(sample3['mask'] == 0))



    def test_filtered_getitem_index_out_of_bounds(self):
         """Test __getitem__ raises IndexError for invalid index on filtered dataset."""
         dataset = FilteredMhcDataset(self.df, self.root_dir, label_of_interest='labelB') # 7 samples
         with self.assertRaises(IndexError):
             dataset[7] # Index 7 is out of bounds for the filtered length
         with self.assertRaises(IndexError):
             dataset[-1] # Negative indices not directly supported by iloc logic, should raise


if __name__ == '__main__':
    # setUpClass handles logging configuration now
    unittest.main(argv=['first-arg-is-ignored'], exit=False)



