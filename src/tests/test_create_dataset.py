import os
import glob
import numpy as np
import pandas as pd
import pytest
from datetime import timedelta

from create import create_dataset, create_synthetic_dfs
from constants import FileType, HKQuantityType, MotionActivityType, HKWorkoutType


# Helper to attempt loading a parquet file from an environment variable.
def load_parquet_from_env(env_var, pattern="*.parquet"):
    path = os.getenv(env_var)
    if path:
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, pattern))
            if files:
                return pd.read_parquet(files[0])
        elif os.path.isfile(path):
            return pd.read_parquet(path)
    return None


#######################################
# REAL DATA
#######################################
@pytest.fixture
def real_dfs():
    """Fixture that returns a dictionary of DataFrames with the correct
    schema (matching the notebook) for each FileType.
    
    For HealthKit, we expect: startTime, endTime, startTime_timezone_offset,
    endTime_timezone_offset, value, type.
    
    For Motion, we expect:
      startTime, endTime, activity, confidence, startTime_timezone_offset,
      source, appVersion, recordId, healthCode.
    
    For Sleep, we expect:
      startTime, endTime, type, "category value", value, unit, source,
      sourceIdentifier, appVersion, startTime_timezone_offset, healthCode, recordId.
    
    For Workout, we expect:
      startTime, endTime, type, workoutType, totalDistance, unitDistance,
      energyConsumed, unitEnergy, source, sourceIdentifier, metadataBase64,
      appVersion, startTime_timezone_offset, endTime_timezone_offset, healthCode, recordId.
    """
    # HealthKit
    hk_df = load_parquet_from_env("HEALTHKIT_TEST_DATA_DIR")
    if hk_df is None or hk_df.empty:
        start = pd.Timestamp("2022-01-01 00:00:00")
        end = start + pd.Timedelta(hours=1)  # one-hour event
        hk_df = pd.DataFrame({
            "startTime": [start],
            "endTime": [end],
            "startTime_timezone_offset": [0],
            "endTime_timezone_offset": [0],
            "value": [3600.0],
            "type": [list(HKQuantityType)[0].value],
        })
        hk_df.index = hk_df["startTime"]
    
    # Motion
    motion_df = load_parquet_from_env("MOTION_TEST_DATA_DIR")
    if motion_df is None or motion_df.empty:
        start = pd.Timestamp("2022-01-01 00:10:00")
        end = pd.Timestamp("2022-01-01 00:20:00")
        motion_df = pd.DataFrame({
            "startTime": [start],
            "endTime": [end],
            "activity": [list(MotionActivityType)[0].value],
            "confidence": [5.0],
            "startTime_timezone_offset": [0],
            "source": ["DummyMotion"],
            "appVersion": ["1.0"],
            "recordId": ["dummy_motion"],
            "healthCode": ["dummy_motion_code"]
        })
        motion_df.index = motion_df["startTime"]
    
    # Sleep
    sleep_df = load_parquet_from_env("SLEEP_TEST_DATA_FILEPATH")
    if sleep_df is None or sleep_df.empty:
        start = pd.Timestamp("2022-01-01 22:00:00")
        end = pd.Timestamp("2022-01-01 23:00:00")
        sleep_df = pd.DataFrame({
            "startTime": [start],
            "endTime": [end],
            "type": ["HKCategoryTypeIdentifierSleepAnalysis"],
            "category value": ["HKCategoryValueSleepAnalysisAsleep"],
            "value": [3600.0],
            "unit": ["s"],
            "source": ["DummySleep"],
            "sourceIdentifier": ["dummy_sleep"],
            "appVersion": ["1.0"],
            "startTime_timezone_offset": [0],
            "healthCode": ["dummy_sleep_code"],
            "recordId": ["dummy_sleep"]
        })
        sleep_df.index = sleep_df["startTime"]
    
    # Workout
    workout_df = load_parquet_from_env("WORKOUT_TEST_DATA_FILEPATH")
    if workout_df is None or workout_df.empty:
        start = pd.Timestamp("2022-01-01 03:00:00")
        end = pd.Timestamp("2022-01-01 03:30:00")
        workout_df = pd.DataFrame({
            "startTime": [start],
            "endTime": [end],
            "type": ["HKWorkoutTypeIdentifier"],
            "workoutType": [HKWorkoutType.HKWorkoutActivityTypeRunning.value],
            "totalDistance": [5000.0],
            "unitDistance": ["m"],
            "energyConsumed": [300.0],
            "unitEnergy": ["kcal"],
            "source": ["DummyWorkout"],
            "sourceIdentifier": ["dummy_workout"],
            "metadataBase64": ["dummyMetadata"],
            "appVersion": ["1.0"],
            "startTime_timezone_offset": [0],
            "endTime_timezone_offset": [0],
            "healthCode": ["dummy_workout_code"],
            "recordId": ["dummy_workout"]
        })
        workout_df.index = workout_df["startTime"]
    
    return {
        FileType.HEALTHKIT: hk_df,
        FileType.MOTION: motion_df,
        FileType.SLEEP: sleep_df,
        FileType.WORKOUT: workout_df,
    }

#######################################
# SYNTHETIC DATA (using the helper function)
#######################################
@pytest.fixture
def synthetic_dfs():
    """Returns synthetic DataFrames (with the notebook schema) via create_synthetic_dfs."""
    return create_synthetic_dfs()

@pytest.fixture
def expected_output_shape():
    """Calculate expected output shape:
    (2, C, 1440) where:
    - First dimension (2): mask and data
    - Second dimension (C): number of channels (HKQuantityType + MotionActivityType + 2 + HKWorkoutType)
    - Third dimension (1440): minutes in a day
    """
    n_channels = len(HKQuantityType) + len(MotionActivityType) + 2 + len(HKWorkoutType)
    return (2, n_channels, 1440)


#######################################
# Synthetic data tests
#######################################
class TestCreateDatasetSynthetic:
    def test_minute_level_transformation(self, synthetic_dfs, tmp_path, expected_output_shape):
        """Test that create_dataset correctly transforms synthetic data based on the notebook schemas.
        We build the expected minute-level arrays from synthetic inputs.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run dataset creation with synthetic data.
        create_dataset(synthetic_dfs, str(output_dir), force_recompute=True)

        # For the synthetic HealthKit record the event is from 00:00 to 01:00 with value 3600.
        # This corresponds to 1.0 per-second for 3600 seconds, which when resampled produces 1.0 for minutes 0-59.
        n_minutes = 1440

        # HEALTHKIT expected (one record exists only for the type provided; others are NaN)
        hk_expected = []
        hk_mask = []
        synthetic_hk_type = synthetic_dfs[FileType.HEALTHKIT].iloc[0]["type"]
        for hk_type in HKQuantityType:
            if hk_type.value == synthetic_hk_type:
                arr = np.full(n_minutes, np.nan, dtype=np.float32)
                arr[:60] = 1.0  # minutes 0 to 59
                arr[60:] = 0.0
                mask = np.zeros_like(arr)
                mask[:60] = 1  # Only the first 60 minutes have valid data
                hk_expected.append(arr)
                hk_mask.append(mask)
            else:
                arr = np.full(n_minutes, np.nan, dtype=np.float32)
                mask = np.zeros_like(arr)  # No valid data
                hk_expected.append(arr)
                hk_mask.append(mask)
        hk_expected = np.array(hk_expected)
        hk_mask = np.array(hk_mask)

        # MOTION expected: synthetic motion record from 00:10:00 to 00:20:00 with confidence 5.0.
        motion_expected = []
        motion_mask = []
        synthetic_motion = synthetic_dfs[FileType.MOTION].iloc[0]["activity"]
        for motion_type in MotionActivityType:
            if motion_type.value == synthetic_motion:
                arr = np.full(n_minutes, np.nan, dtype=np.float32)
                arr[10:20] = 5.0   # Minutes 10 to 19 get 5.0
                mask = np.zeros_like(arr)
                mask[10:20] = 1  # Only these minutes have valid data
                motion_expected.append(arr)
                motion_mask.append(mask)
            else:
                arr = np.full(n_minutes, np.nan, dtype=np.float32)
                mask = np.zeros_like(arr)
                motion_expected.append(arr)
                motion_mask.append(mask)
        motion_expected = np.array(motion_expected)
        motion_mask = np.array(motion_mask)

        # SLEEP expected
        sleep_expected = np.zeros((2, n_minutes), dtype=np.float32)
        sleep_mask = np.zeros((2, n_minutes), dtype=np.float32)
        sleep_expected[0, 1320:1380] = 1.0  # Asleep from 22:00 to 23:00
        sleep_mask[0, 1320:1380] = 1  # Only mark the sleep period as valid
        sleep_mask[1, :] = 0  # No in-bed data

        # WORKOUT expected
        workout_expected = []
        workout_mask = []
        synthetic_workout = synthetic_dfs[FileType.WORKOUT].iloc[0]["workoutType"]
        for wk in HKWorkoutType:
            if wk.value == synthetic_workout:
                arr = np.zeros(n_minutes, dtype=np.float32)
                arr[180:210] = 1.0  # 03:00 to 03:30
                mask = np.zeros_like(arr)
                mask[180:210] = 1  # Only mark the workout period as valid
                workout_expected.append(arr)
                workout_mask.append(mask)
            else:
                arr = np.full(n_minutes, np.nan, dtype=np.float32)
                mask = np.zeros_like(arr)
                workout_expected.append(arr)
                workout_mask.append(mask)
        workout_expected = np.array(workout_expected)
        workout_mask = np.array(workout_mask)

        # Stack expected matrices vertically in the order: HealthKit, Workout, Sleep, Motion
        expected_data = np.vstack([hk_expected, workout_expected, sleep_expected, motion_expected])
        expected_mask = np.vstack([hk_mask, workout_mask, sleep_mask, motion_mask])
        
        # Reshape to match the new 3D format (2, C, 1440)
        expected_matrix = np.stack([expected_mask, expected_data])
        
        assert expected_matrix.shape == expected_output_shape, (
            f"Expected matrix shape {expected_output_shape} but got {expected_matrix.shape}"
        )

        # Load the computed matrix and compare
        output_filepath = os.path.join(str(output_dir), "2022-01-01.npy")
        assert os.path.exists(output_filepath), "Expected output file not found."
        computed_matrix = np.load(output_filepath)

        np.testing.assert_allclose(
            computed_matrix, expected_matrix, rtol=1e-5, atol=1e-5,
            err_msg="Minute level transformation for synthetic data does not match expected output."
        )

    def test_create_dataset_creates_files(self, synthetic_dfs, tmp_path, expected_output_shape):
        """Test that create_dataset successfully creates .npy files for synthetic data,
        with the correct shape and dtype.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        create_dataset(synthetic_dfs, str(output_dir))

        output_files = list(output_dir.glob("*.npy"))
        assert len(output_files) > 0, "No output files were created for synthetic data."

        for file in output_files:
            array = np.load(file)
            assert array.shape == expected_output_shape, (
                f"Output file {file} has shape {array.shape}, expected {expected_output_shape}."
            )
            assert array.dtype == np.float32, (
                f"Output file {file} dtype is {array.dtype}; expected np.float32."
            )

    def test_create_dataset_skip_existing(self, synthetic_dfs, tmp_path, expected_output_shape, capsys):
        """Test that if an output file already exists and force_recompute is False,
        create_dataset will skip processing for synthetic data.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        test_date = pd.Timestamp("2022-01-01")
        filename = test_date.strftime("%Y-%m-%d") + ".npy"
        output_filepath = output_dir / filename

        # Create initial dummy array and metadata
        dummy_array = np.full(expected_output_shape, -1, dtype=np.float32)
        np.save(output_filepath, dummy_array)
        
        # Create dummy metadata
        dummy_metadata = pd.DataFrame({
            "data_coverage": [0.0],
            "n": [0.0],
            "sum": [0.0],
            "sum_of_squares": [0.0],
            "date": [test_date.date()],
            "original_time_offset": [0]
        })
        dummy_metadata.to_parquet(output_dir / "metadata.parquet")

        create_dataset(synthetic_dfs, str(output_dir), force_recompute=False)

        captured = capsys.readouterr().out
        assert "Skipping" in captured, "Expected skip message not printed for synthetic data."

        loaded_array = np.load(output_filepath)
        np.testing.assert_array_equal(
            loaded_array, dummy_array,
            err_msg="Output file was overwritten for synthetic data despite force_recompute being False."
        )

        # Verify metadata wasn't changed
        loaded_metadata = pd.read_parquet(output_dir / "metadata.parquet")
        pd.testing.assert_frame_equal(loaded_metadata, dummy_metadata), \
            "Metadata was modified despite force_recompute being False"

    def test_create_dataset_force_recompute(self, synthetic_dfs, tmp_path, expected_output_shape):
        """Test that when force_recompute is True, preexisting output files are overwritten for synthetic data.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        test_date = pd.Timestamp("2022-01-01")
        filename = test_date.strftime("%Y-%m-%d") + ".npy"
        output_filepath = output_dir / filename

        dummy_array = np.full(expected_output_shape, -1, dtype=np.float32)
        np.save(output_filepath, dummy_array)

        create_dataset(synthetic_dfs, str(output_dir), force_recompute=True)

        loaded_array = np.load(output_filepath)
        assert not np.array_equal(
            loaded_array, dummy_array
        ), "Output file was not overwritten for synthetic data despite force_recompute being True."

    def test_metadata_file_creation(self, synthetic_dfs, tmp_path, expected_output_shape):
        """Test that create_dataset creates a metadata.parquet file with the expected structure."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        create_dataset(synthetic_dfs, str(output_dir), force_recompute=True)

        # Check that metadata file exists
        metadata_path = output_dir / "metadata.parquet"
        assert metadata_path.exists(), "Metadata file was not created"

        # Load and verify metadata
        metadata_df = pd.read_parquet(metadata_path)

        # Check columns
        expected_columns = ["data_coverage", "n", "sum", "sum_of_squares", "date", "original_time_offset"]
        assert all(col in metadata_df.columns for col in expected_columns), \
            f"Metadata missing expected columns. Found {metadata_df.columns}"

        # Check date
        expected_date = pd.Timestamp("2022-01-01").date()
        assert expected_date in metadata_df["date"].values, \
            f"Expected date {expected_date} not found in metadata"

        # Verify data types
        assert metadata_df["data_coverage"].dtype == np.float64, "data_coverage should be float64"
        assert metadata_df["n"].dtype == np.float64, "n should be float64"
        assert metadata_df["sum"].dtype == np.float64, "sum should be float64"
        assert metadata_df["sum_of_squares"].dtype == np.float64, "sum_of_squares should be float64"
        
        # Check that we have the expected number of rows (one per channel)
        expected_channels = len(HKQuantityType) + len(MotionActivityType) + 2 + len(HKWorkoutType)
        assert len(metadata_df) == expected_channels, \
            f"Expected {expected_channels} rows in metadata, got {len(metadata_df)}"

    def test_metadata_values(self, synthetic_dfs, tmp_path):
        """Test specific metadata values for synthetic data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        create_dataset(synthetic_dfs, str(output_dir), force_recompute=True)

        metadata_df = pd.read_parquet(output_dir / "metadata.parquet")

        # For synthetic HealthKit data (steps from 00:00 to 01:00)
        # First row should correspond to first channel (HealthKit steps)
        steps_row = metadata_df.iloc[0]
        assert np.isclose(steps_row["data_coverage"], 60/1440 * 100), \
            "Incorrect data coverage for steps"
        assert np.isclose(steps_row["n"], 60), \
            "Incorrect number of valid minutes for steps"
        assert np.isclose(steps_row["sum"], 60), \
            "Incorrect sum for steps"
        assert np.isclose(steps_row["sum_of_squares"], 60), \
            "Incorrect sum of squares for steps"

        # For synthetic sleep data (asleep from 22:00 to 23:00)
        # Sleep channels start after HealthKit and Workout channels
        n_healthkit = len(HKQuantityType)
        n_workout = len(HKWorkoutType)
        sleep_row_idx = n_healthkit + n_workout  # First sleep channel index
        asleep_row = metadata_df.iloc[sleep_row_idx]  # First sleep row (asleep)
        assert np.isclose(asleep_row["data_coverage"], 60/1440 * 100), \
            "Incorrect data coverage for sleep"
        assert np.isclose(asleep_row["n"], 60), \
            "Incorrect number of valid minutes for sleep"
        assert np.isclose(asleep_row["sum"], 60), \
            "Incorrect sum for sleep"
        assert np.isclose(asleep_row["sum_of_squares"], 60), \
            "Incorrect sum of squares for sleep"

        # Check timezone offset
        assert all(metadata_df["original_time_offset"] == 0), \
            "Incorrect timezone offset in metadata"

    def test_metadata_force_recompute(self, synthetic_dfs, tmp_path):
        """Test that metadata is properly updated when force_recompute is True."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create initial metadata
        create_dataset(synthetic_dfs, str(output_dir))
        initial_metadata = pd.read_parquet(output_dir / "metadata.parquet")

        # Create dummy metadata with different values
        dummy_metadata = initial_metadata.copy()
        dummy_metadata["data_coverage"] = 0.0
        dummy_metadata.to_parquet(output_dir / "metadata.parquet")

        # Recompute with force_recompute=True
        create_dataset(synthetic_dfs, str(output_dir), force_recompute=True, force_recompute_metadata=True)
        recomputed_metadata = pd.read_parquet(output_dir / "metadata.parquet")

        pd.testing.assert_frame_equal(initial_metadata, recomputed_metadata), \
            "Metadata not properly recomputed when force_recompute=True"

    def test_metadata_mixed_existing_and_new(self, synthetic_dfs, tmp_path, expected_output_shape):
        """Test that metadata is correctly calculated when there's a mix of existing and new .npy files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create two dates for testing
        date1 = pd.Timestamp("2022-01-01")
        date2 = pd.Timestamp("2022-01-02")
        
        # Create synthetic dataframes for both dates
        synthetic_dfs_date1 = synthetic_dfs.copy()
        synthetic_dfs_date2 = create_synthetic_dfs()  # Assuming this creates data for 2022-01-01

        # Adjust the synthetic dataframes to represent different dates
        for file_type, df in synthetic_dfs_date1.items():
            synthetic_dfs_date1[file_type]["startTime"] = df["startTime"].apply(lambda x: date1)
            synthetic_dfs_date1[file_type]["endTime"] = df["endTime"].apply(lambda x: date1 + timedelta(hours=1))
        for file_type, df in synthetic_dfs_date2.items():
            synthetic_dfs_date2[file_type]["startTime"] = df["startTime"].apply(lambda x: date2)
            synthetic_dfs_date2[file_type]["endTime"] = df["endTime"].apply(lambda x: date2 + timedelta(hours=1))

        # Create an existing .npy file for date1
        filename1 = date1.strftime("%Y-%m-%d") + ".npy"
        output_filepath1 = output_dir / filename1
        expected_data = np.full(expected_output_shape, 1.0, dtype=np.float32)
        np.save(output_filepath1, expected_data)

        # Call create_dataset with data for both dates
        all_dfs = {k: pd.concat([synthetic_dfs_date1[k], synthetic_dfs_date2[k]]) for k in synthetic_dfs_date1.keys()}
        create_dataset(all_dfs, str(output_dir), force_recompute=False, force_recompute_metadata=True)

        # Check that metadata file exists
        metadata_path = output_dir / "metadata.parquet"
        assert metadata_path.exists(), "Metadata file was not created"

        # Load and verify metadata
        metadata_df = pd.read_parquet(metadata_path)

        # Check that both dates are present in the metadata
        assert date1.date() in metadata_df["date"].values, f"Date {date1.date()} not found in metadata"
        assert date2.date() in metadata_df["date"].values, f"Date {date2.date()} not found in metadata"

        # Check that the data coverage for date1 is 100% (since we pre-populated the .npy file with ones)
        metadata_date1 = metadata_df[metadata_df["date"] == date1.date()]
        assert np.allclose(metadata_date1["data_coverage"], 100.0), "Incorrect data coverage for date1"

        # Check that the data coverage for date2 is as expected for synthetic data
        metadata_date2 = metadata_df[metadata_df["date"] == date2.date()]
        
        # For synthetic HealthKit data (steps from 00:00 to 01:00)
        # First row should correspond to first channel (HealthKit steps)
        steps_row = metadata_date2.iloc[0]
        assert np.isclose(steps_row["data_coverage"], 60/1440 * 100), \
            "Incorrect data coverage for steps on date2"

