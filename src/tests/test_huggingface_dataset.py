import pytest
import numpy as np
import pandas as pd
import datasets
from datasets import Features, Sequence, Value
import os
import shutil
from pathlib import Path
import logging
import datetime
import time # Import time

# Adjust import path based on your project structure
# Assumes tests are run from the root directory
from src.torch_dataset import FlattenedMhcDataset
from src.huggingface_dataset import mhc_timeseries_generator, create_and_save_hf_dataset_as_gluonTS_style

# Configure logging for tests (optional, can help debugging)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants for dummy data
NUM_FEATURES_RAW = 24
NUM_TIME_POINTS = 1440
SELECTED_FEATURES = [0, 5, 10, 15]
NUM_SELECTED_FEATURES = len(SELECTED_FEATURES)

@pytest.fixture(scope="module") # Use module scope for efficiency
def temp_data_dir(tmp_path_factory):
    """Create a temporary directory for dummy data."""
    temp_dir = tmp_path_factory.mktemp("mhc_test_data")
    logger.info(f"Created temporary data directory: {temp_dir}")
    yield temp_dir
    # Teardown: Remove the directory after tests in the module are done
    logger.info(f"Cleaning up temporary data directory: {temp_dir}")
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="module")
def dummy_dataframe():
    """Create a sample pandas DataFrame."""
    data = {
        'healthCode': ['p1', 'p2', 'p1', 'p3'],
        'time_range': [
            '2023-01-01_2023-01-02', # 2 days
            '2023-01-05_2023-01-05', # 1 day
            '2023-01-03_2023-01-04', # 2 days (one file missing)
            'invalid-range'        # Invalid range
        ],
        'file_uris': [
            ['p1/2023-01-01.npy', 'p1/2023-01-02.npy'],
            ['p2/2023-01-05.npy'],
            ['p1/2023-01-03.npy', 'p1/missing_day.npy'], # missing_day won't exist
            ['p3/nonexistent.npy']
        ],
        'label_value': [5.0, 7.0, np.nan, 6.0]
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def setup_dummy_files(temp_data_dir, dummy_dataframe):
    """Create dummy .npy files based on the dataframe."""
    root_dir = temp_data_dir
    participants = dummy_dataframe['healthCode'].unique()
    for p in participants:
        os.makedirs(root_dir / p, exist_ok=True)

    # Create files that should exist
    files_to_create = {
        'p1/2023-01-01.npy',
        'p1/2023-01-02.npy',
        'p2/2023-01-05.npy',
        'p1/2023-01-03.npy',
    }
    for file_rel_path in files_to_create:
        file_abs_path = root_dir / file_rel_path
        # Create array with mask and data channels
        dummy_array = np.random.rand(2, NUM_FEATURES_RAW, NUM_TIME_POINTS).astype(np.float32)
        dummy_array[0, :, :] = np.random.randint(0, 2, size=(NUM_FEATURES_RAW, NUM_TIME_POINTS)).astype(np.float32) # Binary mask
        np.save(file_abs_path, dummy_array)
        logger.debug(f"Created dummy file: {file_abs_path}")

    return root_dir

@pytest.fixture(scope="module")
def flattened_mhc_dataset(dummy_dataframe, setup_dummy_files):
    """Instantiate FlattenedMhcDataset with dummy data."""
    dataset = FlattenedMhcDataset(
        dataframe=dummy_dataframe,
        root_dir=str(setup_dummy_files),
        include_mask=True,
        feature_indices=SELECTED_FEATURES,
        use_cache=False
    )
    return dataset

# --- Tests for mhc_timeseries_generator --- #

def test_generator_yields_dict(flattened_mhc_dataset):
    """Test that the generator yields dictionaries."""
    generator = mhc_timeseries_generator(flattened_mhc_dataset, mask_key='mask')
    first_item = next(generator)
    assert isinstance(first_item, dict)

def test_generator_keys_and_types(flattened_mhc_dataset):
    """Test the keys and basic types of the yielded dictionary."""
    generator = mhc_timeseries_generator(flattened_mhc_dataset, mask_key='mask')
    first_item = next(generator)

    expected_keys = {"target", "start", "freq", "item_id", "feat_dynamic_real"}
    assert set(first_item.keys()) == expected_keys

    assert isinstance(first_item["target"], np.ndarray)
    assert first_item["target"].dtype == np.float32
    assert isinstance(first_item["start"], pd.Timestamp)
    assert first_item["start"].tz == datetime.timezone.utc
    assert isinstance(first_item["freq"], str)
    assert first_item["freq"] == 'T'
    assert isinstance(first_item["item_id"], str)
    assert isinstance(first_item["feat_dynamic_real"], np.ndarray)
    assert first_item["feat_dynamic_real"].dtype == np.float32

def test_generator_target_shape(flattened_mhc_dataset):
    """Test the shape of the target array."""
    generator = mhc_timeseries_generator(flattened_mhc_dataset, mask_key='mask')
    # Sample 0 has 2 days
    item0 = next(generator)
    expected_len0 = 2 * NUM_TIME_POINTS
    assert item0["target"].shape == (NUM_SELECTED_FEATURES, expected_len0)
    assert item0["feat_dynamic_real"].shape == (NUM_SELECTED_FEATURES, expected_len0)

    # Sample 1 has 1 day
    item1 = next(generator)
    expected_len1 = 1 * NUM_TIME_POINTS
    assert item1["target"].shape == (NUM_SELECTED_FEATURES, expected_len1)
    assert item1["feat_dynamic_real"].shape == (NUM_SELECTED_FEATURES, expected_len1)

def test_generator_handles_missing_files(flattened_mhc_dataset):
    """Test generator behavior with missing files (sample 2)."""
    generator = mhc_timeseries_generator(flattened_mhc_dataset, mask_key='mask')
    items = list(generator)
    # Should skip sample 3 (invalid range) and potentially log warnings for missing files in sample 2
    assert len(items) == 3 # p1_0, p2_1, p1_2 (p3_3 is skipped due to invalid range)

    item2 = items[2] # Corresponds to original index 2 (p1, 2023-01-03_2023-01-04)
    assert item2["item_id"] == "p1_2" # Check item_id derived correctly
    expected_len2 = 2 * NUM_TIME_POINTS # Still 2 days expected range
    assert item2["target"].shape == (NUM_SELECTED_FEATURES, expected_len2)
    assert item2["feat_dynamic_real"].shape == (NUM_SELECTED_FEATURES, expected_len2)
    # FlattenedMhcDataset fills missing days with 0.0 (after nan_to_num)
    # Check if the second day's data (indices 1440 onwards) is all zero
    assert np.all(item2["target"][:, NUM_TIME_POINTS:] == 0)
    # Mask for the missing day should also be zero
    assert np.all(item2["feat_dynamic_real"][:, NUM_TIME_POINTS:] == 0)

def test_generator_no_mask(flattened_mhc_dataset):
    """Test generator without requesting the mask."""
    generator = mhc_timeseries_generator(flattened_mhc_dataset, mask_key=None)
    first_item = next(generator)
    assert "feat_dynamic_real" not in first_item

# --- Tests for create_and_save_hf_dataset --- #

@pytest.fixture
def hf_save_path(tmp_path):
    """Create a temporary directory for saving the HF dataset."""
    save_dir = tmp_path / "hf_dataset_output"
    yield str(save_dir)
    # Clean up if exists (optional, tmp_path usually handles it)
    if save_dir.exists():
        shutil.rmtree(save_dir)

def test_create_and_save_basic(flattened_mhc_dataset, hf_save_path):
    """Test basic dataset creation and saving."""
    create_and_save_hf_dataset_as_gluonTS_style(
        torch_dataset=flattened_mhc_dataset,
        save_path=hf_save_path,
        num_features=NUM_SELECTED_FEATURES,
        include_mask_as_dynamic_feature=False,
        num_proc=2 # Force single process saving
    )
    
    # Add a small delay to rule out filesystem timing issues
    time.sleep(0.1)
    
    # Check if essential files exist
    assert os.path.exists(hf_save_path)
    assert os.path.exists(os.path.join(hf_save_path, "dataset_info.json"))
    assert os.path.exists(os.path.join(hf_save_path, "state.json"))
    
    # Force a reload attempt - this will fail if dataset.arrow (or others) is truly missing
    try:
        reloaded_dataset = datasets.load_from_disk(hf_save_path)
        logger.info(f"Successfully reloaded dataset: {reloaded_dataset}")
        # ---> Attempt to access data to force reading the Arrow file <--- 
        logger.info("Attempting to access first element...")
        first_element = reloaded_dataset[0] 
        logger.info(f"Successfully accessed first element: {first_element}")
        # <--------------------------------------------------------------->

        # If reload succeeds, the arrow file must exist, even if os.path.exists failed earlier
        assert len(reloaded_dataset) == 3 # Check expected number of rows
    except Exception as e:
        pytest.fail(f"Failed to reload dataset from {hf_save_path}. Save likely failed silently. Error: {e}")

def test_create_and_save_with_mask(flattened_mhc_dataset, hf_save_path):
    """Test saving with the mask included."""
    create_and_save_hf_dataset_as_gluonTS_style(
        torch_dataset=flattened_mhc_dataset,
        save_path=hf_save_path,
        num_features=NUM_SELECTED_FEATURES,
        include_mask_as_dynamic_feature=True
    )
    assert os.path.exists(hf_save_path)
    reloaded_dataset = datasets.load_from_disk(hf_save_path)
    assert "feat_dynamic_real" in reloaded_dataset.features
    assert len(reloaded_dataset) == 3 # Check number of valid samples
    # Check shapes in the first sample
    sample0 = reloaded_dataset[0]
    assert np.array(sample0["target"]).shape == (NUM_SELECTED_FEATURES, 2 * NUM_TIME_POINTS)
    assert np.array(sample0["feat_dynamic_real"]).shape == (NUM_SELECTED_FEATURES, 2 * NUM_TIME_POINTS)

def test_reload_saved_dataset(flattened_mhc_dataset, hf_save_path):
    """Test reloading the saved dataset and verifying schema and content."""
    create_and_save_hf_dataset_as_gluonTS_style(
        torch_dataset=flattened_mhc_dataset,
        save_path=hf_save_path,
        num_features=NUM_SELECTED_FEATURES,
        include_mask_as_dynamic_feature=True
    )
    reloaded_dataset = datasets.load_from_disk(hf_save_path)

    # Verify features schema
    expected_feature_dict = {
        "target": Sequence(Sequence(Value("float32")), length=NUM_SELECTED_FEATURES),
        "start": Value("timestamp[us]", id='start'),
        "freq": Value("string"),
        "item_id": Value("string"),
        "feat_dynamic_real": Sequence(Sequence(Value("float32")), length=NUM_SELECTED_FEATURES)
    }
    assert reloaded_dataset.features == Features(expected_feature_dict)

    # Verify content of the first sample (more thorough check)
    original_sample_0 = next(mhc_timeseries_generator(flattened_mhc_dataset, mask_key='mask'))
    reloaded_sample_0 = reloaded_dataset[0]

    np.testing.assert_array_equal(np.array(reloaded_sample_0["target"]), original_sample_0["target"])
    assert reloaded_sample_0["start"] == original_sample_0["start"].to_datetime64()
    assert reloaded_sample_0["freq"] == original_sample_0["freq"]
    assert reloaded_sample_0["item_id"] == original_sample_0["item_id"]
    np.testing.assert_array_equal(np.array(reloaded_sample_0["feat_dynamic_real"]), original_sample_0["feat_dynamic_real"])

def test_num_features_inference(flattened_mhc_dataset, hf_save_path):
    """Test if num_features is inferred correctly when not provided."""
    create_and_save_hf_dataset_as_gluonTS_style(
        torch_dataset=flattened_mhc_dataset,
        save_path=hf_save_path,
        num_features=None, # Let the function infer
        include_mask_as_dynamic_feature=False
    )
    reloaded_dataset = datasets.load_from_disk(hf_save_path)
    assert reloaded_dataset.features["target"].length == NUM_SELECTED_FEATURES

def test_create_and_save_with_options(flattened_mhc_dataset, hf_save_path, tmp_path):
    """Test passing optional arguments like cache_dir, num_proc, keep_in_memory."""
    cache_dir = tmp_path / "hf_cache"
    create_and_save_hf_dataset_as_gluonTS_style(
        torch_dataset=flattened_mhc_dataset,
        save_path=hf_save_path,
        num_features=NUM_SELECTED_FEATURES,
        include_mask_as_dynamic_feature=False,
        cache_dir=str(cache_dir), # Pass cache dir
        num_proc=2, # Test multiprocessing (might not have significant effect on small data)
        keep_in_memory=True # Test keeping in memory
    )
    assert os.path.exists(hf_save_path)
    # Basic check that it ran without errors
    reloaded_dataset = datasets.load_from_disk(hf_save_path)
    assert len(reloaded_dataset) == 3

    # Check if cache dir was potentially used (difficult to assert definitively)
    # assert cache_dir.exists() # Cache dir might be created even if not heavily used 