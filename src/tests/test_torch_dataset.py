import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import shutil

# Assume classes are in src/torch_dataset.py
# Adjust import path if needed
from src.torch_dataset import BaseMhcDataset, FilteredMhcDataset

@pytest.fixture(scope="function") # Recreate for each test function
def mhc_data_fixture():
    """Pytest fixture to create temporary data for MHC dataset tests."""
    temp_data_dir = Path("./temp_test_mhc_data")
    temp_data_dir.mkdir(exist_ok=True)

    file_uris = []
    num_samples = 5
    data_shape = (50, 2) # Example: 50 time steps, 2 features
    health_codes = ['HC_0', 'HC_0', 'HC_1', 'HC_1', 'HC_1']

    # Create dummy .npy files
    for i in range(num_samples):
        dummy_data = np.random.rand(*data_shape).astype(np.float32)
        # Ensure file paths match what might be generated
        hc_dir = health_codes[i]
        file_uri = f"{hc_dir}/data_{i}.npy"
        full_path = temp_data_dir / file_uri
        full_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(full_path, dummy_data)
        file_uris.append(file_uri)

    # Create dummy DataFrame
    dummy_df_data = {
        'healthCode': health_codes,
        'time_range': [f'2023-01-{i+1:02d}_2023-01-{i+8:02d}' for i in range(num_samples)],
        'file_uris': file_uris,
        'happiness_value': [5.0, np.nan, 3.0, 4.0, 2.0],
        'happiness_date': pd.to_datetime([f'2023-01-{i+1:02d}' for i in range(num_samples)]),
        'sleep_value': [1.0, 0.0, 1.0, np.nan, 1.0],
        'sleep_date': pd.to_datetime([f'2023-01-{i+1:02d}' for i in range(num_samples)]),
        'extra_col': list(range(num_samples)) # Non-label column
    }
    dummy_df = pd.DataFrame(dummy_df_data)

    # Yield the dataframe and the temp dir path
    yield dummy_df, str(temp_data_dir)

    # Teardown: remove the temporary directory
    shutil.rmtree(temp_data_dir)

# --- Tests for BaseMhcDataset --- 

def test_base_dataset_init_len(mhc_data_fixture):
    """Test initialization and length of BaseMhcDataset."""
    df, root_dir = mhc_data_fixture
    dataset = BaseMhcDataset(dataframe=df, root_dir=root_dir)
    
    assert len(dataset) == 5
    assert isinstance(dataset, Dataset)
    assert hasattr(dataset, 'label_cols')
    # Check if label cols are identified correctly (sorted)
    assert dataset.label_cols == ['happiness_value', 'sleep_value']
    assert dataset.root_dir == Path(root_dir)

def test_base_dataset_getitem(mhc_data_fixture):
    """Test item retrieval from BaseMhcDataset."""
    df, root_dir = mhc_data_fixture
    dataset = BaseMhcDataset(dataframe=df, root_dir=root_dir)
    
    # Test first item (happiness=5.0, sleep=1.0)
    sample_0 = dataset[0]
    assert isinstance(sample_0, dict)
    assert 'data' in sample_0
    assert 'labels' in sample_0
    assert 'metadata' in sample_0
    
    # Check data tensor
    assert isinstance(sample_0['data'], torch.Tensor)
    assert sample_0['data'].shape == (50, 2) # Matches data_shape in fixture
    assert sample_0['data'].dtype == torch.float32

    # Check labels (should match df row 0)
    assert isinstance(sample_0['labels'], dict)
    assert sample_0['labels']['happiness'] == 5.0
    assert sample_0['labels']['sleep'] == 1.0
    assert 'extra' not in sample_0['labels'] # Check only _value cols are labels

    # Check metadata
    assert sample_0['metadata']['healthCode'] == 'HC_0'
    assert sample_0['metadata']['file_uri'] == df.iloc[0]['file_uris']

    # Test second item (happiness=NaN, sleep=0.0)
    sample_1 = dataset[1]
    assert np.isnan(sample_1['labels']['happiness']) # Check NaN handling
    assert sample_1['labels']['sleep'] == 0.0

def test_base_dataset_getitem_errors(mhc_data_fixture):
    """Test error handling during item retrieval."""
    df, root_dir = mhc_data_fixture
    
    # Test index out of bounds
    dataset = BaseMhcDataset(dataframe=df, root_dir=root_dir)
    with pytest.raises(IndexError):
        _ = dataset[5]
    with pytest.raises(IndexError):
        _ = dataset[-1] # Should raise error, not wrap around
        
    # Test file not found (modify a file_uri to be incorrect)
    df_bad_uri = df.copy()
    df_bad_uri.loc[0, 'file_uris'] = 'nonexistent/path/data.npy'
    dataset_bad_uri = BaseMhcDataset(dataframe=df_bad_uri, root_dir=root_dir)
    with pytest.raises(FileNotFoundError):
        _ = dataset_bad_uri[0]
        
# --- Tests for FilteredMhcDataset --- 

def test_filtered_dataset_init_len(mhc_data_fixture):
    """Test initialization and filtering of FilteredMhcDataset."""
    df, root_dir = mhc_data_fixture
    
    # Filter by 'happiness' (should exclude row 1 where happiness is NaN)
    dataset_h = FilteredMhcDataset(dataframe=df, root_dir=root_dir, label_of_interest='happiness')
    assert len(dataset_h) == 4 # Original 5 - 1 NaN = 4
    assert dataset_h.label_of_interest == 'happiness'
    # Check underlying df is filtered
    assert 'happiness_value' in dataset_h.df.columns
    assert not dataset_h.df['happiness_value'].isnull().any()

    # Filter by 'sleep' (should exclude row 3 where sleep is NaN)
    dataset_s = FilteredMhcDataset(dataframe=df, root_dir=root_dir, label_of_interest='sleep')
    assert len(dataset_s) == 4 # Original 5 - 1 NaN = 4
    assert dataset_s.label_of_interest == 'sleep'
    assert not dataset_s.df['sleep_value'].isnull().any()

def test_filtered_dataset_init_errors(mhc_data_fixture):
    """Test initialization errors for FilteredMhcDataset."""
    df, root_dir = mhc_data_fixture
    
    # Test label not found
    with pytest.raises(ValueError, match="Label column 'nonexistent_label_value' not found"):
        _ = FilteredMhcDataset(dataframe=df, root_dir=root_dir, label_of_interest='nonexistent_label')
        
    # Test no samples left after filtering
    df_all_nan = df.copy()
    df_all_nan['happiness_value'] = np.nan # Make all happiness NaN
    with pytest.raises(ValueError, match="No samples found with non-NaN values for label 'happiness'"):
        _ = FilteredMhcDataset(dataframe=df_all_nan, root_dir=root_dir, label_of_interest='happiness')
        
def test_filtered_dataset_getitem(mhc_data_fixture):
    """Test item retrieval from FilteredMhcDataset."""
    df, root_dir = mhc_data_fixture
    
    # Filter by 'happiness'
    dataset_h = FilteredMhcDataset(dataframe=df, root_dir=root_dir, label_of_interest='happiness')
    assert len(dataset_h) == 4
    
    # Get first sample from filtered dataset (should correspond to original index 0)
    sample_0_filt = dataset_h[0]
    assert sample_0_filt['metadata']['healthCode'] == 'HC_0'
    assert sample_0_filt['labels']['happiness'] == 5.0 # Check the label is present
    assert sample_0_filt['labels']['sleep'] == 1.0
    
    # Get second sample from filtered dataset (should correspond to original index 2)
    sample_1_filt = dataset_h[1] 
    assert sample_1_filt['metadata']['healthCode'] == 'HC_1' 
    assert sample_1_filt['labels']['happiness'] == 3.0
    assert sample_1_filt['labels']['sleep'] == 1.0
    
    # Check last sample (original index 4)
    sample_last_filt = dataset_h[3]
    assert sample_last_filt['metadata']['healthCode'] == 'HC_1' 
    assert sample_last_filt['labels']['happiness'] == 2.0
    assert sample_last_filt['labels']['sleep'] == 1.0
    
    # Check index error for filtered dataset
    with pytest.raises(IndexError):
        _ = dataset_h[4] 