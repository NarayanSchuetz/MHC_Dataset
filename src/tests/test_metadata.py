import numpy as np
import pytest

from metadata import compute_array_statistics
from constants import HKQuantityType, MotionActivityType, HKWorkoutType


def __compute_global_mean_and_std(stats_list):
    """
    Calculate global mean and standard deviation from list of array statistics.
    
    Args:
        stats_list (list): List of DataFrames containing:
            - 'n': number of elements
            - 'sum': sum of elements
            - 'sum_of_squares': sum of squared elements
    
    Returns:
        tuple: (global_mean, global_std)
    """
    total_n = sum(df['n'].sum() for df in stats_list)
    total_sum = sum(df['sum'].sum() for df in stats_list)
    total_sum_sq = sum(df['sum_of_squares'].sum() for df in stats_list)
    
    if total_n == 0:
        return 0.0, 1.0  # Return default values if no valid data
        
    global_mean = total_sum / total_n
    global_var = (total_sum_sq / total_n) - (global_mean ** 2)
    global_std = np.sqrt(max(global_var, 1e-8))  # Add small epsilon to prevent division by zero
    
    return global_mean, global_std

def __standardize_array(array, global_mean, global_std):
    """
    Standardize array using precomputed global statistics.
    
    Args:
        array (np.ndarray): Array to standardize
        global_mean (float): Precomputed global mean
        global_std (float): Precomputed global standard deviation
    
    Returns:
        np.ndarray: Standardized array
    """
    return (array - global_mean) / global_std


@pytest.fixture
def synthetic_daily_matrices():
    """Create synthetic daily matrices for testing."""
    n_channels = len(HKQuantityType) + len(MotionActivityType) + 2 + len(HKWorkoutType)
    n_days = 10
    np.random.seed(42)
    
    daily_matrices = []
    for _ in range(n_days):
        # Create random data
        data = np.random.normal(50, 10, size=(n_channels, 1440))
        # Create random mask (some valid, some invalid data)
        mask = np.random.choice([0, 1], size=(n_channels, 1440), p=[0.2, 0.8])
        # Stack mask and data
        daily_matrix = np.stack([mask, data])
        daily_matrices.append(daily_matrix)
    
    return daily_matrices


def test_array_statistics_computation(synthetic_daily_matrices):
    """Test that array statistics are computed correctly."""
    # Compute statistics for each daily matrix
    stats_list = [compute_array_statistics(matrix) for matrix in synthetic_daily_matrices]
    
    # Compute global statistics
    global_mean, global_std = __compute_global_mean_and_std(stats_list)
    
    # Compute direct statistics for comparison
    all_valid_data = []
    for matrix in synthetic_daily_matrices:
        mask = matrix[0]
        data = matrix[1]
        valid_data = data[mask == 1]
        all_valid_data.extend(valid_data)
    
    all_valid_data = np.array(all_valid_data)
    direct_mean = np.mean(all_valid_data)
    direct_std = np.std(all_valid_data)
    
    # Test that computed values match direct calculation within tolerance
    np.testing.assert_allclose(global_mean, direct_mean, rtol=1e-5)
    np.testing.assert_allclose(global_std, direct_std, rtol=1e-5)
    
    # Additional test to verify per-channel calculations
    first_matrix = synthetic_daily_matrices[0]
    first_stats = compute_array_statistics(first_matrix)
    
    # Test shape matches number of channels
    assert len(first_stats) == first_matrix.shape[1]
    
    # Test individual channel calculations
    for channel in range(first_matrix.shape[1]):
        mask = first_matrix[0, channel]
        data = first_matrix[1, channel]
        valid_data = data[mask == 1]
        
        assert first_stats.iloc[channel]['n'] == np.sum(mask)
        assert np.isclose(first_stats.iloc[channel]['sum'], np.sum(valid_data))
        assert np.isclose(first_stats.iloc[channel]['sum_of_squares'], np.sum(valid_data ** 2))


def test_array_standardization(synthetic_daily_matrices):
    """Test that array standardization produces expected statistics."""
    # Compute global statistics
    stats_list = [compute_array_statistics(matrix) for matrix in synthetic_daily_matrices]
    global_mean, global_std = __compute_global_mean_and_std(stats_list)
    
    # Test standardization on random array
    n_channels = len(HKQuantityType) + len(MotionActivityType) + 2 + len(HKWorkoutType)
    np.random.seed(42)
    test_array = np.random.normal(50, 10, size=(n_channels, 1440))
    standardized = __standardize_array(test_array, global_mean, global_std)
    
    # Verify standardization produces approximately zero mean and unit std
    np.testing.assert_allclose(np.mean(standardized), 0, atol=1e-1)
    np.testing.assert_allclose(np.std(standardized), 1, atol=1e-1)