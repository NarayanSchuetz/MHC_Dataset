import pandas as pd
import numpy as np
import os


# def write_metadata_file(output_dir: str, daily_matrix: np.ndarray) -> None:
#     """
#     Creates a metadata file in the output directory.
#     """
#     metadata_df = _create_metadata(daily_matrix)

def calculate_data_coverage(daily_matrix: np.ndarray) -> pd.Series:
    """
    Calculates the data coverage for each file type in the daily matrix.
    
    Data coverage is defined as the percentage of minutes that have valid data
    (neither 0 nor np.nan) for each row in the matrix.
    
    Args:
        daily_matrix: np.ndarray of shape (n_types, 1440) containing minute-level data
        
    Returns:
        pd.Series: Series containing the coverage percentage for each row,
                  with name "data_coverage"
    """
    # Calculate mask for valid data (not zero and not nan)
    valid_data = ~(np.isnan(daily_matrix) | (daily_matrix == 0))
    
    # Calculate percentage of valid data points for each row
    coverage = np.mean(valid_data, axis=1) * 100
    
    # Create series with coverage percentages
    return pd.Series(coverage, name="data_coverage")
    

def compute_array_statistics(daily_minute_level_matrix):
    """
    Compute statistics from a daily_minute_level_matrix for each channel separately.
    
    Args:
        daily_minute_level_matrix (np.ndarray): Array of shape (2, C, 1440) where:
            - First dimension (2): [mask, data]
            - Second dimension (C): number of channels
            - Third dimension (1440): minutes in a day
    
    Returns:
        pd.DataFrame: DataFrame with rows for each channel containing:
            - 'n': number of valid elements (from mask)
            - 'sum': sum of valid elements
            - 'sum_of_squares': sum of squared valid elements
    """
    # Extract mask and data
    mask = daily_minute_level_matrix[0]  # Shape: (C, 1440)
    data = daily_minute_level_matrix[1]  # Shape: (C, 1440)
    
    # Calculate statistics for each channel
    n = np.sum(mask, axis=1)  # Sum along minutes dimension
    
    # Only sum values where mask is 1
    masked_data = data * mask
    sum_values = np.nansum(masked_data, axis=1)
    sum_of_squares = np.nansum((masked_data ** 2), axis=1)
    
    # Create DataFrame with results
    return pd.DataFrame({
        'n': n.astype(float),
        'sum': sum_values.astype(float),
        'sum_of_squares': sum_of_squares.astype(float)
    })