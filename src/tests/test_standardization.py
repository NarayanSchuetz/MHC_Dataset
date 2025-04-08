import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pandas.testing import assert_frame_equal

# Adjust the import path based on your project structure
from src.standardization import calculate_standardization_from_files


@pytest.fixture
def create_metadata_file():
    """Fixture to create a temporary metadata parquet file."""
    temp_files = []

    def _create_file(data, index_name='feature_index', filename_suffix=""):
        df = pd.DataFrame(data)
        if index_name:
            if index_name in df.columns:
                df = df.set_index(index_name)
            else:
                 # Create a dummy index if 'feature_index' not in columns and index_name is expected
                 df.index.name = index_name

        # Ensure correct dtypes if columns exist
        if 'n' in df.columns: df['n'] = df['n'].astype(float)
        if 'sum' in df.columns: df['sum'] = df['sum'].astype(float)
        if 'sum_of_squares' in df.columns: df['sum_of_squares'] = df['sum_of_squares'].astype(float)

        # Use NamedTemporaryFile to handle file creation and cleanup
        # Keep the file open until the test using it is done
        # delete=False is important on some OS (like Windows) when passing the name
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{filename_suffix}.parquet')
        df.to_parquet(temp_file.name)
        temp_files.append(temp_file.name)
        temp_file.close() # Close the file handle but don't delete yet
        return temp_file.name

    yield _create_file

    # Cleanup: Remove all created temporary files
    for f_path in temp_files:
        try:
            os.remove(f_path)
        except OSError:
            pass # Ignore errors if file already deleted or doesn't exist


class TestCalculateStandardizationHappyPath:
    """Tests for successful standardization calculation."""

    def test_single_file(self, create_metadata_file):
        """Test calculation with a single valid metadata file."""
        # Data corresponds to:
        # Feature 0: values [1, 2, 3] -> n=3, sum=6, sum_sq=14 -> mean=2, var=(14/3)-4=2/3, std=sqrt(2/3)
        # Feature 1: values [10, 10] -> n=2, sum=20, sum_sq=200 -> mean=10, var=(200/2)-100=0, std=0
        data = {
            'feature_index': [0, 1],
            'n': [3.0, 2.0],
            'sum': [6.0, 20.0],
            'sum_of_squares': [14.0, 200.0]
        }
        file_path = create_metadata_file(data, filename_suffix="single")

        expected_mean = pd.Series([2.0, 10.0], index=[0, 1], name='mean')
        expected_std = pd.Series([np.sqrt(2/3), 0.0], index=[0, 1], name='std_dev')
        expected_df = pd.DataFrame({'mean': expected_mean, 'std_dev': expected_std})
        expected_df.index.name = 'feature_index'

        result_df = calculate_standardization_from_files([file_path])
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)

    def test_multiple_files(self, create_metadata_file):
        """Test calculation aggregating data from multiple files."""
        # File 1: Feature 0: [1, 2], n=2, sum=3, sum_sq=5
        # File 1: Feature 1: [5],   n=1, sum=5, sum_sq=25
        data1 = {'feature_index': [0, 1], 'n': [2.0, 1.0], 'sum': [3.0, 5.0], 'sum_of_squares': [5.0, 25.0]}
        # File 2: Feature 0: [3],   n=1, sum=3, sum_sq=9
        # File 2: Feature 2: [1, 1], n=2, sum=2, sum_sq=2
        data2 = {'feature_index': [0, 2], 'n': [1.0, 2.0], 'sum': [3.0, 2.0], 'sum_of_squares': [9.0, 2.0]}

        file_path1 = create_metadata_file(data1, filename_suffix="multi1")
        file_path2 = create_metadata_file(data2, filename_suffix="multi2")

        # Aggregated:
        # Feature 0: n=3, sum=6, sum_sq=14 -> mean=2, var=2/3, std=sqrt(2/3)
        # Feature 1: n=1, sum=5, sum_sq=25 -> mean=5, var=0, std=0
        # Feature 2: n=2, sum=2, sum_sq=2 -> mean=1, var=0, std=0
        expected_mean = pd.Series([2.0, 5.0, 1.0], index=[0, 1, 2], name='mean')
        expected_std = pd.Series([np.sqrt(2/3), 0.0, 0.0], index=[0, 1, 2], name='std_dev')
        expected_df = pd.DataFrame({'mean': expected_mean, 'std_dev': expected_std})
        expected_df.index.name = 'feature_index'
        expected_df = expected_df.sort_index() # Ensure sorted index

        result_df = calculate_standardization_from_files([file_path1, file_path2])
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)

    def test_zero_variance(self, create_metadata_file):
        """Test calculation where standard deviation is zero."""
        # Feature 0: [5, 5, 5], n=3, sum=15, sum_sq=75 -> mean=5, var=0, std=0
        data = {'feature_index': [0], 'n': [3.0], 'sum': [15.0], 'sum_of_squares': [75.0]}
        file_path = create_metadata_file(data, filename_suffix="zero_var")

        expected_df = pd.DataFrame({'mean': [5.0], 'std_dev': [0.0]}, index=pd.Index([0], name='feature_index'))

        result_df = calculate_standardization_from_files([file_path])
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)

    def test_index_as_column(self, create_metadata_file):
        """Test when feature_index is a column, not the index."""
        data = {
            'feature_index': [0, 1], # This will become the index
            'n': [3.0, 2.0],
            'sum': [6.0, 20.0],
            'sum_of_squares': [14.0, 200.0]
        }
        # Create file without setting index initially
        file_path = create_metadata_file(data, index_name=None, filename_suffix="idx_col")

        # Expected result is same as test_single_file
        expected_mean = pd.Series([2.0, 10.0], index=[0, 1], name='mean')
        expected_std = pd.Series([np.sqrt(2/3), 0.0], index=[0, 1], name='std_dev')
        expected_df = pd.DataFrame({'mean': expected_mean, 'std_dev': expected_std})
        expected_df.index.name = 'feature_index'

        result_df = calculate_standardization_from_files([file_path])
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)

    def test_unnamed_index(self, create_metadata_file):
        """Test when the index is unnamed but should be used."""
        data = {
            'n': [3.0, 2.0],
            'sum': [6.0, 20.0],
            'sum_of_squares': [14.0, 200.0]
        }
        # Create DataFrame and save without explicit index name
        df = pd.DataFrame(data, index=[0, 1]) # Index is [0, 1], name is None
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='_unnamed_idx.parquet')
        df.to_parquet(temp_file.name)
        file_path = temp_file.name
        temp_file.close()


        # Expected result is same as test_single_file
        expected_mean = pd.Series([2.0, 10.0], index=[0, 1], name='mean')
        expected_std = pd.Series([np.sqrt(2/3), 0.0], index=[0, 1], name='std_dev')
        expected_df = pd.DataFrame({'mean': expected_mean, 'std_dev': expected_std})
        expected_df.index.name = 'feature_index' # Function should assign this name

        result_df = calculate_standardization_from_files([file_path])
        # Clean up the manually created temp file
        os.remove(file_path)
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)


class TestCalculateStandardizationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_file_list(self):
        """Test calculation with an empty list of file paths."""
        result_df = calculate_standardization_from_files([])
        expected_df = pd.DataFrame(columns=['mean', 'std_dev'])
        # Check if index is empty or has expected name based on implementation
        # Current implementation returns empty df without index name set
        assert_frame_equal(result_df, expected_df, check_dtype=False)
        assert result_df.empty
        assert list(result_df.columns) == ['mean', 'std_dev']


    def test_missing_columns(self, create_metadata_file):
        """Test skipping a file if required columns are missing."""
        # File 1 (valid)
        data1 = {'feature_index': [0], 'n': [2.0], 'sum': [4.0], 'sum_of_squares': [10.0]} # mean=2, var=(10/2)-4=1, std=1
        # File 2 (missing 'sum_of_squares')
        data2 = {'feature_index': [0], 'n': [1.0], 'sum': [3.0]}
        # File 3 (valid)
        data3 = {'feature_index': [1], 'n': [3.0], 'sum': [30.0], 'sum_of_squares': [300.0]} # mean=10, var=0, std=0

        file1 = create_metadata_file(data1, filename_suffix="valid1")
        file2 = create_metadata_file(data2, filename_suffix="invalid_cols")
        file3 = create_metadata_file(data3, filename_suffix="valid2")

        # Only file1 and file3 should be processed
        expected_df = pd.DataFrame(
            {'mean': [2.0, 10.0], 'std_dev': [1.0, 0.0]},
            index=pd.Index([0, 1], name='feature_index')
        )

        result_df = calculate_standardization_from_files([file1, file2, file3])
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)


    def test_incorrect_index_name(self, create_metadata_file):
        """Test skipping file with wrong index name and no 'feature_index' column."""
        data_correct = {'feature_index': [0], 'n': [1.], 'sum': [1.], 'sum_of_squares': [1.]}
        data_wrong_index = {'n': [1.], 'sum': [2.], 'sum_of_squares': [4.]} # Index will be [0]

        file_correct = create_metadata_file(data_correct, filename_suffix="correct_idx")
        # Create the incorrect file manually to set a specific wrong index name
        df_wrong = pd.DataFrame(data_wrong_index, index=pd.Index([1], name='wrong_index'))
        temp_file_wrong = tempfile.NamedTemporaryFile(delete=False, suffix='_wrong_idx.parquet')
        df_wrong.to_parquet(temp_file_wrong.name)
        file_wrong = temp_file_wrong.name
        temp_file_wrong.close()


        expected_df = pd.DataFrame({'mean': [1.0], 'std_dev': [0.0]}, index=pd.Index([0], name='feature_index'))

        result_df = calculate_standardization_from_files([file_correct, file_wrong])
        os.remove(file_wrong) # Manual cleanup
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)

    def test_non_numeric_data(self, create_metadata_file):
        """Test skipping file with non-numeric data in stat columns."""
        data_valid = {'feature_index': [0], 'n': [1.0], 'sum': [5.0], 'sum_of_squares': [25.0]}
        # Create invalid data directly in DataFrame before saving
        df_invalid = pd.DataFrame({
            'feature_index': [1],
            'n': [2.0],
            'sum': ['not a number'], # Invalid data
            'sum_of_squares': [50.0]
        }).set_index('feature_index')

        file_valid = create_metadata_file(data_valid, filename_suffix="valid_num")
        temp_file_invalid = tempfile.NamedTemporaryFile(delete=False, suffix='_non_numeric.parquet')
        df_invalid.to_parquet(temp_file_invalid.name)
        file_invalid = temp_file_invalid.name
        temp_file_invalid.close()

        expected_df = pd.DataFrame({'mean': [5.0], 'std_dev': [0.0]}, index=pd.Index([0], name='feature_index'))

        result_df = calculate_standardization_from_files([file_valid, file_invalid])
        os.remove(file_invalid) # Manual cleanup
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)


    def test_nan_stats(self, create_metadata_file):
        """Test skipping rows with NaN values in statistics."""
        data = {
            'feature_index': [0, 1, 2],
            'n': [2.0, np.nan, 3.0], # Feature 1 has NaN n
            'sum': [4.0, 5.0, 6.0],
            'sum_of_squares': [10.0, 25.0, np.nan] # Feature 2 has NaN sum_sq
        }
        file_path = create_metadata_file(data, filename_suffix="nan_stats")

        # Only feature 0 should be processed (mean=2, var=1, std=1)
        expected_df = pd.DataFrame(
            {'mean': [2.0], 'std_dev': [1.0]},
            index=pd.Index([0], name='feature_index')
        )

        result_df = calculate_standardization_from_files([file_path])
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)


    def test_zero_n(self, create_metadata_file):
        """Test calculation results in NaN when total n is 0 for a feature."""
        # File 1: Feature 0 has n=0
        data1 = {'feature_index': [0, 1], 'n': [0.0, 2.0], 'sum': [0.0, 10.0], 'sum_of_squares': [0.0, 50.0]}
        # File 2: Feature 0 also has n=0
        data2 = {'feature_index': [0, 2], 'n': [0.0, 1.0], 'sum': [0.0, 5.0], 'sum_of_squares': [0.0, 25.0]}

        file1 = create_metadata_file(data1, filename_suffix="zero_n1")
        file2 = create_metadata_file(data2, filename_suffix="zero_n2")

        # Feature 0: total_n = 0 -> mean=NaN, std=NaN
        # Feature 1: n=2, sum=10, sum_sq=50 -> mean=5, var=0, std=0
        # Feature 2: n=1, sum=5, sum_sq=25 -> mean=5, var=0, std=0
        expected_df = pd.DataFrame(
            {'mean': [np.nan, 5.0, 5.0], 'std_dev': [np.nan, 0.0, 0.0]},
            index=pd.Index([0, 1, 2], name='feature_index')
        ).sort_index()

        result_df = calculate_standardization_from_files([file1, file2])
        # Need to check NaNs carefully
        assert_frame_equal(result_df.dropna(), expected_df.dropna(), check_dtype=False, atol=1e-6)
        assert pd.isna(result_df.loc[0, 'mean'])
        assert pd.isna(result_df.loc[0, 'std_dev'])
        assert result_df.index.name == 'feature_index'


    def test_file_not_found(self, create_metadata_file, capsys):
        """Test calculation continues when a file path does not exist."""
        data_valid = {'feature_index': [0], 'n': [1.0], 'sum': [5.0], 'sum_of_squares': [25.0]}
        file_valid = create_metadata_file(data_valid, filename_suffix="exists")
        file_nonexistent = "/path/to/non/existent/file.parquet"

        expected_df = pd.DataFrame({'mean': [5.0], 'std_dev': [0.0]}, index=pd.Index([0], name='feature_index'))

        result_df = calculate_standardization_from_files([file_valid, file_nonexistent])
        captured = capsys.readouterr()

        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)
        assert f"Error: File not found {file_nonexistent}" in captured.out # Check warning

    def test_mixed_validity(self, create_metadata_file, capsys):
        """Test with a mix of valid, invalid (missing cols), and non-existent files."""
        data_valid1 = {'feature_index': [0], 'n': [2.], 'sum': [4.], 'sum_of_squares': [10.]} # mean=2, std=1
        data_invalid_cols = {'feature_index': [1], 'n': [1.], 'sum': [5.]} # Missing sum_sq
        data_valid2 = {'feature_index': [0], 'n': [1.], 'sum': [3.], 'sum_of_squares': [9.]} # mean=3, std=0
        file_nonexistent = "nonexistent_mix.parquet"

        file1 = create_metadata_file(data_valid1, filename_suffix="mix_valid1")
        file2 = create_metadata_file(data_invalid_cols, filename_suffix="mix_invalid")
        file3 = create_metadata_file(data_valid2, filename_suffix="mix_valid2")

        # Aggregate Feature 0: n=3, sum=7, sum_sq=19 -> mean=7/3, var=(19/3)-(49/9)=(57-49)/9=8/9, std=sqrt(8)/3
        expected_mean = 7/3
        expected_std = np.sqrt(8)/3
        expected_df = pd.DataFrame(
            {'mean': [expected_mean], 'std_dev': [expected_std]},
            index=pd.Index([0], name='feature_index')
        )

        result_df = calculate_standardization_from_files([file1, file2, file_nonexistent, file3])
        captured = capsys.readouterr()

        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-6)
        assert f"Skipping {file2}. Missing required columns: {{'sum_of_squares'}}" in captured.out
        assert f"Error: File not found {file_nonexistent}" in captured.out

    def test_calculation_precision(self, create_metadata_file):
        """Test variance calculation handles potential floating point issues."""
        # Case where sum_sq/n might be slightly less than mean^2 due to precision
        # Let mean = 1/3, mean^2 = 1/9
        # Let n = 3, sum = 1
        # Let sum_sq = 1/3 (so exact variance is (1/3)/3 - (1/3)^2 = 1/9 - 1/9 = 0)
        # Simulate slight precision error making sum_sq/n < mean^2
        n = 3.0
        sum_val = 1.0
        mean_val = sum_val / n # 1/3
        # Make sum_of_squares slightly less than what would give variance 0
        sum_sq_val = n * (mean_val**2) - 1e-12 # Slightly less than 1/3

        data = {'feature_index': [0], 'n': [n], 'sum': [sum_val], 'sum_of_squares': [sum_sq_val]}
        file_path = create_metadata_file(data, filename_suffix="precision")

        # Expect variance to be clamped to 0, so std_dev is 0
        expected_df = pd.DataFrame(
            {'mean': [mean_val], 'std_dev': [0.0]},
            index=pd.Index([0], name='feature_index')
        )

        result_df = calculate_standardization_from_files([file_path])
        assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-9) # Use tighter tolerance for mean check if needed
        assert result_df['std_dev'].iloc[0] == 0.0 # Std dev should be exactly 0
