import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Union, Tuple # Added Union, Tuple
from datetime import datetime, timedelta # Added for date range handling
import warnings # Add warnings import
import logging # Add logging import

# Set up logger
logger = logging.getLogger(__name__)

# NOTE: the way of accessing the data is not very optimized (pandas iloc is not the fastest, we could speed this up by even just using a dict).


class BaseMhcDataset(Dataset):
    """
    PyTorch Dataset for loading MHC data from a denormalized DataFrame.

    Assumptions:
    - DataFrame contains 'healthCode', 'time_range' (YYYY-MM-DD_YYYY-MM-DD),
      'file_uris' (List[str]), and label columns ('*_value').
    - 'file_uris' contains relative paths to daily .npy files, where the filename
      is the date (e.g., 'YYYY-MM-DD.npy'), relative to a participant-specific
      subdirectory within root_dir (e.g., root_dir/healthCode/YYYY-MM-DD.npy).
    - Each daily .npy file is a NumPy array with at least 3 dimensions (Mask+Data, Features, Time),
      where the desired data slice is array[1, :, :] (shape (24, 1440))
      and the mask slice (if requested) is array[0, :, :] (shape (24, 1440)).
    - Handles missing daily files within the 'time_range' by inserting a
      NaN placeholder tensor of shape (24, 1440) for data, and potentially
      a corresponding zero mask tensor if include_mask=True.

    Output sample['data'] shape: (num_days, 24, 1440)
    Output sample['mask'] shape (if include_mask=True): (num_days, 24, 1440)
    """

    def __init__(self, dataframe: pd.DataFrame, root_dir: str, include_mask: bool = False):
        """
        Args:
            dataframe (pd.DataFrame): The denormalized dataframe.
            root_dir (str): The root directory containing participant subdirectories.
            include_mask (bool): Whether to load and include a mask channel
                                 (from index 0 of the npy file) alongside the data.
                                 Defaults to False.
        """
        super().__init__()
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        required_cols = ['healthCode', 'time_range', 'file_uris']
        for col in required_cols:
            if col not in dataframe.columns:
                raise ValueError(f"DataFrame must contain column '{col}'.")
        # Basic check for file_uris format (optional but recommended)
        if len(dataframe) > 0 and not isinstance(dataframe['file_uris'].iloc[0], list):
             # Allow empty lists though
            if not (isinstance(dataframe['file_uris'].iloc[0], list) or pd.isna(dataframe['file_uris'].iloc[0])):
                 # Check if it might be a string representation of a list
                 try:
                     import ast
                     # Attempt to evaluate the string - USE WITH CAUTION if source isn't trusted
                     first_val = ast.literal_eval(dataframe['file_uris'].iloc[0])
                     if not isinstance(first_val, list):
                         raise ValueError("Expected 'file_uris' column to contain lists.")
                     # Use warnings.warn for deprecation/usage issues
                     warnings.warn("Warning: 'file_uris' appears to contain string representations of lists. Consider pre-processing.", UserWarning)
                 except (ValueError, SyntaxError, TypeError):
                      raise ValueError("Expected 'file_uris' column to contain lists.")


        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(os.path.expanduser(root_dir))
        self.include_mask = include_mask

        # Identify label columns
        self.label_cols = sorted([col for col in self.df.columns if col.endswith('_value')])
        # Use logging for info messages
        logger.info(f"Initialized Dataset with {len(self.df)} samples.")
        logger.info(f"Found label columns: {self.label_cols}")
        logger.info(f"Root directory set to: {self.root_dir.resolve()}")
        logger.info(f"Include mask: {self.include_mask}") # Log mask status

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def _load_and_slice_npy(
        self,
        file_path: Path,
        expected_shape: tuple = (24, 1440),
        include_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Loads, validates, and slices a single npy file.

        Args:
            file_path (Path): Path to the .npy file.
            expected_shape (tuple): The expected shape of *each* slice (mask or data).
            include_mask (bool): If True, return both mask (index 0) and data (index 1).
                                 Otherwise, return only data (index 1). The mask at index 0 typically 
                                 indicates which data points are valid (1) or missing (0).

        Returns:
            np.ndarray: The data slice if include_mask is False.
            Tuple[np.ndarray, np.ndarray]: The mask slice and data slice if include_mask is True.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the array dimensions or shapes are incorrect.
            IOError: If there's an error loading or processing the file.
        """
        if not file_path.is_file():
             # This case should ideally be handled by the caller checking existence first,
             # but added as a safeguard.
             # This is an unrecoverable error for this function, so raise directly
             raise FileNotFoundError(f"File not found during load attempt: {file_path}")

        try:
            data_array = np.load(file_path).astype(np.float32)

            # Perform the slicing: index 1 for data, index 0 for mask
            # Add checks for sufficient dimensions
            if data_array.ndim < 3:
                 # This indicates bad data, raise ValueError
                 raise ValueError(f"Expected >=3 dimensions in {file_path}, but got shape {data_array.shape}")
            # Check if first dimension has at least 2 elements (for mask and data)
            if data_array.shape[0] < 2:
                 raise ValueError(f"Expected >=2 elements in first dimension (mask+data) in {file_path}, but got shape {data_array.shape}")

            data_slice = data_array[1, :, :] # Data is always at index 1

            # Validate data shape after slicing
            if data_slice.shape != expected_shape:
                # This indicates bad data or wrong expectation, raise ValueError
                raise ValueError(f"Expected data shape {expected_shape} after slicing {file_path}, but got {data_slice.shape}")

            if include_mask:
                mask_slice = data_array[0, :, :] # Mask is at index 0
                 # Validate mask shape after slicing
                if mask_slice.shape != expected_shape:
                    raise ValueError(f"Expected mask shape {expected_shape} after slicing {file_path}, but got {mask_slice.shape}")
                return mask_slice, data_slice
            else:
                return data_slice

        except Exception as e:
            # Log the error before re-raising
            logger.error(f"Could not load or process file {file_path}. Error: {e}")
            # Re-raise with context as IOError, standard practice for file issues
            raise IOError(f"Failed to load or process data from {file_path}") from e

    @staticmethod
    def _generate_date_range(start_date_str: str, end_date_str: str) -> List[str]:
        """Generates a list of dates (YYYY-MM-DD) between start and end (inclusive)."""
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        delta = end_date - start_date
        return [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]

    @staticmethod
    def _create_placeholder(
        shape: tuple = (24, 1440),
        include_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Creates a placeholder array (or arrays).

        Args:
            shape (tuple): The desired shape for the data slice (e.g., (24, 1440)).
            include_mask (bool): If True, returns a tuple of (mask_placeholder, data_placeholder).
                                 Otherwise, returns just the data_placeholder.

        Returns:
            np.ndarray: Data placeholder filled with NaNs.
            Tuple[np.ndarray, np.ndarray]: Mask placeholder (zeros) and data placeholder (NaNs).
        """
        data_placeholder = np.full(shape, np.nan, dtype=np.float32)
        if include_mask:
            # Mask for missing data should be zeros
            mask_placeholder = np.zeros(shape, dtype=np.float32)
            return mask_placeholder, data_placeholder
        else:
            return data_placeholder


    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Loads daily data based on 'time_range', slicing loaded files and
        inserting placeholders for missing days. Stacks daily data along axis 0.
        Optionally includes a corresponding mask tensor.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'data' (torch.Tensor): Shape (num_days, 24, 1440).
                - 'mask' (torch.Tensor, optional): Shape (num_days, 24, 1440). Included if
                  `include_mask` was True during initialization.
                - 'labels' (dict): Label values.
                - 'metadata' (dict): Includes 'healthCode', 'time_range', 'file_uris'.
        """
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.df)}")

        row = self.df.iloc[idx]
        health_code = row['healthCode']
        time_range_str = row['time_range']
        file_uris_list = row['file_uris'] # Assumed to be a list

        # 1. Parse time range and generate expected dates
        try:
            start_date_str, end_date_str = time_range_str.split('_')
            expected_dates = self._generate_date_range(start_date_str, end_date_str)
        except ValueError:
             # This indicates bad config in dataframe, raise
             logger.error(f"Invalid time_range format for sample {idx}: '{time_range_str}'. Expected 'YYYY-MM-DD_YYYY-MM-DD'.")
             raise ValueError(f"Invalid time_range format for sample {idx}: '{time_range_str}'. Expected 'YYYY-MM-DD_YYYY-MM-DD'.")

        # 2. Create a lookup for provided files based on date
        provided_files_map = {}
        participant_dir = self.root_dir / health_code # <-- This line is kept for context but is no longer used in the loop below for constructing the full path
        # Handle cases where file_uris might be NaN or not a list
        if isinstance(file_uris_list, list):
            for uri in file_uris_list:
                # Assume filename is the date: YYYY-MM-DD.npy
                try:
                    # Extract date from filename (robustly handle potential path components)
                    filename = Path(uri).name
                    date_str = filename.split('.npy')[0]
                    # Validate date format extracted from filename
                    datetime.strptime(date_str, "%Y-%m-%d")
                    # Store full path against date string - CORRECTED LINE:
                    provided_files_map[date_str] = self.root_dir / uri # Construct full path directly from root_dir and uri
                except (ValueError, IndexError):
                     # Log warning for files that cannot be parsed, skip them
                     logger.warning(f"Could not parse date from file URI '{uri}' for sample {idx}. Skipping this file.")
                     continue # Skip files with unexpected names
        elif pd.isna(file_uris_list):
            logger.warning(f"file_uris is NaN for sample {idx}. No files will be loaded.")
        else:
            # This case should ideally be caught by __init__, but log just in case
            logger.warning(f"Unexpected type for file_uris for sample {idx}: {type(file_uris_list)}. No files will be loaded.")


        # 3. Load or create placeholder for each expected date
        daily_data = []
        daily_masks = [] # List to store masks if needed
        expected_slice_shape = (24, 1440)
        placeholder = self._create_placeholder(expected_slice_shape, self.include_mask) # Create placeholder once

        for date_str in expected_dates:
            if date_str in provided_files_map:
                file_path = provided_files_map[date_str]
                try:
                    # Check existence *before* trying to load
                    if file_path.is_file():
                         loaded_data = self._load_and_slice_npy(file_path, expected_slice_shape, self.include_mask)
                         if self.include_mask:
                             mask_slice, data_slice = loaded_data
                             daily_masks.append(mask_slice)
                             daily_data.append(data_slice)
                         else:
                             daily_data.append(loaded_data) # Only data slice returned
                    else:
                         # Log warning for listed files that are missing
                         logger.warning(f"File listed in 'file_uris' not found at {file_path} for sample {idx}, date {date_str}. Using placeholder.")
                         if self.include_mask:
                             mask_ph, data_ph = placeholder
                             daily_masks.append(mask_ph)
                             daily_data.append(data_ph)
                         else:
                             daily_data.append(placeholder) # Only data placeholder
                except (ValueError, IOError, Exception) as e: # Catch specific load/process errors
                     # Log error for file that failed processing, use placeholder
                     logger.error(f"Failed loading/slicing file {file_path} for sample {idx}, date {date_str}. Using placeholder. Error: {e}")
                     if self.include_mask:
                         mask_ph, data_ph = placeholder
                         daily_masks.append(mask_ph)
                         daily_data.append(data_ph)
                     else:
                         daily_data.append(placeholder) # Only data placeholder
            else:
                # File for this date was not listed or not found, use placeholder
                if self.include_mask:
                     mask_ph, data_ph = placeholder
                     daily_masks.append(mask_ph)
                     daily_data.append(data_ph)
                else:
                    daily_data.append(placeholder) # Only data placeholder

        # 4. Stack daily data and convert to tensor(s)
        if not daily_data:
             # This should only happen if time range was invalid or no dates generated
             logger.error(f"No daily data could be processed for sample {idx} within time range {time_range_str}. This might indicate an empty date range.")
             # Raise error as this indicates a problem with the sample definition
             raise ValueError(f"No daily data could be processed for sample {idx} within time range {time_range_str}. Check time_range and file availability.")

        # Prepare result dictionary
        result_dict = {}

        try:
            # Stack data along a new first dimension (axis=0)
            final_data_array = np.stack(daily_data, axis=0)
            result_dict['data'] = torch.from_numpy(final_data_array)

            # Stack masks if included
            if self.include_mask:
                if not daily_masks: # Should not happen if daily_data is not empty
                    raise ValueError(f"Internal error: daily_masks is empty but daily_data is not for sample {idx}.")
                final_mask_array = np.stack(daily_masks, axis=0)
                result_dict['mask'] = torch.from_numpy(final_mask_array)

        except ValueError as e:
             # This might happen if placeholders and loaded data have inconsistent shapes
             data_shapes = [arr.shape for arr in daily_data]
             mask_shapes = [arr.shape for arr in daily_masks] if self.include_mask else "N/A"
             logger.error(f"Could not stack daily arrays for sample {idx}. Data shapes: {data_shapes}. Mask shapes: {mask_shapes}. Error: {e}")
             # Re-raise as it indicates a fundamental issue
             raise ValueError(f"Failed to stack daily arrays for sample {idx}. Data shapes: {data_shapes}, Mask shapes: {mask_shapes}") from e


        # 5. Extract labels (same as before)
        labels = {}
        for label_col in self.label_cols:
            label_name = label_col.replace('_value', '')
            label_value = float(row[label_col]) if pd.notna(row[label_col]) else np.nan
            labels[label_name] = label_value

        # 6. Include metadata
        metadata = {
            'healthCode': health_code,
            'time_range': time_range_str,
            #'file_uris': file_uris_list # Store original list
        }

        # Add labels and metadata to the result dict
        result_dict['labels'] = labels
        result_dict['metadata'] = metadata

        return result_dict



class FilteredMhcDataset(BaseMhcDataset):
    """
    A subclass of BaseMhcDataset that filters the dataframe to only include samples
    where a specified label of interest is not NaN.
    
    This dataset is useful when training models for a specific prediction task
    where you only want samples that have the target label available.
    
    Usage Example:
        >>> # Assume 'full_df' is your loaded denormalized dataframe
        >>> # Assume 'data_root' is the path to the directory with .npy files
        >>> happiness_dataset = FilteredMhcDataset(
        ...     dataframe=full_df, 
        ...     root_dir=data_root, 
        ...     label_of_interest='happiness',
        ...     include_mask=True  # Optionally include the mask
        ... )
        >>> # Now happiness_dataset only contains samples with a non-NaN happiness_value
        >>> # You can use it with a DataLoader:
        >>> # from torch.utils.data import DataLoader
        >>> # data_loader = DataLoader(happiness_dataset, batch_size=32)
    """

    def __init__(self, dataframe: pd.DataFrame, root_dir: str, label_of_interest: str, include_mask: bool = False):
        """
        Args:
            dataframe (pd.DataFrame): The denormalized dataframe containing metadata and labels.
            root_dir (str): The root directory where the .npy files specified in 'file_uris' are located.
            label_of_interest (str): The label to filter by (without '_value' suffix).
                                     Only samples where this label is not NaN will be included.
            include_mask (bool): Whether to load and include a mask channel. Passed to BaseMhcDataset.
                                 Defaults to False.
        """
        # Ensure the label_of_interest has the '_value' suffix for filtering
        label_col = f"{label_of_interest}_value" if not label_of_interest.endswith('_value') else label_of_interest
        
        # Check if the label exists in the dataframe
        if label_col not in dataframe.columns:
            # This is a configuration error
            logger.error(f"Label column '{label_col}' not found in the dataframe for filtering.")
            raise ValueError(f"Label column '{label_col}' not found in the dataframe.")
        
        # Filter the dataframe to only include rows where the label is not NaN
        original_len = len(dataframe)
        filtered_df = dataframe[dataframe[label_col].notna()].copy()
        
        if len(filtered_df) == 0:
            # This might be expected or an error depending on data
            logger.error(f"No samples found with non-NaN values for label '{label_of_interest}'. Filtering removed all samples.")
            raise ValueError(f"No samples found with non-NaN values for label '{label_of_interest}'.")
        
        # Log info about the filtering result
        logger.info(f"Filtered dataset from {original_len} to {len(filtered_df)} samples with non-NaN '{label_of_interest}' values.")
        
        # Initialize the parent class with the filtered dataframe and pass include_mask
        # Parent's __init__ will log its own messages
        super().__init__(filtered_df, root_dir, include_mask=include_mask)
        
        # Store the label of interest for reference
        self.label_of_interest = label_of_interest
