import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Union, Tuple, Optional, Callable # Added Union, Tuple, Optional, Callable
from datetime import datetime, timedelta # Added for date range handling
import warnings # Add warnings import
import logging # Add logging import
import pickle # Add pickle import for caching

# Set up logger
logger = logging.getLogger(__name__)

# NOTE: the way of accessing the data is not very optimized (pandas iloc is not the fastest, we could speed this up by even just using a dict).
# NOTE: Assume a fixed number of features (24) based on current slicing logic.

_EXPECTED_RAW_FEATURES = 24
_EXPECTED_TIME_POINTS = 1440


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
      where the raw data slice is array[1, :, :] (shape (24, 1440))
      and the mask slice (if requested) is array[0, :, :] (shape (24, 1440)).
      The number of features (24) is assumed unless feature_indices is specified.
    - Handles missing daily files within the 'time_range' by inserting a
      NaN placeholder tensor of shape (num_features, 1440) for data, and potentially
      a corresponding zero mask tensor if include_mask=True.
    - If feature_indices is provided, only those features are loaded/returned.
    - If feature_stats is provided, specified feature channels will be standardized
      using the provided means and standard deviations *after* feature selection.

    Output sample['data'] shape: (num_days, num_selected_features, 1440)
    Output sample['mask'] shape (if include_mask=True): (num_days, num_selected_features, 1440)
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 root_dir: str,
                 include_mask: bool = False,
                 feature_indices: Optional[List[int]] = None,
                 feature_stats: Optional[dict] = None,
                 postprocessors: Optional[List[Callable]] = None,
                 use_cache: bool = False,
                 force_recompute: bool = False):
        """
        Args:
            dataframe (pd.DataFrame): The denormalized dataframe.
            root_dir (str): The root directory containing participant subdirectories.
            include_mask (bool): Whether to load and include a mask channel
                                 (from index 0 of the npy file) alongside the data.
                                 Defaults to False.
            feature_indices (Optional[List[int]]): Optional list of integer indices for the features
                                                    to select. If None, all 24 features are used.
                                                    Indices must be within [0, 23].
            feature_stats (Optional[dict]): Optional dictionary mapping *original* feature indices
                                             to tuples of (mean, std) for feature-wise standardization.
                                             If None, no standardization is applied.
                                             Example: {0: (0.5, 1.0), 1: (0.0, 2.0)} for standardizing
                                             original features 0 and 1. Standardization happens
                                             *after* feature selection.
            postprocessors (Optional[List[Callable]]): A list of callable objects (e.g., functions
                                                       or instances with __call__) that will be applied
                                                       sequentially to the sample dictionary in __getitem__
                                                       after all other processing.
            use_cache (bool): Whether to use caching for processed samples. If True, processed samples 
                              will be saved to disk and loaded from cache in the same directory as the 
                              source .npy files.
            force_recompute (bool): If True, cached samples will be ignored and recomputed. New cache files
                                   will still be saved if use_cache is True. Defaults to False.
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

        # Validate feature_indices
        self.feature_indices = None
        self.num_features = _EXPECTED_RAW_FEATURES # Start assuming all features
        if feature_indices is not None:
            if not isinstance(feature_indices, list) or not all(isinstance(i, int) for i in feature_indices):
                raise TypeError("feature_indices must be a list of integers.")
            if not feature_indices:
                 raise ValueError("feature_indices cannot be an empty list.")
            # Sort and remove duplicates
            unique_indices = sorted(list(set(feature_indices)))
            if not all(0 <= i < _EXPECTED_RAW_FEATURES for i in unique_indices):
                raise ValueError(f"All feature_indices must be between 0 and {_EXPECTED_RAW_FEATURES - 1}.")
            self.feature_indices = unique_indices
            self.num_features = len(self.feature_indices) # Update number of features
            logger.info(f"Selecting {self.num_features} features with indices: {self.feature_indices}")
        else:
             logger.info(f"Using all {_EXPECTED_RAW_FEATURES} features.")


        # Validate feature_stats and create remapped version
        self.feature_stats = feature_stats # Store original for metadata maybe?
        self._remapped_feature_stats = None
        if feature_stats is not None:
            if not isinstance(feature_stats, dict):
                raise TypeError("feature_stats must be a dictionary.")
            
            valid_stats = {}
            original_indices_to_standardize = []
            for idx, stats in feature_stats.items():
                if not isinstance(idx, int):
                     raise TypeError(f"Keys in feature_stats must be integer indices, found {type(idx)} ({idx}).")
                if not (0 <= idx < _EXPECTED_RAW_FEATURES):
                    logger.warning(f"Feature index {idx} in feature_stats is out of the expected range [0, {_EXPECTED_RAW_FEATURES-1}]. Skipping.")
                    continue
                if not isinstance(stats, tuple) or len(stats) != 2:
                    raise ValueError(f"Feature stats for index {idx} must be a tuple of (mean, std).")
                if not all(isinstance(x, (int, float)) for x in stats):
                    raise ValueError(f"Mean and std for feature {idx} must be numeric.")
                
                # Check if this original index is among the selected features (if applicable)
                if self.feature_indices is None or idx in self.feature_indices:
                    valid_stats[idx] = stats
                    original_indices_to_standardize.append(idx)
                else:
                    logger.warning(f"Feature index {idx} in feature_stats is not in the selected feature_indices. Skipping standardization for this feature.")

            if valid_stats:
                 self._remapped_feature_stats = {}
                 if self.feature_indices is None:
                     # No remapping needed if using all features
                     self._remapped_feature_stats = valid_stats
                 else:
                     # Create mapping from original index to new index
                     original_to_new_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.feature_indices)}
                     for original_idx in original_indices_to_standardize:
                         new_idx = original_to_new_index_map[original_idx]
                         self._remapped_feature_stats[new_idx] = valid_stats[original_idx]
                 logger.info(f"Feature standardization enabled for {len(self._remapped_feature_stats)} selected features (original indices: {list(valid_stats.keys())}).")
            else:
                 logger.info("Feature standardization provided but no applicable features found or selected.")


        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(os.path.expanduser(root_dir))
        self.include_mask = include_mask
        self.postprocessors = postprocessors if postprocessors is not None else [] # Store postprocessors
        
        # Caching setup
        self.use_cache = use_cache
        self.force_recompute = force_recompute
        
        if use_cache:
            if force_recompute:
                logger.info("Caching enabled, but force_recompute is True. Cached files will be regenerated.")
            else:
                logger.info("Caching enabled. Cache files will be stored alongside source files and used if available.")
        else:
            if force_recompute:
                logger.warning("force_recompute is True but use_cache is False. Setting has no effect.")
            logger.info("Caching disabled.")

        # Identify label columns
        self.label_cols = sorted([col for col in self.df.columns if col.endswith('_value')])
        # Use logging for info messages
        logger.info(f"Initialized Dataset with {len(self.df)} samples.")
        logger.info(f"Found label columns: {self.label_cols}")
        logger.info(f"Root directory set to: {self.root_dir.resolve()}")
        logger.info(f"Include mask: {self.include_mask}") # Log mask status
        # Logging for standardization is handled above


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def _load_and_slice_npy(
        self,
        file_path: Path,
        expected_raw_shape: tuple = (_EXPECTED_RAW_FEATURES, _EXPECTED_TIME_POINTS), # Expect raw shape before selection
        include_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Loads, validates, and slices the raw data and mask from a single npy file.
        This method returns the full feature set (e.g., 24 features).
        Feature selection happens later in __getitem__.

        Args:
            file_path (Path): Path to the .npy file.
            expected_raw_shape (tuple): The expected shape of the raw data/mask slices
                                        (e.g., (24, 1440)) before any feature selection.
            include_mask (bool): If True, return both mask (index 0) and data (index 1).
                                 Otherwise, return only data (index 1).

        Returns:
            np.ndarray: The raw data slice (shape expected_raw_shape) if include_mask is False.
            Tuple[np.ndarray, np.ndarray]: The mask slice and data slice (both shape expected_raw_shape)
                                           if include_mask is True.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the array dimensions or shapes are incorrect.
            IOError: If there's an error loading or processing the file.
        """
        if not file_path.is_file():
             raise FileNotFoundError(f"File not found during load attempt: {file_path}")

        try:
            data_array = np.load(file_path).astype(np.float32)

            # Perform the slicing: index 1 for data, index 0 for mask
            if data_array.ndim < 3:
                 raise ValueError(f"Expected >=3 dimensions in {file_path}, but got shape {data_array.shape}")
            if data_array.shape[0] < 2:
                 raise ValueError(f"Expected >=2 elements in first dimension (mask+data) in {file_path}, but got shape {data_array.shape}")

            data_slice = data_array[1, :, :] # Data is always at index 1

            # Validate raw data shape before selection
            if data_slice.shape != expected_raw_shape:
                raise ValueError(f"Expected raw data shape {expected_raw_shape} after slicing {file_path}, but got {data_slice.shape}")

            if include_mask:
                mask_slice = data_array[0, :, :] # Mask is at index 0
                if mask_slice.shape != expected_raw_shape:
                    raise ValueError(f"Expected raw mask shape {expected_raw_shape} after slicing {file_path}, but got {mask_slice.shape}")
                return mask_slice, data_slice
            else:
                return data_slice

        except Exception as e:
            logger.error(f"Could not load or process file {file_path}. Error: {e}")
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
        raw_shape: tuple = (_EXPECTED_RAW_FEATURES, _EXPECTED_TIME_POINTS), # Create raw shape placeholder
        include_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Creates a placeholder array (or arrays) with the raw feature shape.
        Feature selection happens later in __getitem__.

        Args:
            raw_shape (tuple): The desired raw shape for the data slice (e.g., (24, 1440)).
            include_mask (bool): If True, returns a tuple of (mask_placeholder, data_placeholder).
                                 Otherwise, returns just the data_placeholder.

        Returns:
            np.ndarray: Data placeholder (shape raw_shape) filled with NaNs.
            Tuple[np.ndarray, np.ndarray]: Mask placeholder (zeros, shape raw_shape) and
                                           data placeholder (NaNs, shape raw_shape).
        """
        data_placeholder = np.full(raw_shape, np.nan, dtype=np.float32)
        if include_mask:
            mask_placeholder = np.zeros(raw_shape, dtype=np.float32)
            return mask_placeholder, data_placeholder
        else:
            return data_placeholder

    def _get_cache_path(self, idx: int, file_path: Path) -> Path:
        """
        Generate the file path for a cached sample. The cached file will be stored
        in the same directory as the original source file.
        
        Args:
            idx (int): The index of the sample.
            file_path (Path): The path to the source .npy file to determine cache location.
            
        Returns:
            Path: The path where the cached sample should be stored.
        """
        if not self.use_cache:
            return None
            
        # Get class name
        class_name = self.__class__.__name__
        
        # Get processor names if any
        processor_names = []
        for processor in self.postprocessors:
            if hasattr(processor, '__name__'):
                processor_names.append(processor.__name__)
            else:
                processor_names.append(processor.__class__.__name__)
        
        # Create cache identifier with class and processor names
        cache_identifier = f"{class_name}"
        if processor_names:
            cache_identifier += "-" + "-".join(processor_names)
            
        # Use the directory of the source file
        parent_dir = file_path.parent
        
        # Create a unique filename for this sample with class and processor info
        filename = f"{file_path.stem}_cached-{cache_identifier}.pkl"
        
        return parent_dir / filename
        
    def _check_cache(self, idx: int, file_path: Path) -> Optional[dict]:
        """
        Check if a cached version of the sample exists and load it if so.
        Will skip checking if force_recompute is True.
        
        Args:
            idx (int): The index of the sample to check.
            file_path (Path): The path to the source .npy file.
            
        Returns:
            Optional[dict]: The cached sample dict if found, None otherwise.
        """
        if not self.use_cache or file_path is None or self.force_recompute:
            return None
            
        cache_path = self._get_cache_path(idx, file_path)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_sample = pickle.load(f)
                logger.debug(f"Loaded cached sample for index {idx} from {cache_path}")
                return cached_sample
            except Exception as e:
                logger.warning(f"Failed to load cached sample for index {idx}: {e}")
                return None
        return None
        
    def _save_to_cache(self, idx: int, sample: dict, file_path: Path) -> None:
        """
        Save the processed sample to cache in the same directory as the source file.
        
        Args:
            idx (int): The index of the sample.
            sample (dict): The processed sample to cache.
            file_path (Path): The path to the source .npy file.
        """
        if not self.use_cache or file_path is None:
            return
            
        cache_path = self._get_cache_path(idx, file_path)
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(sample, f)
                logger.debug(f"Saved sample for index {idx} to cache at {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save sample for index {idx} to cache: {e}")

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Loads daily data based on 'time_range', slicing loaded files and
        inserting placeholders for missing days. Stacks daily data along axis 0.
        Selects features based on `feature_indices` (if provided).
        Optionally includes a corresponding mask tensor. If `feature_stats` was provided,
        selected feature channels will be standardized using the provided means and standard deviations.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'data' (torch.Tensor): Shape (num_days, num_selected_features, 1440).
                - 'mask' (torch.Tensor, optional): Shape (num_days, num_selected_features, 1440).
                  Included if `include_mask` was True.
                - 'labels' (dict): Label values.
                - 'metadata' (dict): Includes 'healthCode', 'time_range', 'feature_indices'.
        """
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.df)}")

        row = self.df.iloc[idx]
        health_code = row['healthCode']
        time_range_str = row['time_range']
        file_uris_list = row['file_uris'] # Assumed to be a list

        # Get the first valid file path to use for cache location
        cache_reference_path = None
        if self.use_cache and isinstance(file_uris_list, list) and file_uris_list:
            for uri in file_uris_list:
                try:
                    file_path = self.root_dir / uri
                    if file_path.exists():
                        cache_reference_path = file_path
                        break
                except:
                    continue
            
            # Check if the sample is already cached (skips if force_recompute is True)
            if cache_reference_path:
                cached_sample = self._check_cache(idx, cache_reference_path)
                if cached_sample is not None:
                    return cached_sample

        # 1. Parse time range and generate expected dates
        try:
            start_date_str, end_date_str = time_range_str.split('_')
            expected_dates = self._generate_date_range(start_date_str, end_date_str)
        except ValueError:
             logger.error(f"Invalid time_range format for sample {idx}: '{time_range_str}'. Expected 'YYYY-MM-DD_YYYY-MM-DD'.")
             raise ValueError(f"Invalid time_range format for sample {idx}: '{time_range_str}'. Expected 'YYYY-MM-DD_YYYY-MM-DD'.")

        # 2. Create a lookup for provided files based on date
        provided_files_map = {}
        # Handle cases where file_uris might be NaN or not a list
        if isinstance(file_uris_list, list):
            for uri in file_uris_list:
                try:
                    filename = Path(uri).name
                    date_str = filename.split('.npy')[0]
                    datetime.strptime(date_str, "%Y-%m-%d")
                    provided_files_map[date_str] = self.root_dir / uri
                except (ValueError, IndexError):
                     logger.warning(f"Could not parse date from file URI '{uri}' for sample {idx}. Skipping this file.")
                     continue
        elif pd.isna(file_uris_list):
            logger.warning(f"file_uris is NaN for sample {idx}. No files will be loaded.")
        else:
            logger.warning(f"Unexpected type for file_uris for sample {idx}: {type(file_uris_list)}. No files will be loaded.")


        # 3. Load or create placeholder for each expected date (using raw shape)
        daily_data = []
        daily_masks = [] # List to store masks if needed
        raw_slice_shape = (_EXPECTED_RAW_FEATURES, _EXPECTED_TIME_POINTS)
        placeholder = self._create_placeholder(raw_slice_shape, self.include_mask) # Create raw placeholder once

        for date_str in expected_dates:
            if date_str in provided_files_map:
                file_path = provided_files_map[date_str]
                try:
                    if file_path.is_file():
                         loaded_data = self._load_and_slice_npy(file_path, raw_slice_shape, self.include_mask)
                         if self.include_mask:
                             mask_slice, data_slice = loaded_data
                             daily_masks.append(mask_slice)
                             daily_data.append(data_slice)
                         else:
                             daily_data.append(loaded_data)
                    else:
                         logger.warning(f"File listed in 'file_uris' not found at {file_path} for sample {idx}, date {date_str}. Using placeholder.")
                         if self.include_mask:
                             mask_ph, data_ph = placeholder
                             daily_masks.append(mask_ph)
                             daily_data.append(data_ph)
                         else:
                             daily_data.append(placeholder)
                except (ValueError, IOError, Exception) as e:
                     logger.error(f"Failed loading/slicing raw file {file_path} for sample {idx}, date {date_str}. Using placeholder. Error: {e}")
                     if self.include_mask:
                         mask_ph, data_ph = placeholder
                         daily_masks.append(mask_ph)
                         daily_data.append(data_ph)
                     else:
                         daily_data.append(placeholder)
            else:
                # File for this date was not listed or not found, use placeholder
                if self.include_mask:
                     mask_ph, data_ph = placeholder
                     daily_masks.append(mask_ph)
                     daily_data.append(data_ph)
                else:
                    daily_data.append(placeholder)

        # 4. Stack daily data and convert to tensor(s)
        if not daily_data:
             logger.error(f"No daily data could be processed for sample {idx} within time range {time_range_str}. This might indicate an empty date range.")
             raise ValueError(f"No daily data could be processed for sample {idx} within time range {time_range_str}. Check time_range and file availability.")

        # Prepare result dictionary
        result_dict = {}

        try:
            # Stack raw data along axis 0
            stacked_data_array = np.stack(daily_data, axis=0) # Shape: (num_days, 24, 1440)
            stacked_mask_array = None
            if self.include_mask:
                if not daily_masks:
                    raise ValueError(f"Internal error: daily_masks is empty but daily_data is not for sample {idx}.")
                stacked_mask_array = np.stack(daily_masks, axis=0) # Shape: (num_days, 24, 1440)

            # ---> 4b. Feature Selection <---
            if self.feature_indices is not None:
                stacked_data_array = stacked_data_array[:, self.feature_indices, :]
                if self.include_mask and stacked_mask_array is not None:
                    stacked_mask_array = stacked_mask_array[:, self.feature_indices, :]
                logger.debug(f"Sample {idx}: Selected features. New data shape: {stacked_data_array.shape}")


            # ---> 4d. Convert final arrays to tensors <---
            result_dict['data'] = torch.from_numpy(stacked_data_array.copy()) # Use copy for safety

            if self.include_mask and stacked_mask_array is not None:
                result_dict['mask'] = torch.from_numpy(stacked_mask_array.copy())

        except ValueError as e:
             # Catch potential errors during stacking or slicing
             raw_data_shapes = [arr.shape for arr in daily_data]
             raw_mask_shapes = [arr.shape for arr in daily_masks] if self.include_mask else "N/A"
             logger.error(f"Could not stack or process daily arrays for sample {idx}. Raw Data shapes: {raw_data_shapes}. Raw Mask shapes: {raw_mask_shapes}. Feature indices: {self.feature_indices}. Error: {e}")
             raise ValueError(f"Failed to stack or process daily arrays for sample {idx}. Raw Shapes: Data={raw_data_shapes}, Mask={raw_mask_shapes}. Error: {e}") from e


        # 5. Extract labels
        labels = {}
        for label_col in self.label_cols:
            label_name = label_col.replace('_value', '')
            label_value = float(row[label_col]) if pd.notna(row[label_col]) else np.nan
            labels[label_name] = label_value

        # 6. Include metadata
        metadata = {
            'healthCode': health_code,
            'time_range': time_range_str,
            # 'file_uris': file_uris_list # Maybe too much data?
            'feature_indices': self.feature_indices # Pass selected indices for postprocessors
        }

        # Add labels and metadata to the result dict
        result_dict['labels'] = labels
        result_dict['metadata'] = metadata

        # Replace NaNs with 0.0 in final tensors
        if 'data' in result_dict and isinstance(result_dict['data'], torch.Tensor):
            result_dict['data'] = torch.nan_to_num(result_dict['data'], nan=0.0)
        if 'mask' in result_dict and isinstance(result_dict['mask'], torch.Tensor):
            # Although mask placeholders are zeros, apply defensively
            result_dict['mask'] = torch.nan_to_num(result_dict['mask'], nan=0.0)

        # 7. Apply postprocessors sequentially
        for processor in self.postprocessors:
             if callable(processor):
                 try:
                     # Processor should take the sample dict and return the modified dict
                     result_dict = processor(result_dict)
                 except Exception as e:
                     logger.error(f"Error applying postprocessor {type(processor).__name__} to sample {idx}: {e}", exc_info=True)
                     # Reraise to make the problem visible during development/debugging
                     raise RuntimeError(f"Postprocessor {type(processor).__name__} failed for sample {idx}") from e
             else:
                 logger.warning(f"Item in postprocessors list is not callable: {type(processor)}. Skipping.")

        # Apply feature-wise standardization (using remapped stats)
        if self._remapped_feature_stats is not None and 'data' in result_dict:
            stacked_data_array = result_dict['data'].numpy()
            # Data shape is now (num_days, num_selected_features, 1440)
            current_num_features = stacked_data_array.shape[1]
            for remapped_feature_idx, (mean, std) in self._remapped_feature_stats.items():
                # remapped_feature_idx is the index in the *selected* feature array
                if remapped_feature_idx < 0 or remapped_feature_idx >= current_num_features:
                    # This check should theoretically be redundant due to validation in __init__
                    logger.warning(f"Remapped feature index {remapped_feature_idx} out of bounds for data with {current_num_features} selected features (Sample {idx}). Skipping standardization for this feature.")
                    continue
                # Avoid division by zero or near-zero std dev
                if std is None or abs(std) < 1e-9:
                     logger.warning(f"Standard deviation for feature index {remapped_feature_idx} is zero or too small ({std}). Skipping standardization for this feature.")
                     continue
                # Standardize this feature across all days and time points
                stacked_data_array[:, remapped_feature_idx, :] = (stacked_data_array[:, remapped_feature_idx, :] - mean) / std
            logger.debug(f"Sample {idx}: Applied standardization to {len(self._remapped_feature_stats)} features.")
            result_dict['data'] = torch.from_numpy(stacked_data_array.copy())

        # Apply mask to data: explicitly zero out data values where mask is 0
        if 'data' in result_dict and 'mask' in result_dict:
            result_dict['data'] = result_dict['data'] * result_dict['mask']

        # Save the processed sample to cache
        if self.use_cache and cache_reference_path:
            self._save_to_cache(idx, result_dict, cache_reference_path)

        return result_dict



class FilteredMhcDataset(BaseMhcDataset):
    """
    A subclass of BaseMhcDataset that filters the dataframe to only include samples
    where a specified label of interest is not NaN. Also supports feature selection
    and standardization via parameters passed to BaseMhcDataset.

    This dataset is useful when training models for a specific prediction task
    where you only want samples that have the target label available.

    Usage Example:
        >>> # Assume 'full_df' is your loaded denormalized dataframe
        >>> # Assume 'data_root' is the path to the directory with .npy files
        >>> # Select only features 0, 5, 10
        >>> feature_subset = [0, 5, 10]
        >>> # Standardize feature 5 (original index)
        >>> stats = {5: (mean_of_5, std_of_5)}
        >>> happiness_dataset = FilteredMhcDataset(
        ...     dataframe=full_df,
        ...     root_dir=data_root,
        ...     label_of_interest='happiness',
        ...     include_mask=True,
        ...     feature_indices=feature_subset,
        ...     feature_stats=stats
        ... )
        >>> # Now happiness_dataset only contains samples with a non-NaN happiness_value
        >>> # and only includes data/mask for features 0, 5, 10, with feature 5 standardized.
        >>> # Use it with a DataLoader:
        >>> # from torch.utils.data import DataLoader
        >>> # data_loader = DataLoader(happiness_dataset, batch_size=32)
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 root_dir: str,
                 label_of_interest: str,
                 include_mask: bool = False,
                 feature_indices: Optional[List[int]] = None,
                 feature_stats: Optional[dict] = None,
                 postprocessors: Optional[List[Callable]] = None):
        """
        Args:
            dataframe (pd.DataFrame): The denormalized dataframe containing metadata and labels.
            root_dir (str): The root directory where the .npy files are located.
            label_of_interest (str): The label to filter by (without '_value' suffix).
                                     Only samples where this label is not NaN will be included.
            include_mask (bool): Whether to load and include a mask channel. Passed to BaseMhcDataset.
            feature_indices (Optional[List[int]]): Optional list of feature indices to select.
                                                    Passed to BaseMhcDataset.
            feature_stats (Optional[dict]): Optional dictionary for feature standardization.
                                            Passed to BaseMhcDataset.
            postprocessors (Optional[List[Callable]]): Postprocessors passed to BaseMhcDataset.
        """
        # Ensure the label_of_interest has the '_value' suffix for filtering
        label_col = f"{label_of_interest}_value" if not label_of_interest.endswith('_value') else label_of_interest

        # Check if the label exists in the dataframe
        if label_col not in dataframe.columns:
            logger.error(f"Label column '{label_col}' not found in the dataframe for filtering.")
            raise ValueError(f"Label column '{label_col}' not found in the dataframe.")

        # Filter the dataframe to only include rows where the label is not NaN
        original_len = len(dataframe)
        filtered_df = dataframe[dataframe[label_col].notna()].copy()

        if len(filtered_df) == 0:
            logger.error(f"No samples found with non-NaN values for label '{label_of_interest}'. Filtering removed all samples.")
            # Consider if raising ValueError is always desired, maybe warning is sufficient?
            # For now, retain ValueError as it indicates the dataset is unusable for this label.
            raise ValueError(f"No samples found with non-NaN values for label '{label_of_interest}'.")

        # Log info about the filtering result
        logger.info(f"Filtered dataset for label '{label_of_interest}' from {original_len} to {len(filtered_df)} samples.")

        # Initialize the parent class with the filtered dataframe and pass through other args
        # Parent's __init__ will handle validation and logging for mask, features, stats
        super().__init__(
            dataframe=filtered_df,
            root_dir=root_dir,
            include_mask=include_mask,
            feature_indices=feature_indices,
            feature_stats=feature_stats,
            postprocessors=postprocessors
        )

        # Store the label of interest for reference
        self.label_of_interest = label_of_interest


class FlattenedMhcDataset(BaseMhcDataset):
    """
    A subclass of BaseMhcDataset that flattens the time dimension.

    It loads the data identically to BaseMhcDataset but reshapes the output
    'data' and 'mask' tensors to combine the day and time_point dimensions.

    Output sample['data'] shape: (num_selected_features, num_days * 1440)
    Output sample['mask'] shape (if include_mask=True): (num_selected_features, num_days * 1440)
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 root_dir: str,
                 include_mask: bool = False,
                 feature_indices: Optional[List[int]] = None,
                 feature_stats: Optional[dict] = None,
                 postprocessors: Optional[List[Callable]] = None):
        """
        Initializes the FlattenedMhcDataset. Arguments are passed directly
        to the BaseMhcDataset constructor.

        Args:
            dataframe (pd.DataFrame): The denormalized dataframe.
            root_dir (str): The root directory containing participant subdirectories.
            include_mask (bool): Whether to load and include a mask channel.
            feature_indices (Optional[List[int]]): Optional list of feature indices.
            feature_stats (Optional[dict]): Optional dictionary for feature standardization.
            postprocessors (Optional[List[Callable]]): Postprocessors passed to BaseMhcDataset.
        """
        super().__init__(dataframe=dataframe,
                         root_dir=root_dir,
                         include_mask=include_mask,
                         feature_indices=feature_indices,
                         feature_stats=feature_stats,
                         postprocessors=postprocessors)
        logger.info("Initialized FlattenedMhcDataset. Output data/mask shape will be (num_features, num_days * 1440).")

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset and reshapes the data/mask tensors.

        Calls the parent __getitem__ and then reshapes the 'data' and 'mask'
        tensors from (num_days, num_features, 1440) to
        (num_features, num_days * 1440).

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'data' (torch.Tensor): Shape (num_selected_features, num_days * 1440).
                - 'mask' (torch.Tensor, optional): Shape (num_selected_features, num_days * 1440).
                - 'labels' (dict): Label values.
                - 'metadata' (dict): Metadata.
        """
        # Get the original sample with shape (D, F, T)
        sample = super().__getitem__(idx)

        data_tensor = sample['data'] # Shape (D, F, T)
        num_days, num_features, time_points = data_tensor.shape
        total_time_points = num_days * time_points

        # Reshape data: (D, F, T) -> (F, D, T) -> (F, D * T)
        reshaped_data = data_tensor.permute(1, 0, 2).reshape(num_features, total_time_points)
        sample['data'] = reshaped_data

        # Reshape mask if it exists
        if 'mask' in sample:
            mask_tensor = sample['mask'] # Shape (D, F, T)
            # Ensure mask shape matches data shape before permuting
            if mask_tensor.shape == (num_days, num_features, time_points):
                 reshaped_mask = mask_tensor.permute(1, 0, 2).reshape(num_features, total_time_points)
                 sample['mask'] = reshaped_mask
            else:
                 logger.warning(f"Mask shape {mask_tensor.shape} does not match data shape {(num_days, num_features, time_points)} for sample {idx}. Skipping mask reshape.")

        return sample


class ForecastingEvaluationDataset(BaseMhcDataset):
    """
    A subclass of BaseMhcDataset designed for forecasting tasks.

    It takes the time series data loaded by the parent class and splits it
    into input sequences (data_x) and target sequences (data_y) based on
    specified lengths and overlap.

    The splitting occurs along the time dimension (axis 2, typically 1440 points).

    Assumes the base data loaded by `BaseMhcDataset.__getitem__` has shape
    (num_days, num_selected_features, num_time_points).

    Output `data_x` shape: (num_days, num_selected_features, sequence_len)
    Output `data_y` shape: (num_days, num_selected_features, prediction_horizon)
    Output `mask_x`/`mask_y` shapes follow `data_x`/`data_y` if `include_mask=True`.

    Raises:
        ValueError: If `sequence_len`, `prediction_horizon`, or `overlap` parameters
                    are invalid or result in indices outside the available time points
                    (assumed to be `_EXPECTED_TIME_POINTS`).
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 root_dir: str,
                 sequence_len: int,
                 prediction_horizon: int,
                 overlap: int = 0,
                 include_mask: bool = False,
                 feature_indices: Optional[List[int]] = None,
                 feature_stats: Optional[dict] = None,
                 postprocessors: Optional[List[Callable]] = None):
        """
        Args:
            dataframe (pd.DataFrame): The denormalized dataframe.
            root_dir (str): The root directory containing participant subdirectories.
            sequence_len (int): The length of the input sequence (x) along the time axis.
            prediction_horizon (int): The length of the target sequence (y) along the time axis.
            overlap (int): The number of time points the end of x overlaps with the start of y.
                           Can be positive (overlap), zero (adjacent), or negative (gap).
                           Defaults to 0.
            include_mask (bool): Whether to load and include a mask channel. Passed to BaseMhcDataset.
                                 If True, 'mask_x' and 'mask_y' will be included in the output.
            feature_indices (Optional[List[int]]): Optional list of feature indices to select.
                                                    Passed to BaseMhcDataset.
            feature_stats (Optional[dict]): Optional dictionary for feature standardization.
                                            Passed to BaseMhcDataset.
            postprocessors (Optional[List[Callable]]): Postprocessors passed to BaseMhcDataset.
        """
        # --- Parameter Validation ---
        if not isinstance(sequence_len, int) or sequence_len <= 0:
            raise ValueError("sequence_len must be a positive integer.")
        if not isinstance(prediction_horizon, int) or prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be a positive integer.")
        if not isinstance(overlap, int):
             raise ValueError("overlap must be an integer.")

        # Check if the required time span fits within the expected time points
        x_start = 0
        x_end = sequence_len
        y_start = sequence_len - overlap
        y_end = y_start + prediction_horizon

        if x_start < 0:
             raise ValueError("Calculated x_start index is negative (should not happen with current logic).")
        if y_start < 0:
             raise ValueError(f"Calculated y_start index ({y_start}) is negative. "
                              f"Check sequence_len ({sequence_len}) and overlap ({overlap}).")
                              
        # Check for divisibility by _EXPECTED_TIME_POINTS (1440)
        if sequence_len % _EXPECTED_TIME_POINTS != 0:
            raise ValueError(f"sequence_len ({sequence_len}) must be a multiple of points_per_day ({_EXPECTED_TIME_POINTS}) for day-based splitting.")
        if prediction_horizon % _EXPECTED_TIME_POINTS != 0:
            raise ValueError(f"prediction_horizon ({prediction_horizon}) must be a multiple of points_per_day ({_EXPECTED_TIME_POINTS}) for day-based splitting.")
        if overlap % _EXPECTED_TIME_POINTS != 0 and overlap != 0:
            raise ValueError(f"overlap ({overlap}) must be zero or a multiple of points_per_day ({_EXPECTED_TIME_POINTS}) for day-based splitting.")

        # --- Store Forecasting Parameters ---
        self.sequence_len = sequence_len
        self.prediction_horizon = prediction_horizon
        self.overlap = overlap

        # --- Initialize Parent Class ---
        # This handles validation/setup for dataframe, root_dir, include_mask,
        # feature_indices, and feature_stats.
        super().__init__(
            dataframe=dataframe,
            root_dir=root_dir,
            include_mask=include_mask,
            feature_indices=feature_indices,
            feature_stats=feature_stats,
            postprocessors=postprocessors
        )
        logger.info(f"Initialized ForecastingEvaluationDataset with sequence_len={sequence_len}, "
                    f"prediction_horizon={prediction_horizon}, overlap={overlap}.")
        logger.info(f"Input sequence (x) will span indices [{x_start}:{x_end}].")
        logger.info(f"Target sequence (y) will span indices [{y_start}:{y_end}].")


    def __getitem__(self, idx):
        """
        Retrieves a sample and splits its time series data into input (x) and target (y).

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'data_x' (torch.Tensor): Input sequence, shape (num_days, num_features, sequence_len).
                - 'data_y' (torch.Tensor): Target sequence, shape (num_days, num_features, prediction_horizon).
                - 'mask_x' (torch.Tensor, optional): Mask for input sequence. Included if `include_mask=True`.
                - 'mask_y' (torch.Tensor, optional): Mask for target sequence. Included if `include_mask=True`.
                - 'labels' (dict): Label values from the original sample.
                - 'metadata' (dict): Metadata from the original sample.
        """
        # 1. Get the full sample from the base class
        base_sample = super().__getitem__(idx)

        # 2. Extract full data and mask (if applicable)
        full_data = base_sample['data'] # Shape: (num_days, num_features, 1440)
        full_mask = base_sample.get('mask') # Shape: (num_days, num_features, 1440) or None

        # 3. Get dimensions and check divisibility constraints
        num_days_total, num_features, points_per_day = full_data.shape
        if points_per_day != _EXPECTED_TIME_POINTS:
            # This should ideally not happen if base dataset works correctly
            logger.warning(f"Unexpected points_per_day ({points_per_day}) for sample {idx}. Expected {_EXPECTED_TIME_POINTS}. Proceeding, but check data integrity.")
        
        if self.sequence_len % points_per_day != 0:
             raise ValueError(f"sequence_len ({self.sequence_len}) must be a multiple of points_per_day ({points_per_day}) for day-based splitting.")
        if self.prediction_horizon % points_per_day != 0:
            raise ValueError(f"prediction_horizon ({self.prediction_horizon}) must be a multiple of points_per_day ({points_per_day}) for day-based splitting.")
        if self.overlap % points_per_day != 0:
             # Allow zero overlap even if not divisible, otherwise require divisibility
             if self.overlap != 0:
                raise ValueError(f"overlap ({self.overlap}) must be zero or a multiple of points_per_day ({points_per_day}) for day-based splitting.")

        # 4. Calculate split indices based on days
        num_days_x = self.sequence_len // points_per_day
        num_days_y = self.prediction_horizon // points_per_day
        overlap_days = self.overlap // points_per_day if self.overlap != 0 else 0

        x_start_day = 0
        x_end_day = num_days_x
        y_start_day = num_days_x - overlap_days
        y_end_day = y_start_day + num_days_y

        # 5. Validate day indices against total available days
        if x_end_day > num_days_total:
            raise ValueError(f"Required input days ({num_days_x}) exceeds total available days ({num_days_total}) for sample index {idx}. (sequence_len={self.sequence_len})")
        if y_end_day > num_days_total:
             raise ValueError(f"Required target end day ({y_end_day}) exceeds total available days ({num_days_total}) for sample index {idx}. "
                              f"(input_days={num_days_x}, target_days={num_days_y}, overlap_days={overlap_days})")
        if y_start_day < 0:
             # This check might be redundant if overlap validation is strict, but good safety measure
             raise ValueError(f"Calculated y_start_day ({y_start_day}) is negative for sample index {idx}. Check sequence_len/overlap.")


        # 6. Perform the split on the day dimension (axis 0)
        data_x = full_data[x_start_day:x_end_day, :, :]
        data_y = full_data[y_start_day:y_end_day, :, :]

        # 7. Prepare the result dictionary, copying labels and metadata
        result_dict = {
            # Note the restored shape: (num_days_x, num_features, 1440)
            'data_x': data_x,
            # Note the restored shape: (num_days_y, num_features, 1440)
            'data_y': data_y,
            'labels': base_sample['labels'],
            'metadata': base_sample['metadata']
        }

        # 8. Perform the split on the mask if it exists and was requested
        if self.include_mask:
             if full_mask is not None:
                 mask_x = full_mask[x_start_day:x_end_day, :, :]
                 mask_y = full_mask[y_start_day:y_end_day, :, :]
                 # Shape: (num_days_x, num_features, 1440)
                 result_dict['mask_x'] = mask_x
                 # Shape: (num_days_y, num_features, 1440)
                 result_dict['mask_y'] = mask_y
             else:
                 logger.warning(f"Mask was requested (include_mask=True) but not found in base_sample "
                                f"for index {idx}. 'mask_x' and 'mask_y' will be missing.")


        return result_dict

class FlattenedForecastingDataset(FlattenedMhcDataset):
    """
    A subclass of FlattenedMhcDataset designed for forecasting tasks with flattened data.

    It takes the flattened time series data loaded by the parent class and splits it
    into input sequences (data_x) and target sequences (data_y) based on
    specified lengths and overlap.

    Assumes the base data loaded by `FlattenedMhcDataset.__getitem__` has shape
    (num_selected_features, num_days * num_time_points).

    Output `data_x` shape: (num_selected_features, sequence_len)
    Output `data_y` shape: (num_selected_features, prediction_horizon)
    Output `mask_x`/`mask_y` shapes follow `data_x`/`data_y` if `include_mask=True`.

    Raises:
        ValueError: If `sequence_len`, `prediction_horizon`, or `overlap` parameters
                    are invalid or result in indices outside the available time points.
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 root_dir: str,
                 sequence_len: int,
                 prediction_horizon: int,
                 overlap: int = 0,
                 include_mask: bool = False,
                 feature_indices: Optional[List[int]] = None,
                 feature_stats: Optional[dict] = None,
                 postprocessors: Optional[List[Callable]] = None):
        """
        Args:
            dataframe (pd.DataFrame): The denormalized dataframe.
            root_dir (str): The root directory containing participant subdirectories.
            sequence_len (int): The length of the input sequence (x) along the time axis.
            prediction_horizon (int): The length of the target sequence (y) along the time axis.
            overlap (int): The number of time points the end of x overlaps with the start of y.
                           Can be positive (overlap), zero (adjacent), or negative (gap).
                           Defaults to 0.
            include_mask (bool): Whether to load and include a mask channel. Passed to FlattenedMhcDataset.
                                 If True, 'mask_x' and 'mask_y' will be included in the output.
            feature_indices (Optional[List[int]]): Optional list of feature indices to select.
                                                    Passed to FlattenedMhcDataset.
            feature_stats (Optional[dict]): Optional dictionary for feature standardization.
                                            Passed to FlattenedMhcDataset.
            postprocessors (Optional[List[Callable]]): Postprocessors passed to FlattenedMhcDataset.
        """
        # --- Parameter Validation ---
        if not isinstance(sequence_len, int) or sequence_len <= 0:
            raise ValueError("sequence_len must be a positive integer.")
        if not isinstance(prediction_horizon, int) or prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be a positive integer.")
        if not isinstance(overlap, int):
             raise ValueError("overlap must be an integer.")

        # Check if the required time span fits within the expected time points
        x_start = 0
        x_end = sequence_len
        y_start = sequence_len - overlap
        y_end = y_start + prediction_horizon

        if x_start < 0:
             raise ValueError("Calculated x_start index is negative (should not happen with current logic).")
        if y_start < 0:
             raise ValueError(f"Calculated y_start index ({y_start}) is negative. "
                              f"Check sequence_len ({sequence_len}) and overlap ({overlap}).")

        # --- Store Forecasting Parameters ---
        self.sequence_len = sequence_len
        self.prediction_horizon = prediction_horizon
        self.overlap = overlap

        # --- Initialize Parent Class ---
        # This handles validation/setup for dataframe, root_dir, include_mask,
        # feature_indices, and feature_stats.
        super().__init__(
            dataframe=dataframe,
            root_dir=root_dir,
            include_mask=include_mask,
            feature_indices=feature_indices,
            feature_stats=feature_stats,
            postprocessors=postprocessors
        )
        logger.info(f"Initialized FlattenedForecastingDataset with sequence_len={sequence_len}, "
                    f"prediction_horizon={prediction_horizon}, overlap={overlap}.")
        logger.info(f"Input sequence (x) will span indices [{x_start}:{x_end}].")
        logger.info(f"Target sequence (y) will span indices [{y_start}:{y_end}].")


    def __getitem__(self, idx):
        """
        Retrieves a sample and splits its flattened time series data into input (x) and target (y).

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'data_x' (torch.Tensor): Input sequence, shape (num_features, sequence_len).
                - 'data_y' (torch.Tensor): Target sequence, shape (num_features, prediction_horizon).
                - 'mask_x' (torch.Tensor, optional): Mask for input sequence. Included if `include_mask=True`.
                - 'mask_y' (torch.Tensor, optional): Mask for target sequence. Included if `include_mask=True`.
                - 'labels' (dict): Label values from the original sample.
                - 'metadata' (dict): Metadata from the original sample.
        """
        # 1. Get the full sample from the parent class (already flattened)
        base_sample = super().__getitem__(idx)

        # 2. Extract full flattened data and mask (if applicable)
        full_data = base_sample['data']  # Shape: (num_features, num_days * 1440)
        full_mask = base_sample.get('mask')  # Shape: (num_features, num_days * 1440) or None

        # 3. Calculate split indices for the flattened dimension
        x_start = 0
        x_end = self.sequence_len
        y_start = self.sequence_len - self.overlap
        y_end = y_start + self.prediction_horizon

        # 4. Validate indices against total available time points
        num_features, total_time_points = full_data.shape
        if x_end > total_time_points:
            raise ValueError(f"Required input sequence length ({self.sequence_len}) exceeds total available time points ({total_time_points}) for sample index {idx}.")
        if y_end > total_time_points:
            raise ValueError(f"Required target end point ({y_end}) exceeds total available time points ({total_time_points}) for sample index {idx}.")
        if y_start < 0:
            raise ValueError(f"Calculated y_start ({y_start}) is negative for sample index {idx}. Check sequence_len/overlap.")

        # 5. Perform the split on the flattened time dimension (axis 1)
        data_x = full_data[:, x_start:x_end]
        data_y = full_data[:, y_start:y_end]

        # 6. Prepare the result dictionary, copying labels and metadata
        result_dict = {
            'data_x': data_x,  # Shape: (num_features, sequence_len)
            'data_y': data_y,  # Shape: (num_features, prediction_horizon)
            'labels': base_sample['labels'],
            'metadata': base_sample['metadata']
        }

        # 7. Perform the split on the mask if it exists and was requested
        if self.include_mask and full_mask is not None:
            mask_x = full_mask[:, x_start:x_end]
            mask_y = full_mask[:, y_start:y_end]
            result_dict['mask_x'] = mask_x
            result_dict['mask_y'] = mask_y
        elif self.include_mask:
            logger.warning(f"Mask was requested (include_mask=True) but not found in base_sample "
                          f"for index {idx}. 'mask_x' and 'mask_y' will be missing.")

        return result_dict