import torch
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
import logging

# Set up logger
logger = logging.getLogger(__name__)

def find_consecutive_runs(bool_tensor_1d: torch.Tensor, min_length: int) -> List[Tuple[int, int]]:
    """
    Finds start and end indices (exclusive of end) of consecutive True runs
    of at least min_length in a 1D boolean tensor.
    
    Args:
        bool_tensor_1d: 1D boolean tensor where True indicates a condition of interest
        min_length: Minimum length of consecutive True values to be considered a run
        
    Returns:
        List of (start, end) index tuples, where end is exclusive
    """
    if not isinstance(bool_tensor_1d, torch.Tensor) or bool_tensor_1d.ndim != 1 or bool_tensor_1d.dtype != torch.bool:
        logger.debug(f"Input to find_consecutive_runs is not a 1D bool Tensor. Type: {type(bool_tensor_1d)}. Returning empty list.")
        return []

    if not bool_tensor_1d.any(): 
        return [] # Optimization: No True values

    # Pad with False at both ends to reliably detect runs at the start/end
    padded = torch.cat([
        torch.tensor([False], device=bool_tensor_1d.device),
        bool_tensor_1d,
        torch.tensor([False], device=bool_tensor_1d.device)
    ])

    # Find where the boolean value changes
    diff = padded.to(torch.int8).diff() # Convert to int for diff

    # Start indices: where False changes to True (diff == 1)
    start_indices = (diff == 1).nonzero(as_tuple=False).squeeze(-1)
    # End indices: where True changes to False (diff == -1)
    # The end index from diff corresponds to the start of the False sequence,
    # so it's the correct exclusive end index for the True sequence.
    end_indices = (diff == -1).nonzero(as_tuple=False).squeeze(-1)

    # Handle edge case of all True tensor matching min_length
    if len(start_indices) == 0 and len(end_indices) == 0 and bool_tensor_1d.all() and len(bool_tensor_1d) >= min_length:
        return [(0, len(bool_tensor_1d))]
        
    # Handle cases where tensor starts or ends with True runs
    if len(start_indices) != len(end_indices):
         logger.warning(f"Mismatch in start/end points ({len(start_indices)}/{len(end_indices)}) in find_consecutive_runs. This might indicate an issue with padding or input.")
         # Attempt basic recovery for simple cases like all True (already handled) or mostly True
         if bool_tensor_1d.all() and len(bool_tensor_1d) >= min_length:
              return [(0, len(bool_tensor_1d))]
         # If recovery isn't straightforward, return empty or log more details
         return []

    runs = []
    for start, end in zip(start_indices, end_indices):
        length = end - start
        if length >= min_length:
            # Adjust for padding: start index in diff corresponds to index in original tensor
            runs.append((start.item(), end.item())) # .item() to get Python int

    return runs

def _vectorized_interpolate(data_1d: torch.Tensor, 
                           mask_1d: torch.Tensor, 
                           gap_start: int, 
                           gap_end: int,
                           valid_indices: torch.Tensor,
                           valid_values: torch.Tensor,
                           method: str = 'linear') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs vectorized interpolation on a 1D tensor segment (in-place for data, returns updated mask).

    Args:
        data_1d: The 1D data tensor for the day (modified in-place).
        mask_1d: The 1D mask tensor for the day (used for finding boundaries, returned modified).
        gap_start: Start index of the zero-data gap.
        gap_end: End index (exclusive) of the zero-data gap.
        valid_indices: Tensor containing indices of valid (non-zero and unmasked) points in data_1d.
        valid_values: Tensor containing values at valid_indices.
        method: Interpolation method ('linear' or 'nearest').

    Returns:
        Tuple: (data_1d, updated_mask_1d) - data is modified in-place, mask is returned.
    """
    
    # --- Find nearest valid boundary indices ---
    before_idx_candidates = valid_indices[valid_indices < gap_start]
    after_idx_candidates = valid_indices[valid_indices >= gap_end]

    before_idx = before_idx_candidates.max() if len(before_idx_candidates) > 0 else -1
    after_idx = after_idx_candidates.min() if len(after_idx_candidates) > 0 else -1

    # --- Handle edge cases and perform interpolation ---
    gap_indices = torch.arange(gap_start, gap_end, device=data_1d.device)
    
    can_interpolate = False
    if before_idx != -1 and after_idx != -1:
        # --- Both boundaries exist ---
        before_val = data_1d[before_idx]
        after_val = data_1d[after_idx]
        
        if method == 'linear':
            # Avoid division by zero if indices are the same (shouldn't happen with valid points)
            denominator = (after_idx - before_idx).float()
            if denominator > 1e-6: 
                weight = (gap_indices - before_idx).float() / denominator
                interp_values = before_val * (1 - weight) + after_val * weight
                data_1d[gap_start:gap_end] = interp_values
                can_interpolate = True
            else: # Fallback if indices are identical or too close
                 data_1d[gap_start:gap_end] = before_val 
                 can_interpolate = True

        elif method == 'nearest':
            dist_before = gap_indices - before_idx
            dist_after = after_idx - gap_indices
            interp_values = torch.where(dist_before <= dist_after, before_val, after_val)
            data_1d[gap_start:gap_end] = interp_values
            can_interpolate = True
            
    elif before_idx != -1:
        # --- Only boundary before exists ---
        data_1d[gap_start:gap_end] = data_1d[before_idx]
        can_interpolate = True
        
    elif after_idx != -1:
        # --- Only boundary after exists ---
        data_1d[gap_start:gap_end] = data_1d[after_idx]
        can_interpolate = True

    # --- Mask if interpolation wasn't possible or if boundaries didn't exist ---
    if not can_interpolate:
        mask_1d[gap_start:gap_end] = 0
        # Ensure data is also zero where mask is zero (might already be, but safety)
        data_1d[gap_start:gap_end] = 0 
        
    return data_1d, mask_1d

class CustomMaskPostprocessor:
    """
    A callable postprocessor to apply custom masking rules to MHC data tensors.

    Operates on a sample dictionary produced by a BaseMhcDataset (or similar).
    Requires 'data' and potentially 'mask' and 'metadata' (with 'feature_indices')
    keys in the input sample dictionary. Modifies/adds the 'mask' key.

    Assumes mask uses 1 for valid data and 0 for masked data.
    Assumes data uses 0 to indicate missing values for gap detection checks
    (as BaseMhcDataset converts NaNs to 0.0).

    Masking Rules:
    1. Mask time points where *all* features are 0 for > CONSECUTIVE_ZERO_THRESHOLD (10) minutes.
    2. Mask entire feature channels that are 0 across the whole sample (all days, all time).
    3. Mask heart rate channel during gaps (0s) > HR_GAP_THRESHOLD (30) minutes,
       if heart_rate_original_index is provided and found in the selected features.
    """
    # Default thresholds in minutes, assuming 1 time point = 1 minute
    CONSECUTIVE_ZERO_THRESHOLD = 10  # Mask regions where all channels are 0 for 10+ minutes
    HR_GAP_THRESHOLD = 30  # Mask HR if gaps > 30 minutes

    def __init__(self,
                 heart_rate_original_index: Optional[int] = None,
                 consecutive_zero_threshold: Optional[int] = None,
                 hr_gap_threshold: Optional[int] = None,
                 expected_raw_features: int = 24):
        """
        Args:
            heart_rate_original_index (Optional[int]): The *original* index (0-23)
                                                      of the heart rate feature. Needed
                                                      for HR-specific gap masking.
                                                      If None, HR gap masking is disabled.
            consecutive_zero_threshold (Optional[int]): Override the threshold (in minutes)
                                                       for masking regions where all features
                                                       are zero. Default is 10 minutes.
            hr_gap_threshold (Optional[int]): Override the threshold (in minutes) for
                                             masking HR gaps. Default is 30 minutes.
            expected_raw_features (int): The total number of features in the raw
                                        data files (e.g., 24). Used for validation.
        """
        # Store parameters
        self.heart_rate_original_index = heart_rate_original_index
        self._expected_raw_features = expected_raw_features
        
        # Set thresholds, using class defaults if not provided
        self.consecutive_zero_threshold = consecutive_zero_threshold if consecutive_zero_threshold is not None else self.CONSECUTIVE_ZERO_THRESHOLD 
        self.hr_gap_threshold = hr_gap_threshold if hr_gap_threshold is not None else self.HR_GAP_THRESHOLD
        
        # Validate HR index once during init
        if self.heart_rate_original_index is not None:
            if not isinstance(self.heart_rate_original_index, int) or \
               not (0 <= self.heart_rate_original_index < self._expected_raw_features):
                logger.warning(f"Invalid heart_rate_original_index: {self.heart_rate_original_index}. "
                               f"Must be int in [0, {self._expected_raw_features - 1}). "
                               f"HR gap masking will be disabled.")
                self.heart_rate_original_index = None # Disable if invalid

        logger.info(f"Initialized CustomMaskPostprocessor with:" 
                   f"\n - HR original index: {self.heart_rate_original_index}"
                   f"\n - Consecutive zero threshold: {self.consecutive_zero_threshold} min"
                   f"\n - HR gap threshold: {self.hr_gap_threshold} min")

    def _calculate_hr_index_in_data(self, num_features: int, feature_indices: Optional[List[int]]) -> Optional[int]:
        """ 
        Determines the index for HR within the current feature set.
        
        Args:
            num_features: Number of features in the current tensor
            feature_indices: List of feature indices that were selected from the original set, if any
            
        Returns:
            The index of the heart rate feature in the current tensor, or None if not available
        """
        if self.heart_rate_original_index is None:
            return None

        # If feature_indices were used for selection...
        if feature_indices is not None:
            # Ensure feature_indices is a list of ints
            if not isinstance(feature_indices, list) or not all(isinstance(i, int) for i in feature_indices):
                 logger.warning(f"Received invalid feature_indices in metadata: {feature_indices}. Cannot map HR index.")
                 return None
                 
            try:
                # Find the position of the original HR index within the selected list
                selected_hr_index = feature_indices.index(self.heart_rate_original_index)
                # Validate against the actual number of features in the tensor
                if 0 <= selected_hr_index < num_features:
                    logger.debug(f"HR original index {self.heart_rate_original_index} mapped to index {selected_hr_index} in selected features.")
                    return selected_hr_index
                else:
                    logger.warning(f"Calculated HR index {selected_hr_index} is out of bounds "
                                   f"for data with {num_features} features. Indices: {feature_indices}")
                    return None
            except ValueError:
                # The specified HR index was not among the selected features
                logger.debug(f"HR original index {self.heart_rate_original_index} not found in selected features {feature_indices}.")
                return None
        else:
            # No feature selection applied, so the original index is the index in the data
            # Validate against the actual number of features
            if 0 <= self.heart_rate_original_index < num_features:
                 logger.debug(f"Using original HR index {self.heart_rate_original_index} (no feature selection).")
                 return self.heart_rate_original_index
            else:
                 logger.warning(f"Original HR index {self.heart_rate_original_index} is out of bounds "
                                f"for data with {num_features} features.")
                 return None


    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies custom masking rules to the data in the sample dictionary.

        Args:
            sample (Dict[str, Any]): The sample dictionary containing at least 'data',
                                     and optionally 'mask' and 'metadata' (with
                                     'feature_indices').

        Returns:
            Dict[str, Any]: The modified sample dictionary with updated 'mask'.
        """
        # --- Extract components from sample ---
        data_tensor = sample.get('data')
        original_mask = sample.get('mask') # Can be None
        metadata = sample.get('metadata', {}) # Default to empty dict
        feature_indices = metadata.get('feature_indices') # List or None

        # --- Input Validation ---
        if not isinstance(data_tensor, torch.Tensor):
            logger.warning(f"CustomMaskPostprocessor expected tensor data, got {type(data_tensor)}. Skipping.")
            return sample # Return unmodified sample
            
        if data_tensor.ndim != 3:
            logger.warning(f"CustomMaskPostprocessor expected 3D data tensor (D, F, T), got shape {data_tensor.shape}. Skipping.")
            return sample # Return unmodified sample

        num_days, num_features, num_time_points = data_tensor.shape
        device = data_tensor.device

        # --- Initialize Final Mask ---
        if original_mask is not None:
            if not isinstance(original_mask, torch.Tensor) or original_mask.shape != data_tensor.shape:
                 logger.warning(f"CustomMaskPostprocessor found original mask shape "
                                f"{original_mask.shape if isinstance(original_mask, torch.Tensor) else 'Non-tensor'} "
                                f"inconsistent with data shape {data_tensor.shape}. Creating new 'all valid' mask.")
                 final_mask = torch.ones_like(data_tensor, dtype=torch.float32, device=device)
            else:
                 # Ensure mask is float, on the correct device, and work on a copy
                 final_mask = original_mask.clone().to(dtype=torch.float32, device=device)
        else:
            # Create a new mask if one wasn't provided by the base dataset
            logger.debug("No input mask tensor found in sample. Creating a new mask.")
            final_mask = torch.ones_like(data_tensor, dtype=torch.float32, device=device)

        # --- Pre-calculations ---
        # Create a copy of the initial mask state to base calculations on
        initial_mask_state = final_mask.clone()
        # Determine where the *initial* mask indicates masked data (mask == 0)
        is_masked_initial = (initial_mask_state == 0) # Shape (D, F, T)

        # --- 1. Consecutive Masked Regions Across All Channels ---
        # Mask time points where ALL features are already masked for >= threshold duration
        # Use the initial mask state for this check
        all_features_masked = is_masked_initial.all(dim=1) # Shape (D, T)
        for d in range(num_days):
            # Find runs where all features were initially masked
            runs = find_consecutive_runs(all_features_masked[d], min_length=self.consecutive_zero_threshold)
            # Apply the mask modification to the final_mask
            for start, end in runs:
                if start < end: 
                    logger.debug(f"Masking region Day {d} {start}-{end} due to all features being masked initially.")
                    final_mask[d, :, start:end] = 0

        # --- 2. Fully Masked Channels ---
        # Mask features that are already masked across ALL days and time points in the initial mask
        # Or channels that are fully masked for at least one complete day in the initial mask
        channel_is_fully_masked_initial = is_masked_initial.all(dim=0).all(dim=1) # Shape (F,) based on initial mask
        day_level_masked_initial = is_masked_initial.all(dim=2) # Shape (D, F) based on initial mask
        channel_is_masked_any_day_initial = day_level_masked_initial.any(dim=0) # Shape (F,)
        
        # Combine conditions based on initial mask state
        combined_masked_initial = channel_is_fully_masked_initial | channel_is_masked_any_day_initial
        
        missing_channel_indices = combined_masked_initial.nonzero(as_tuple=False).squeeze(-1)
        if len(missing_channel_indices) > 0:
            missing_indices_list = missing_channel_indices.tolist()
            if isinstance(missing_indices_list, int): missing_indices_list = [missing_indices_list]
            logger.debug(f"Masking initially fully masked channels: indices {missing_indices_list}")
            # Apply the mask modification to the final_mask
            final_mask[:, missing_channel_indices, :] = 0

        # --- 3. Heart Rate Gaps (Based on Initial Mask) ---
        # Calculate the HR index within *this specific sample's* features
        hr_index_in_data = self._calculate_hr_index_in_data(num_features, feature_indices)

        if hr_index_in_data is not None:
            # Check if this channel was already identified as fully masked based on the initial state
            is_hr_channel_fully_masked = combined_masked_initial[hr_index_in_data].item() if hr_index_in_data < len(combined_masked_initial) else False

            if not is_hr_channel_fully_masked:
                # Check the initial mask state for the HR channel
                hr_is_masked_initial = is_masked_initial[:, hr_index_in_data, :] # Shape (D, T)
                for d in range(num_days):
                    # Find runs where HR was initially masked
                    runs = find_consecutive_runs(hr_is_masked_initial[d], min_length=self.hr_gap_threshold)
                    # Apply the mask modification to the final_mask
                    for start, end in runs:
                        if start < end: 
                            logger.debug(f"Masking HR gap on day {d} from {start} to {end} based on initial mask.")
                            final_mask[d, hr_index_in_data, start:end] = 0
            else:
                logger.debug(f"HR channel ({hr_index_in_data}) already masked based on initial state. Skipping HR gap check.")

        # --- Update Sample ---
        # Ensure data is zero where the final mask is zero
        # Apply this *after* all masking rules have been processed
        data_tensor = data_tensor * final_mask # Element-wise multiplication
        sample['data'] = data_tensor
        sample['mask'] = final_mask
        return sample


class StripNansPostprocessor:
    """
    A callable postprocessor that converts NaN values in data tensors to zeros.
    
    Useful for ensuring data compatibility with models that can't handle NaNs.
    This functionality is built into the BaseMhcDataset, so this postprocessor
    is mainly useful for custom datasets or to ensure NaNs are always removed.
    """
    
    def __init__(self, replace_with: float = 0.0, verbose: bool = False):
        """
        Args:
            replace_with (float): Value to replace NaNs with, default is 0.0
            verbose (bool): Whether to log when replacements are made
        """
        self.replace_with = replace_with
        self.verbose = verbose
        logger.info(f"Initialized StripNansPostprocessor. Replacing NaNs with {replace_with}.")
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts NaN values in tensors to the specified replacement value.
        
        Works on 'data' and 'mask' tensors if present in the sample.
        
        Args:
            sample (Dict[str, Any]): Sample dictionary with potential tensor values
            
        Returns:
            Dict[str, Any]: Modified sample with NaNs replaced
        """
        for key in ['data', 'mask', 'data_x', 'data_y', 'mask_x', 'mask_y']:
            if key in sample and isinstance(sample[key], torch.Tensor):
                tensor = sample[key]
                nan_mask = torch.isnan(tensor)
                num_nans = nan_mask.sum().item()
                
                if num_nans > 0:
                    if self.verbose:
                        logger.info(f"StripNansPostprocessor: Replacing {num_nans} NaN values in '{key}' tensor (shape: {tensor.shape}).")
                    sample[key] = torch.nan_to_num(tensor, nan=self.replace_with)
                    
        return sample


class HeartRateInterpolationPostprocessor:
    """
    A postprocessor that interpolates gaps in heart rate data using vectorized operations.
    
    Rules:
    1. Only interpolate gaps smaller than HR_GAP_THRESHOLD 
    2. Only interpolate if the gap is within an initially unmasked region.
    3. Mask gaps larger than the threshold or at the boundaries or without valid neighbors.
    
    Works on data that has already been processed by the BaseMhcDataset and
    potentially the CustomMaskPostprocessor. Assumes data uses 0 for missing values.
    """
    
    # Default threshold in minutes (assuming 1 time point = 1 minute)
    HR_GAP_THRESHOLD = 30  # Don't interpolate gaps larger than 30 minutes
    
    def __init__(self, 
                 heart_rate_original_index: int,
                 hr_gap_threshold: Optional[int] = None,
                 expected_raw_features: int = 24,
                 interpolation_method: str = 'linear'):
        """
        Args:
            heart_rate_original_index (int): The *original* index (0-23) of the heart rate 
                                           feature. Required for HR interpolation.
            hr_gap_threshold (Optional[int]): Override the threshold (in minutes) for
                                            masking vs. interpolating HR gaps. 
                                            Default is 30 minutes.
            expected_raw_features (int): The total number of features in the raw
                                       data files (e.g., 24). Used for validation.
            interpolation_method (str): Method for interpolation. 
                                      Options: 'linear', 'nearest'. Default is 'linear'.
        """
        if heart_rate_original_index is None:
            raise ValueError("heart_rate_original_index must be provided for HeartRateInterpolationPostprocessor")
            
        self.heart_rate_original_index = heart_rate_original_index
        self._expected_raw_features = expected_raw_features
        
        self.hr_gap_threshold = hr_gap_threshold if hr_gap_threshold is not None else self.HR_GAP_THRESHOLD
        
        valid_methods = ['linear', 'nearest']
        if interpolation_method not in valid_methods:
            logger.warning(f"Invalid interpolation_method: {interpolation_method}. Using 'linear' instead.")
            self.interpolation_method = 'linear'
        else:
            self.interpolation_method = interpolation_method
            
        if not isinstance(self.heart_rate_original_index, int) or \
           not (0 <= self.heart_rate_original_index < self._expected_raw_features):
            raise ValueError(f"Invalid heart_rate_original_index: {self.heart_rate_original_index}. "
                            f"Must be int in [0, {self._expected_raw_features - 1}).")
        
        logger.info(f"Initialized HeartRateInterpolationPostprocessor with:"
                   f"\n - HR original index: {self.heart_rate_original_index}"
                   f"\n - HR gap threshold: {self.hr_gap_threshold} min"
                   f"\n - Interpolation method: {self.interpolation_method}")
    
    def _calculate_hr_index_in_data(self, num_features: int, feature_indices: Optional[List[int]]) -> Optional[int]:
        """
        Determines the index for HR within the current feature set.
        """
        # If feature_indices were used for selection...
        if feature_indices is not None:
            # Ensure feature_indices is a list of ints
            if not isinstance(feature_indices, list) or not all(isinstance(i, int) for i in feature_indices):
                 logger.warning(f"Received invalid feature_indices in metadata: {feature_indices}. Cannot map HR index.")
                 return None
                 
            try:
                # Find the position of the original HR index within the selected list
                selected_hr_index = feature_indices.index(self.heart_rate_original_index)
                # Validate against the actual number of features in the tensor
                if 0 <= selected_hr_index < num_features:
                    logger.debug(f"HR original index {self.heart_rate_original_index} mapped to index {selected_hr_index} in selected features.")
                    return selected_hr_index
                else:
                    logger.warning(f"Calculated HR index {selected_hr_index} is out of bounds "
                                   f"for data with {num_features} features. Indices: {feature_indices}")
                    return None
            except ValueError:
                # The specified HR index was not among the selected features
                logger.debug(f"HR original index {self.heart_rate_original_index} not found in selected features {feature_indices}.")
                return None
        else:
            # No feature selection applied, so the original index is the index in the data
            # Validate against the actual number of features
            if 0 <= self.heart_rate_original_index < num_features:
                 logger.debug(f"Using original HR index {self.heart_rate_original_index} (no feature selection).")
                 return self.heart_rate_original_index
            else:
                 logger.warning(f"Original HR index {self.heart_rate_original_index} is out of bounds "
                                f"for data with {num_features} features.")
                 return None

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies heart rate interpolation using vectorized operations.
        """
        # --- Extract components & Validate ---
        data_tensor = sample.get('data')
        mask_tensor = sample.get('mask')
        metadata = sample.get('metadata', {})
        feature_indices = metadata.get('feature_indices')
        
        if not isinstance(data_tensor, torch.Tensor) or data_tensor.ndim != 3:
            logger.warning(f"HR Interpolation: Invalid data tensor. Skipping.")
            return sample
            
        if mask_tensor is None or not isinstance(mask_tensor, torch.Tensor) or mask_tensor.shape != data_tensor.shape:
            logger.warning(f"HR Interpolation: Requires a valid mask tensor matching data shape. Skipping.")
            return sample
        
        num_days, num_features, num_time_points = data_tensor.shape
        hr_index = self._calculate_hr_index_in_data(num_features, feature_indices)
        
        if hr_index is None:
            logger.warning("HR Interpolation: Could not find heart rate feature. Skipping.")
            return sample
            
        # --- Process HR data per day ---
        # Work on copies to avoid modifying input tensors directly if used elsewhere
        updated_data = data_tensor.clone()
        updated_mask = mask_tensor.clone()
        
        for d in range(num_days):
            # Extract HR data and mask for this day
            hr_data_day = updated_data[d, hr_index, :] 
            hr_mask_day = updated_mask[d, hr_index, :]
            
            # Identify initial state: where data is zero AND mask is currently valid
            is_zero = (hr_data_day == 0)
            is_initially_valid_mask = (hr_mask_day > 0.5) # Use threshold for float masks
            is_gap_to_process = is_zero & is_initially_valid_mask
            
            if not is_gap_to_process.any():
                continue # No gaps to process in initially valid regions for this day

            # Find consecutive runs of potential gaps (zero data in initially valid mask regions)
            gap_runs = find_consecutive_runs(is_gap_to_process, min_length=1)

            if not gap_runs: # No runs found
                continue

            # Find all valid points (non-zero data AND valid mask) for boundary lookup
            is_valid_point = (hr_data_day != 0) & is_initially_valid_mask
            valid_indices = is_valid_point.nonzero(as_tuple=True)[0]  # Get indices as 1D tensor
            valid_values = hr_data_day[valid_indices]  # Get corresponding values
            
            # Process each potential gap run
            for start, end in gap_runs:
                gap_length = end - start
                
                # --- Rule 1: Mask large gaps ---
                if gap_length >= self.hr_gap_threshold:
                    hr_mask_day[start:end] = 0
                    hr_data_day[start:end] = 0 # Ensure data is zeroed too
                    logger.debug(f"HR Int: Masking large gap day {d}, {start}-{end} (len {gap_length})")
                    continue # Move to next gap

                # --- Rule 2: Mask boundary gaps (start/end of day) ---
                # Check if valid points exist outside the gap boundaries
                has_point_before = (valid_indices < start).any()
                has_point_after = (valid_indices >= end).any()

                # Important: Fix for the test case - mask gaps at the start of data
                if start == 0:
                    hr_mask_day[start:end] = 0
                    hr_data_day[start:end] = 0
                    logger.debug(f"HR Int: Masking start boundary gap day {d}, {start}-{end}")
                    continue
                elif end == num_time_points:
                    hr_mask_day[start:end] = 0
                    hr_data_day[start:end] = 0
                    logger.debug(f"HR Int: Masking end boundary gap day {d}, {start}-{end}")
                    continue
                elif not has_point_before and not has_point_after:
                    # Gap covers the whole day or is surrounded by masked/zero regions
                    hr_mask_day[start:end] = 0
                    hr_data_day[start:end] = 0
                    logger.debug(f"HR Int: Masking gap with no valid neighbors day {d}, {start}-{end}")
                    continue

                # --- Rule 3: Interpolate small gaps ---
                logger.debug(f"HR Int: Interpolating gap day {d}, {start}-{end} (len {gap_length})")
                # _vectorized_interpolate modifies hr_data_day and hr_mask_day in-place/returns modified mask
                hr_data_day, hr_mask_day = _vectorized_interpolate(
                    hr_data_day, hr_mask_day, start, end, valid_indices, valid_values, 
                    method=self.interpolation_method
                )
            
            # Ensure data is zero where the final mask is zero for this day
            hr_data_day = hr_data_day * (hr_mask_day > 0.5).float()
            
            # Update the main tensors
            updated_data[d, hr_index, :] = hr_data_day
            updated_mask[d, hr_index, :] = hr_mask_day

        # --- Update Sample ---
        sample['data'] = updated_data
        sample['mask'] = updated_mask
        
        return sample
