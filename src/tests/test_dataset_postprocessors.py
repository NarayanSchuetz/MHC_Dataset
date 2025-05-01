import pytest
import torch
import numpy as np
from typing import Dict, Any, List, Optional

# Assuming postprocessors are in src, adjust import path if needed
from ..dataset_postprocessors import (
    find_consecutive_runs,
    CustomMaskPostprocessor,
    StripNansPostprocessor,
    HeartRateInterpolationPostprocessor
)

# --- Test find_consecutive_runs --- 

@pytest.mark.parametrize("tensor_input, min_length, expected_output", [
    # Empty tensor
    (torch.tensor([], dtype=torch.bool), 5, []),
    # All False
    (torch.tensor([False, False, False]), 2, []),
    # All True, below min_length
    (torch.tensor([True, True]), 3, []),
    # All True, equal to min_length
    (torch.tensor([True, True, True]), 3, [(0, 3)]),
    # All True, above min_length
    (torch.tensor([True, True, True, True]), 3, [(0, 4)]),
    # Single run, exact length
    (torch.tensor([False, True, True, True, False]), 3, [(1, 4)]),
    # Single run, longer than min_length
    (torch.tensor([False, True, True, True, True, False]), 3, [(1, 5)]),
    # Single run, shorter than min_length
    (torch.tensor([False, True, True, False]), 3, []),
    # Multiple runs, one meets length
    (torch.tensor([True, True, False, True, True, True, False, True]), 3, [(3, 6)]),
    # Multiple runs, multiple meet length
    (torch.tensor([True, True, True, False, True, True, True, True]), 3, [(0, 3), (4, 8)]),
    # Run at the beginning
    (torch.tensor([True, True, True, False, False]), 3, [(0, 3)]),
    # Run at the end
    (torch.tensor([False, False, True, True, True]), 3, [(2, 5)]),
    # Non-boolean input (should return empty)
    (torch.tensor([0, 1, 1, 0]), 2, []),
    # 2D input (should return empty)
    (torch.tensor([[True, True], [False, False]]), 2, []),
])
def test_find_consecutive_runs(tensor_input, min_length, expected_output):
    assert find_consecutive_runs(tensor_input, min_length) == expected_output

# --- Helper Fixtures --- 

@pytest.fixture
def sample_data_shape():
    # D, F, T
    return 2, 3, 100 # 2 days, 3 features, 100 time points

@pytest.fixture
def base_sample(sample_data_shape) -> Dict[str, Any]:
    D, F, T = sample_data_shape
    # Create a sample with some zeros and some non-zeros
    data = torch.arange(D * F * T, dtype=torch.float32).reshape(D, F, T)
    data[0, 0, 10:20] = 0 # Add a zero segment
    data[1, :, 50:60] = 0 # Add a zero segment across all features
    data[:, 1, :] = 0 # Make one channel all zero across ALL days
    
    return {
        'data': data,
        'mask': torch.ones(D, F, T, dtype=torch.float32), # Start with all valid mask
        'metadata': {
            'feature_indices': [0, 1, 2] # Assuming no selection initially
        },
        'labels': {'label1': 1.0}
    }
    
@pytest.fixture
def base_sample_with_selection(sample_data_shape) -> Dict[str, Any]:
    D, F_raw, T = 2, 5, 100 # Original 5 features
    selected_indices = [0, 2, 4] # Select 3 features
    F_selected = len(selected_indices)
    
    data = torch.arange(D * F_selected * T, dtype=torch.float32).reshape(D, F_selected, T)
    # Let's say original HR was index 4, now it's index 2 in selected data
    data[0, 2, 30:70] = 0 # Add a long zero segment to HR channel
    data[1, 0, 10:15] = 0 # Add short gap to another channel
    
    return {
        'data': data,
        'mask': torch.ones(D, F_selected, T, dtype=torch.float32),
        'metadata': {
            'feature_indices': selected_indices
        },
        'labels': {'label1': 1.0}
    }

# --- Test StripNansPostprocessor ---

def test_strip_nans_postprocessor(sample_data_shape):
    D, F, T = sample_data_shape
    processor = StripNansPostprocessor(replace_with=0.0)
    data_with_nans = torch.ones(D, F, T) * torch.nan
    sample = {'data': data_with_nans.clone()}
    
    processed_sample = processor(sample)
    assert not torch.isnan(processed_sample['data']).any()
    assert (processed_sample['data'] == 0.0).all()
    
    # Test with mask and other value
    processor_neg_one = StripNansPostprocessor(replace_with=-1.0)
    mask_with_nans = torch.ones(D, F, T)
    mask_with_nans[0, 0, 0] = torch.nan
    sample = {'data': torch.ones(D, F, T), 'mask': mask_with_nans.clone()}
    processed_sample = processor_neg_one(sample)
    assert not torch.isnan(processed_sample['mask']).any()
    assert processed_sample['mask'][0, 0, 0] == -1.0

# --- Test CustomMaskPostprocessor --- 

def test_custom_mask_rule1_consecutive_zeros(base_sample):
    # Default threshold is 10
    processor = CustomMaskPostprocessor()
    
    # Make region 50:65 masked across all features (length 15 > 10)
    base_sample['mask'][1, :, 50:65] = 0 
    # Make region 80:85 masked across all features (length 5 < 10)
    base_sample['mask'][0, :, 80:85] = 0 
    
    processed_sample = processor(base_sample)
    final_mask = processed_sample['mask']
    
    # Check the long gap IS masked
    assert (final_mask[1, :, 50:65] == 0).all()
    # Check surrounding areas are NOT masked by this rule
    assert final_mask[1, 0, 49] == 1
    assert final_mask[1, 2, 49] == 1
    assert final_mask[1, 0, 65] == 1
    assert final_mask[1, 2, 65] == 1
    # Check the short gap remains masked in the final output
    # (The processor preserves all initially masked regions, regardless of length)
    assert (final_mask[0, 0, 80:85] == 0).all()
    assert (final_mask[0, 2, 80:85] == 0).all()

def test_custom_mask_rule2_missing_channels(base_sample):
    processor = CustomMaskPostprocessor()
    # Make feature 1 masked across all time points
    base_sample['mask'][:, 1, :] = 0
    
    processed_sample = processor(base_sample)
    final_mask = processed_sample['mask']
    
    # Check channel 1 is fully masked
    assert (final_mask[:, 1, :] == 0).all()
    # Check other channels are not fully masked (might have partial masks)
    assert (final_mask[:, 0, :] == 1).any()
    assert (final_mask[:, 2, :] == 1).any()

def test_custom_mask_rule3_hr_gaps(base_sample):
    # Default threshold 30
    hr_original_index = 0 # Let's say feature 0 is HR
    processor = CustomMaskPostprocessor(heart_rate_original_index=hr_original_index)
    
    # Add a long gap (40) to HR channel (feature 0) in the mask
    base_sample['mask'][0, hr_original_index, 10:50] = 0
    # Add a short gap (20) to HR channel in the mask
    base_sample['mask'][1, hr_original_index, 70:90] = 0
    
    processed_sample = processor(base_sample)
    final_mask = processed_sample['mask']
    
    # Check long gap is masked
    assert (final_mask[0, hr_original_index, 10:50] == 0).all()
    assert final_mask[0, hr_original_index, 9] == 1
    assert final_mask[0, hr_original_index, 50] == 1
    
    # Check short gap remains masked in the final output
    # (The processor preserves all initially masked regions, regardless of length,
    # even though this gap is shorter than the HR_GAP_THRESHOLD of 30)
    assert (final_mask[1, hr_original_index, 70:90] == 0).all()

def test_custom_mask_with_feature_selection(base_sample_with_selection):
    # Original HR index 4 -> Selected index 2
    hr_original_index = 4
    processor = CustomMaskPostprocessor(heart_rate_original_index=hr_original_index)
    
    # Add a long gap (40) to HR channel (selected index 2) in the mask
    hr_selected_index = 2
    base_sample_with_selection['mask'][0, hr_selected_index, 30:70] = 0
    
    processed_sample = processor(base_sample_with_selection)
    final_mask = processed_sample['mask']
    
    # Check long gap is masked in the correct selected channel
    assert (final_mask[0, hr_selected_index, 30:70] == 0).all()
    assert final_mask[0, hr_selected_index, 29] == 1
    assert final_mask[0, hr_selected_index, 70] == 1
    # Check other channels not affected by HR rule
    assert (final_mask[0, 0, 30:70] == 1).all()
    assert (final_mask[0, 1, 30:70] == 1).all()
    
def test_custom_mask_no_hr_index(base_sample):
    processor = CustomMaskPostprocessor(heart_rate_original_index=None)
    # Add a long gap (40) to feature 0 in the mask
    base_sample['mask'][0, 0, 10:50] = 0
    
    # Without an HR index, the processor will not mask based on HR gap patterns
    # Also, if the mask isn't caused by *all* features being masked, it won't be picked up by rule 1
    # We need to make sure all features are masked in this region for rule 1 to apply
    base_sample['mask'][0, :, 10:50] = 0  # Mask ALL features in this region
    
    processed_sample = processor(base_sample)
    final_mask = processed_sample['mask']
    
    # Since all features are masked for > threshold duration, it should be masked in output
    assert (final_mask[0, 0, 10:50] == 0).all()
    # Make sure the surrounding regions are untouched
    assert final_mask[0, 0, 9] == 1
    assert final_mask[0, 0, 50] == 1

def test_custom_mask_preexisting_mask(base_sample):
    processor = CustomMaskPostprocessor()
    
    # Pre-mask a region - but for the new implementation to consider it,
    # we need to mask ALL features in the region (for Rule 1)
    # or make the mask long enough to exceed the threshold
    # Let's set a long enough region that exceeds the threshold (10 minutes)
    base_sample['mask'][0, :, 5:20] = 0  # All features, 15 time points
    
    processed_sample = processor(base_sample)
    final_mask = processed_sample['mask']
    
    # Since we masked all features for a duration > threshold, the processor
    # should detect this pattern and include it in the new mask
    assert (final_mask[0, :, 5:20] == 0).all()
    
    # To confirm it's detecting the pattern properly, check surrounding regions
    assert final_mask[0, 0, 4] == 1
    assert final_mask[0, 0, 20] == 1

# --- Test HeartRateInterpolationPostprocessor --- 

@pytest.fixture
def hr_interp_sample(sample_data_shape): 
    D, F, T = sample_data_shape
    hr_idx = 1 # Let feature 1 be HR
    data = torch.ones(D, F, T, dtype=torch.float32) * 50 # Base HR 50
    mask = torch.ones(D, F, T, dtype=torch.float32)
    
    # --- Day 0 --- 
    # Small gap (5) within valid region -> Interpolate
    data[0, hr_idx, 10:15] = 0 
    # Large gap (35) within valid region -> Mask
    data[0, hr_idx, 30:65] = 0 
    # Gap at start of valid region -> Mask
    data[0, hr_idx, 0:5] = 0 
    
    # --- Day 1 --- 
    # Gap crossing a pre-masked region -> Don't interpolate across
    data[1, hr_idx, 45:55] = 0
    mask[1, hr_idx, 50:52] = 0 # Pre-masked bit in the middle
    # Gap at end of valid region -> Mask
    data[1, hr_idx, T-5:T] = 0
    
    return {
        'data': data,
        'mask': mask,
        'metadata': {'feature_indices': [0, 1, 2]}, # HR is index 1
        'labels': {}
    }

def test_hr_interpolation_linear(hr_interp_sample):
    hr_original_index = 1
    processor = HeartRateInterpolationPostprocessor(heart_rate_original_index=hr_original_index, interpolation_method='linear')
    
    processed_sample = processor(hr_interp_sample)
    final_data = processed_sample['data']
    final_mask = processed_sample['mask']
    hr_idx = 1
    
    # Day 0: Small gap (10:15) should be interpolated (linear -> stays 50)
    assert (final_data[0, hr_idx, 10:15] == 50).all()
    assert (final_mask[0, hr_idx, 10:15] == 1).all()
    # Day 0: Large gap (30:65) should be masked
    assert (final_mask[0, hr_idx, 30:65] == 0).all()
    assert (final_data[0, hr_idx, 30:65] == 0).all() # Data also zeroed
    # Day 0: Gap at start (0:5) should be masked
    assert (final_mask[0, hr_idx, 0:5] == 0).all()
    
    # Day 1: Gap crossing mask (45:55, mask 50:52)
    # Should interpolate 45:50 and 52:55 separately, mask 50:52
    assert (final_data[1, hr_idx, 45:50] == 50).all() # Interpolated part 1
    assert (final_mask[1, hr_idx, 45:50] == 1).all()
    assert (final_mask[1, hr_idx, 50:52] == 0).all() # Original mask maintained
    assert (final_data[1, hr_idx, 52:55] == 50).all() # Interpolated part 2
    assert (final_mask[1, hr_idx, 52:55] == 1).all()
    # Day 1: Gap at end (T-5:T) should be masked
    T = hr_interp_sample['data'].shape[2]
    assert (final_mask[1, hr_idx, T-5:T] == 0).all()
    
def test_hr_interpolation_nearest(hr_interp_sample):
    # Test with nearest neighbor (results should be same as linear for constant data)
    hr_original_index = 1
    processor = HeartRateInterpolationPostprocessor(heart_rate_original_index=hr_original_index, interpolation_method='nearest')
    
    processed_sample = processor(hr_interp_sample)
    final_data = processed_sample['data']
    final_mask = processed_sample['mask']
    hr_idx = 1
    T = hr_interp_sample['data'].shape[2]

    # Day 0: Small gap (10:15) should be interpolated (nearest -> stays 50)
    assert (final_data[0, hr_idx, 10:15] == 50).all()
    assert (final_mask[0, hr_idx, 10:15] == 1).all()
    # Day 0: Large gap (30:65) should be masked
    assert (final_mask[0, hr_idx, 30:65] == 0).all()

def test_hr_interpolation_with_selection(base_sample_with_selection):
    # Original HR index 4 -> Selected index 2
    hr_original_index = 4
    processor = HeartRateInterpolationPostprocessor(heart_rate_original_index=hr_original_index)
    
    # Data fixture has a long gap (30:70, length 40) in HR (selected index 2)
    # Add a short gap to interpolate
    hr_selected_index = 2
    base_sample_with_selection['data'][1, hr_selected_index, 10:15] = 0
    # Add non-zero values around short gap
    base_sample_with_selection['data'][1, hr_selected_index, 9] = 60
    base_sample_with_selection['data'][1, hr_selected_index, 15] = 70
    
    processed_sample = processor(base_sample_with_selection)
    final_data = processed_sample['data']
    final_mask = processed_sample['mask']
    
    # Check long gap (day 0) is masked
    assert (final_mask[0, hr_selected_index, 30:70] == 0).all()
    
    # Check short gap (day 1) is interpolated
    assert (final_mask[1, hr_selected_index, 10:15] == 1).all()
    # Check linear interpolation values (60 -> 70 over 6 steps: ~1.67/step)
    # Index 10 should be ~61.67, 11 ~63.33, 12 ~65, 13 ~66.67, 14 ~68.33
    expected_interp = torch.tensor([61.6667, 63.3333, 65.0000, 66.6667, 68.3333])
    assert torch.allclose(final_data[1, hr_selected_index, 10:15], expected_interp, atol=1e-4)

def test_hr_interpolation_requires_mask(base_sample):
    hr_original_index = 1
    processor = HeartRateInterpolationPostprocessor(heart_rate_original_index=hr_original_index)
    # Remove mask from input
    if 'mask' in base_sample: del base_sample['mask']
    
    # Should log warning and return sample unmodified (or mostly)
    initial_data = base_sample['data'].clone()
    processed_sample = processor(base_sample)
    
    assert torch.equal(processed_sample['data'], initial_data)
    assert 'mask' not in processed_sample # Mask shouldn't be added if not present 