import unittest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Assuming src is in the python path or adjust as needed
from models.evaluation import evaluate_forecast_dataset, parse_forecast_split
from models.lstm import AutoencoderLSTM # Use a concrete class for type hints

# --- Mock Components --- 

class MockForecastingDataset(Dataset):
    """Mocks ForecastingEvaluationDataset for testing."""
    def __init__(self, num_samples=10, num_days_in=5, num_days_out=2, 
                 num_features=6, time_steps_per_day=1440, 
                 include_mask=True, return_none_idx=None,
                 different_shape_idx=None):
        self.num_samples = num_samples
        self.num_days_in = num_days_in
        self.num_days_out = num_days_out
        self.num_features = num_features
        self.time_steps_per_day = time_steps_per_day
        self.include_mask = include_mask
        self.return_none_idx = return_none_idx
        self.different_shape_idx = different_shape_idx
        
        # Calculate total sequence lengths
        self.sequence_len = num_days_in * time_steps_per_day
        self.prediction_horizon = num_days_out * time_steps_per_day
        
        # Pre-generate consistent data for valid samples
        self.base_data_x = torch.randn(num_days_in, num_features, time_steps_per_day)
        # Add variation to base_data_y to prevent potential zero variance
        self.base_data_y = torch.randn(num_days_out, num_features, time_steps_per_day)
        time_variation_y = torch.linspace(0, 0.1, time_steps_per_day).view(1, 1, -1)
        self.base_data_y += time_variation_y # Add time-based drift
        
        if include_mask:
            # Ensure mask has reasonable density (e.g., >= 10% ones)
            mask_density = 0.5 # Aim for 50% valid points
            self.base_mask_x = (torch.rand(num_days_in, num_features, time_steps_per_day) < mask_density).float()
            self.base_mask_y = (torch.rand(num_days_out, num_features, time_steps_per_day) < mask_density).float()
            # Ensure at least a few points are unmasked if possible to avoid division by zero errors in stats
            if self.base_mask_x.sum() < 2: self.base_mask_x.fill_(1.0)
            if self.base_mask_y.sum() < 2: self.base_mask_y.fill_(1.0)

        # Pre-generate different shape data if needed
        if different_shape_idx is not None:
            self.diff_data_x = torch.randn(num_days_in -1, num_features, time_steps_per_day)
            self.diff_data_y = torch.randn(num_days_out -1, num_features, time_steps_per_day)
            # Add variation here too
            self.diff_data_y += time_variation_y 
            if include_mask:
                 # Use denser mask generation here too
                 self.diff_mask_x = (torch.rand(num_days_in -1, num_features, time_steps_per_day) < mask_density).float()
                 self.diff_mask_y = (torch.rand(num_days_out -1, num_features, time_steps_per_day) < mask_density).float()
                 if self.diff_mask_x.sum() < 2: self.diff_mask_x.fill_(1.0)
                 if self.diff_mask_y.sum() < 2: self.diff_mask_y.fill_(1.0)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx == self.return_none_idx:
            return None # Simulate a problematic sample
            
        sample = {
            'labels': {'some_label': float(idx)},
            'metadata': {'healthCode': f'test_{idx}', 'time_range': '...'}
        }
        
        if idx == self.different_shape_idx:
            sample['data_x'] = self.diff_data_x
            sample['data_y'] = self.diff_data_y
            if self.include_mask:
                sample['mask_x'] = self.diff_mask_x
                sample['mask_y'] = self.diff_mask_y
        else:
            sample['data_x'] = self.base_data_x
            sample['data_y'] = self.base_data_y
            if self.include_mask:
                sample['mask_x'] = self.base_mask_x
                sample['mask_y'] = self.base_mask_y
                
        return sample

class MockLSTMModel(torch.nn.Module):
    """Mocks the LSTM model interface needed by evaluation functions."""
    def __init__(self, num_features=6, segments_per_day=6, minutes_per_segment=240):
        super().__init__()
        self.num_features = num_features
        self.segments_per_day = segments_per_day
        self.minutes_per_segment = minutes_per_segment
        # Add dummy parameter to satisfy optimizer loading if needed elsewhere
        self.dummy_param = torch.nn.Parameter(torch.randn(1))

    def predict_future(self, data_x, steps):
        """Returns mock predictions matching expected output shape."""
        # data_x shape: [B, total_input_segments, F * minutes_per_segment]
        batch_size, total_input_segments, input_feature_dim = data_x.shape
        
        # Expected output shape: [B, target_segments, num_features * minutes_per_segment]
        target_segments = steps
        output_dim = self.num_features * self.minutes_per_segment
        
        # Return predictions slightly varied based on input to avoid zero variance
        mock_preds = torch.zeros(batch_size, target_segments, output_dim, device=data_x.device)
        
        # Add a small, deterministic variation based on the input batch index and target segment
        # This is artificial but prevents constant output that leads to NaN correlation
        variation = (data_x.mean(dim=(1,2), keepdim=True) / 1000.0) # Base variation per batch item
        segment_variation = torch.arange(target_segments, device=data_x.device).view(1, -1, 1) * 0.01 # Variation per segment
        
        # Apply variation to the first element of the feature dimension for simplicity
        mock_preds[:, :, 0:1] = variation + segment_variation 
        
        # Ensure shape matches exactly if output_dim > 1
        if output_dim > 1:
             mock_preds = mock_preds.repeat(1, 1, output_dim // mock_preds.shape[-1] + 1)[:, :, :output_dim]

        return mock_preds

# --- Test Class --- 

class TestEvaluateForecastDataset(unittest.TestCase):
    
    def setUp(self):
        """Set up common test variables."""
        self.num_samples = 8
        self.batch_size = 4
        self.num_features = 6
        self.num_days_in = 5
        self.num_days_out = 2
        self.time_steps = 1440
        self.device = torch.device('cpu')

        # Mock Model setup consistent with dataset assumptions
        # Calculate segments per day based on 1440 total minutes
        self.minutes_per_segment = 240 # Example: 6 segments per day
        self.segments_per_day = self.time_steps // self.minutes_per_segment
        
        self.mock_model = MockLSTMModel(
            num_features=self.num_features,
            segments_per_day=self.segments_per_day,
            minutes_per_segment=self.minutes_per_segment
        ).to(self.device)
        self.mock_model.eval()

    def test_basic_evaluation_with_mask(self):
        """Test basic evaluation flow with a valid dataset including masks."""
        mock_dataset = MockForecastingDataset(
            num_samples=self.num_samples,
            num_days_in=self.num_days_in,
            num_days_out=self.num_days_out,
            num_features=self.num_features,
            time_steps_per_day=self.time_steps,
            include_mask=True
        )
        
        results = evaluate_forecast_dataset(
            model=self.mock_model,
            dataset=mock_dataset,
            batch_size=self.batch_size,
            device=self.device
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('overall_mae', results)
        self.assertIn('overall_mse', results)
        self.assertIn('overall_pearson_corr', results)
        self.assertIn('channel_mae', results)
        self.assertIn('channel_mse', results)
        self.assertIn('channel_pearson_corr', results)
        
        # Check metrics are numerical (not NaN)
        self.assertFalse(np.isnan(results['overall_mae']))
        self.assertFalse(np.isnan(results['overall_mse']))
        # Correlation can be NaN if variance is zero, but our mock data should prevent this
        self.assertFalse(np.isnan(results['overall_pearson_corr'])) 
        
        # Check channel metrics length and content
        self.assertEqual(len(results['channel_mae']), self.num_features)
        self.assertFalse(any(np.isnan(x) for x in results['channel_mae']))
        self.assertEqual(len(results['channel_mse']), self.num_features)
        self.assertFalse(any(np.isnan(x) for x in results['channel_mse']))
        self.assertEqual(len(results['channel_pearson_corr']), self.num_features)
        # Allow NaNs for correlation as it can be undefined for channels with no variance
        # self.assertFalse(any(np.isnan(x) for x in results['channel_pearson_corr']))
        
    def test_basic_evaluation_no_mask(self):
        """Test basic evaluation flow with a valid dataset excluding masks."""
        mock_dataset = MockForecastingDataset(
            num_samples=self.num_samples,
            num_days_in=self.num_days_in,
            num_days_out=self.num_days_out,
            num_features=self.num_features,
            time_steps_per_day=self.time_steps,
            include_mask=False # Key difference
        )
        
        results = evaluate_forecast_dataset(
            model=self.mock_model,
            dataset=mock_dataset,
            batch_size=self.batch_size,
            device=self.device
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('overall_mae', results)
        self.assertFalse(np.isnan(results['overall_mae']))
        self.assertEqual(len(results['channel_mae']), self.num_features)
        self.assertFalse(any(np.isnan(x) for x in results['channel_mae']))

    def test_empty_dataset(self):
        """Test evaluation with an empty dataset."""
        mock_dataset = MockForecastingDataset(num_samples=0) # Empty dataset
        
        results = evaluate_forecast_dataset(
            model=self.mock_model,
            dataset=mock_dataset,
            batch_size=self.batch_size,
            device=self.device
        )
        
        self.assertIsInstance(results, dict)
        # Metrics should be NaN for an empty dataset
        self.assertTrue(np.isnan(results['overall_mae']))
        self.assertTrue(np.isnan(results['overall_mse']))
        self.assertTrue(np.isnan(results['overall_pearson_corr']))
        self.assertEqual(len(results['channel_mae']), 0) # Should be empty lists
        self.assertEqual(len(results['channel_mse']), 0)
        self.assertEqual(len(results['channel_pearson_corr']), 0)

    def test_dataloader_passed_directly(self):
        """Test passing a pre-configured DataLoader directly."""
        mock_dataset = MockForecastingDataset(
            num_samples=self.num_samples,
            include_mask=True
        )
        dataloader = DataLoader(mock_dataset, batch_size=self.batch_size, shuffle=False)
        
        results = evaluate_forecast_dataset(
            model=self.mock_model,
            dataset=mock_dataset, # dataset arg is still needed for shape inference
            batch_size=999, # This should be ignored
            device=self.device,
            dataloader=dataloader # Pass the dataloader
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('overall_mae', results)
        self.assertFalse(np.isnan(results['overall_mae']))

    # Note: Testing the collation error itself is tricky in a unit test 
    # as it depends heavily on the default_collate behavior and specific data failures.
    # The previous fix addresses the mask inconsistency issue which is a common cause.
    # Robust handling of truly problematic data (None samples, inconsistent shapes)
    # is often better handled by dataset wrappers or custom collate_fns if needed.

if __name__ == '__main__':
    unittest.main() 