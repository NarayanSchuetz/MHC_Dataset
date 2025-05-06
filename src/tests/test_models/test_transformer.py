import pytest
import torch
import numpy as np

from src.models.transformer import (
    ForecastingTransformer,
    RevInForecastingTransformer
)


class TestForecastingTransformer:
    """Test cases for the ForecastingTransformer model."""
    
    @pytest.fixture
    def mock_batch_data(self):
        """
        Create mock batch data of shape (batch_size=2, num_days=3, features=24, minutes=1440).
        """
        return torch.randn(2, 3, 24, 1440)
    
    @pytest.fixture
    def mock_batch_mask(self):
        """Create mock binary mask for testing, same shape as mock_batch_data."""
        mask = torch.randint(0, 2, (2, 3, 24, 1440)).float()
        return mask
    
    @pytest.fixture
    def mock_batch(self, mock_batch_data, mock_batch_mask):
        """Create a mock batch dictionary with data, mask, and a label dictionary."""
        batch_size = mock_batch_data.shape[0]
        return {
            'data': mock_batch_data,
            'mask': mock_batch_mask,
            'labels': {
                'default': torch.tensor([0.5] * batch_size)
            }
        }
    
    @pytest.fixture
    def model_with_mask(self):
        """Create an ForecastingTransformer with masked loss enabled."""
        return ForecastingTransformer(
            d_model=64,
            n_heads=4,
            num_layers=1,
            use_masked_loss=True
        )
    
    @pytest.fixture
    def model_without_mask(self):
        """Create an ForecastingTransformer with masked loss disabled."""
        return ForecastingTransformer(
            d_model=64,
            n_heads=4,
            num_layers=1,
            use_masked_loss=False
        )
    
    def test_init(self, model_with_mask, model_without_mask):
        """Test model initialization for masked vs non-masked versions."""
        assert model_with_mask.use_masked_loss is True
        assert model_without_mask.use_masked_loss is False
        assert model_with_mask.d_model == 64
        assert len(model_with_mask.blocks) == 1
    
    def test_preprocess_batch_without_mask(self, model_without_mask, mock_batch_data):
        """Test preprocessing of batch data without mask."""
        preprocessed = model_without_mask.preprocess_batch(mock_batch_data)
        
        # Expect shape (batch_size, num_days*48, 24*30)
        batch_size = mock_batch_data.shape[0]
        num_days = mock_batch_data.shape[1]
        features = mock_batch_data.shape[2]
        minutes_per_day = mock_batch_data.shape[3]
        
        segments_per_day = minutes_per_day // model_without_mask.minutes_per_segment  # 48
        num_segments = num_days * segments_per_day
        output_features = features * model_without_mask.minutes_per_segment
        
        expected_shape = (batch_size, num_segments, output_features)
        assert preprocessed.shape == expected_shape
        
        # Check for NaNs
        assert not torch.isnan(preprocessed).any()
    
    def test_preprocess_batch_with_mask(self, model_with_mask, mock_batch_data, mock_batch_mask):
        """Test preprocessing of batch data with mask."""
        preprocessed_data, preprocessed_mask = model_with_mask.preprocess_batch(mock_batch_data, mock_batch_mask)
        
        # Expect shape (batch_size, num_days*48, 24*30)
        batch_size = mock_batch_data.shape[0]
        num_days = mock_batch_data.shape[1]
        features = mock_batch_data.shape[2]
        minutes_per_day = mock_batch_data.shape[3]
        
        segments_per_day = minutes_per_day // model_with_mask.minutes_per_segment
        num_segments = num_days * segments_per_day
        output_features = features * model_with_mask.minutes_per_segment
        
        expected_shape = (batch_size, num_segments, output_features)
        assert preprocessed_data.shape == expected_shape
        assert preprocessed_mask.shape == expected_shape
        # Check no NaNs
        assert not torch.isnan(preprocessed_data).any()
        # Check mask is binary
        assert torch.all((preprocessed_mask == 0) | (preprocessed_mask == 1))
    
    def test_forward_without_mask(self, model_without_mask, mock_batch):
        """Test forward pass without masked loss."""
        output = model_without_mask(mock_batch)
        assert 'sequence_output' in output
        assert 'target_segments' in output
        assert 'label_predictions' in output
        assert 'target_mask' not in output
        
        # Check shapes
        batch_size = mock_batch['data'].shape[0]
        num_days = mock_batch['data'].shape[1]
        minutes_per_day = mock_batch['data'].shape[3]
        
        segments_per_day = minutes_per_day // model_without_mask.minutes_per_segment
        num_segments = num_days * segments_per_day
        
        # shape => (batch_size, num_segments - prediction_horizon, features_per_segment)
        pred_horizon = model_without_mask.prediction_horizon
        expected_output_shape = (batch_size, num_segments - pred_horizon, model_without_mask.features_per_segment)
        assert output['sequence_output'].shape == expected_output_shape
        assert output['target_segments'].shape == expected_output_shape
        assert output['label_predictions']['default'].shape == (batch_size,)
    
    def test_forward_with_mask(self, model_with_mask, mock_batch):
        """Test forward pass with masked loss."""
        output = model_with_mask(mock_batch)
        assert 'sequence_output' in output
        assert 'target_segments' in output
        assert 'target_mask' in output
        assert 'label_predictions' in output
        
        batch_size = mock_batch['data'].shape[0]
        num_days = mock_batch['data'].shape[1]
        minutes_per_day = mock_batch['data'].shape[3]
        
        segments_per_day = minutes_per_day // model_with_mask.minutes_per_segment
        num_segments = num_days * segments_per_day
        
        pred_horizon = model_with_mask.prediction_horizon
        expected_output_shape = (batch_size, num_segments - pred_horizon, model_with_mask.features_per_segment)
        assert output['sequence_output'].shape == expected_output_shape
        assert output['target_segments'].shape == expected_output_shape
        assert output['target_mask'].shape == expected_output_shape
        assert output['label_predictions']['default'].shape == (batch_size,)
    
    def test_compute_loss_without_mask(self, model_without_mask, mock_batch):
        """Test loss computation without masked loss."""
        output = model_without_mask(mock_batch)
        loss = model_without_mask.compute_loss(output, mock_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_compute_loss_with_mask(self, model_with_mask, mock_batch):
        """Test loss computation with masked loss."""
        output = model_with_mask(mock_batch)
        loss = model_with_mask.compute_loss(output, mock_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_masked_loss_ignores_masked_values(self):
        """
        Test that masked loss calculation ignores masked values.
        """
        model = ForecastingTransformer(use_masked_loss=True)
        batch_size = 1
        seq_len = 2
        feats = model.features_per_segment
        
        # Simple outputs
        sequence_output = torch.ones(batch_size, seq_len, feats)
        target_segments = torch.zeros(batch_size, seq_len, feats)
        
        # All observed => MSE should be 1.0
        mask_all_ones = torch.ones(batch_size, seq_len, feats)
        model_output_all = {
            'sequence_output': sequence_output,
            'target_segments': target_segments,
            'target_mask': mask_all_ones
        }
        loss_all = model.compute_loss(model_output_all, {})
        assert abs(loss_all.item() - 1.0) < 1e-5
        
        # Half observed
        mask_half = torch.zeros_like(mask_all_ones)
        half_size = feats // 2
        mask_half[:, :, :half_size] = 1.0
        model_output_half = {
            'sequence_output': sequence_output,
            'target_segments': target_segments,
            'target_mask': mask_half
        }
        loss_half = model.compute_loss(model_output_half, {})
        assert abs(loss_half.item() - 1.0) < 1e-5
        
        # No observed
        mask_zero = torch.zeros(batch_size, seq_len, feats)
        model_output_none = {
            'sequence_output': sequence_output,
            'target_segments': target_segments,
            'target_mask': mask_zero
        }
        loss_none = model.compute_loss(model_output_none, {})
        assert loss_none.item() < 1e-5
    
    def test_predict_future(self, model_without_mask):
        """
        Test the future prediction functionality.
        """
        model_without_mask.eval()
        
        batch_size = 2
        seq_len = 5
        feats = model_without_mask.features_per_segment
        
        input_seq = torch.randn(batch_size, seq_len, feats)
        
        steps = 3
        future = model_without_mask.predict_future(input_seq, steps)
        
        # Should produce shape (batch_size, steps, feats)
        assert future.shape == (batch_size, steps, feats)
        
        # Check determinism in eval mode
        future2 = model_without_mask.predict_future(input_seq, steps)
        assert torch.allclose(future, future2)
    
    def test_sequential_data_preprocessing(self):
        """
        Test that data is segmented properly in a sequential manner.
        """
        model = ForecastingTransformer()
        
        batch_size = 1
        num_days = 1
        features = 24
        minutes = 1440
        
        sequential_data = torch.arange(batch_size * num_days * features * minutes, dtype=torch.float32)
        sequential_data = sequential_data.reshape(batch_size, num_days, features, minutes)
        
        preprocessed = model.preprocess_batch(sequential_data)
        
        segments_per_day = minutes // model.minutes_per_segment
        num_segments = num_days * segments_per_day
        feats_per_seg = model.features_per_segment
        
        assert preprocessed.shape == (batch_size, num_segments, feats_per_seg)
        
        # Check first segment's first feature
        # segment0 => minutes [0..29] for feature0
        # segment1 => minutes [30..59] for feature0, etc.
        segment0 = preprocessed[0, 0, :].numpy()
        segment1 = preprocessed[0, 1, :].numpy()
        
        # The first 30 values of segment0 correspond to the first 30 minutes of feature 0
        assert np.allclose(segment0[:30], np.arange(30))
        # The next 30 values in segment0 correspond to the first 30 minutes of feature 1, which starts at 1440
        # etc. (same logic as in the LSTM tests).
        
        # Just ensure no data is lost or scrambled in a major way:
        original_sum = sequential_data.sum().item()
        preproc_sum = preprocessed.sum().item()
        assert abs(original_sum - preproc_sum) < 1e-5, "Preprocessing should preserve sum of values"


class TestRevInForecastingTransformer:
    """Test the RevInForecastingTransformer model."""
    
    @pytest.fixture
    def mock_batch_data(self):
        return torch.randn(2, 3, 24, 1440)
    
    @pytest.fixture
    def mock_batch_mask(self):
        return torch.randint(0, 2, (2, 3, 24, 1440)).float()
    
    @pytest.fixture
    def mock_batch(self, mock_batch_data, mock_batch_mask):
        batch_size = mock_batch_data.shape[0]
        return {
            'data': mock_batch_data,
            'mask': mock_batch_mask,
            'labels': {
                'default': torch.tensor([0.5] * batch_size)
            }
        }
    
    @pytest.fixture
    def model_with_mask(self):
        return RevInForecastingTransformer(
            d_model=64,
            n_heads=4,
            num_layers=1,
            use_masked_loss=True
        )
    
    def test_forward_pass(self, model_with_mask, mock_batch):
        output = model_with_mask(mock_batch)
        assert 'sequence_output' in output
        assert 'target_segments' in output
        assert 'target_mask' in output
        assert 'label_predictions' in output
    
    def test_loss(self, model_with_mask, mock_batch):
        output = model_with_mask(mock_batch)
        loss = model_with_mask.compute_loss(output, mock_batch)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_predict_future(self, model_with_mask):
        model_with_mask.eval()
        batch_size = 2
        seq_len = 5
        feats = model_with_mask.features_per_segment
        input_seq = torch.randn(batch_size, seq_len, feats)
        
        steps = 3
        future = model_with_mask.predict_future(input_seq, steps)
        assert future.shape == (batch_size, steps, feats)
    
    def test_revin_round_trip(self, model_with_mask):
        """
        Test that normalizing + denormalizing returns the original data (approximately).
        """
        model = model_with_mask
        batch_size, seq_len = 2, 10
        feats = model.features_per_segment
        x = torch.randn(batch_size, seq_len, feats) * 5.0 + 10.0
        
        x_norm = model.rev_in(x, mode='norm')
        x_denorm = model.rev_in(x_norm, mode='denorm')
        
        assert torch.allclose(x, x_denorm, rtol=1e-4, atol=1e-4)
