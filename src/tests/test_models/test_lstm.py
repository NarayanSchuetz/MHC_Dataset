import pytest
import torch
import numpy as np
from src.models.lstm import AutoencoderLSTM, RevInAutoencoderLSTM
from src.models.revin import RevIN # Keep this import for direct RevIN tests if needed


class TestAutoencoderLSTM:
    """Test cases for AutoencoderLSTM model"""
    
    @pytest.fixture
    def mock_batch_data(self):
        """Create mock batch data for testing"""
        # Shape: batch_size=2, num_days=3, features=24, minutes=1440
        return torch.randn(2, 3, 24, 1440)
    
    @pytest.fixture
    def mock_batch_mask(self):
        """Create mock binary mask for testing"""
        # Shape matches mock_batch_data
        # Create a random binary mask
        mask = torch.randint(0, 2, (2, 3, 24, 1440)).float()
        return mask
    
    @pytest.fixture
    def mock_batch(self, mock_batch_data, mock_batch_mask):
        """Create a mock batch dictionary"""
        batch_size = mock_batch_data.shape[0] # Get batch size from data
        batch = {
            'data': mock_batch_data,
            'mask': mock_batch_mask,
            'labels': {
                # Create label tensor matching batch size
                'default': torch.tensor([0.5] * batch_size) 
            }
        }
        return batch
    
    @pytest.fixture
    def model_with_mask(self):
        """Create an AutoencoderLSTM model with masked loss enabled"""
        return AutoencoderLSTM(
            hidden_size=64,
            encoding_dim=32,
            num_layers=1,
            use_masked_loss=True
        )
    
    @pytest.fixture
    def model_without_mask(self):
        """Create an AutoencoderLSTM model with masked loss disabled"""
        return AutoencoderLSTM(
            hidden_size=64,
            encoding_dim=32,
            num_layers=1,
            use_masked_loss=False
        )
    
    def test_init(self, model_with_mask, model_without_mask):
        """Test model initialization"""
        # Test model with masked loss
        assert model_with_mask.use_masked_loss is True
        assert model_with_mask.hidden_size == 64
        assert model_with_mask.encoding_dim == 32
        
        # Test model without masked loss
        assert model_without_mask.use_masked_loss is False
    
    def test_preprocess_batch_without_mask(self, model_without_mask, mock_batch_data):
        """Test preprocessing of batch data without mask"""
        preprocessed = model_without_mask.preprocess_batch(mock_batch_data)
        
        # Calculate expected shape using the new correct approach
        batch_size = mock_batch_data.shape[0]
        num_days = mock_batch_data.shape[1]
        features = mock_batch_data.shape[2]
        minutes_per_day = mock_batch_data.shape[3]
        
        # Each day has 48 segments (1440 minutes ÷ 30 minutes)
        segments_per_day = minutes_per_day // model_without_mask.minutes_per_segment
        num_segments = num_days * segments_per_day
        
        # Output combines features with minutes per segment
        output_features = features * model_without_mask.minutes_per_segment
        
        expected_shape = (batch_size, num_segments, output_features)
        assert preprocessed.shape == expected_shape
        
        # Verify the calculation is correct
        assert segments_per_day == 48, "Each day should have 48 thirty-minute segments"
        assert num_segments == 144, "3 days × 48 segments = 144 segments total"
        assert output_features == 24 * 30, "Each segment has 24 features × 30 minutes"
        
        # Check NaN values are replaced with zeros
        assert not torch.isnan(preprocessed).any()
    
    def test_preprocess_batch_with_mask(self, model_with_mask, mock_batch_data, mock_batch_mask):
        """Test preprocessing of batch data with mask"""
        preprocessed_data, preprocessed_mask = model_with_mask.preprocess_batch(mock_batch_data, mock_batch_mask)
        
        # Calculate expected shape using the new correct approach
        batch_size = mock_batch_data.shape[0]
        num_days = mock_batch_data.shape[1]
        features = mock_batch_data.shape[2]
        minutes_per_day = mock_batch_data.shape[3]
        
        # Each day has 48 segments (1440 minutes ÷ 30 minutes)
        segments_per_day = minutes_per_day // model_with_mask.minutes_per_segment
        num_segments = num_days * segments_per_day
        
        # Output combines features with minutes per segment
        output_features = features * model_with_mask.minutes_per_segment
        
        expected_shape = (batch_size, num_segments, output_features)
        assert preprocessed_data.shape == expected_shape
        assert preprocessed_mask.shape == expected_shape
        
        # Check no NaNs in data
        assert not torch.isnan(preprocessed_data).any()
        
        # Check mask remains binary
        assert torch.all((preprocessed_mask == 0) | (preprocessed_mask == 1))
    
    def test_forward_without_mask(self, model_without_mask, mock_batch):
        """Test forward pass without masked loss"""
        output = model_without_mask(mock_batch)
        
        # Check output keys
        assert 'sequence_output' in output
        assert 'target_segments' in output
        assert 'label_predictions' in output
        assert 'target_mask' not in output  # Should not have mask in output
        
        # Check shapes based on correct preprocessing
        batch_size = mock_batch['data'].shape[0]
        num_days = mock_batch['data'].shape[1]
        features = mock_batch['data'].shape[2]
        minutes_per_day = mock_batch['data'].shape[3]
        
        segments_per_day = minutes_per_day // model_without_mask.minutes_per_segment
        num_segments = num_days * segments_per_day
        output_features = features * model_without_mask.minutes_per_segment
        
        # Output has shape (batch_size, num_segments - prediction_horizon, output_features)
        expected_output_shape = (batch_size, num_segments - model_without_mask.prediction_horizon, output_features)
        assert output['sequence_output'].shape == expected_output_shape
        assert output['target_segments'].shape == expected_output_shape
        assert output['label_predictions']['default'].shape == (batch_size,)
    
    def test_forward_with_mask(self, model_with_mask, mock_batch):
        """Test forward pass with masked loss"""
        output = model_with_mask(mock_batch)
        
        # Check output keys
        assert 'sequence_output' in output
        assert 'target_segments' in output
        assert 'target_mask' in output  # Should have mask in output
        assert 'label_predictions' in output
        
        # Check shapes based on correct preprocessing
        batch_size = mock_batch['data'].shape[0]
        num_days = mock_batch['data'].shape[1]
        features = mock_batch['data'].shape[2]
        minutes_per_day = mock_batch['data'].shape[3]
        
        segments_per_day = minutes_per_day // model_with_mask.minutes_per_segment
        num_segments = num_days * segments_per_day
        output_features = features * model_with_mask.minutes_per_segment
        
        # Output has shape (batch_size, num_segments - prediction_horizon, output_features)
        expected_output_shape = (batch_size, num_segments - model_with_mask.prediction_horizon, output_features)
        assert output['sequence_output'].shape == expected_output_shape
        assert output['target_segments'].shape == expected_output_shape
        assert output['target_mask'].shape == expected_output_shape
        assert output['label_predictions']['default'].shape == (batch_size,)
        
        # Check mask is binary
        assert torch.all((output['target_mask'] == 0) | (output['target_mask'] == 1))
    
    def test_compute_loss_without_mask(self, model_without_mask, mock_batch):
        """Test loss computation without masked loss"""
        output = model_without_mask(mock_batch)
        loss = model_without_mask.compute_loss(output, mock_batch)
        
        # Check loss is scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0  # Loss should be positive for random data
    
    def test_compute_loss_with_mask(self, model_with_mask, mock_batch):
        """Test loss computation with masked loss"""
        output = model_with_mask(mock_batch)
        loss = model_with_mask.compute_loss(output, mock_batch)
        
        # Check loss is scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0  # Loss should be positive for random data

    def test_masked_loss_ignores_masked_values(self):
        """Test that masked loss calculation correctly ignores masked values"""
        # Create a model with masked loss
        model = AutoencoderLSTM(use_masked_loss=True)
        
        # Create a simple batch with known values
        # For simplicity, we'll create data directly in the preprocessed form
        batch_size = 1
        num_segments = 2
        # With new preprocessing, output features would be 24 features × 30 minutes = 720
        output_features = 24 * 30
        
        # Create prediction and target tensors with simple values
        sequence_output = torch.ones(batch_size, num_segments, output_features)
        target_segments = torch.zeros(batch_size, num_segments, output_features)
        
        # Case 1: All values observed (mask all ones)
        # MSE should be 1.0 ((1-0)^2 = 1)
        mask_all_ones = torch.ones(batch_size, num_segments, output_features)
        model_output = {
            'sequence_output': sequence_output,
            'target_segments': target_segments,
            'target_mask': mask_all_ones
        }
        loss_all_observed = model.compute_loss(model_output, {})
        
        # Case 2: Half of the values masked (not observed)
        # Create a mask with half zeros and half ones
        mask_half = torch.zeros(batch_size, num_segments, output_features)
        mask_half[:, :, :output_features//2] = 1.0  # First half observed, second half not observed
        
        model_output_half = {
            'sequence_output': sequence_output,
            'target_segments': target_segments,
            'target_mask': mask_half
        }
        loss_half_observed = model.compute_loss(model_output_half, {})
        
        # Case 3: No values observed (mask all zeros)
        # Should result in almost zero loss (just the epsilon to avoid division by zero)
        mask_all_zeros = torch.zeros(batch_size, num_segments, output_features)
        model_output_zeros = {
            'sequence_output': sequence_output,
            'target_segments': target_segments,
            'target_mask': mask_all_zeros
        }
        loss_none_observed = model.compute_loss(model_output_zeros, {})
        
        # Assertions:
        # 1. All observed loss should be roughly 1.0
        assert abs(loss_all_observed.item() - 1.0) < 1e-5
        
        # 2. Half observed loss should be roughly 1.0 (since we're averaging only over observed values)
        assert abs(loss_half_observed.item() - 1.0) < 1e-5
        
        # 3. No observed values should give close to zero loss
        assert loss_none_observed.item() < 1e-5

    def test_teacher_forcing_initialization(self):
        """Test teacher forcing ratio initialization"""
        # Test default teacher forcing ratio
        model_default = AutoencoderLSTM()
        assert model_default.teacher_forcing_ratio == 0.5
        
        # Test custom teacher forcing ratio
        model_custom = AutoencoderLSTM(teacher_forcing_ratio=0.7)
        assert model_custom.teacher_forcing_ratio == 0.7
        
        # Test boundary values
        model_zero = AutoencoderLSTM(teacher_forcing_ratio=0.0)
        assert model_zero.teacher_forcing_ratio == 0.0
        
        model_one = AutoencoderLSTM(teacher_forcing_ratio=1.0)
        assert model_one.teacher_forcing_ratio == 1.0
    
    def test_teacher_forcing_training_vs_eval(self, mock_batch):
        """Test that teacher forcing behaves differently in training vs evaluation modes"""
        # Create a single model instance
        model = AutoencoderLSTM(teacher_forcing_ratio=0.0)
        
        # Set model to training mode
        model.train()
        
        # Set fixed seed for reproducibility
        torch.manual_seed(42)
        
        # Get output with teacher_forcing_ratio=0.0
        with torch.no_grad():
            output_train_without_tf = model(mock_batch)
        
        # Change teacher forcing ratio to 1.0
        model.teacher_forcing_ratio = 1.0
        
        # Reset seed
        torch.manual_seed(42)
        
        # Get output with teacher_forcing_ratio=1.0
        with torch.no_grad():
            output_train_with_tf = model(mock_batch)
        
        # The outputs should be different due to teacher forcing
        assert not torch.allclose(
            output_train_with_tf['sequence_output'], 
            output_train_without_tf['sequence_output']
        )
        
        # Now test in evaluation mode
        model.eval()
        
        # First with teacher_forcing_ratio=1.0 (already set)
        torch.manual_seed(42)
        with torch.no_grad():
            output_eval_with_tf = model(mock_batch)
        
        # Then with teacher_forcing_ratio=0.0
        model.teacher_forcing_ratio = 0.0
        torch.manual_seed(42)
        with torch.no_grad():
            output_eval_without_tf = model(mock_batch)
        
        # The outputs should be the same in eval mode regardless of teacher_forcing_ratio
        # This is because teacher forcing is only applied during training
        assert torch.allclose(
            output_eval_with_tf['sequence_output'], 
            output_eval_without_tf['sequence_output']
        )
    
    def test_teacher_forcing_consistency(self, mock_batch):
        """Test that outputs are consistent in eval mode regardless of teacher_forcing_ratio"""
        # Create a single model
        model = AutoencoderLSTM()
        
        # Set model to eval mode
        model.eval()
        
        # Try different teacher forcing ratios with the same model
        teacher_forcing_ratios = [0.0, 0.3, 0.5, 0.7, 1.0]
        outputs = []
        
        for ratio in teacher_forcing_ratios:
            # Update teacher forcing ratio
            model.teacher_forcing_ratio = ratio
            
            # Set the same seed before each run
            torch.manual_seed(42)
            
            with torch.no_grad():
                outputs.append(model(mock_batch)['sequence_output'])
        
        # All outputs should be the same in eval mode
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i])
        
        # Set model to train mode to test randomness in teacher forcing
        model.train()
        model.teacher_forcing_ratio = 0.5
        
        # Run with two different seeds to demonstrate randomness in teacher forcing
        torch.manual_seed(42)
        with torch.no_grad():
            output_train1 = model(mock_batch)['sequence_output']
            
        # Use a different seed
        torch.manual_seed(100)
        with torch.no_grad():
            output_train2 = model(mock_batch)['sequence_output']
        
        # Due to random teacher forcing, the two training runs should produce different outputs
        assert not torch.allclose(output_train1, output_train2)

    def test_predict_future(self, model_without_mask):
        """Test future prediction functionality"""
        # Create a simple input sequence with proper dimensions for the model
        batch_size = 2
        num_segments = 10
        output_features = 24 * 30  # 24 features × 30 minutes
        input_sequence = torch.randn(batch_size, num_segments, output_features)
        
        # Predict 3 steps into future
        steps = 3
        future = model_without_mask.predict_future(input_sequence, steps)
        
        # Check shape is correct
        assert future.shape == (batch_size, steps, output_features)
        
        # Ensure predictions are deterministic in eval mode
        model_without_mask.eval()
        future1 = model_without_mask.predict_future(input_sequence, steps)
        future2 = model_without_mask.predict_future(input_sequence, steps)
        assert torch.allclose(future1, future2)

    def test_sequential_data_preprocessing(self):
        """
        Test preprocessing with sequential data to verify correct arrangement of time series data.
        
        Creates an input tensor with sequential values (0, 1, 2, ...) and checks the
        first 3 segments after preprocessing to verify that time and feature dimensions
        are handled correctly.
        """
        # Create a model without mask
        model = AutoencoderLSTM(use_masked_loss=False)
        
        # Create a small test tensor with sequential values
        # Shape: (batch_size=1, num_days=1, features=24, minutes=1440)
        batch_size = 1
        num_days = 1
        features = 24
        minutes = 1440
        
        # Create tensor with sequential values (0, 1, 2, ...)
        sequential_tensor = torch.arange(batch_size * num_days * features * minutes, 
                                         dtype=torch.float32)
        sequential_tensor = sequential_tensor.reshape(batch_size, num_days, features, minutes)
        
        # Verify first few values of the input tensor
        assert sequential_tensor[0, 0, 0, 0] == 0  # First value
        assert sequential_tensor[0, 0, 0, 1] == 1  # Second value
        assert sequential_tensor[0, 0, 1, 0] == minutes  # First value of second feature

        # Preprocess the tensor
        preprocessed = model.preprocess_batch(sequential_tensor)
        
        # Calculate expected shapes
        segments_per_day = minutes // model.minutes_per_segment  # Should be 48
        num_segments = num_days * segments_per_day  # Should be 48 for 1 day
        output_features = features * model.minutes_per_segment  # Should be 24*30 = 720
        
        # Verify output shape
        expected_shape = (batch_size, num_segments, output_features)
        assert preprocessed.shape == expected_shape
        
        # Extract the first 3 segments for detailed verification
        segment0 = preprocessed[0, 0, :].numpy()  # First segment, all features
        segment1 = preprocessed[0, 1, :].numpy()  # Second segment, all features
        segment2 = preprocessed[0, 2, :].numpy()  # Third segment, all features
        
        # Verify the first segment (first 30 minutes with all features)
        # For the first feature (feature 0):
        # - Values should be [0, 1, 2, ..., 29] (first 30 minutes of feature 0)
        first_feature_segment0 = segment0[:30]
        expected_first_feature = np.arange(30)
        np.testing.assert_array_equal(first_feature_segment0, expected_first_feature)
        
        # For the second feature (feature 1):
        # - Values should be [1440, 1441, 1442, ..., 1469] (first 30 minutes of feature 1)
        second_feature_segment0 = segment0[30:60]
        expected_second_feature = np.arange(minutes, minutes + 30)
        np.testing.assert_array_equal(second_feature_segment0, expected_second_feature)
        
        # Verify the second segment (minutes 30-59 with all features)
        # For the first feature (feature 0):
        # - Values should be [30, 31, 32, ..., 59] (second 30 minutes of feature 0)
        first_feature_segment1 = segment1[:30]
        expected_first_feature_seg1 = np.arange(30, 60)
        np.testing.assert_array_equal(first_feature_segment1, expected_first_feature_seg1)
        
        # Instead of just printing, raise an assertion with the segment values to force pytest to display them
        message = "\n"
        message += f"Segment 0 (first 60 values): {segment0[:60]}\n"
        message += f"Segment 1 (first 60 values): {segment1[:60]}\n"
        message += f"Segment 2 (first 60 values): {segment2[:60]}"
        
        # Use a dummy assertion to force pytest to show the values even when the test passes
        assert True, message
        
        # Verify that all values in the preprocessed tensor are maintained
        # (no NaN replacement for this test)
        original_sum = sequential_tensor.sum().item()
        preprocessed_sum = preprocessed.sum().item()
        assert abs(original_sum - preprocessed_sum) < 1e-5


# Script to manually visualize the preprocessing of sequential data
if __name__ == "__main__":
    print("\n=== SEQUENTIAL DATA PREPROCESSING VISUALIZATION ===\n")
    
    # Create a model
    model = AutoencoderLSTM(use_masked_loss=False)
    
    # Create a small test tensor with sequential values
    # Shape: (batch_size=1, num_days=1, features=24, minutes=1440)
    batch_size = 1
    num_days = 1
    features = 24
    minutes = 1440
    
    # Create tensor with sequential values (0, 1, 2, ...)
    sequential_tensor = torch.arange(batch_size * num_days * features * minutes, dtype=torch.float32)
    sequential_tensor = sequential_tensor.reshape(batch_size, num_days, features, minutes)
    
    # Display structure of input tensor
    print(f"Input tensor shape: {sequential_tensor.shape}")
    print("First few values for feature 0, first minutes:")
    print(sequential_tensor[0, 0, 0, :10].numpy())
    print("First few values for feature 1, first minutes:")
    print(sequential_tensor[0, 0, 1, :10].numpy())
    print("First few values for feature 2, first minutes:")
    print(sequential_tensor[0, 0, 2, :10].numpy())
    
    # Preprocess the tensor
    preprocessed = model.preprocess_batch(sequential_tensor)
    
    # Display the output shape
    print(f"\nPreprocessed tensor shape: {preprocessed.shape}")
    
    # Extract the first 3 segments for detailed inspection
    segment0 = preprocessed[0, 0, :].numpy()  # First segment (minutes 0-29)
    segment1 = preprocessed[0, 1, :].numpy()  # Second segment (minutes 30-59)
    segment2 = preprocessed[0, 2, :].numpy()  # Third segment (minutes 60-89)
    
    # Display segments
    print("\nSegment 0 (first segment, minutes 0-29):")
    print("  Feature 0 values (first 10):", segment0[:10])
    print("  Feature 1 values (first 10):", segment0[30:40])
    print("  Feature 2 values (first 10):", segment0[60:70])
    
    print("\nSegment 1 (second segment, minutes 30-59):")
    print("  Feature 0 values (first 10):", segment1[:10])
    print("  Feature 1 values (first 10):", segment1[30:40])
    print("  Feature 2 values (first 10):", segment1[60:70])
    
    print("\nSegment 2 (third segment, minutes 60-89):")
    print("  Feature 0 values (first 10):", segment2[:10])
    print("  Feature 1 values (first 10):", segment2[30:40])
    print("  Feature 2 values (first 10):", segment2[60:70])


class TestRevInRevInAutoencoderLSTM:
    """Test cases for RevInAutoencoderLSTM model"""
    
    @pytest.fixture
    def mock_batch_data(self):
        """Create mock batch data for testing"""
        # Shape: batch_size=2, num_days=3, features=24, minutes=1440
        return torch.randn(2, 3, 24, 1440)

    @pytest.fixture
    def mock_batch_data_with_zeros(self):
        """Create mock batch data where feature 1 is all zeros"""
        data = torch.randn(2, 3, 24, 1440)
        data[:, :, 1, :] = 0.0 # Set feature 1 to all zeros
        return data

    @pytest.fixture
    def mock_batch_mask(self):
        """Create mock binary mask for testing"""
        mask = torch.randint(0, 2, (2, 3, 24, 1440)).float()
        return mask
    
    @pytest.fixture
    def mock_batch(self, mock_batch_data, mock_batch_mask):
        """Create a mock batch dictionary with random data"""
        batch_size = mock_batch_data.shape[0]
        batch = {
            'data': mock_batch_data,
            'mask': mock_batch_mask,
            'labels': {'default': torch.tensor([0.5] * batch_size)}
        }
        return batch

    @pytest.fixture
    def mock_batch_zeros(self, mock_batch_data_with_zeros, mock_batch_mask):
        """Create a mock batch dictionary with feature 1 as zeros"""
        batch_size = mock_batch_data_with_zeros.shape[0]
        batch = {
            'data': mock_batch_data_with_zeros,
            'mask': mock_batch_mask, # Use the same random mask
            'labels': {'default': torch.tensor([0.5] * batch_size)}
        }
        return batch
    
    @pytest.fixture
    def model_base(self):
        """Base RevInAutoencoderLSTM model"""
        return RevInAutoencoderLSTM(
            hidden_size=64, encoding_dim=32, num_layers=1, use_masked_loss=False,
            rev_in_affine=False, rev_in_subtract_last=False, rev_in_eps=1e-5
        )

    @pytest.fixture
    def model_with_mask(self):
        """RevInAutoencoderLSTM with masked loss enabled"""
        return RevInAutoencoderLSTM(
            hidden_size=64, encoding_dim=32, num_layers=1, use_masked_loss=True,
            rev_in_affine=False, rev_in_subtract_last=False, rev_in_eps=1e-5
        )
    
    @pytest.fixture
    def model_with_affine(self):
        """RevInAutoencoderLSTM with affine parameters"""
        return RevInAutoencoderLSTM(
            hidden_size=64, encoding_dim=32, num_layers=1, use_masked_loss=False,
            rev_in_affine=True, rev_in_subtract_last=False, rev_in_eps=1e-5
        )
    
    @pytest.fixture
    def model_with_subtract_last(self):
        """RevInAutoencoderLSTM with subtract_last mode"""
        return RevInAutoencoderLSTM(
            hidden_size=64, encoding_dim=32, num_layers=1, use_masked_loss=False,
            rev_in_affine=False, rev_in_subtract_last=True, rev_in_eps=1e-5
        )
    
    def test_init_with_eps(self, model_base):
        """Test initialization with rev_in_eps"""
        assert model_base.rev_in_eps == 1e-5
        assert model_base.rev_in.eps == 1e-5
        assert hasattr(model_base, 'rev_in')
        assert model_base.rev_in.affine is False
        assert model_base.rev_in.subtract_last is False
    
    def test_init_configurations(self, model_with_mask, model_with_affine, model_with_subtract_last):
        """Test model initialization with different RevIN configurations"""
        assert model_with_mask.use_masked_loss is True
        assert model_with_affine.rev_in.affine is True
        assert hasattr(model_with_affine.rev_in, 'affine_weight')
        assert model_with_subtract_last.rev_in.subtract_last is True
    
    def test_forward_pass(self, model_base, mock_batch):
        """Test forward pass with standard RevIN Time Series"""
        output = model_base(mock_batch)
        assert 'sequence_output' in output
        assert 'target_segments' in output
        assert 'target_mask' not in output # Masked loss is false
        assert 'label_predictions' in output

        # Basic shape checks
        batch_size = mock_batch['data'].shape[0]
        num_days = mock_batch['data'].shape[1]
        features = mock_batch['data'].shape[2]
        minutes_per_day = mock_batch['data'].shape[3]
        segments_per_day = minutes_per_day // model_base.minutes_per_segment
        num_segments = num_days * segments_per_day
        output_features = features * model_base.minutes_per_segment
        expected_output_shape = (batch_size, num_segments - model_base.prediction_horizon, output_features)

        assert output['sequence_output'].shape == expected_output_shape
        assert output['target_segments'].shape == expected_output_shape
        assert output['label_predictions']['default'].shape == (batch_size,)
    
    def test_compute_loss(self, model_base, mock_batch):
        """Test loss computation"""
        output = model_base(mock_batch)
        loss = model_base.compute_loss(output, mock_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss) and not torch.isinf(loss)
        assert loss.item() >= 0

    def test_revin_roundtrip(self, model_base, mock_batch_data):
        """Test that RevIN norm/denorm is reversible on time series features"""
        model = model_base
        x_orig = mock_batch_data
        batch_size, num_days, num_features, minutes_per_day = x_orig.shape
        time_steps = num_days * minutes_per_day

        # Reshape and clean
        x_reshaped = x_orig.permute(0, 2, 1, 3).reshape(batch_size, num_features, time_steps)
        x_reshaped = x_reshaped.permute(0, 2, 1) # (B, T, F)
        x_clean = torch.nan_to_num(x_reshaped, nan=0.0)

        # Check variance and normalize conditionally
        stdev = torch.std(x_clean, dim=1, keepdim=True)
        skip_revin_mask = (stdev < model.rev_in_eps)
        x_norm_all = model.rev_in(x_clean, mode='norm')
        x_norm_final = torch.where(skip_revin_mask, x_clean, x_norm_all)

        # Denormalize conditionally
        x_denorm_all = model.rev_in(x_norm_final, mode='denorm') # Pass normalized data to denorm
        x_denorm_final = torch.where(skip_revin_mask, x_norm_final, x_denorm_all) # Use original if skipped

        # Check that denormalization recovers the cleaned original data
        assert torch.allclose(x_clean, x_denorm_final, rtol=1e-4, atol=1e-4)

    def test_forward_skips_zero_variance_features(self, model_base, mock_batch_zeros):
        """Test that forward pass correctly handles features with zero variance"""
        model = model_base
        x_orig_zeros = mock_batch_zeros['data'] # Has feature 1 as all zeros
        batch_size, num_days, num_features, minutes_per_day = x_orig_zeros.shape
        time_steps = num_days * minutes_per_day

        # Manually perform the conditional normalization part of forward
        x_reshaped = x_orig_zeros.permute(0, 2, 1, 3).reshape(batch_size, num_features, time_steps)
        x_reshaped = x_reshaped.permute(0, 2, 1)
        x_clean = torch.nan_to_num(x_reshaped, nan=0.0)
        stdev = torch.std(x_clean, dim=1, keepdim=True)
        skip_revin_mask = (stdev < model.rev_in_eps) # Shape (B, 1, F)

        # Verify feature 1 is marked for skipping
        assert torch.all(skip_revin_mask[:, :, 1] == True)
        # Verify feature 0 is *not* marked for skipping (assuming random data has variance)
        assert torch.all(skip_revin_mask[:, :, 0] == False)

        # Apply conditional normalization
        x_norm_all = model.rev_in(x_clean, mode='norm')
        x_normalized_final = torch.where(skip_revin_mask, x_clean, x_norm_all)

        # Check feature 1 in normalized data - should be unchanged (all zeros)
        assert torch.all(x_normalized_final[:, :, 1] == 0.0)
        # Check feature 0 - should be normalized (mean approx 0, std approx 1)
        assert torch.mean(x_normalized_final[:, :, 0]).abs().item() < 0.1 # Mean close to 0
        assert torch.std(x_normalized_final[:, :, 0]).item() > 0.5 # Std deviation non-zero

        # --- Now run the full forward pass ---
        model.eval() # Ensure deterministic behavior
        with torch.no_grad():
            output = model(mock_batch_zeros)
            loss = model.compute_loss(output, mock_batch_zeros) # Also check loss computation

        # Check loss is valid
        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss should not be NaN or Inf with zero-variance features"


    def test_predict_future_with_zero_variance(self, model_base):
        """Test future prediction with zero-variance features in input"""
        model = model_base
        model.eval()

        # Create input sequence segments with feature 1 as zero
        batch_size = 2
        num_segments = 10 # History length
        num_features = model.num_original_features # e.g., 24
        mins_per_seg = model.minutes_per_segment # e.g., 30
        features_per_segment = model.features_per_segment # e.g., 720

        input_sequence = torch.randn(batch_size, num_segments, features_per_segment)

        # Manually set parts corresponding to feature 1 to zero
        # Reshape to access feature dimension: (B, N_seg, F * MinPerSeg) -> (B, N_seg, F, MinPerSeg)
        input_reshaped_for_zeroing = input_sequence.view(batch_size, num_segments, num_features, mins_per_seg)
        input_reshaped_for_zeroing[:, :, 1, :] = 0.0 # Set feature 1 to zero
        input_sequence_zeros = input_reshaped_for_zeroing.view(batch_size, num_segments, features_per_segment)

        # Predict future steps
        steps = 3
        with torch.no_grad():
            future = model.predict_future(input_sequence_zeros, steps) # Shape (B, steps, F_seg)

        # Check shape
        assert future.shape == (batch_size, steps, features_per_segment)

        # Check the zero-variance feature in the prediction
        # Reshape prediction: (B, steps, F * MinPerSeg) -> (B, steps, F, MinPerSeg)
        future_reshaped = future.view(batch_size, steps, num_features, mins_per_seg)


if __name__ == "__main__":
    # ... (previous visualization code) ...

    print("\n=== RevIN TIME SERIES NORMALIZATION VISUALIZATION ===\n")

    # Create a RevIN Time Series model
    revin_ts_model = RevInAutoencoderLSTM(
        hidden_size=64, encoding_dim=32, num_layers=1, use_masked_loss=False, num_features=3 # Use 3 features for simplicity
    )

    # Create a simple tensor (B, D, F, M) -> (1, 1, 3, 10) for easier visualization
    batch_size = 1
    num_days = 1
    num_features = 3
    minutes_per_day = 10
    time_steps = num_days * minutes_per_day

    x_orig = torch.randn(batch_size, num_days, num_features, minutes_per_day)
    # Make feature 1 constant (zero variance)
    x_orig[:, :, 1, :] = 5.0 # Constant value 5.0

    print(f"Original Data (Feature 0): {x_orig[0, 0, 0, :].numpy()}")
    print(f"Original Data (Feature 1): {x_orig[0, 0, 1, :].numpy()}")
    print(f"Original Data (Feature 2): {x_orig[0, 0, 2, :].numpy()}")

    # Reshape and clean
    x_reshaped = x_orig.permute(0, 2, 1, 3).reshape(batch_size, num_features, time_steps)
    x_reshaped = x_reshaped.permute(0, 2, 1)
    x_clean = torch.nan_to_num(x_reshaped, nan=0.0)

    # Check variance and normalize conditionally
    stdev = torch.std(x_clean, dim=1, keepdim=True)
    skip_revin_mask = (stdev < revin_ts_model.rev_in_eps)
    print(f"\nSkip RevIN Mask (True means skip): {skip_revin_mask.numpy()}")

    x_norm_all = revin_ts_model.rev_in(x_clean, mode='norm')
    x_norm_final = torch.where(skip_revin_mask, x_clean, x_norm_all)

    print(f"\nConditionally Normalized (Feature 0): {x_norm_final[0, :, 0].numpy()}")
    print(f"Conditionally Normalized (Feature 1): {x_norm_final[0, :, 1].numpy()}") # Should be 5.0
    print(f"Conditionally Normalized (Feature 2): {x_norm_final[0, :, 2].numpy()}")

    # Denormalize conditionally
    x_denorm_all = revin_ts_model.rev_in(x_norm_final, mode='denorm')
    x_denorm_final = torch.where(skip_revin_mask, x_norm_final, x_denorm_all)

    print(f"\nConditionally De-normalized (Feature 0): {x_denorm_final[0, :, 0].numpy()}")
    print(f"Conditionally De-normalized (Feature 1): {x_denorm_final[0, :, 1].numpy()}") # Should be 5.0
    print(f"Conditionally De-normalized (Feature 2): {x_denorm_final[0, :, 2].numpy()}")

    # Verify original cleaned and denormalized tensors are close
    diff = torch.abs(x_clean - x_denorm_final).max().item()
    print(f"\nMaximum absolute difference after roundtrip: {diff:.8f}")
