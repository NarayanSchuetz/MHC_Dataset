import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from models.revin import RevIN
from torch.utils.data import DataLoader
from torch_dataset import FlattenedForecastingDataset
from utils import ForecastSplit, CommonSplits


class AutoencoderLSTM(nn.Module):
    """
    Autoencoder LSTM model for processing MHC dataset time series data.
    
    Takes input of shape (batch_size, num_days, 24, 1440) from the MHC dataset,
    where 24 is the number of features per minute and 1440 is the total minutes in a day.
    The preprocessing divides each day into 48 segments of 30 minutes each,
    resulting in a tensor of shape (batch_size, num_days*48, 24*30).
    Each segment is then encoded to a lower-dimensional vector, processed with LSTM,
    and decoded back to predict future segments.
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        encoding_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        target_labels: Optional[List[str]] = None,
        prediction_horizon: int = 1,  # Number of future 30-min segments to predict
        use_masked_loss: bool = False,  # Whether to use the mask for loss calculation
        teacher_forcing_ratio: float = 0.5,  # Ratio of steps to use teacher forcing
        num_features: int = 24,  # Number of features per minute (default is all 24)
        force_teacher_forcing: bool = False,  # Whether to force using teacher forcing regardless of training mode
        l2_weight: float = 0.0,  # Weight for L2 regularization (0.0 means disabled)
    ):
        """
        Initialize the Autoencoder LSTM model.
        
        Args:
            hidden_size: Hidden dimension size for the LSTM
            encoding_dim: Dimension of the encoded 30-minute segment vector
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
            target_labels: List of target labels to predict (without '_value' suffix)
            prediction_horizon: Number of future 30-min segments to predict
            use_masked_loss: Whether to use the binary mask to exclude missing values from loss calculation
            teacher_forcing_ratio: Probability of using teacher forcing during training (0.0 to 1.0)
            num_features: Number of original features per minute (e.g., 24)
            force_teacher_forcing: Whether to force using teacher forcing regardless of training mode
            l2_weight: Weight for L2 regularization (0.0 means disabled)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.target_labels = target_labels if target_labels else ["default"]
        self.prediction_horizon = prediction_horizon
        self.use_masked_loss = use_masked_loss
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.num_original_features = num_features # Store the original number of features (e.g., 24)
        self.force_teacher_forcing = force_teacher_forcing
        self.l2_weight = l2_weight  # L2 regularization weight
        
        # Constants for data structure
        self.minutes_per_segment = 30
        self.segments_per_day = (24 * 60) // self.minutes_per_segment  # 48 segments per day
        self.time_points = 24 * 60  # 1440 minutes per day, used in evaluate_forecast
        # features_per_segment is based on the original number of features
        self.features_per_segment = self.num_original_features * self.minutes_per_segment # e.g., 24*30 = 720
        
        # Encoder: Compress each 30-minute segment (with all features) to encoding_dim
        # Input shape for each 30-min segment remains (features_per_segment)
        self.encoder = nn.Sequential(
            nn.Linear(self.features_per_segment, encoding_dim),
        )
        
        # LSTM processes the encoded segments
        self.lstm = nn.LSTM(
            input_size=self.encoding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Decoder: Convert LSTM output back to original 30-minute segment dimensions
        lstm_output_size = hidden_size * self.num_directions
        self.decoder = nn.Sequential(
            nn.Linear(lstm_output_size, self.features_per_segment),
        )
        
        # Output layers for label prediction (optional targets)
        self.output_layers = nn.ModuleDict({
            label: nn.Linear(lstm_output_size, 1) for label in self.target_labels
        })
        
    def preprocess_batch(self, batch_data: torch.Tensor, batch_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess batch data from either:
        1. Original shape (batch_size, num_days, num_original_features, 1440)
        2. Flattened shape (batch_size, num_original_features, num_days * 1440)
        to 30-minute segments suitable for encoding.

        Args:
            batch_data: Input tensor of either shape above
            batch_mask: Optional mask tensor of same shape

        Returns:
            If batch_mask is None:
                Processed tensor of shape (batch_size, num_segments, features_per_segment)
            Else:
                Tuple of (processed_data, processed_mask) both of shape (batch_size, num_segments, features_per_segment)
        """
        # Determine input format based on tensor shape
        if len(batch_data.shape) == 4:
            # Original shape: (batch_size, num_days, num_original_features, 1440)
            batch_size, num_days, features, minutes_per_day = batch_data.shape

            # Verify features match expectation
            if features != self.num_original_features:
                raise ValueError(f"Expected {self.num_original_features} features per minute, got {features}")
            if minutes_per_day != 24 * 60:
                raise ValueError(f"Expected 1440 minutes per day, got {minutes_per_day}")

            segments_per_day = minutes_per_day // self.minutes_per_segment # Should be 48
            num_segments = num_days * segments_per_day

            # Reshape to create 30-minute segments
            # From (B, D, F, M) to (B, D * S, F * MinPerSeg) = (B, num_segments, features_per_segment)
            x = batch_data.view(batch_size, num_days, features, segments_per_day, self.minutes_per_segment)
            x = x.permute(0, 1, 3, 2, 4) # -> (B, D, S, F, MinPerSeg)
            x = x.reshape(batch_size, num_segments, features * self.minutes_per_segment) # -> (B, num_segments, features_per_segment)

            # Replace NaN values with zeros in data
            x = torch.nan_to_num(x, nan=0.0)

            if batch_mask is not None:
                # Reshape mask similarly
                mask = batch_mask.view(batch_size, num_days, features, segments_per_day, self.minutes_per_segment)
                mask = mask.permute(0, 1, 3, 2, 4) # -> (B, D, S, F, MinPerSeg)
                mask = mask.reshape(batch_size, num_segments, features * self.minutes_per_segment) # -> (B, num_segments, features_per_segment)
                return x, mask

            return x
            
        elif len(batch_data.shape) == 3:
            # Flattened shape: (batch_size, num_original_features, num_days * 1440)
            batch_size, features, total_time_points = batch_data.shape
            
            # Verify features match expectation
            if features != self.num_original_features:
                raise ValueError(f"Expected {self.num_original_features} features per minute, got {features}")
                
            # Calculate how many days we have based on total time points
            if total_time_points % (24 * 60) != 0:
                raise ValueError(f"Total time points ({total_time_points}) is not divisible by minutes per day (1440)")
                
            num_days = total_time_points // (24 * 60)
            minutes_per_day = 24 * 60
            segments_per_day = minutes_per_day // self.minutes_per_segment # Should be 48
            num_segments = num_days * segments_per_day
            
            # Reshape flattened data to create 30-minute segments
            # From (B, F, D*M) to (B, F, D, M) to (B, F, D, S, MinPerSeg)
            x = batch_data.view(batch_size, features, num_days, minutes_per_day)
            x = x.view(batch_size, features, num_days, segments_per_day, self.minutes_per_segment)
            # Permute to (B, D, S, F, MinPerSeg)
            x = x.permute(0, 2, 3, 1, 4)
            # Reshape to (B, D*S, F*MinPerSeg) = (B, num_segments, features_per_segment)
            x = x.reshape(batch_size, num_segments, features * self.minutes_per_segment)
            
            # Replace NaN values with zeros
            x = torch.nan_to_num(x, nan=0.0)
            
            if batch_mask is not None:
                # Reshape mask similarly
                mask = batch_mask.view(batch_size, features, num_days, minutes_per_day)
                mask = mask.view(batch_size, features, num_days, segments_per_day, self.minutes_per_segment)
                mask = mask.permute(0, 2, 3, 1, 4)
                mask = mask.reshape(batch_size, num_segments, features * self.minutes_per_segment)
                return x, mask
                
            return x
        else:
            raise ValueError(f"Unexpected input shape: {batch_data.shape}. Expected either (B, D, F, M) or (B, F, D*M)")
            
    def forward(self, batch: Dict[str, Union[torch.Tensor, Dict]], return_predictions: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the base AutoencoderLSTM model (without RevIN).
        """
        x_orig = batch['data']  # Shape: (B, D, F, M)
        mask_orig = batch.get('mask') if self.use_masked_loss else None

        # Preprocess original data (and mask) into segments
        if mask_orig is not None:
            x_segmented, mask_segmented = self.preprocess_batch(x_orig, mask_orig)
        else:
            x_segmented = self.preprocess_batch(x_orig)
            mask_segmented = None

        batch_size, num_segments, _ = x_segmented.shape

        # Split into input and target segments (using original, non-normalized data)
        input_segments = x_segmented[:, :-self.prediction_horizon, :]
        target_segments = x_segmented[:, self.prediction_horizon:, :]

        target_mask = None
        if mask_segmented is not None:
            target_mask = mask_segmented[:, self.prediction_horizon:, :]

        # --- Standard LSTM Autoencoder Logic ---
        encoded_input = self.encoder(input_segments[:, 0:1, :]) # Encode first segment
        h, c = None, None
        outputs = []
        seq_len = input_segments.size(1) # Length of sequence to process

        for t in range(seq_len):
            if h is None and c is None:
                lstm_out, (h, c) = self.lstm(encoded_input)
            else:
                lstm_out, (h, c) = self.lstm(encoded_input, (h, c))

            decoded = self.decoder(lstm_out) # Decode LSTM output
            outputs.append(decoded)

            if t < seq_len - 1:
                # Check for teacher forcing - either in training mode or if forced
                use_teacher_forcing = (self.training or self.force_teacher_forcing) and (torch.rand(1).item() < self.teacher_forcing_ratio)
                if use_teacher_forcing:
                    next_input = input_segments[:, t+1:t+2, :] # Ground truth segment
                else:
                    next_input = decoded # Model's own prediction
                encoded_input = self.encoder(next_input) # Encode for next step
        # --- End Standard LSTM Autoencoder Logic ---

        decoded_segments = torch.cat(outputs, dim=1) # Concatenate predictions

        # Prepare result dictionary
        result = {
            'sequence_output': decoded_segments,
            'target_segments': target_segments,
        }
        if self.use_masked_loss and target_mask is not None:
            result['target_mask'] = target_mask

        # Add label predictions if requested
        if return_predictions:
            if self.bidirectional:
                final_hidden = h.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)[-1]
                final_hidden = final_hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            else:
                final_hidden = h[-1] # Final hidden state of last layer

            label_predictions = {}
            for label in self.target_labels:
                label_predictions[label] = self.output_layers[label](final_hidden).squeeze(-1)
            result['label_predictions'] = label_predictions

        return result
    
    def compute_loss(self, model_output: Dict[str, torch.Tensor], 
                      batch: Dict[str, Union[torch.Tensor, Dict]]) -> torch.Tensor:
        """
        Compute combined loss for sequence prediction and label prediction.
        
        If use_masked_loss=True, the loss for sequence prediction is only computed 
        on observed values (where target_mask=1), ignoring missing values (where target_mask=0).
        
        Args:
            model_output: Dictionary from forward() containing:
                - 'sequence_output': predicted segments
                - 'target_segments': actual segments to predict
                - 'target_mask': mask for target segments (if use_masked_loss=True)
                - 'label_predictions': dictionary of label predictions (optional)
            batch: Dictionary containing the input data and 'labels' dictionary
            
        Returns:
            Combined loss tensor
        """
        # Initialize total loss
        total_loss = 0
        
        # Get prediction outputs and targets
        sequence_output = model_output['sequence_output']
        target_segments = model_output['target_segments']
        
        # Check if we're using masked loss and have a mask
        if self.use_masked_loss and 'target_mask' in model_output:
            # Get the mask for target segments
            target_mask = model_output['target_mask']
            
            # Calculate squared error between predictions and targets
            squared_error = (sequence_output - target_segments) ** 2
            
            # Apply mask to only include observed values in the loss
            masked_squared_error = squared_error * target_mask
            
            # Calculate mean over observed values only (sum of errors / sum of mask)
            # Add small epsilon to avoid division by zero
            mask_sum = target_mask.sum() + 1e-10
            sequence_loss = masked_squared_error.sum() / mask_sum
            
        else:
            # Standard MSE loss if not using masking
            sequence_loss = F.mse_loss(sequence_output, target_segments)
            
        total_loss += sequence_loss
        
        # Add label prediction loss if available
        if 'label_predictions' in model_output and 'labels' in batch:
            label_predictions = model_output['label_predictions']
            labels_dict = batch['labels']
            
            for label in self.target_labels:
                # Check if label exists and handle tensor vs scalar cases
                if label in labels_dict:
                    label_value = labels_dict[label]
                    
                    # Convert to tensor if not already
                    if not isinstance(label_value, torch.Tensor):
                        label_value = torch.tensor([label_value], device=label_predictions[label].device)
                    elif label_value.dim() == 0:  # scalar tensor
                        label_value = label_value.unsqueeze(0)
                    
                    # Create a mask for non-NaN values
                    valid_mask = ~torch.isnan(label_value)
                    
                    # Only compute loss for non-NaN labels
                    if valid_mask.any():
                        # Get predictions for current samples and filter out NaNs
                        pred = label_predictions[label]
                        
                        # Replace NaNs with zeros for computation
                        label_value_clean = torch.where(valid_mask, label_value, torch.zeros_like(label_value))
                        
                        # Compute loss only on valid entries (where the mask is True)
                        label_loss = F.mse_loss(
                            pred[valid_mask], 
                            label_value_clean[valid_mask]
                        )
                        total_loss += label_loss
                    
        # Add L2 regularization if weight > 0
        if self.l2_weight > 0:
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)
            total_loss += self.l2_weight * l2_reg
                    
        return total_loss
    
    def predict_future(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future time steps given an input sequence.
        Assumes input_sequence is already segmented (batch_size, num_segments, features_per_segment).
        If input_sequence is flattened format (batch_size, num_features, time_points), it's
        preprocessed to convert to the segmented format.
        
        Args:
            input_sequence: Input sequence in segmented format (B, S, F*M) or
                            flattened format (B, F, T)
            steps: Number of future steps to predict
            
        Returns:
            torch.Tensor: Predicted future segments (B, steps, F*M)
        """
        self.eval()
        
        # Check if input is already in segmented format or needs preprocessing
        if len(input_sequence.shape) == 3:
            batch_size, dim1, dim2 = input_sequence.shape
            
            # If dim1 < dim2 and dim1 == num_features, it's likely in flattened format (B, F, T)
            if dim1 < dim2 and dim1 == self.num_original_features:
                # Input is in flattened format (B, F, T), preprocess it
                input_sequence = self.preprocess_batch(input_sequence)
        
        batch_size = input_sequence.shape[0]
        
        with torch.no_grad():
            future_predictions = []
            current_input = input_sequence # Shape: (B, num_segments, features_per_segment)
            h, c = None, None
            
            # Process the initial input sequence to get the starting hidden state
            encoded = self.encoder(current_input) # Shape: (B, num_segments, encoding_dim)
            lstm_out, (h, c) = self.lstm(encoded) # Get final state from full sequence

            # Use the last segment's output for the first prediction step input
            # The prediction loop needs an input of shape (B, 1, features_per_segment)
            last_segment_input = current_input[:, -1:, :] # Shape: (B, 1, features_per_segment)

            for _ in range(steps):
                # Encode the last known segment (either from input or last prediction)
                encoded_last_segment = self.encoder(last_segment_input) # Shape: (B, 1, encoding_dim)

                # Pass through LSTM using the previous hidden state
                # Input to LSTM should be (B, 1, encoding_dim)
                lstm_out_step, (h, c) = self.lstm(encoded_last_segment, (h, c))

                # Decode the output of this single step
                decoded = self.decoder(lstm_out_step) # Shape: (B, 1, features_per_segment)
                future_predictions.append(decoded)

                # Update last_segment_input for the next iteration
                last_segment_input = decoded

            # Stack all predictions
            return torch.cat(future_predictions, dim=1) # Shape: (B, steps, features_per_segment)

    def evaluate_forecast(
        self,
        dataframe: pd.DataFrame,
        root_dir: str,
        split: Union[str, 'ForecastSplit'] = CommonSplits.FIVE_TO_TWO,
        batch_size: int = 16,
        include_mask: bool = True,
        feature_indices: Optional[List[int]] = None,
        feature_stats: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Evaluate forecasting performance of the model using flattened data.
        
        Args:
            dataframe: DataFrame with MHC dataset metadata
            root_dir: Root directory for data files
            split: Either a ForecastSplit instance or string like "5-2d"
            batch_size: Batch size for evaluation
            include_mask: Whether to include masks (required for proper evaluation)
            feature_indices: Optional list of feature indices to select
            feature_stats: Optional dictionary for feature standardization
            device: Device to run evaluation on. If None, uses model's current device.
        
        Returns:
            Dictionary with MAE metrics per channel and overall
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Convert string to ForecastSplit if needed
        if isinstance(split, str):
            split = ForecastSplit.from_string(split)
        
        # Create the flattened forecast dataset
        forecast_dataset = FlattenedForecastingDataset(
            dataframe=dataframe,
            root_dir=root_dir,
            sequence_len=split.sequence_len,
            prediction_horizon=split.prediction_horizon,
            overlap=split.overlap,
            include_mask=include_mask,
            feature_indices=feature_indices,
            feature_stats=feature_stats
        )
        
        # Create data loader
        dataloader = DataLoader(
            forecast_dataset,
            batch_size=batch_size,
            shuffle=False  # Don't shuffle for evaluation
        )
        
        # Ensure model is in evaluation mode
        self.eval()
        
        # Initialize metrics
        all_errors = []
        all_masks = []
        
        # Track per-channel errors
        num_features = len(feature_indices) if feature_indices is not None else self.num_original_features
        channel_errors = [[] for _ in range(num_features)]
        channel_masks = [[] for _ in range(num_features)]
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                data_x = batch['data_x'].to(device)  # Shape: (B, F, sequence_len)
                data_y = batch['data_y'].to(device)  # Shape: (B, F, prediction_horizon)
                
                # Get the mask if available
                mask_y = batch.get('mask_y')
                if mask_y is not None:
                    mask_y = mask_y.to(device)
                
                # Get dimensions from flattened data
                B, F, sequence_len = data_x.shape
                _, _, prediction_horizon = data_y.shape
                
                # Generate predictions using the model's prediction method
                # Input to predict_future should be (B, F, sequence_len) 
                # No need to permute as our preprocess_batch will handle flattened input
                
                # Generate predictions - the predict_future method will internally
                # preprocess the flattened data
                predictions_segmented = self.predict_future(data_x, steps=prediction_horizon // self.minutes_per_segment)
                
                # Reshape predictions to match data_y's flattened shape (B, F, prediction_horizon)
                # First get the number of segments we predicted
                B_pred, num_steps, features_per_segment = predictions_segmented.shape
                
                # Reshape to get features separated
                # (B, steps, F*M) -> (B, steps, F, M)
                reshaped_preds = predictions_segmented.reshape(B_pred, num_steps, self.num_original_features, self.minutes_per_segment)
                
                # Permute and reshape to flattened format (B, F, T)
                # (B, steps, F, M) -> (B, F, steps*M)
                final_preds = reshaped_preds.permute(0, 2, 1, 3).reshape(B_pred, self.num_original_features, num_steps * self.minutes_per_segment)
                
                # Verify shapes match
                if final_preds.shape[2] != data_y.shape[2]:
                    raise ValueError(f"Prediction shape {final_preds.shape} doesn't match target shape {data_y.shape}")
                
                # Calculate absolute error (shapes are now compatible)
                abs_error = torch.abs(final_preds - data_y)
                
                # Apply mask if available
                if mask_y is not None:
                    # For calculating overall MAE
                    masked_abs_error = abs_error * mask_y
                    valid_mask = (mask_y > 0)
                    
                    # For per-channel MAE
                    for f in range(num_features):
                        channel_error = abs_error[:, f, :]
                        channel_mask = mask_y[:, f, :]
                        
                        # Check if we have any observed values for this channel
                        if torch.any(channel_mask > 0):
                            masked_channel_error = channel_error * channel_mask
                            channel_errors[f].append(masked_channel_error.cpu().numpy())
                            channel_masks[f].append(channel_mask.cpu().numpy())
                    
                    # Add to overall metrics
                    all_errors.append(masked_abs_error.cpu().numpy())
                    all_masks.append(valid_mask.cpu().numpy())
                else:
                    # Without a mask, use all data
                    for f in range(num_features):
                        channel_error = abs_error[:, f, :]
                        channel_errors[f].append(channel_error.cpu().numpy())
                    
                    all_errors.append(abs_error.cpu().numpy())
        
        # Calculate metrics
        results = {}
        
        # Overall MAE
        if all_masks:
            # If masks were used
            all_errors_array = np.concatenate(all_errors, axis=0)
            all_masks_array = np.concatenate(all_masks, axis=0)
            
            # Calculate MAE only over observed values
            mask_sum = np.sum(all_masks_array)
            if mask_sum > 0:
                overall_mae = np.sum(all_errors_array) / mask_sum
            else:
                overall_mae = np.nan
        else:
            # If no masks were used
            all_errors_array = np.concatenate(all_errors, axis=0)
            overall_mae = np.mean(all_errors_array)
        
        results['overall_mae'] = float(overall_mae)
        
        # Per-channel MAE
        channel_maes = []
        for f in range(num_features):
            if channel_masks and len(channel_masks[f]) > 0:
                # With masks
                try:
                    channel_errors_array = np.concatenate(channel_errors[f], axis=0)
                    channel_masks_array = np.concatenate(channel_masks[f], axis=0)
                    
                    mask_sum = np.sum(channel_masks_array)
                    if mask_sum > 0:
                        channel_mae = np.sum(channel_errors_array) / mask_sum
                    else:
                        channel_mae = np.nan
                except ValueError:
                    # Handle case where no data was available for this channel
                    channel_mae = np.nan
            elif len(channel_errors[f]) > 0:
                # Without masks
                channel_errors_array = np.concatenate(channel_errors[f], axis=0)
                channel_mae = np.mean(channel_errors_array)
            else:
                channel_mae = np.nan
            
            channel_maes.append(float(channel_mae))
        
        results['channel_mae'] = channel_maes
        
        return results


# New class incorporating RevIN applied to original time series features
class RevInAutoencoderLSTM(AutoencoderLSTM):
    """
    Autoencoder LSTM model incorporating Reversible Instance Normalization (RevIN)
    applied to the original time series features *before* segmentation.
    Inherits from AutoencoderLSTM.
    """
    def __init__(
        self,
        hidden_size: int = 128,
        encoding_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        target_labels: Optional[List[str]] = None,
        prediction_horizon: int = 1,
        use_masked_loss: bool = False,
        teacher_forcing_ratio: float = 0.5,
        num_features: int = 24, # This is num_original_features
        force_teacher_forcing: bool = False, # Whether to force using teacher forcing regardless of training mode
        rev_in_affine: bool = False,
        rev_in_subtract_last: bool = False,
        rev_in_eps: float = 1e-5,
        l2_weight: float = 0.0, # Weight for L2 regularization
    ):
        """
        Initialize the RevIN Time Series Autoencoder LSTM model.

        Args:
            num_features: Number of original features per minute (e.g., 24)
            force_teacher_forcing: Whether to force using teacher forcing regardless of training mode
            rev_in_affine: If True, RevIN layer has learnable affine parameters
            rev_in_subtract_last: If True, RevIN subtracts last element instead of mean
            rev_in_eps: Small epsilon for numerical stability in RevIN calculations
            l2_weight: Weight for L2 regularization (0.0 means disabled)
            Other args are passed to AutoencoderLSTM.__init__
        """
        super().__init__(
            hidden_size=hidden_size,
            encoding_dim=encoding_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            target_labels=target_labels,
            prediction_horizon=prediction_horizon,
            use_masked_loss=use_masked_loss,
            teacher_forcing_ratio=teacher_forcing_ratio,
            num_features=num_features, # Passes num_original_features to parent
            force_teacher_forcing=force_teacher_forcing,
            l2_weight=l2_weight,  # Pass L2 weight to parent class
        )

        # Instantiate RevIN layer - operates on the original features dimension
        # It expects input shape (batch, seq_len, features)
        # We will reshape data to (batch, time_steps, num_original_features)
        self.rev_in_eps = rev_in_eps
        
        self.rev_in = RevIN(
            num_features=self.num_original_features, # Use original feature count
            affine=rev_in_affine,
            subtract_last=rev_in_subtract_last,
            eps=self.rev_in_eps
        )

    def forward(self, batch: Dict[str, Union[torch.Tensor, Dict]], return_predictions: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with conditional RevIN based on feature variance.
        Handles both original shape (B, D, F, M) and flattened shape (B, F, D*M)
        """
        x_orig = batch['data']
        mask_orig = batch.get('mask') if self.use_masked_loss else None
        
        # Determine input format based on tensor shape
        if len(x_orig.shape) == 4:
            # Original shape: (batch_size, num_days, num_original_features, 1440)
            batch_size, num_days, features, minutes_per_day = x_orig.shape
            time_steps = num_days * minutes_per_day

            # --- Reshape and Clean Input ---
            # Reshape: (B, D, F, M) -> (B, T, F)
            x_reshaped = x_orig.permute(0, 2, 1, 3).reshape(batch_size, self.num_original_features, time_steps)
            x_reshaped = x_reshaped.permute(0, 2, 1)
        
        elif len(x_orig.shape) == 3:
            # Flattened shape: (batch_size, num_original_features, num_days * 1440)
            batch_size, features, time_steps = x_orig.shape
            
            # Already in the right shape for RevIN, just permute
            # From (B, F, T) to (B, T, F)
            x_reshaped = x_orig.permute(0, 2, 1)
            
            # Calculate num_days for later use
            minutes_per_day = 24 * 60
            if time_steps % minutes_per_day != 0:
                raise ValueError(f"Total time points ({time_steps}) is not divisible by minutes per day (1440)")
            num_days = time_steps // minutes_per_day
        
        else:
            raise ValueError(f"Unexpected input shape: {x_orig.shape}. Expected either (B, D, F, M) or (B, F, D*M)")

        # Handle NaNs before variance check
        x_clean = torch.nan_to_num(x_reshaped, nan=0.0)

        # --- Detect Zero-Variance Features ---
        # Calculate std dev along time dim (T) for each feature (F) in each batch (B)
        stdev = torch.std(x_clean, dim=1, keepdim=True) # Shape: (B, 1, F)
        # Mask for features with variance < eps (True means skip RevIN)
        skip_revin_mask = (stdev < self.rev_in_eps) # Shape: (B, 1, F)

        # --- Conditional RevIN Normalization ---
        # Apply RevIN normalization (will calculate stats internally)
        # The RevIN layer itself should handle the eps for division safety
        x_normalized_all = self.rev_in(x_clean, mode='norm')

        # Keep original (cleaned) values for zero-variance features
        # Use the mask to select between original cleaned data and normalized data
        x_normalized_final = torch.where(skip_revin_mask, x_clean, x_normalized_all)

        # Reshape back based on original input format
        if len(x_orig.shape) == 4:
            # Original shape: reshape back to (B, D, F, M)
            x_normalized = x_normalized_final.permute(0, 2, 1).reshape(batch_size, self.num_original_features, num_days, minutes_per_day)
            x_normalized = x_normalized.permute(0, 2, 1, 3)
        else:
            # Flattened shape: reshape back to (B, F, T)
            x_normalized = x_normalized_final.permute(0, 2, 1)
        # --- End Conditional Normalization ---

        # Preprocess normalized data (and original mask) into segments
        if mask_orig is not None:
            x_norm_segmented, mask_segmented = self.preprocess_batch(x_normalized, mask_orig)
        else:
            x_norm_segmented = self.preprocess_batch(x_normalized)
            mask_segmented = None

        # Preprocess original data for targets (handle NaNs)
        x_orig_segmented = self.preprocess_batch(x_orig)

        # Split segments
        input_segments_norm = x_norm_segmented[:, :-self.prediction_horizon, :]
        target_segments_orig = x_orig_segmented[:, self.prediction_horizon:, :]
        target_mask = None
        if mask_segmented is not None:
            target_mask = mask_segmented[:, self.prediction_horizon:, :]

        # --- LSTM Autoencoder Logic (operates on conditionally normalized data) ---
        encoded_input = self.encoder(input_segments_norm[:, 0:1, :])
        h, c = None, None
        outputs_conditionally_normalized = []
        seq_len = input_segments_norm.size(1)

        for t in range(seq_len):
            if h is None and c is None:
                lstm_out, (h, c) = self.lstm(encoded_input)
            else:
                lstm_out, (h, c) = self.lstm(encoded_input, (h, c))

            decoded_cond_norm = self.decoder(lstm_out) # Output is conditionally normalized
            outputs_conditionally_normalized.append(decoded_cond_norm)

            if t < seq_len - 1:
                # Check for teacher forcing - either in training mode or if forced
                use_teacher_forcing = (self.training or self.force_teacher_forcing) and (torch.rand(1).item() < self.teacher_forcing_ratio)
                if use_teacher_forcing:
                    next_input_cond_norm = input_segments_norm[:, t+1:t+2, :]
                else:
                    next_input_cond_norm = decoded_cond_norm
                encoded_input = self.encoder(next_input_cond_norm)
        # --- End LSTM Autoencoder Logic ---

        # Concatenate predictions (still conditionally normalized)
        decoded_segments_cond_norm = torch.cat(outputs_conditionally_normalized, dim=1) # (B, seq_len, F_seg)

        # --- Conditional RevIN De-normalization ---
        # Reshape predictions back into time series format: (B, seq_len, F*MinPerSeg) -> (B, pred_T, F)
        num_predicted_segments = decoded_segments_cond_norm.shape[1]
        predicted_time_steps = num_predicted_segments * self.minutes_per_segment
        decoded_cond_norm_reshaped = decoded_segments_cond_norm.view(batch_size, num_predicted_segments, self.num_original_features, self.minutes_per_segment)
        decoded_cond_norm_reshaped = decoded_cond_norm_reshaped.permute(0, 1, 3, 2)
        decoded_cond_norm_reshaped = decoded_cond_norm_reshaped.reshape(batch_size, predicted_time_steps, self.num_original_features) # (B, pred_T, F)

        # Apply RevIN de-normalization (using stored stats)
        # This will attempt to de-normalize all features based on stored stats
        decoded_denorm_all = self.rev_in(decoded_cond_norm_reshaped, mode='denorm')

        # Keep the non-de-normalized (i.e., the direct output from LSTM) values
        # for features that were originally skipped.
        # Expand skip_revin_mask to match prediction time steps: (B, 1, F) -> (B, pred_T, F)
        skip_revin_mask_expanded = skip_revin_mask.expand(-1, predicted_time_steps, -1)
        decoded_denorm_final = torch.where(
            skip_revin_mask_expanded,
            decoded_cond_norm_reshaped, # Use the non-denormalized value if RevIN was skipped
            decoded_denorm_all          # Use the de-normalized value if RevIN was applied
        )

        # Reshape de-normalized time series back into segments
        decoded_denorm_segments = decoded_denorm_final.view(batch_size, num_predicted_segments, self.minutes_per_segment, self.num_original_features)
        decoded_denorm_segments = decoded_denorm_segments.permute(0, 1, 3, 2)
        decoded_denorm_segments = decoded_denorm_segments.reshape(batch_size, num_predicted_segments, self.features_per_segment)
        # --- End Conditional De-normalization ---

        # Prepare result dictionary
        result = {
            'sequence_output': decoded_denorm_segments, # Final output in original scale
            'target_segments': target_segments_orig,    # Target in original scale
        }
        if self.use_masked_loss and target_mask is not None:
            result['target_mask'] = target_mask

        # Add label predictions
        if return_predictions:
            if self.bidirectional:
                final_hidden = h.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)[-1]
                final_hidden = final_hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            else:
                final_hidden = h[-1]
            label_predictions = {}
            for label in self.target_labels:
                label_predictions[label] = self.output_layers[label](final_hidden).squeeze(-1)
            result['label_predictions'] = label_predictions

        return result

    # compute_loss remains the same

    # predict_future would need similar conditional logic:
    # 1. Reshape input segments -> time series
    # 2. Clean NaNs
    # 3. Calculate std dev and skip_revin_mask
    # 4. Apply RevIN norm conditionally -> normalized_input_final
    # 5. Reshape normalized_input_final -> segments for LSTM
    # 6. Run prediction loop in conditionally normalized space
    # 7. Reshape predictions -> time series (conditionally normalized)
    # 8. Apply RevIN de-norm conditionally using the *original* skip_revin_mask -> final denormalized predictions
    # 9. Reshape back to segments
    def predict_future(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future time steps auto-regressively using the RevIN LSTM model.

        This method handles input sequences in two formats:
        1. Flattened time series: (batch_size, num_original_features, time_points) - (B, F, T)
        2. Pre-segmented format: (batch_size, num_segments, features_per_segment) - (B, S, F*M)

        The prediction process involves these key steps:
        1.  Input Handling: Detects the input format. If flattened, reshapes to (B, T, F).
            If segmented, reshapes to time series format (B, T, F) where T = S * M.
        2.  Conditional Normalization:
            - Calculates the standard deviation for each feature across time for each batch instance.
            - Creates a mask (`skip_revin_mask`) to identify features with near-zero variance.
            - Applies RevIN normalization to the time series data (B, T, F).
            - Uses the `skip_revin_mask` to keep the original (cleaned) values for zero-variance features,
              resulting in a 'conditionally normalized' time series.
        3.  Segmentation: Reshapes the conditionally normalized time series back into the
            segmented format (B, S, F*M) expected by the encoder/LSTM.
        4.  LSTM State Initialization: Encodes the entire conditionally normalized input sequence
            and processes it through the LSTM to obtain the final hidden state (`h`, `c`). This
            state captures the history encoded in the input sequence.
        5.  Auto-regressive Prediction Loop:
            - Starts with the last segment of the conditionally normalized input sequence.
            - Iterates `steps` times:
                - Encodes the current segment.
                - Feeds the encoded segment and the current hidden state (`h`, `c`) to the LSTM.
                - Decodes the LSTM output to get the next predicted segment (still conditionally normalized).
                - Stores the prediction and uses it as the input for the next iteration.
        6.  Conditional De-normalization:
            - Concatenates the predicted segments (which are in the normalized space).
            - Reshapes the predicted segments back into a time series format (B, pred_T, F).
            - Applies RevIN de-normalization using the statistics stored during the initial normalization step.
            - Crucially, uses the *original* `skip_revin_mask` (computed from the input sequence)
              to ensure that features that were initially skipped during normalization are *not*
              de-normalized, preserving their values directly from the prediction loop.
        7.  Output Formatting: Reshapes the final, de-normalized time series predictions back
            into the segmented format (B, steps, F*M).

        Args:
            input_sequence: Input sequence tensor in either flattened (B, F, T) or
                            segmented (B, S, F*M) format.
            steps: Number of future 30-minute segments to predict.

        Returns:
            torch.Tensor: Predicted future segments in the original data scale,
                          with shape (batch_size, steps, features_per_segment).
        """
        self.eval()

        # Check if input is already in segmented format or needs preprocessing
        if len(input_sequence.shape) == 3:
            batch_size, dim1, dim2 = input_sequence.shape
            
            # If dim1 < dim2 and dim1 == num_features, it's likely flattened format (B, F, T)
            # Otherwise, assume it's already segmented (B, S, F*M)
            is_flattened = dim1 < dim2 and dim1 == self.num_original_features
            
            if is_flattened:
                # Input is in flattened format (B, F, T)
                batch_size, features, total_time_points = input_sequence.shape
                input_time_steps = total_time_points
                
                # Reshape to (B, T, F) for RevIN
                input_reshaped = input_sequence.permute(0, 2, 1)
                input_clean = torch.nan_to_num(input_reshaped, nan=0.0)
                
                # Detect zero-variance features
                stdev = torch.std(input_clean, dim=1, keepdim=True)
                skip_revin_mask = (stdev < self.rev_in_eps)  # Shape: (B, 1, F) - Crucial mask for later
                
                # Apply conditional normalization
                normalized_input_all = self.rev_in(input_clean, mode='norm')
                normalized_input_final = torch.where(skip_revin_mask, input_clean, normalized_input_all)
                
                # Reshape back to (B, F, T) for preprocessing
                normalized_input_flattened = normalized_input_final.permute(0, 2, 1)
                
                # Preprocess to segmented format
                normalized_input_segmented = self.preprocess_batch(normalized_input_flattened)
            else:
                # Already segmented, handle as in original method
                batch_size = input_sequence.shape[0]
                num_input_segments = input_sequence.shape[1]
                input_time_steps = num_input_segments * self.minutes_per_segment
                
                # Reshape for variance detection and normalization
                input_reshaped = input_sequence.view(batch_size, num_input_segments, self.num_original_features, self.minutes_per_segment)
                input_reshaped = input_reshaped.permute(0, 1, 3, 2).reshape(batch_size, input_time_steps, self.num_original_features)
                input_clean = torch.nan_to_num(input_reshaped, nan=0.0)
                
                # Detect zero-variance features
                stdev = torch.std(input_clean, dim=1, keepdim=True)
                skip_revin_mask = (stdev < self.rev_in_eps)  # Shape: (B, 1, F) - Crucial mask for later
                
                # Apply conditional normalization
                normalized_input_all = self.rev_in(input_clean, mode='norm')
                normalized_input_final = torch.where(skip_revin_mask, input_clean, normalized_input_all)
                
                # Reshape normalized time series back into segments
                normalized_input_segmented = normalized_input_final.view(batch_size, num_input_segments, self.minutes_per_segment, self.num_original_features)
                normalized_input_segmented = normalized_input_segmented.permute(0, 1, 3, 2).reshape(batch_size, num_input_segments, self.features_per_segment)
        else:
            raise ValueError(f"Unexpected input shape: {input_sequence.shape}. Expected either (B, S, F*M) or (B, F, T)")

        with torch.no_grad():
            # --- Prediction Loop (operates on conditionally normalized segments) ---
            future_predictions_cond_norm = []
            current_cond_norm_input = normalized_input_segmented # Start with the normalized input history
            h, c = None, None
            encoded_cond_norm_hist = self.encoder(current_cond_norm_input)
            _, (h, c) = self.lstm(encoded_cond_norm_hist) # Get final state
            last_cond_norm_segment = current_cond_norm_input[:, -1:, :] # Shape: (B, 1, features_per_segment)

            for _ in range(steps):
                encoded_last_cond_norm_segment = self.encoder(last_cond_norm_segment) # Shape: (B, 1, encoding_dim)
                lstm_out_cond_norm_step, (h, c) = self.lstm(encoded_last_cond_norm_segment, (h, c))
                decoded_cond_norm = self.decoder(lstm_out_cond_norm_step) # Conditionally normalized
                future_predictions_cond_norm.append(decoded_cond_norm)
                last_cond_norm_segment = decoded_cond_norm
            # --- End Prediction Loop ---

            # Concatenate predictions (still conditionally normalized)
            final_cond_norm_predictions_segmented = torch.cat(future_predictions_cond_norm, dim=1) # (B, steps, F_seg)

            # --- Conditional De-normalization ---
            # Reshape predictions back into time series format
            predicted_time_steps = steps * self.minutes_per_segment
            final_cond_norm_pred_reshaped = final_cond_norm_predictions_segmented.view(batch_size, steps, self.num_original_features, self.minutes_per_segment)
            final_cond_norm_pred_reshaped = final_cond_norm_pred_reshaped.permute(0, 1, 3, 2).reshape(batch_size, predicted_time_steps, self.num_original_features) # (B, pred_T, F)

            # Apply RevIN de-norm (using stats stored during normalization)
            final_denorm_pred_all = self.rev_in(final_cond_norm_pred_reshaped, mode='denorm')

            # Use the *original* skip_revin_mask calculated from the input sequence
            skip_revin_mask_expanded = skip_revin_mask.expand(-1, predicted_time_steps, -1)
            final_denorm_pred_final = torch.where(
                skip_revin_mask_expanded,
                final_cond_norm_pred_reshaped, # Use non-denormalized if RevIN was skipped
                final_denorm_pred_all          # Use de-normalized otherwise
            )

            # Reshape back into segments
            final_predictions_segmented = final_denorm_pred_final.view(batch_size, steps, self.minutes_per_segment, self.num_original_features)
            final_predictions_segmented = final_predictions_segmented.permute(0, 1, 3, 2).reshape(batch_size, steps, self.features_per_segment)
            # --- End Conditional De-normalization ---

            return final_predictions_segmented
        

def run_forecast_evaluation(
    model: Union[AutoencoderLSTM, RevInAutoencoderLSTM],
    dataframe: pd.DataFrame,
    root_dir: str,
    split: Union[str, 'ForecastSplit'] = CommonSplits.FIVE_TO_TWO,
    batch_size: int = 16,
    include_mask: bool = True,
    feature_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    feature_stats: Optional[Dict] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Run forecast evaluation and print formatted results.
    
    Args:
        model: The LSTM model to evaluate
        dataframe: DataFrame with MHC dataset metadata
        root_dir: Root directory for data files
        split: Either a ForecastSplit instance or string like "5-2d"
        batch_size: Batch size for evaluation
        include_mask: Whether to include masks
        feature_indices: Optional list of feature indices to select
        feature_names: Optional list of feature names for reporting
        feature_stats: Optional dictionary for feature standardization
        device: Device to run evaluation on. If None, uses model's current device.
    
    Returns:
        Dictionary with evaluation results
    """
    # Convert string to ForecastSplit if needed
    if isinstance(split, str):
        split = ForecastSplit.from_string(split)
    
    print(f"Running forecast evaluation with split: {split}")
    
    # Run evaluation
    results = model.evaluate_forecast(
        dataframe=dataframe,
        root_dir=root_dir,
        split=split,
        batch_size=batch_size,
        include_mask=include_mask,
        feature_indices=feature_indices,
        feature_stats=feature_stats,
        device=device
    )
    
    # Print overall results
    print(f"\nOverall MAE: {results['overall_mae']:.4f}")
    
    # Print per-channel results
    print("\nPer-Channel MAE:")
    channel_maes = results['channel_mae']
    
    # Use feature names if provided, otherwise use indices
    if feature_names and len(feature_names) == len(channel_maes):
        feature_labels = feature_names
    elif feature_indices and len(feature_indices) == len(channel_maes):
        feature_labels = [f"Feature {i}" for i in feature_indices]
    else:
        feature_labels = [f"Feature {i}" for i in range(len(channel_maes))]
    
    # Print table header
    print(f"{'Feature':<20} {'MAE':<10}")
    print("-" * 30)
    
    # Print each channel's MAE
    for label, mae in zip(feature_labels, channel_maes):
        mae_str = f"{mae:.4f}" if not np.isnan(mae) else "NaN"
        print(f"{label:<20} {mae_str:<10}")
    
    return results


class LSTMTrainer:
    """
    Trainer class for the AutoencoderLSTM model.
    
    Handles training loop, validation, and inference.
    Works with both AutoencoderLSTM and RevInTimeSeriesAutoencoderLSTM.
    """
    
    def __init__(
        self,
        model: Union[AutoencoderLSTM, RevInAutoencoderLSTM], # Accept either model type
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The AutoencoderLSTM or RevInTimeSeriesAutoencoderLSTM model to train
            optimizer: Optimizer for training
            device: Device to use for training
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        # Move model to device
        self.model.to(device)
        
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader providing batches from the MHC dataset
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move necessary batch data to device
            batch_data = batch['data'].to(self.device)
            batch['data'] = batch_data # Keep batch dict structure

            # Move labels to device if they exist (handled within compute_loss)
            if 'labels' in batch:
                 # Ensure labels are on the correct device inside compute_loss if needed
                 pass 

            # Move mask to device if it exists and model uses masked loss
            if self.model.use_masked_loss and 'mask' in batch:
                batch_mask = batch['mask'].to(self.device)
                batch['mask'] = batch_mask
            else:
                 # Ensure 'mask' key is absent if not used, to avoid errors later
                 if 'mask' in batch:
                     del batch['mask']

            
            # Forward pass
            self.optimizer.zero_grad()
            # Pass entire batch dict to model's forward
            model_output = self.model(batch) 
            
            # Compute loss (pass model output and original batch dict)
            loss = self.model.compute_loss(model_output, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        # Update learning rate if scheduler exists
        if self.scheduler is not None:
            self.scheduler.step()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: torch.utils.data.DataLoader, use_teacher_forcing: Optional[bool] = None) -> float:
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader providing batches from the MHC dataset
            use_teacher_forcing: Whether to use teacher forcing during validation.
                                If None, uses normal eval mode behavior (no teacher forcing).
                                If True, forces teacher forcing with the model's teacher_forcing_ratio.
                                If False, ensures teacher forcing is disabled.
            
        Returns:
            Average validation loss
        """
        self.model.eval()  # Always set to eval mode for validation
        total_loss = 0
        
        # Store original force_teacher_forcing flag to restore later
        original_force_tf = self.model.force_teacher_forcing
        
        # Set force_teacher_forcing based on user preference
        if use_teacher_forcing is not None:
            self.model.force_teacher_forcing = use_teacher_forcing
        
        with torch.no_grad():
            for batch in dataloader:
                # Move necessary batch data to device
                batch_data = batch['data'].to(self.device)
                batch['data'] = batch_data # Keep batch dict structure

                # Move labels to device if they exist (handled within compute_loss)
                if 'labels' in batch:
                    # Ensure labels are on the correct device inside compute_loss if needed
                    pass

                # Move mask to device if it exists and model uses masked loss
                if self.model.use_masked_loss and 'mask' in batch:
                    batch_mask = batch['mask'].to(self.device)
                    batch['mask'] = batch_mask
                else:
                    if 'mask' in batch:
                        del batch['mask']
                
                # Forward pass
                model_output = self.model(batch)
                
                # Compute loss
                loss = self.model.compute_loss(model_output, batch)
                
                total_loss += loss.item()
        
        # Restore original force_teacher_forcing flag
        self.model.force_teacher_forcing = original_force_tf
                
        return total_loss / len(dataloader)
    

def load_checkpoint(
    checkpoint_path: str, 
    model: Union[AutoencoderLSTM, RevInAutoencoderLSTM], # Allow loading into either type
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> Tuple[Union[AutoencoderLSTM, RevInAutoencoderLSTM], Dict]:
    """
    Load a model checkpoint saved during training.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model instance to load weights into
        device: Device to load the model on ('cpu' or 'cuda')
        verbose: Whether to print status messages
        
    Returns:
        Tuple of (loaded_model, checkpoint_data)
        where checkpoint_data contains additional information like epoch, losses, etc.
    """
    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from state dict
    state_dict = checkpoint['model_state_dict']
    
    # Load the state dictionary
    try:
        model.load_state_dict(state_dict)
        if verbose:
            print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        # Try a partial load if full load fails
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Partial weight loading successful")
        except Exception as e2:
            print(f"Partial loading also failed: {e2}")
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    if verbose:
        print(f"Model loaded and ready on {device}")
    
    # Return model and checkpoint data (which includes optimizer state, epoch, losses)
    return model, checkpoint
