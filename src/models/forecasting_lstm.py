import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from torch.utils.data import DataLoader
from torch_dataset import ForecastingEvaluationDataset, FlattenedForecastingDataset
from utils import ForecastSplit, CommonSplits


class ForecastingLSTM(nn.Module):
    """
    LSTM model specifically designed for forecasting tasks with the MHC dataset.
    
    This model works with ForecastingEvaluationDataset and FlattenedForecastingDataset,
    which provide separate input sequences (data_x) and target sequences (data_y).
    
    The model encodes the input sequence, processes it with LSTM layers, and then
    generates a forecast sequence matching the length of the target sequence.
    
    Unlike AutoencoderLSTM, this model does not use teacher forcing during training,
    as it's designed to generate pure forecasts based only on the input context.
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        encoding_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        target_labels: Optional[List[str]] = None,
        use_masked_loss: bool = False,
        num_features: int = 24,  # Number of features per minute (default is all 24)
        l2_weight: float = 0.0,  # Weight for L2 regularization (0.0 means disabled)
    ):
        """
        Initialize the Forecasting LSTM model.
        
        Args:
            hidden_size: Hidden dimension size for the LSTM
            encoding_dim: Dimension of the encoded 30-minute segment vector
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
            target_labels: List of target labels to predict (without '_value' suffix)
            use_masked_loss: Whether to use the binary mask to exclude missing values from loss calculation
            num_features: Number of original features per minute (e.g., 24)
            l2_weight: Weight for L2 regularization (0.0 means disabled)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.target_labels = target_labels if target_labels else []
        self.use_masked_loss = use_masked_loss
        self.num_original_features = num_features
        self.l2_weight = l2_weight  # L2 regularization weight
        
        # Constants for data structure
        self.minutes_per_segment = 30
        self.segments_per_day = (24 * 60) // self.minutes_per_segment  # 48 segments per day
        self.time_points = 24 * 60  # 1440 minutes per day
        # features_per_segment is based on the original number of features
        self.features_per_segment = self.num_original_features * self.minutes_per_segment
        
        # Encoder: Compress each 30-minute segment (with all features) to encoding_dim
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
        if self.target_labels:
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
        # Check shape of batch_data to determine preprocessing approach
        if len(batch_data.shape) == 4:
            # Original shape: (batch_size, num_days, num_original_features, 1440)
            batch_size, num_days, features, minutes_per_day = batch_data.shape
            
            if features != self.num_original_features:
                raise ValueError(f"Expected {self.num_original_features} features, but got {features}")
            
            # Reshape to segment-oriented format
            # Formula: (batch_size, num_days, features, 1440) -> (batch_size, num_days*48, features*30)
            segments_per_day = self.segments_per_day
            num_segments = num_days * segments_per_day
            
            # View as batch_size, num_days, features, segment_count, minutes_per_segment
            x = batch_data.view(batch_size, num_days, features, segments_per_day, self.minutes_per_segment)
            
            # Permute to batch_size, num_days, segment_count, features, minutes_per_segment
            x = x.permute(0, 1, 3, 2, 4)
            
            # Reshape to (batch_size, num_segments, features * minutes_per_segment)
            x = x.reshape(batch_size, num_segments, features * self.minutes_per_segment)
            
            if batch_mask is not None:
                # Reshape mask similarly
                mask = batch_mask.view(batch_size, num_days, features, segments_per_day, self.minutes_per_segment)
                mask = mask.permute(0, 1, 3, 2, 4)
                mask = mask.reshape(batch_size, num_segments, features * self.minutes_per_segment)
                return x, mask
            
            return x
            
        elif len(batch_data.shape) == 3:
            # Flattened shape: (batch_size, num_original_features, num_days * 1440)
            batch_size, features, flattened_minutes = batch_data.shape
            
            if features != self.num_original_features:
                raise ValueError(f"Expected {self.num_original_features} features, but got {features}")
            
            # Calculate num_days
            minutes_per_day = 24 * 60
            if flattened_minutes % minutes_per_day != 0:
                raise ValueError(f"Flattened minutes ({flattened_minutes}) is not divisible by minutes per day ({minutes_per_day})")
            
            num_days = flattened_minutes // minutes_per_day
            segments_per_day = self.segments_per_day
            num_segments = num_days * segments_per_day
            
            # Reshape to batch_size, features, num_days, minutes_per_day
            x = batch_data.view(batch_size, features, num_days, minutes_per_day)
            
            # Reshape to include segment dimension: (B, F, D, S, M_per_S)
            x = x.view(batch_size, features, num_days, segments_per_day, self.minutes_per_segment)
            
            # Permute to (B, D, S, F, M_per_S)
            x = x.permute(0, 2, 3, 1, 4)
            
            # Reshape to (B, D*S, F*M_per_S)
            x = x.reshape(batch_size, num_segments, features * self.minutes_per_segment)
            
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
        Forward pass through the ForecastingLSTM model.
        
        Takes separate input (data_x) and target (data_y) sequences and forecasts
        a sequence of the same length as data_y based on the context in data_x.
        
        Args:
            batch: Dictionary containing 'data_x', 'data_y', and optionally 'mask_x' and 'mask_y'
            return_predictions: Whether to include label predictions in the output
            
        Returns:
            dict: A dictionary containing:
                - 'sequence_output': Forecasted segments matching target_segments
                - 'target_segments': Segmented ground truth from data_y
                - 'target_mask': Mask for target segments (if include_mask=True)
                - 'label_predictions': Dictionary of label predictions (if return_predictions=True)
        """
        # Extract input and target data from batch
        x_input = batch['data_x']  # Shape: (B, D_x, F, M) or (B, F, D_x*M)
        x_target = batch['data_y']  # Shape: (B, D_y, F, M) or (B, F, D_y*M)
        
        # Get masks if provided and masked loss is enabled
        mask_input = batch.get('mask_x') if self.use_masked_loss else None
        mask_target = batch.get('mask_y') if self.use_masked_loss else None
        
        # Preprocess input and target data into segments
        if mask_input is not None:
            input_segments, input_mask_segments = self.preprocess_batch(x_input, mask_input)
        else:
            input_segments = self.preprocess_batch(x_input)
        
        if mask_target is not None:
            target_segments, target_mask_segments = self.preprocess_batch(x_target, mask_target)
        else:
            target_segments = self.preprocess_batch(x_target)
            target_mask_segments = None
        
        batch_size = input_segments.shape[0]
        num_input_segments = input_segments.shape[1]
        num_target_segments = target_segments.shape[1]
        
        # --- Context Encoding Phase ---
        # Encode all input segments
        encoded_input = self.encoder(input_segments)  # Shape: (B, num_input_segments, encoding_dim)
        
        # Process entire input sequence with LSTM to get the final state
        _, (h_final, c_final) = self.lstm(encoded_input)  # Process all input segments at once
        
        # --- Forecast Generation Phase ---
        # Use the final state to generate forecasts without teacher forcing
        predicted_segments_list = []
        
        # Create the initial decoder input - we'll use the encoding of the last input segment
        last_input_segment = input_segments[:, -1:, :]  # Shape: (B, 1, features_per_segment)
        current_lstm_input = self.encoder(last_input_segment)  # Shape: (B, 1, encoding_dim)
        
        # Set current hidden and cell states from the context encoding
        current_h, current_c = h_final, c_final
        
        # Generate forecast segments one by one
        for _ in range(num_target_segments):
            # Generate a single forecast step using the current state
            lstm_out_step, (current_h, current_c) = self.lstm(current_lstm_input, (current_h, current_c))
            # lstm_out_step shape: (B, 1, hidden_size * num_directions)
            
            # Decode the LSTM output to get the forecasted segment
            decoded_segment = self.decoder(lstm_out_step)  # Shape: (B, 1, features_per_segment)
            predicted_segments_list.append(decoded_segment)
            
            # Update the LSTM input for the next iteration using the predicted segment
            current_lstm_input = self.encoder(decoded_segment)
        
        # Concatenate all forecasted segments
        sequence_output = torch.cat(predicted_segments_list, dim=1)  # Shape: (B, num_target_segments, features_per_segment)
        
        # Prepare result dictionary
        result = {
            'sequence_output': sequence_output,
            'target_segments': target_segments,
        }
        
        if self.use_masked_loss and target_mask_segments is not None:
            result['target_mask'] = target_mask_segments
        
        # Add label predictions if requested
        if return_predictions and self.target_labels:
            if self.bidirectional:
                final_hidden = h_final.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)[-1]
                final_hidden = final_hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            else:
                final_hidden = h_final[-1]  # Final hidden state of last layer
            
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
        
        # Calculate loss for label predictions if available
        if 'label_predictions' in model_output and 'labels' in batch:
            label_loss = 0
            label_predictions = model_output['label_predictions']
            for label, prediction in label_predictions.items():
                if label in batch['labels']:
                    # Get target value, assuming batch['labels'] is a dict or a tensor
                    if isinstance(batch['labels'], dict):
                        target_value = batch['labels'].get(label)
                        if target_value is not None:
                            # Convert to tensor if not already
                            if not isinstance(target_value, torch.Tensor):
                                target_value = torch.tensor(target_value, device=prediction.device, dtype=prediction.dtype)
                    else:
                        # Assume it's a tensor indexed by label name
                        target_value = batch['labels'][label].to(prediction.device)
                        
                    # Calculate MSE loss for this label
                    label_loss += F.mse_loss(prediction, target_value)
            
            # Add label loss to total loss (weight equally with sequence loss for now)
            if self.target_labels:  # Only add if we have target labels
                total_loss += label_loss
        
        # Add L2 regularization if weight > 0
        if self.l2_weight > 0:
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)
            total_loss += self.l2_weight * l2_reg
        
        return total_loss
    
    def predict(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future time steps given an input sequence.
        
        Args:
            input_sequence: Input sequence in original format (B, D, F, M) or
                           flattened format (B, F, D*M)
            steps: Number of future 30-min segments to predict
            
        Returns:
            torch.Tensor: Predicted future segments (B, steps, F*M)
        """
        self.eval()
        
        # Preprocess input if needed
        input_segments = self.preprocess_batch(input_sequence)
        batch_size = input_segments.shape[0]
        
        with torch.no_grad():
            # Encode the entire input sequence
            encoded_input = self.encoder(input_segments)  # Shape: (B, num_segments, encoding_dim)
            
            # Process with LSTM to get final state
            _, (h_final, c_final) = self.lstm(encoded_input)
            
            # Create the initial decoder input
            last_segment = input_segments[:, -1:, :]  # Shape: (B, 1, features_per_segment)
            current_lstm_input = self.encoder(last_segment)  # Shape: (B, 1, encoding_dim)
            
            # Set current hidden and cell states
            current_h, current_c = h_final, c_final
            
            # Generate predictions
            predictions = []
            for _ in range(steps):
                # Pass through LSTM using the previous hidden state
                lstm_out_step, (current_h, current_c) = self.lstm(current_lstm_input, (current_h, current_c))
                
                # Decode the output of this single step
                decoded = self.decoder(lstm_out_step)  # Shape: (B, 1, features_per_segment)
                predictions.append(decoded)
                
                # Update input for next iteration
                current_lstm_input = self.encoder(decoded)
            
            # Stack all predictions
            return torch.cat(predictions, dim=1)  # Shape: (B, steps, features_per_segment)
            
    def evaluate_forecast(
        self,
        dataframe: pd.DataFrame,
        root_dir: str,
        sequence_len: int,
        prediction_horizon: int,
        overlap: int = 0,
        batch_size: int = 16,
        include_mask: bool = True,
        feature_indices: Optional[List[int]] = None,
        feature_stats: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Evaluates the model's forecasting performance on a dataset.
        
        Args:
            dataframe: DataFrame with MHC dataset metadata
            root_dir: Root directory for data files
            sequence_len: Length of input sequence in time points
            prediction_horizon: Length of prediction horizon in time points
            overlap: Overlap between input and target sequences
            batch_size: Batch size for evaluation
            include_mask: Whether to include masks for evaluation
            feature_indices: Optional list of feature indices to select
            feature_stats: Optional dictionary for feature standardization
            device: Device to run evaluation on
            
        Returns:
            Dictionary with evaluation metrics
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Create the forecasting dataset
        forecast_dataset = ForecastingEvaluationDataset(
            dataframe=dataframe,
            root_dir=root_dir,
            sequence_len=sequence_len,
            prediction_horizon=prediction_horizon,
            overlap=overlap,
            include_mask=include_mask,
            feature_indices=feature_indices,
            feature_stats=feature_stats
        )
        
        # Create dataloader
        dataloader = DataLoader(
            forecast_dataset,
            batch_size=batch_size,
            shuffle=False  # No need to shuffle for evaluation
        )
        
        # Initialize metrics
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        # Set model to evaluation mode
        self.eval()
        self.to(device)
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                batch_data_x = batch['data_x'].to(device)
                batch_data_y = batch['data_y'].to(device)
                
                # Add data to batch dictionary expected by model
                model_batch = {
                    'data_x': batch_data_x,
                    'data_y': batch_data_y
                }
                
                # Add masks if needed
                if include_mask:
                    if 'mask_x' in batch:
                        model_batch['mask_x'] = batch['mask_x'].to(device)
                    if 'mask_y' in batch:
                        model_batch['mask_y'] = batch['mask_y'].to(device)
                
                # Forward pass
                output = self(model_batch)
                
                # Get predicted and target segments
                predictions = output['sequence_output']
                targets = output['target_segments']
                
                # Calculate metrics
                # For MSE
                mse = F.mse_loss(predictions, targets, reduction='none')
                
                # For MAE
                mae = torch.abs(predictions - targets)
                
                # If using masks, apply them
                if include_mask and 'target_mask' in output:
                    mask = output['target_mask']
                    mse = mse * mask
                    mae = mae * mask
                    # Normalize by sum of mask
                    batch_mse = mse.sum() / (mask.sum() + 1e-10)
                    batch_mae = mae.sum() / (mask.sum() + 1e-10)
                else:
                    # Take mean across all dimensions
                    batch_mse = mse.mean()
                    batch_mae = mae.mean()
                
                # Accumulate metrics
                batch_size = predictions.size(0)
                total_mse += batch_mse.item() * batch_size
                total_mae += batch_mae.item() * batch_size
                total_samples += batch_size
        
        # Calculate final metrics
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        
        return {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': np.sqrt(avg_mse)
        }


class RevInForecastingLSTM(ForecastingLSTM):
    """
    ForecastingLSTM model incorporating Reversible Instance Normalization (RevIN).
    
    RevIN is applied to the input time series before segmentation and to the output
    forecasts prior to loss calculation. This makes the model more robust to distribution
    shifts and helps maintain the distributional properties of the time series.
    
    This class extends ForecastingLSTM, adding RevIN normalization and de-normalization
    steps in the forward pass.
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        encoding_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        target_labels: Optional[List[str]] = None,
        use_masked_loss: bool = False,
        num_features: int = 24,  # Number of features per minute (default is all 24)
        rev_in_affine: bool = False,
        rev_in_subtract_last: bool = False,
        rev_in_eps: float = 1e-5,
        l2_weight: float = 0.0,  # Weight for L2 regularization
    ):
        """
        Initialize the RevIN Forecasting LSTM model.
        
        Args:
            hidden_size: Hidden dimension size for the LSTM
            encoding_dim: Dimension of the encoded 30-minute segment vector
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
            target_labels: List of target labels to predict (without '_value' suffix)
            use_masked_loss: Whether to use the binary mask for loss calculation
            num_features: Number of original features per minute (e.g., 24)
            rev_in_affine: If True, RevIN layer has learnable affine parameters
            rev_in_subtract_last: If True, RevIN subtracts last element instead of mean
            rev_in_eps: Small epsilon for numerical stability in RevIN calculations
            l2_weight: Weight for L2 regularization (0.0 means disabled)
        """
        # Initialize the parent ForecastingLSTM class
        super().__init__(
            hidden_size=hidden_size,
            encoding_dim=encoding_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            target_labels=target_labels,
            use_masked_loss=use_masked_loss,
            num_features=num_features,
            l2_weight=l2_weight,  # Pass L2 weight to parent class
        )
        
        # RevIN configuration
        self.rev_in_eps = rev_in_eps
        
        # Import RevIN module here to avoid circular imports
        from models.revin import RevIN
        
        # Instantiate RevIN layer - operates on the original features dimension
        # It expects input shape (batch, seq_len, features)
        self.rev_in = RevIN(
            num_features=self.num_original_features,
            affine=rev_in_affine,
            subtract_last=rev_in_subtract_last,
            eps=self.rev_in_eps
        )
    
    def forward(self, batch: Dict[str, Union[torch.Tensor, Dict]], return_predictions: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with RevIN normalization and de-normalization.
        
        Args:
            batch: Dictionary containing 'data_x', 'data_y', and optionally 'mask_x' and 'mask_y'
            return_predictions: Whether to include label predictions in the output
            
        Returns:
            dict: A dictionary containing:
                - 'sequence_output': Forecasted segments (de-normalized)
                - 'target_segments': Segmented ground truth from data_y
                - 'target_mask': Mask for target segments (if include_mask=True)
                - 'label_predictions': Dictionary of label predictions (if return_predictions=True)
        """
        # Extract input and target data from batch
        x_input_orig = batch['data_x']  # Shape: (B, D_x, F, M) or (B, F, D_x*M)
        x_target_orig = batch['data_y']  # Shape: (B, D_y, F, M) or (B, F, D_y*M)
        
        # Get masks if provided and masked loss is enabled
        mask_input = batch.get('mask_x') if self.use_masked_loss else None
        mask_target = batch.get('mask_y') if self.use_masked_loss else None
        
        # --- Normalize input data using RevIN ---
        # Reshape input data for RevIN
        if len(x_input_orig.shape) == 4:
            # Original shape: (batch_size, num_days, num_original_features, 1440)
            batch_size, num_days_x, features, minutes_per_day = x_input_orig.shape
            time_steps_x = num_days_x * minutes_per_day
            
            # Reshape to (B, T, F)
            x_input_reshaped = x_input_orig.permute(0, 2, 1, 3).reshape(batch_size, features, time_steps_x)
            x_input_reshaped = x_input_reshaped.permute(0, 2, 1)
            
        elif len(x_input_orig.shape) == 3:
            # Flattened shape: (batch_size, num_original_features, num_days * 1440)
            batch_size, features, time_steps_x = x_input_orig.shape
            
            # Reshape to (B, T, F)
            x_input_reshaped = x_input_orig.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected input shape: {x_input_orig.shape}")
        
        # Handle NaNs
        x_input_clean = torch.nan_to_num(x_input_reshaped, nan=0.0)
        
        # Detect zero-variance features
        stdev = torch.std(x_input_clean, dim=1, keepdim=True)
        skip_revin_mask = (stdev < self.rev_in_eps)  # Shape: (B, 1, F)
        
        # Apply RevIN normalization
        x_input_normalized_all = self.rev_in(x_input_clean, mode='norm')
        
        # Keep original (cleaned) values for zero-variance features
        x_input_normalized = torch.where(skip_revin_mask, x_input_clean, x_input_normalized_all)
        
        # Reshape back to original format
        if len(x_input_orig.shape) == 4:
            # Original shape: reshape back to (B, D, F, M)
            x_input_norm = x_input_normalized.permute(0, 2, 1).reshape(batch_size, features, num_days_x, minutes_per_day)
            x_input_norm = x_input_norm.permute(0, 2, 1, 3)
        else:
            # Flattened shape: reshape back to (B, F, T)
            x_input_norm = x_input_normalized.permute(0, 2, 1)
            
        # Preprocess normalized input data and target data into segments
        if mask_input is not None:
            input_segments, input_mask_segments = self.preprocess_batch(x_input_norm, mask_input)
        else:
            input_segments = self.preprocess_batch(x_input_norm)
        
        # Also preprocess original target data (not normalized) for ground truth
        if mask_target is not None:
            target_segments, target_mask_segments = self.preprocess_batch(x_target_orig, mask_target)
        else:
            target_segments = self.preprocess_batch(x_target_orig)
            target_mask_segments = None
        
        batch_size = input_segments.shape[0]
        num_input_segments = input_segments.shape[1]
        num_target_segments = target_segments.shape[1]
        
        # --- Context Encoding Phase ---
        # Encode all normalized input segments
        encoded_input = self.encoder(input_segments)  # Shape: (B, num_input_segments, encoding_dim)
        
        # Process entire input sequence with LSTM to get the final state
        _, (h_final, c_final) = self.lstm(encoded_input)  # Process all input segments at once
        
        # --- Forecast Generation Phase ---
        # Use the final state to generate forecasts
        predicted_segments_norm_list = []
        
        # Create the initial decoder input - we'll use the encoding of the last input segment
        last_input_segment = input_segments[:, -1:, :]  # Shape: (B, 1, features_per_segment)
        current_lstm_input = self.encoder(last_input_segment)  # Shape: (B, 1, encoding_dim)
        
        # Set current hidden and cell states from the context encoding
        current_h, current_c = h_final, c_final
        
        # Generate forecast segments one by one (still in normalized space)
        for _ in range(num_target_segments):
            # Generate a single forecast step using the current state
            lstm_out_step, (current_h, current_c) = self.lstm(current_lstm_input, (current_h, current_c))
            # lstm_out_step shape: (B, 1, hidden_size * num_directions)
            
            # Decode the LSTM output to get the forecasted segment (still normalized)
            decoded_segment_norm = self.decoder(lstm_out_step)  # Shape: (B, 1, features_per_segment)
            predicted_segments_norm_list.append(decoded_segment_norm)
            
            # Update the LSTM input for the next iteration using the predicted segment
            current_lstm_input = self.encoder(decoded_segment_norm)
        
        # Concatenate all forecasted segments (still normalized)
        predicted_segments_norm = torch.cat(predicted_segments_norm_list, dim=1)  
        # Shape: (B, num_target_segments, features_per_segment)
        
        # --- De-normalize the predictions using RevIN ---
        # Reshape predictions to time series format for de-normalization
        num_predicted_segments = predicted_segments_norm.shape[1]
        predicted_time_steps = num_predicted_segments * self.minutes_per_segment
        
        # Reshape from (B, num_target_segments, features_per_segment) to (B, T, F)
        pred_norm_reshaped = predicted_segments_norm.view(
            batch_size, num_predicted_segments, self.num_original_features, self.minutes_per_segment
        )
        pred_norm_reshaped = pred_norm_reshaped.permute(0, 1, 3, 2)
        pred_norm_reshaped = pred_norm_reshaped.reshape(batch_size, predicted_time_steps, self.num_original_features)
        
        # Apply RevIN de-normalization (using stored stats from normalization)
        pred_denorm_all = self.rev_in(pred_norm_reshaped, mode='denorm')
        
        # Keep the non-de-normalized values for features that were skipped
        # Expand skip_revin_mask to match prediction time steps
        skip_revin_mask_expanded = skip_revin_mask.expand(-1, predicted_time_steps, -1)
        pred_denorm_final = torch.where(
            skip_revin_mask_expanded,
            pred_norm_reshaped,  # Use the non-denormalized value if RevIN was skipped
            pred_denorm_all      # Use the de-normalized value if RevIN was applied
        )
        
        # Reshape de-normalized time series back into segments
        pred_denorm_segments = pred_denorm_final.view(
            batch_size, num_predicted_segments, self.minutes_per_segment, self.num_original_features
        )
        pred_denorm_segments = pred_denorm_segments.permute(0, 1, 3, 2)
        pred_denorm_segments = pred_denorm_segments.reshape(
            batch_size, num_predicted_segments, self.features_per_segment
        )
        
        # Prepare result dictionary
        result = {
            'sequence_output': pred_denorm_segments,  # Final output in original scale
            'target_segments': target_segments,       # Target in original scale
        }
        
        if self.use_masked_loss and target_mask_segments is not None:
            result['target_mask'] = target_mask_segments
        
        # Add label predictions if requested
        if return_predictions and self.target_labels:
            if self.bidirectional:
                final_hidden = h_final.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)[-1]
                final_hidden = final_hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            else:
                final_hidden = h_final[-1]  # Final hidden state of last layer
            
            label_predictions = {}
            for label in self.target_labels:
                label_predictions[label] = self.output_layers[label](final_hidden).squeeze(-1)
            result['label_predictions'] = label_predictions
        
        return result
    
    def predict(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future time steps given an input sequence, with RevIN normalization.
        
        Args:
            input_sequence: Input sequence in original format (B, D, F, M) or
                           flattened format (B, F, D*M)
            steps: Number of future 30-min segments to predict
            
        Returns:
            torch.Tensor: Predicted future segments (B, steps, F*M), de-normalized
        """
        self.eval()
        
        # --- Normalize input with RevIN ---
        # Reshape input data for RevIN
        if len(input_sequence.shape) == 4:
            # Original shape: (B, D, F, M)
            batch_size, num_days, features, minutes_per_day = input_sequence.shape
            time_steps = num_days * minutes_per_day
            
            # Reshape to (B, T, F)
            input_reshaped = input_sequence.permute(0, 2, 1, 3).reshape(batch_size, features, time_steps)
            input_reshaped = input_reshaped.permute(0, 2, 1)
            
        elif len(input_sequence.shape) == 3:
            # Could be either:
            # - Flattened format: (B, F, D*M)
            # - Already segmented: (B, S, F*M)
            batch_size, dim1, dim2 = input_sequence.shape
            
            # If dim1 < dim2 and dim1 == num_features, it's likely flattened format
            if dim1 < dim2 and dim1 == self.num_original_features:
                # Flattened format: (B, F, T) -> (B, T, F)
                input_reshaped = input_sequence.permute(0, 2, 1)
                time_steps = dim2
            else:
                # Already segmented - we need to reshape to time series format first
                # (B, S, F*M) -> (B, S, F, M) -> (B, T, F)
                num_segments = dim1
                input_view = input_sequence.view(batch_size, num_segments, self.num_original_features, self.minutes_per_segment)
                input_view = input_view.permute(0, 1, 3, 2)
                time_steps = num_segments * self.minutes_per_segment
                input_reshaped = input_view.reshape(batch_size, time_steps, self.num_original_features)
        else:
            raise ValueError(f"Unexpected input shape: {input_sequence.shape}")
        
        # Handle NaNs
        input_clean = torch.nan_to_num(input_reshaped, nan=0.0)
        
        # Detect zero-variance features
        with torch.no_grad():
            stdev = torch.std(input_clean, dim=1, keepdim=True)
            skip_revin_mask = (stdev < self.rev_in_eps)  # Shape: (B, 1, F)
            
            # Apply RevIN normalization
            input_normalized_all = self.rev_in(input_clean, mode='norm')
            
            # Keep original (cleaned) values for zero-variance features
            input_normalized = torch.where(skip_revin_mask, input_clean, input_normalized_all)
            
            # Reshape back to original format
            if len(input_sequence.shape) == 4:
                # Original shape: reshape back to (B, D, F, M)
                input_norm = input_normalized.permute(0, 2, 1).reshape(batch_size, features, num_days, minutes_per_day)
                input_norm = input_norm.permute(0, 2, 1, 3)
            elif len(input_sequence.shape) == 3 and dim1 < dim2 and dim1 == self.num_original_features:
                # Flattened shape: reshape back to (B, F, T)
                input_norm = input_normalized.permute(0, 2, 1)
            else:
                # Already segmented: reshape back to segments
                # (B, T, F) -> (B, S, F, M) -> (B, S, F*M)
                num_segments = time_steps // self.minutes_per_segment
                input_norm_view = input_normalized.reshape(batch_size, num_segments, self.minutes_per_segment, self.num_original_features)
                input_norm_view = input_norm_view.permute(0, 1, 3, 2)
                input_norm = input_norm_view.reshape(batch_size, num_segments, self.features_per_segment)
            
            # Preprocess normalized input if needed (if not already segmented)
            if len(input_sequence.shape) != 3 or (dim1 < dim2 and dim1 == self.num_original_features):
                input_segments = self.preprocess_batch(input_norm)
            else:
                input_segments = input_norm
            
            # Encode the entire input sequence
            encoded_input = self.encoder(input_segments)  # Shape: (B, num_segments, encoding_dim)
            
            # Process with LSTM to get final state
            _, (h_final, c_final) = self.lstm(encoded_input)
            
            # Create the initial decoder input
            last_segment = input_segments[:, -1:, :]  # Shape: (B, 1, features_per_segment)
            current_lstm_input = self.encoder(last_segment)  # Shape: (B, 1, encoding_dim)
            
            # Set current hidden and cell states
            current_h, current_c = h_final, c_final
            
            # Generate predictions (still in normalized space)
            predictions_norm = []
            for _ in range(steps):
                # Pass through LSTM using the previous hidden state
                lstm_out_step, (current_h, current_c) = self.lstm(current_lstm_input, (current_h, current_c))
                
                # Decode the output of this single step
                decoded_norm = self.decoder(lstm_out_step)  # Shape: (B, 1, features_per_segment)
                predictions_norm.append(decoded_norm)
                
                # Update input for next iteration
                current_lstm_input = self.encoder(decoded_norm)
            
            # Stack all predictions (still normalized)
            pred_norm = torch.cat(predictions_norm, dim=1)  # Shape: (B, steps, features_per_segment)
            
            # --- De-normalize predictions ---
            # Reshape to time series format for de-normalization
            predicted_time_steps = steps * self.minutes_per_segment
            
            # Reshape from (B, steps, features_per_segment) to (B, T, F)
            pred_norm_reshaped = pred_norm.view(
                batch_size, steps, self.num_original_features, self.minutes_per_segment
            )
            pred_norm_reshaped = pred_norm_reshaped.permute(0, 1, 3, 2)
            pred_norm_reshaped = pred_norm_reshaped.reshape(batch_size, predicted_time_steps, self.num_original_features)
            
            # Apply RevIN de-normalization
            pred_denorm_all = self.rev_in(pred_norm_reshaped, mode='denorm')
            
            # Keep the non-de-normalized values for features that were skipped
            skip_revin_mask_expanded = skip_revin_mask.expand(-1, predicted_time_steps, -1)
            pred_denorm_final = torch.where(
                skip_revin_mask_expanded,
                pred_norm_reshaped,  # Use non-denormalized value if RevIN was skipped
                pred_denorm_all      # Use de-normalized value if RevIN was applied
            )
            
            # Reshape back to segments
            pred_denorm_segments = pred_denorm_final.view(
                batch_size, steps, self.minutes_per_segment, self.num_original_features
            )
            pred_denorm_segments = pred_denorm_segments.permute(0, 1, 3, 2)
            pred_denorm_segments = pred_denorm_segments.reshape(
                batch_size, steps, self.features_per_segment
            )
            
            return pred_denorm_segments 