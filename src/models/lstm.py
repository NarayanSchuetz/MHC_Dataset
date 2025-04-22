import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from models.revin import RevIN


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
        
        # Constants for data structure
        self.minutes_per_segment = 30
        self.segments_per_day = (24 * 60) // self.minutes_per_segment  # 48 segments per day
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
        Preprocess batch data from shape (batch_size, num_days, num_original_features, 1440)
        to 30-minute segments suitable for encoding.

        Args:
            batch_data: Input tensor of shape (batch_size, num_days, num_original_features, 1440)
            batch_mask: Optional mask tensor of same shape

        Returns:
            If batch_mask is None:
                Processed tensor of shape (batch_size, num_segments, features_per_segment)
            Else:
                Tuple of (processed_data, processed_mask) both of shape (batch_size, num_segments, features_per_segment)
        """
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
                use_teacher_forcing = self.training and (torch.rand(1).item() < self.teacher_forcing_ratio)
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
                    
        return total_loss
    
    def predict_future(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future time steps given an input sequence.
        Assumes input_sequence is already segmented (batch_size, num_segments, features_per_segment).
        Base implementation without RevIN.
        """
        self.eval()
        batch_size = input_sequence.shape[0]
        
        with torch.no_grad():
            future_predictions = []
            current_input = input_sequence # Shape: (B, num_segments, features_per_segment)
            h, c = None, None
            
            # Process the initial input sequence to get the starting hidden state
            encoded = self.encoder(current_input) # Shape: (B, num_segments, encoding_dim)
            lstm_out, (h, c) = self.lstm(encoded) # Get final state from full sequence

            # Use the last segment's output for the first prediction step input
            # Or maybe just use the last segment of the input directly? Let's use the last segment
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
        rev_in_affine: bool = False,
        rev_in_subtract_last: bool = False
    ):
        """
        Initialize the RevIN Time Series Autoencoder LSTM model.

        Args:
            num_features: Number of original features per minute (e.g., 24)
            rev_in_affine: If True, RevIN layer has learnable affine parameters
            rev_in_subtract_last: If True, RevIN subtracts last element instead of mean
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
            num_features=num_features # Passes num_original_features to parent
        )

        # Instantiate RevIN layer - operates on the original features dimension
        # It expects input shape (batch, seq_len, features)
        # We will reshape data to (batch, time_steps, num_original_features)
        self.rev_in = RevIN(
            num_features=self.num_original_features, # Use original feature count
            affine=rev_in_affine,
            subtract_last=rev_in_subtract_last
        )

    def forward(self, batch: Dict[str, Union[torch.Tensor, Dict]], return_predictions: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RevIN Time Series Autoencoder LSTM model.
        Applies RevIN normalization to original features before segmentation,
        and de-normalization after decoding.
        """
        x_orig = batch['data']  # Shape: (B, D, F, M) = (batch_size, num_days, num_original_features, minutes_per_day)
        mask_orig = batch.get('mask') if self.use_masked_loss else None

        batch_size, num_days, _, minutes_per_day = x_orig.shape
        time_steps = num_days * minutes_per_day

        # --- RevIN Normalization ---
        # Reshape for RevIN: (B, D, F, M) -> (B, F, D*M) -> (B, D*M, F)
        x_reshaped = x_orig.permute(0, 2, 1, 3).reshape(batch_size, self.num_original_features, time_steps)
        x_reshaped = x_reshaped.permute(0, 2, 1) # Shape: (B, time_steps, F)

        # Apply RevIN normalization
        x_normalized_reshaped = self.rev_in(x_reshaped, mode='norm')

        # Reshape back to original format: (B, time_steps, F) -> (B, F, time_steps) -> (B, D, F, M)
        x_normalized = x_normalized_reshaped.permute(0, 2, 1).reshape(batch_size, self.num_original_features, num_days, minutes_per_day)
        x_normalized = x_normalized.permute(0, 2, 1, 3) # Shape: (B, D, F, M)
        # --- End RevIN Normalization ---

        # Preprocess normalized data (and original mask) into segments
        # Mask remains associated with original structure, not normalized values
        if mask_orig is not None:
            x_norm_segmented, mask_segmented = self.preprocess_batch(x_normalized, mask_orig)
        else:
            x_norm_segmented = self.preprocess_batch(x_normalized)
            mask_segmented = None

        # Preprocess original data (without mask needed here) to get targets in original scale
        x_orig_segmented = self.preprocess_batch(x_orig)

        # Split segments
        # Input for LSTM is normalized
        input_segments_norm = x_norm_segmented[:, :-self.prediction_horizon, :]
        # Target for loss is original scale
        target_segments_orig = x_orig_segmented[:, self.prediction_horizon:, :]

        target_mask = None
        if mask_segmented is not None:
            target_mask = mask_segmented[:, self.prediction_horizon:, :] # Mask corresponds to original structure

        # --- LSTM Autoencoder Logic (operates on normalized data) ---
        encoded_input = self.encoder(input_segments_norm[:, 0:1, :]) # Encode first normalized segment
        h, c = None, None
        outputs_normalized = [] # Store normalized decoder outputs
        seq_len = input_segments_norm.size(1) # Length of sequence to process

        for t in range(seq_len):
            if h is None and c is None:
                lstm_out, (h, c) = self.lstm(encoded_input)
            else:
                lstm_out, (h, c) = self.lstm(encoded_input, (h, c))

            decoded_normalized = self.decoder(lstm_out) # Still normalized
            outputs_normalized.append(decoded_normalized)

            if t < seq_len - 1:
                use_teacher_forcing = self.training and (torch.rand(1).item() < self.teacher_forcing_ratio)
                if use_teacher_forcing:
                    next_input_normalized = input_segments_norm[:, t+1:t+2, :] # Ground truth (normalized)
                else:
                    next_input_normalized = decoded_normalized # Model's own prediction (normalized)
                encoded_input = self.encoder(next_input_normalized) # Encode for next step
        # --- End LSTM Autoencoder Logic ---

        # Concatenate normalized predictions
        # Shape: (B, seq_len, features_per_segment)
        decoded_segments_normalized = torch.cat(outputs_normalized, dim=1)

        # --- RevIN De-normalization ---
        # Reshape normalized decoded segments back into time series format
        # (B, seq_len, F*MinPerSeg) -> (B, seq_len*MinPerSeg, F)
        num_predicted_segments = decoded_segments_normalized.shape[1]
        predicted_time_steps = num_predicted_segments * self.minutes_per_segment

        decoded_norm_reshaped = decoded_segments_normalized.view(batch_size, num_predicted_segments, self.num_original_features, self.minutes_per_segment)
        decoded_norm_reshaped = decoded_norm_reshaped.permute(0, 1, 3, 2) # -> (B, seq_len, MinPerSeg, F)
        decoded_norm_reshaped = decoded_norm_reshaped.reshape(batch_size, predicted_time_steps, self.num_original_features) # -> (B, predicted_time_steps, F)

        # Apply RevIN de-normalization
        # Important: Need to pass the reshaped *normalized* data corresponding to the prediction window
        # We use the statistics stored from the *full* input sequence normalization.
        # RevIN's denorm uses the stored mean/std, not recalculating.
        decoded_denorm_reshaped = self.rev_in(decoded_norm_reshaped, mode='denorm')

        # Reshape de-normalized time series back into segments
        # (B, predicted_time_steps, F) -> (B, seq_len, F*MinPerSeg)
        decoded_denorm_segments = decoded_denorm_reshaped.view(batch_size, num_predicted_segments, self.minutes_per_segment, self.num_original_features)
        decoded_denorm_segments = decoded_denorm_segments.permute(0, 1, 3, 2) # -> (B, seq_len, F, MinPerSeg)
        decoded_denorm_segments = decoded_denorm_segments.reshape(batch_size, num_predicted_segments, self.features_per_segment) # -> (B, seq_len, features_per_segment)
        # --- End RevIN De-normalization ---


        # Prepare result dictionary
        result = {
            'sequence_output': decoded_denorm_segments, # De-normalized output
            'target_segments': target_segments_orig,  # Original scale target
        }
        if self.use_masked_loss and target_mask is not None:
            result['target_mask'] = target_mask # Use original mask

        # Add label predictions if requested (based on final LSTM hidden state from normalized data)
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

    # compute_loss remains the same as it operates on the de-normalized sequence_output
    # and the original scale target_segments.

    def predict_future(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future time steps using RevIN applied to original features.
        Normalizes input, predicts in normalized space using segmented LSTM,
        and de-normalizes the final prediction.

        Args:
            input_sequence: Input tensor of shape (batch_size, num_segments, features_per_segment).
                            Assumes this is already preprocessed (segmented) *in original scale*.
            steps: Number of future 30-min segments to predict

        Returns:
            Tensor of shape (batch_size, steps, features_per_segment) with de-normalized predictions
        """
        self.eval()
        batch_size = input_sequence.shape[0]
        num_input_segments = input_sequence.shape[1]
        input_time_steps = num_input_segments * self.minutes_per_segment

        with torch.no_grad():
            # --- RevIN Normalization ---
            # Reshape input segments back to time series format: (B, num_seg, F*MinPerSeg) -> (B, num_seg*MinPerSeg, F)
            input_reshaped = input_sequence.view(batch_size, num_input_segments, self.num_original_features, self.minutes_per_segment)
            input_reshaped = input_reshaped.permute(0, 1, 3, 2) # -> (B, num_seg, MinPerSeg, F)
            input_reshaped = input_reshaped.reshape(batch_size, input_time_steps, self.num_original_features) # -> (B, input_time_steps, F)

            # Apply RevIN normalization
            normalized_input_reshaped = self.rev_in(input_reshaped, mode='norm')

            # Reshape normalized time series back into segments for LSTM processing
            # (B, input_time_steps, F) -> (B, num_seg, features_per_segment)
            normalized_input_segmented = normalized_input_reshaped.view(batch_size, num_input_segments, self.minutes_per_segment, self.num_original_features)
            normalized_input_segmented = normalized_input_segmented.permute(0, 1, 3, 2) # -> (B, num_seg, F, MinPerSeg)
            normalized_input_segmented = normalized_input_segmented.reshape(batch_size, num_input_segments, self.features_per_segment) # -> (B, num_seg, features_per_segment)
            # --- End RevIN Normalization ---

            # --- Prediction Loop (operates on normalized segments) ---
            future_predictions_normalized = []
            current_normalized_input = normalized_input_segmented # Start with the normalized input history
            h, c = None, None

            # Process the initial normalized history to get starting hidden state
            encoded_norm_hist = self.encoder(current_normalized_input)
            _, (h, c) = self.lstm(encoded_norm_hist) # Get final state

            # Use the last segment of the normalized history as the first input for prediction loop
            last_normalized_segment = current_normalized_input[:, -1:, :] # Shape: (B, 1, features_per_segment)

            for _ in range(steps):
                encoded_last_norm_segment = self.encoder(last_normalized_segment) # Shape: (B, 1, encoding_dim)

                # Pass through LSTM using previous state
                lstm_out_normalized_step, (h, c) = self.lstm(encoded_last_norm_segment, (h, c))

                # Decode the output (still normalized)
                decoded_normalized = self.decoder(lstm_out_normalized_step) # Shape: (B, 1, features_per_segment)
                future_predictions_normalized.append(decoded_normalized)

                # Update last_normalized_segment for next iteration
                last_normalized_segment = decoded_normalized
            # --- End Prediction Loop ---

            # Concatenate normalized predictions
            # Shape: (batch_size, steps, features_per_segment)
            final_normalized_predictions_segmented = torch.cat(future_predictions_normalized, dim=1)

            # --- RevIN De-normalization ---
            # Reshape normalized predictions back into time series format
            # (B, steps, F*MinPerSeg) -> (B, steps*MinPerSeg, F)
            predicted_time_steps = steps * self.minutes_per_segment
            final_norm_pred_reshaped = final_normalized_predictions_segmented.view(batch_size, steps, self.num_original_features, self.minutes_per_segment)
            final_norm_pred_reshaped = final_norm_pred_reshaped.permute(0, 1, 3, 2) # -> (B, steps, MinPerSeg, F)
            final_norm_pred_reshaped = final_norm_pred_reshaped.reshape(batch_size, predicted_time_steps, self.num_original_features) # -> (B, predicted_time_steps, F)

            # Apply RevIN de-normalization using stats from original normalization
            final_denorm_pred_reshaped = self.rev_in(final_norm_pred_reshaped, mode='denorm')

            # Reshape de-normalized time series back into segments
            # (B, predicted_time_steps, F) -> (B, steps, features_per_segment)
            final_predictions_segmented = final_denorm_pred_reshaped.view(batch_size, steps, self.minutes_per_segment, self.num_original_features)
            final_predictions_segmented = final_predictions_segmented.permute(0, 1, 3, 2) # -> (B, steps, F, MinPerSeg)
            final_predictions_segmented = final_predictions_segmented.reshape(batch_size, steps, self.features_per_segment) # -> (B, steps, features_per_segment)
            # --- End RevIN De-normalization ---

            return final_predictions_segmented


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
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader providing batches from the MHC dataset
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
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
