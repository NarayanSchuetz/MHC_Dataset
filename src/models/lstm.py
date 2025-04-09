import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union

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
        
        # Constants for data structure
        self.minutes_per_segment = 30
        self.segments_per_day = (24 * 60) // self.minutes_per_segment  # 48 segments per day
        self.features_per_minute = 24  # 24 features per minute
        self.features_per_segment = self.features_per_minute * self.minutes_per_segment  # 24*30 = 720
        
        # Encoder: Compress each 30-minute segment (with all features) to encoding_dim
        # Input shape for each 30-min segment is (24 features * 30 minutes) = 720
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
        Preprocess batch data from shape (batch_size, num_days, 24, 1440) 
        to 30-minute segments suitable for encoding.
        
        Args:
            batch_data: Input tensor of shape (batch_size, num_days, 24, 1440)
                        where 24 is the number of features per minute and 1440 is minutes per day
            batch_mask: Optional mask tensor of shape (batch_size, num_days, 24, 1440)
                        where 1 indicates observed values, 0 indicates missing values
            
        Returns:
            If batch_mask is None:
                Processed tensor of shape (batch_size, num_segments, minutes_per_segment)
            Else:
                Tuple of (processed_data, processed_mask) both of shape (batch_size, num_segments, minutes_per_segment)
        """
        batch_size, num_days, features, minutes_per_day = batch_data.shape
        
        # Verify that minutes_per_day is as expected (1440 minutes per day)
        if minutes_per_day != 24 * 60:
            raise ValueError(f"Expected 1440 minutes per day, got {minutes_per_day}")
        
        # Calculate number of 30-minute segments per day
        segments_per_day = minutes_per_day // self.minutes_per_segment  # Should be 48
        
        # Reshape to separate days and create 30-minute segments
        # From (batch_size, num_days, features, minutes_per_day) to 
        # (batch_size, num_days * segments_per_day, features, minutes_per_segment)
        x = batch_data.reshape(batch_size, num_days, features, segments_per_day, self.minutes_per_segment)
        x = x.permute(0, 1, 3, 2, 4)  # -> (batch_size, num_days, segments_per_day, features, minutes_per_segment)
        x = x.reshape(batch_size, num_days * segments_per_day, features * self.minutes_per_segment)
        
        # Replace NaN values with zeros
        x = torch.nan_to_num(x, nan=0.0)
        
        # If we have a mask, preprocess it the same way
        if batch_mask is not None:
            mask = batch_mask.reshape(batch_size, num_days, features, segments_per_day, self.minutes_per_segment)
            mask = mask.permute(0, 1, 3, 2, 4)
            mask = mask.reshape(batch_size, num_days * segments_per_day, features * self.minutes_per_segment)
            return x, mask
        
        return x
            
    def forward(self, batch: Dict[str, Union[torch.Tensor, Dict]], return_predictions: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing:
                - 'data' tensor of shape (batch_size, num_days, 24, 1440)
                - 'mask' tensor of same shape (if self.use_masked_loss=True)
                - 'labels' dictionary
            return_predictions: Whether to return label predictions (False to just return sequence outputs)
                  
        Returns:
            Dictionary containing:
              - 'sequence_output': Tensor of shape (batch_size, num_segments, minutes_per_segment)
                representing reconstructed/predicted segments
              - 'target_segments': Target segments to predict
              - 'target_mask': Mask for target segments (if self.use_masked_loss=True)
              - 'label_predictions': Dictionary mapping target labels to predictions (if return_predictions=True)
        """
        # Extract data tensor from the batch
        x = batch['data']  # Shape: (batch_size, num_days, 24, 1440)
        
        # Get mask if using masked loss
        mask = batch.get('mask') if self.use_masked_loss else None
        
        # Preprocess data (and mask if provided)
        if mask is not None:
            x, mask = self.preprocess_batch(x, mask)  # Both shape: (batch_size, num_segments, minutes_per_segment)
        else:
            x = self.preprocess_batch(x)  # Shape: (batch_size, num_segments, minutes_per_segment)
        
        # Get batch size and number of segments
        batch_size, num_segments, _ = x.shape
        
        # Split into input sequence (all but last prediction_horizon segments) 
        # Input segments sequence to process
        input_segments = x[:, :-self.prediction_horizon, :]  # Shape: (batch_size, num_segments-prediction_horizon, minutes_per_segment)
        
        # Target segments to predict
        target_segments = x[:, self.prediction_horizon:, :]  # Shape: (batch_size, num_segments-prediction_horizon, minutes_per_segment)
        
        # Also split the mask if provided
        target_mask = None
        if mask is not None:
            target_mask = mask[:, self.prediction_horizon:, :]  # Shape: (batch_size, num_segments-prediction_horizon, minutes_per_segment)
        
        # Encode each 30-min segment to a 100d vector
        # Process all segments in batch at once
        encoded_segments = self.encoder(input_segments)  # Shape: (batch_size, num_segments-prediction_horizon, encoding_dim)
        
        # Apply LSTM to encoded sequence
        lstm_out, (h_n, c_n) = self.lstm(encoded_segments)  # lstm_out: (batch_size, num_segments-prediction_horizon, hidden_size*num_directions)
        
        # Decode each output timestep back to original dimensions
        decoded_segments = self.decoder(lstm_out)  # Shape: (batch_size, num_segments-prediction_horizon, minutes_per_segment)
        
        # Prepare result dictionary
        result = {
            'sequence_output': decoded_segments,
            'target_segments': target_segments,
        }
        
        # Include mask if using masked loss
        if self.use_masked_loss and target_mask is not None:
            result['target_mask'] = target_mask
        
        # Add label predictions if requested
        if return_predictions:
            # Use the final hidden state for label prediction
            if self.bidirectional:
                # Combine forward and backward final hidden states
                final_hidden = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
                final_hidden = final_hidden[-1].transpose(0, 1).contiguous().view(batch_size, -1)
            else:
                # Just use the final layer's hidden state
                final_hidden = h_n[-1]
            
            # Generate predictions for each target label
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
                if label in labels_dict:
                    # Get ground truth and handle NaN values
                    ground_truth = torch.tensor([
                        labels_dict[label] if not torch.isnan(torch.tensor(labels_dict[label])) 
                        else 0.0
                    ], device=label_predictions[label].device)
                    
                    # Only compute loss for non-NaN labels
                    if not torch.isnan(ground_truth).any():
                        # MSE loss for regression
                        label_loss = F.mse_loss(label_predictions[label], ground_truth)
                        total_loss += label_loss
                    
        return total_loss
    
    def predict_future(self, input_sequence: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future time steps given an input sequence.
        
        Args:
            input_sequence: Input tensor of shape (batch_size, num_segments, minutes_per_segment)
            steps: Number of future 30-min segments to predict
            
        Returns:
            Tensor of shape (batch_size, steps, minutes_per_segment) with predictions
        """
        self.eval()
        batch_size = input_sequence.shape[0]
        
        with torch.no_grad():
            # Initialize lists to store predictions
            future_predictions = []
            
            # Current input is the provided sequence
            current_input = input_sequence
            
            # Initialize hidden state
            h, c = None, None
            
            for _ in range(steps):
                # Encode the current input
                encoded = self.encoder(current_input)
                
                # Pass through LSTM
                if h is None and c is None:
                    lstm_out, (h, c) = self.lstm(encoded)
                else:
                    lstm_out, (h, c) = self.lstm(encoded, (h, c))
                
                # Decode the output
                decoded = self.decoder(lstm_out[:, -1:, :])  # Only use last timestep
                future_predictions.append(decoded)
                
                # Update current input for next iteration (remove oldest, add newest prediction)
                current_input = torch.cat([current_input[:, 1:, :], decoded], dim=1)
            
            # Stack all predictions
            return torch.cat(future_predictions, dim=1)


class LSTMTrainer:
    """
    Trainer class for the AutoencoderLSTM model.
    
    Handles training loop, validation, and inference.
    """
    
    def __init__(
        self,
        model: AutoencoderLSTM,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The AutoencoderLSTM model to train
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
            # Move batch data to device
            batch_data = batch['data'].to(self.device)
            batch['data'] = batch_data
            
            # Move mask to device if it exists and model uses masked loss
            if self.model.use_masked_loss and 'mask' in batch:
                batch_mask = batch['mask'].to(self.device)
                batch['mask'] = batch_mask
            
            # Forward pass
            self.optimizer.zero_grad()
            model_output = self.model(batch)
            
            # Compute loss
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
                # Move batch data to device
                batch_data = batch['data'].to(self.device)
                batch['data'] = batch_data
                
                # Move mask to device if it exists and model uses masked loss
                if self.model.use_masked_loss and 'mask' in batch:
                    batch_mask = batch['mask'].to(self.device)
                    batch['mask'] = batch_mask
                
                # Forward pass
                model_output = self.model(batch)
                
                # Compute loss
                loss = self.model.compute_loss(model_output, batch)
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
