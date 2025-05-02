import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F # Added for mse_loss

from torch_dataset import ForecastingEvaluationDataset
from models.lstm import AutoencoderLSTM, RevInAutoencoderLSTM


def parse_forecast_split(split_str: str) -> Tuple[int, int, int]:
    """
    Parse a forecast split string into sequence length and prediction horizon.
    
    Args:
        split_str: String in format "X-Yd" where X is input days, Y is forecast days
                  e.g., "5-2d" means 5 days of input, 2 days of forecast
    
    Returns:
        Tuple of (sequence_len, prediction_horizon, overlap) in minutes
    """
    parts = split_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid split format: {split_str}. Expected format like '5-2d'")
    
    try:
        input_days = int(parts[0])
        forecast_part = parts[1]
        if not forecast_part.endswith('d'):
            raise ValueError(f"Invalid forecast format: {forecast_part}. Expected format like '2d'")
        
        forecast_days = int(forecast_part[:-1])
        
        # Convert to minutes
        minutes_per_day = 24 * 60
        sequence_len = input_days * minutes_per_day
        prediction_horizon = forecast_days * minutes_per_day
        
        # Default to no overlap
        overlap = 0
        
        return sequence_len, prediction_horizon, overlap
    except ValueError as e:
        raise ValueError(f"Error parsing split string '{split_str}': {str(e)}")


def _predict_batch(
    model: Union[AutoencoderLSTM, RevInAutoencoderLSTM],
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    prediction_horizon: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Handles reshaping and prediction for a single batch.

    Args:
        model: The forecasting model.
        data_x: Input data tensor [B, D_in, F, T_x].
        data_y: Target data tensor [B, D_out, F, T_pred].
        prediction_horizon: Total prediction horizon in minutes.
        device: Device to run on.

    Returns:
        Tuple of (reshaped_predictions, data_y, valid_mask_elements).
        Returns (None, data_y, None) if reshaping or prediction fails.
    """
    # --- Reshape input data for prediction ---
    batch_size, num_days_in, _, time_steps_x = data_x.shape
    _, _, num_features, time_steps_y = data_y.shape # Infer num_features from data_y

    # Get model parameters
    segments_per_day = model.segments_per_day
    minutes_per_segment = model.minutes_per_segment

    # Check consistency for input
    if time_steps_x != segments_per_day * minutes_per_segment:
         segments_per_day = time_steps_x // minutes_per_segment # Adjust based on data

    total_input_segments = num_days_in * segments_per_day

    # Reshape data_x: [B, D_in, F, T_x] -> [B, total_input_segments, F * minutes_per_segment]
    try:
        reshaped_data_x = data_x.view(
            batch_size, num_days_in, num_features, segments_per_day, minutes_per_segment
        ).permute(0, 1, 3, 2, 4).reshape(
            batch_size, total_input_segments, num_features * minutes_per_segment
        )
    except RuntimeError as e:
         return None, data_y, None # Indicate failure

    # --- Generate predictions ---
    target_segments = prediction_horizon // minutes_per_segment
    try:
        predictions = model.predict_future(reshaped_data_x, steps=target_segments)
        # Predictions shape: [B, target_segments, num_features * minutes_per_segment]
    except Exception as e:
        return None, data_y, None # Indicate failure

    # --- Reshape predictions to match data_y shape ---
    # Target data_y shape: [B, D_out, F, T_steps_per_day == time_steps_y]
    batch_size_pred, num_days_out_actual, num_features_pred, T_steps_per_day = data_y.shape

    # Recalculate expected segments per day based on output shape if necessary
    calculated_segments_per_day_out = T_steps_per_day // minutes_per_segment
    if calculated_segments_per_day_out * minutes_per_segment != T_steps_per_day:
        return None, data_y, None

    pred_days_out_calc = target_segments // calculated_segments_per_day_out
    if target_segments % calculated_segments_per_day_out != 0:
         pred_days_out = num_days_out_actual
    else:
        pred_days_out = pred_days_out_calc
        if pred_days_out != num_days_out_actual:
             pred_days_out = num_days_out_actual # Prioritize actual data shape

    # Check if total segments in prediction match expected based on target shape
    expected_total_segments = pred_days_out * calculated_segments_per_day_out
    if predictions.shape[1] != expected_total_segments:
        target_segments = expected_total_segments # Adjust target segments for reshape

    # [B, target_segments, F * min_per_seg] -> [B, D_out, F, T_steps_per_day]
    try:
        reshaped_preds = predictions.view(
            batch_size, target_segments, num_features, minutes_per_segment # B, T_seg_tot, F, M_per_seg
        ).view(
            batch_size, pred_days_out, calculated_segments_per_day_out, num_features, minutes_per_segment # B, D_out, S_per_day, F, M_per_seg
        ).permute(
            0, 1, 3, 2, 4 # B, D_out, F, S_per_day, M_per_seg
        ).reshape(
            batch_size, pred_days_out, num_features, T_steps_per_day # B, D_out, F, T_steps_day
        )

        # Final shape check and alignment
        if reshaped_preds.shape != data_y.shape:
            min_d = min(reshaped_preds.shape[1], data_y.shape[1])
            min_f = min(reshaped_preds.shape[2], data_y.shape[2])
            min_t = min(reshaped_preds.shape[3], data_y.shape[3])
            reshaped_preds = reshaped_preds[:, :min_d, :min_f, :min_t]
            data_y = data_y[:, :min_d, :min_f, :min_t] # Align labels too

    except (RuntimeError, ValueError) as e:
        return None, data_y, None # Indicate failure

    return reshaped_preds, data_y, None # Return aligned preds/labels, mask handled separately


def _calculate_mae_mse_batch(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor, # Boolean mask
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates MAE and MSE for a batch, handling masking and NaNs.

    Args:
        predictions: Predicted values [B, D, F, T].
        labels: True values [B, D, F, T].
        mask: Boolean mask for valid elements [B, D, F, T].
        device: Torch device.

    Returns:
        Tuple: (batch_mae_sum, batch_mse_sum, batch_channel_mae_sum,
                batch_channel_mse_sum, batch_channel_element_count)
               Sums are float64, count is int64. Returns tensors of zeros if inputs are invalid.
    """
    if predictions is None or labels is None or mask is None:
        # Determine num_features safely
        num_features = labels.shape[2] if labels is not None and len(labels.shape) > 2 else 0
        zero_f64 = torch.tensor(0.0, dtype=torch.float64, device=device)
        zero_i64 = torch.tensor(0, dtype=torch.int64, device=device)
        zero_f64_ch = torch.zeros(num_features, dtype=torch.float64, device=device)
        zero_i64_ch = torch.zeros(num_features, dtype=torch.int64, device=device)
        return zero_f64, zero_f64, zero_f64_ch, zero_f64_ch, zero_i64_ch

    abs_error = torch.abs(predictions - labels)
    sq_error = (predictions - labels)**2

    # Apply mask and handle potential NaNs/Infs resulting from 0 * Inf etc.
    masked_abs_error = torch.nan_to_num(abs_error * mask, nan=0.0, posinf=0.0, neginf=0.0)
    masked_sq_error = torch.nan_to_num(sq_error * mask, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate sums and counts
    batch_mae_sum = masked_abs_error.sum().to(torch.float64)
    batch_mse_sum = masked_sq_error.sum().to(torch.float64)
    batch_element_count = mask.sum() # Total valid elements in batch

    # Per-channel sums and counts
    batch_channel_mae_sum = masked_abs_error.sum(dim=(0, 1, 3)).to(torch.float64) # Sum over B, D_out, T_pred
    batch_channel_mse_sum = masked_sq_error.sum(dim=(0, 1, 3)).to(torch.float64)
    batch_channel_element_count = mask.sum(dim=(0, 1, 3)).to(torch.int64)

    return (batch_mae_sum, batch_mse_sum, batch_element_count,
            batch_channel_mae_sum, batch_channel_mse_sum, batch_channel_element_count)


def _calculate_pearson_batch(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor, # Boolean mask
    epsilon: float = 1e-9,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates Pearson Correlation for a batch, handling masking.

    Args:
        predictions: Predicted values [B, D, F, T].
        labels: True values [B, D, F, T].
        mask: Boolean mask for valid elements [B, D, F, T].
        epsilon: Small value for numerical stability.
        device: Torch device.

    Returns:
        Tuple: (batch_corr_sum, batch_corr_count,
                batch_channel_corr_sum, batch_channel_corr_count)
               Sums are float64, counts are int64. Returns tensors of zeros if inputs are invalid.
    """
    if predictions is None or labels is None or mask is None or predictions.numel() == 0 or labels.numel() == 0:
        num_features = labels.shape[2] if labels is not None and len(labels.shape) > 2 else 0
        zero_f64 = torch.tensor(0.0, dtype=torch.float64, device=device)
        zero_i64 = torch.tensor(0, dtype=torch.int64, device=device)
        zero_f64_ch = torch.zeros(num_features, dtype=torch.float64, device=device)
        zero_i64_ch = torch.zeros(num_features, dtype=torch.int64, device=device)
        return zero_f64, zero_i64, zero_f64_ch, zero_i64_ch

    b, d, f, t = predictions.shape
    time_dim = d * t

    # Reshape for correlation: [B, D, F, T] -> [B, F, D * T]
    preds_flat = predictions.permute(0, 2, 1, 3).reshape(b, f, time_dim)
    labels_flat = labels.permute(0, 2, 1, 3).reshape(b, f, time_dim)
    mask_flat = mask.permute(0, 2, 1, 3).reshape(b, f, time_dim) # Boolean mask

    # Clean potential NaNs in labels *outside* the mask
    labels_flat = torch.where(mask_flat, labels_flat, torch.zeros_like(labels_flat))
    # Clean predictions as well (though ideally they shouldn't have NaNs if model is stable)
    preds_flat = torch.where(mask_flat, preds_flat, torch.zeros_like(preds_flat))

    # Calculate needed stats only over valid time points
    n_valid = mask_flat.sum(dim=2, dtype=torch.float64) # Shape (B, F)

    # Mask out invalid time steps (redundant due to `torch.where` above, but safe)
    preds_masked = preds_flat * mask_flat
    labels_masked = labels_flat * mask_flat

    # Calculate means over valid time points
    sum_x = preds_masked.sum(dim=2)
    sum_y = labels_masked.sum(dim=2)

    mean_x = torch.where(n_valid > 0, sum_x / n_valid, torch.zeros_like(sum_x)) # Shape (B, F)
    mean_y = torch.where(n_valid > 0, sum_y / n_valid, torch.zeros_like(sum_y))

    # Calculate centered values (only valid time steps contribute)
    centered_x = (preds_flat - mean_x.unsqueeze(2)) * mask_flat
    centered_y = (labels_flat - mean_y.unsqueeze(2)) * mask_flat

    # Calculate terms for correlation formula
    numerator = (centered_x * centered_y).sum(dim=2) # Sum over time: (B, F)
    denom_x_sq = (centered_x**2).sum(dim=2) # (B, F)
    denom_y_sq = (centered_y**2).sum(dim=2) # (B, F)

    # Calculate correlation, handling potential division by zero/low variance
    denominator = torch.sqrt(denom_x_sq * denom_y_sq)
    correlation = torch.where(
        denominator > epsilon,
        numerator / (denominator + epsilon), # Add epsilon for safety
        torch.zeros_like(denominator) # Set corr to 0 if variance is near zero
    ) # Shape (B, F)

    # Mask for valid correlations (need at least 2 points and non-zero variance)
    valid_corr_mask = (n_valid >= 2) & (denom_x_sq > epsilon) & (denom_y_sq > epsilon) # Shape (B, F)

    # Apply mask to correlations (invalid correlations become 0)
    valid_correlations = correlation * valid_corr_mask

    # Update running sums and counts for correlation
    batch_corr_sum = valid_correlations.sum().to(torch.float64)
    batch_corr_count = valid_corr_mask.sum().to(torch.int64) # Count of valid (B, F) pairs

    batch_channel_corr_sum = valid_correlations.sum(dim=0).to(torch.float64) # Sum over B
    batch_channel_corr_count = valid_corr_mask.sum(dim=0).to(torch.int64) # Sum over B

    return batch_corr_sum, batch_corr_count, batch_channel_corr_sum, batch_channel_corr_count


def evaluate_forecast(
    model: Union[AutoencoderLSTM, RevInAutoencoderLSTM],
    dataframe: pd.DataFrame,
    root_dir: str,
    split: str = "5-2d",
    batch_size: int = 16,
    include_mask: bool = True,
    feature_indices: Optional[List[int]] = None,
    feature_stats: Optional[Dict] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Dict:
    """
    Evaluate forecasting performance of a model using running metrics.
    
    Args:
        model: The LSTM model (AutoencoderLSTM or RevInAutoencoderLSTM)
        dataframe: DataFrame with MHC dataset metadata
        root_dir: Root directory for data files
        split: String defining the forecast split (e.g., "5-2d")
        batch_size: Batch size for evaluation
        include_mask: Whether to include masks (required for proper evaluation)
        feature_indices: Optional list of feature indices to select
        feature_stats: Optional dictionary for feature standardization
        device: Device to run evaluation on
    
    Returns:
        Dictionary with MAE, MSE, Pearson Correlation metrics per channel and overall.
    """
    # Parse the split string
    sequence_len, prediction_horizon, overlap = parse_forecast_split(split)
    
    # Create the forecast dataset
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
    
    # Use the dataset-based evaluation
    return evaluate_forecast_dataset(
        model=model,
        dataset=forecast_dataset,
        batch_size=batch_size,
        device=device
    )


def evaluate_forecast_dataset(
    model: Union[AutoencoderLSTM, RevInAutoencoderLSTM],
    dataset,  # ForecastingEvaluationDataset or compatible
    batch_size: int = 16,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    dataloader = None  # Optional pre-configured dataloader
) -> Dict:
    """
    Evaluate forecasting performance of a model using a pre-created dataset.
    
    Args:
        model: The LSTM model (AutoencoderLSTM or RevInAutoencoderLSTM)
        dataset: Pre-configured dataset with forecasting data
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        dataloader: Optional pre-configured DataLoader (if provided, batch_size is ignored)
    
    Returns:
        Dictionary with MAE, MSE, Pearson Correlation metrics per channel and overall.
    """
    # Create data loader if not provided
    if dataloader is None:
        # Use default collate, assuming dataset handles problematic samples or they are filtered out
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            # collate_fn can be added here if needed, e.g., custom_collate_fn
        )
    
    # Ensure model is in evaluation mode
    model.to(device)
    model.eval()
    
    # Determine number of features and prediction horizon
    if len(dataset) > 0:
        sample = dataset[0] # Get a sample to infer dimensions
        # Extract prediction_horizon from the dataset's sample or attributes
        if hasattr(dataset, 'prediction_horizon'):
            prediction_horizon = dataset.prediction_horizon
        elif 'data_y' in sample and len(sample['data_y'].shape) >= 3:
             # Infer from shape if not directly available
             time_dim_index = -1 # Usually the last dimension
             prediction_horizon = sample['data_y'].shape[time_dim_index]
        else:
            prediction_horizon = 2880 # Fallback (e.g., 2 days in minutes)
            print("Warning: Could not determine prediction_horizon from dataset. Using default.")
                
        # Get number of features
        if 'data_y' in sample and len(sample['data_y'].shape) >= 3:
            # Shape could be [D_out, F, T_pred] or [B, D_out, F, T_pred]
            # Feature dimension is typically the second-to-last or third-to-last
            if len(sample['data_y'].shape) == 4: # Assume [B, D_out, F, T_pred]
                num_features = sample['data_y'].shape[2]
            elif len(sample['data_y'].shape) == 3: # Assume [D_out, F, T_pred]
                num_features = sample['data_y'].shape[1]
            else:
                num_features = 24 # Fallback
                print("Warning: Unexpected data_y shape. Defaulting to 24 features.")
        elif 'data_x' in sample and len(sample['data_x'].shape) >= 3: # Try data_x
             if len(sample['data_x'].shape) == 4: # Assume [B, D_in, F, T_in]
                num_features = sample['data_x'].shape[2]
             elif len(sample['data_x'].shape) == 3: # Assume [D_in, F, T_in]
                num_features = sample['data_x'].shape[1]
             else:
                num_features = 24 # Fallback
        else:
            num_features = 24  # Fallback if structure doesn't match expectations
            print("Warning: Could not determine num_features from dataset sample. Using default.")
    else:
        # Handle empty dataset case explicitly
        num_features = 0
        prediction_horizon = 0 # No horizon if no data
        print("Warning: Dataset is empty. Metrics will be NaN.")

    # Initialize running metrics (use float64 for sums)
    # Check if num_features is valid before initializing channel tensors
    if num_features > 0:
        running_mae_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        running_mse_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        running_cor_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        
        running_element_count = torch.tensor(0, dtype=torch.int64, device=device) # Count for MAE/MSE
        running_cor_count = torch.tensor(0, dtype=torch.int64, device=device) # Count of valid (sample, feature) pairs for Corr

        channel_mae_sum = torch.zeros(num_features, dtype=torch.float64, device=device)
        channel_mse_sum = torch.zeros(num_features, dtype=torch.float64, device=device)
        channel_cor_sum = torch.zeros(num_features, dtype=torch.float64, device=device)
        
        channel_element_count = torch.zeros(num_features, dtype=torch.int64, device=device) # Count for MAE/MSE per channel
        channel_cor_count = torch.zeros(num_features, dtype=torch.int64, device=device) # Count for Corr per channel
    
        epsilon = 1e-9 # For numerical stability in correlation

        with torch.no_grad():
            for batch in dataloader:
                # Check if batch is empty (might happen with custom collate_fn)
                if not batch or 'data_x' not in batch or batch['data_x'].numel() == 0:
                    print("Skipping empty batch.")
                    continue
                    
                # Move data to device
                data_x = batch['data_x'].to(device) # Shape: [B, D_in, F, T_x]
                data_y = batch['data_y'].to(device) # Shape: [B, D_out, F, T_pred]
                
                # Get the mask if available AND consistently present in the collated batch
                mask_y = None
                # Check if 'mask_y' key exists in the collated batch dictionary
                if 'mask_y' in batch:
                    mask_y = batch['mask_y'] # Get the collated mask tensor
                    if mask_y is not None:
                        mask_y = mask_y.to(device)
                
                # Determine valid elements based on the successfully collated mask (if any)
                if mask_y is not None:
                    # Ensure mask is boolean for helper functions
                    valid_mask_elements = (mask_y > 0) 
                else:
                    # If no mask was collated (key missing or inconsistent before collation), assume all elements are valid
                    valid_mask_elements = torch.ones_like(data_y, dtype=torch.bool, device=device)

                # --- Predict Batch ---
                # Need prediction_horizon here!
                if prediction_horizon == 0 and len(dataset) > 0:
                    # Try to re-infer if it was missed initially but dataset is not empty
                    sample = dataset[0]
                    if hasattr(dataset, 'prediction_horizon'):
                        prediction_horizon = dataset.prediction_horizon
                    elif 'data_y' in sample and len(sample['data_y'].shape) >= 3:
                        time_dim_index = -1
                        prediction_horizon = sample['data_y'].shape[time_dim_index]
                    else:
                         prediction_horizon = data_y.shape[-1] # Infer from batch data
                
                predictions, data_y_aligned, _ = _predict_batch(
                    model, data_x, data_y, prediction_horizon, device
                )

                # Skip batch if prediction/reshaping failed
                if predictions is None:
                    print("Skipping batch due to prediction/reshaping error.")
                    continue
                    
                # Align mask if predictions/labels were aligned
                if predictions.shape != valid_mask_elements.shape:
                     print(f"Aligning mask shape from {valid_mask_elements.shape} to {predictions.shape}")
                     # Ensure batch dimensions match before attempting alignment
                     if predictions.shape[0] != valid_mask_elements.shape[0]:
                          print(f"ERROR: Batch size mismatch between predictions ({predictions.shape[0]}) and mask ({valid_mask_elements.shape[0]}). Cannot align mask.")
                          continue # Skip this problematic batch
                          
                     min_d = min(predictions.shape[1], valid_mask_elements.shape[1])
                     min_f = min(predictions.shape[2], valid_mask_elements.shape[2])
                     min_t = min(predictions.shape[3], valid_mask_elements.shape[3])

                     # Slice the mask to match the potentially smaller prediction shape
                     valid_mask_elements = valid_mask_elements[:, :min_d, :min_f, :min_t]
                     # Ensure data_y is also aligned if needed (though _predict_batch should handle data_y alignment)
                     if predictions.shape != data_y_aligned.shape:
                          data_y_aligned = data_y_aligned[:, :min_d, :min_f, :min_t]


                # --- Calculate MAE/MSE ---
                (batch_mae_sum, batch_mse_sum, batch_element_count,
                 batch_channel_mae_sum, batch_channel_mse_sum, batch_channel_element_count) = _calculate_mae_mse_batch(
                    predictions, data_y_aligned, valid_mask_elements, device
                )
                
                # Update running MAE/MSE sums and counts
                running_mae_sum += batch_mae_sum
                running_mse_sum += batch_mse_sum
                running_element_count += batch_element_count # Use the count returned by the function

                channel_mae_sum += batch_channel_mae_sum
                channel_mse_sum += batch_channel_mse_sum
                channel_element_count += batch_channel_element_count # Use the channel count

                # --- Calculate Pearson Correlation ---
                (batch_corr_sum, batch_corr_count,
                 batch_channel_corr_sum, batch_channel_corr_count) = _calculate_pearson_batch(
                    predictions, data_y_aligned, valid_mask_elements, epsilon, device
                )

                # Update running Correlation sums and counts
                running_cor_sum += batch_corr_sum
                running_cor_count += batch_corr_count

                channel_cor_sum += batch_channel_corr_sum
                channel_cor_count += batch_channel_corr_count

    # --- Calculate Final Metrics --- 
    results = {}
    
    # Handle case where num_features might be 0 (empty dataset)
    if num_features > 0 and running_element_count > 0:
        overall_mae = (running_mae_sum / running_element_count)
        overall_mse = (running_mse_sum / running_element_count)
    else:
        overall_mae = torch.tensor(float('nan'))
        overall_mse = torch.tensor(float('nan'))
        
    if num_features > 0 and running_cor_count > 0:
        overall_cor = (running_cor_sum / running_cor_count)
    else:
        overall_cor = torch.tensor(float('nan'))
    
    results['overall_mae'] = overall_mae.item()
    results['overall_mse'] = overall_mse.item()
    results['overall_pearson_corr'] = overall_cor.item()

    # Per-channel metrics
    if num_features > 0:
        channel_mae = torch.where(channel_element_count > 0, channel_mae_sum / channel_element_count, torch.tensor(float('nan'), device=device))
        channel_mse = torch.where(channel_element_count > 0, channel_mse_sum / channel_element_count, torch.tensor(float('nan'), device=device))
        channel_cor = torch.where(channel_cor_count > 0, channel_cor_sum / channel_cor_count, torch.tensor(float('nan'), device=device))
        results['channel_mae'] = channel_mae.cpu().tolist()
        results['channel_mse'] = channel_mse.cpu().tolist()
        results['channel_pearson_corr'] = channel_cor.cpu().tolist()
    else:
        # Return empty lists if dataset was empty
        results['channel_mae'] = []
        results['channel_mse'] = []
        results['channel_pearson_corr'] = []
    
    return results


def run_forecast_evaluation(
    model: Union[AutoencoderLSTM, RevInAutoencoderLSTM],
    dataframe: pd.DataFrame = None,
    root_dir: str = None,
    split: str = "5-2d",
    batch_size: int = 16,
    include_mask: bool = True,
    feature_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    feature_stats: Optional[Dict] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    dataset = None  # Optional pre-configured dataset
) -> Dict: # Return results dict
    """
    Run forecast evaluation and print formatted results.
    
    Args:
        model: The LSTM model to evaluate
        dataframe: DataFrame with MHC dataset metadata (not needed if dataset is provided)
        root_dir: Root directory for data files (not needed if dataset is provided)
        split: String defining the forecast split (e.g., "5-2d") (not needed if dataset is provided)
        batch_size: Batch size for evaluation
        include_mask: Whether to include masks (not needed if dataset is provided)
        feature_indices: Optional list of feature indices to select (not needed if dataset is provided)
        feature_names: Optional list of feature names for reporting
        feature_stats: Optional dictionary for feature standardization (not needed if dataset is provided)
        device: Device to run evaluation on
        dataset: Optional pre-configured ForecastingEvaluationDataset or compatible dataset
                (if provided, dataframe, root_dir, split, include_mask, feature_indices, and
                feature_stats parameters are ignored)
        
    Returns:
        Dictionary containing the evaluation results.
    """
    # Run evaluation
    if dataset is not None:
        # Use provided dataset
        results = evaluate_forecast_dataset(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device
        )
    else:
        # Create and use dataset from parameters
        if dataframe is None or root_dir is None:
            raise ValueError("If dataset is not provided, dataframe and root_dir must be specified")
            
        results = evaluate_forecast(
            model=model,
            dataframe=dataframe,
            root_dir=root_dir,
            split=split,
            batch_size=batch_size,
            include_mask=include_mask,
            feature_indices=feature_indices,
            feature_stats=feature_stats,
            device=device
        )
    
    # --- Print Overall Results ---
    print("\n--- Overall Metrics ---")
    print(f"MAE:            {results.get('overall_mae', float('nan')):.4f}")
    print(f"MSE:            {results.get('overall_mse', float('nan')):.4f}")
    print(f"Pearson Corr.:  {results.get('overall_pearson_corr', float('nan')):.4f}")
    
    # --- Print Per-Channel Results ---
    print("\n--- Per-Channel Metrics ---")
    channel_maes = results.get('channel_mae', [])
    channel_mses = results.get('channel_mse', [])
    channel_cors = results.get('channel_pearson_corr', [])
    num_channels = len(channel_maes)

    # Determine feature labels
    if feature_names and len(feature_names) == num_channels:
        feature_labels = feature_names
    elif feature_indices and len(feature_indices) == num_channels:
        feature_labels = [f"Feature {i}" for i in feature_indices]
    else:
        feature_labels = [f"Feature {i}" for i in range(num_channels)]
    
    # Print table header
    header = f"{'Feature':<20} {'MAE':<10} {'MSE':<10} {'Corr':<10}"
    print(header)
    print("-" * len(header))
    
    # Print each channel's metrics
    for i in range(num_channels):
        label = feature_labels[i]
        mae = channel_maes[i] if i < len(channel_maes) else float('nan')
        mse = channel_mses[i] if i < len(channel_mses) else float('nan')
        corr = channel_cors[i] if i < len(channel_cors) else float('nan')
        
        mae_str = f"{mae:.4f}" if not np.isnan(mae) else "NaN"
        mse_str = f"{mse:.4f}" if not np.isnan(mse) else "NaN"
        corr_str = f"{corr:.4f}" if not np.isnan(corr) else "NaN"
        
        print(f"{label:<20} {mae_str:<10} {mse_str:<10} {corr_str:<10}")
        
    print("-" * len(header))
    
    return results # Return the results dictionary


# Usage example:
if __name__ == "__main__":
    import argparse
    from models.lstm import load_checkpoint, RevInAutoencoderLSTM
    from torch_dataset import ForecastingEvaluationDataset
    
    parser = argparse.ArgumentParser(description='Evaluate LSTM model forecasting')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of MHC data')
    parser.add_argument('--split', type=str, default='5-2d', help='Forecast split (e.g., "5-2d")')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--no_mask', action='store_true', help='Disable using mask during evaluation')
    parser.add_argument('--use_dataset_api', action='store_true', help='Use the dataset-based API instead of dataframe')
    args = parser.parse_args()
    
    # Load the CSV file
    try:
        df = pd.read_csv(args.data_csv)
        if 'split' in df.columns:
             test_df = df[df['split'] == 'test'].reset_index(drop=True)
             if len(test_df) == 0:
                 test_df = df
        else:
             test_df = df
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        exit(1)
    
    # Create model (will be filled with checkpoint weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_instance = RevInAutoencoderLSTM(
        hidden_size=128,
        encoding_dim=100,
        num_layers=2
    ) 
    
    # Load checkpoint
    try:
        model, optimizer_state, epoch, loss = load_checkpoint(args.checkpoint, model_instance, None, device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
    
    # Feature names
    num_features_loaded = model.num_features
    feature_names = [f"Feature_{i}" for i in range(num_features_loaded)]
    
    # Run evaluation
    if args.use_dataset_api:
        # Parse the split string for dataset creation
        sequence_len, prediction_horizon, overlap = parse_forecast_split(args.split)
        
        # Create the dataset directly
        print(f"Creating evaluation dataset with split {args.split}...")
        dataset = ForecastingEvaluationDataset(
            dataframe=test_df,
            root_dir=args.data_root,
            sequence_len=sequence_len,
            prediction_horizon=prediction_horizon,
            overlap=overlap,
            include_mask=(not args.no_mask)
        )
        
        print(f"Using dataset-based API with {len(dataset)} samples")
        evaluation_results = run_forecast_evaluation(
            model=model,
            batch_size=args.batch_size,
            feature_names=feature_names,
            device=device,
            dataset=dataset  # Pass the pre-configured dataset
        )
    else:
        # Use the original API
        print("Using dataframe-based API")
        evaluation_results = run_forecast_evaluation(
            model=model,
            dataframe=test_df,
            root_dir=args.data_root,
            split=args.split,
            batch_size=args.batch_size,
            include_mask=(not args.no_mask),
            feature_names=feature_names,
            device=device
        )