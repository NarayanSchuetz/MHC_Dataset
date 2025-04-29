import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
import pandas as pd

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
    Evaluate forecasting performance of a model.
    
    Args:
        model: The LSTM model (AutoencoderLSTM or RevInAutoencoderLSTM)
        dataframe: DataFrame with MHC dataset metadata
        root_dir: Root directory for data files
        split: String defining the forecast split (e.g., "5-2d" for 5 days input, 2 days forecast)
        batch_size: Batch size for evaluation
        include_mask: Whether to include masks (required for proper evaluation)
        feature_indices: Optional list of feature indices to select
        feature_stats: Optional dictionary for feature standardization
        device: Device to run evaluation on
    
    Returns:
        Dictionary with MAE metrics per channel and overall
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
    
    # Create data loader
    dataloader = DataLoader(
        forecast_dataset,
        batch_size=batch_size,
        shuffle=False  # Don't shuffle for evaluation
    )
    
    # Ensure model is in evaluation mode
    model.to(device)
    model.eval()
    
    # Initialize metrics
    all_errors = []
    all_masks = []
    
    # Track per-channel errors
    num_features = len(feature_indices) if feature_indices is not None else 24
    channel_errors = [[] for _ in range(num_features)]
    channel_masks = [[] for _ in range(num_features)]
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            data_x = batch['data_x'].to(device)
            data_y = batch['data_y'].to(device)
            
            # Get the mask if available
            mask_y = batch.get('mask_y')
            if mask_y is not None:
                mask_y = mask_y.to(device)
            
            # Reshape data for prediction
            batch_size, num_days, num_features, time_steps_x = data_x.shape
            
            # Reshape to [B, num_days*segments_per_day, num_features*minutes_per_segment]
            # For the LSTM model's input format
            segments_per_day = model.segments_per_day
            minutes_per_segment = model.minutes_per_segment
            
            # Calculate how many segments we have in the input
            input_segments = (num_days * time_steps_x) // (minutes_per_segment)
            
            # Reshape data_x to match model's expected input shape for segments
            reshaped_data_x = data_x.view(batch_size, num_days, num_features, input_segments, minutes_per_segment)
            reshaped_data_x = reshaped_data_x.permute(0, 1, 3, 2, 4)  # -> [B, num_days, segments, num_features, minutes_per_segment]
            reshaped_data_x = reshaped_data_x.reshape(batch_size, input_segments, num_features * minutes_per_segment)
            
            # Calculate how many segments we need to predict
            # Convert prediction_horizon (in minutes) to number of segments
            target_segments = prediction_horizon // minutes_per_segment
            
            # Generate predictions
            predictions = model.predict_future(reshaped_data_x, steps=target_segments)
            
            # Reshape predictions to match data_y shape for comparison
            # Predictions shape: [B, target_segments, num_features*minutes_per_segment]
            pred_days = target_segments // segments_per_day
            reshaped_preds = predictions.view(batch_size, target_segments, num_features, minutes_per_segment)
            reshaped_preds = reshaped_preds.permute(0, 1, 3, 2)  # -> [B, target_segments, minutes_per_segment, num_features]
            reshaped_preds = reshaped_preds.reshape(batch_size, pred_days, num_features, prediction_horizon)
            
            # Calculate absolute error
            abs_error = torch.abs(reshaped_preds - data_y)
            
            # Apply mask if available
            if mask_y is not None:
                # For calculating overall MAE
                masked_abs_error = abs_error * mask_y
                valid_mask = (mask_y > 0)
                
                # For per-channel MAE
                for f in range(num_features):
                    channel_error = abs_error[:, :, f, :]
                    channel_mask = mask_y[:, :, f, :]
                    
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
                    channel_error = abs_error[:, :, f, :]
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


def run_forecast_evaluation(
    model: Union[AutoencoderLSTM, RevInAutoencoderLSTM],
    dataframe: pd.DataFrame,
    root_dir: str,
    split: str = "5-2d",
    batch_size: int = 16,
    include_mask: bool = True,
    feature_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    feature_stats: Optional[Dict] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> None:
    """
    Run forecast evaluation and print formatted results.
    
    Args:
        model: The LSTM model to evaluate
        dataframe: DataFrame with MHC dataset metadata
        root_dir: Root directory for data files
        split: String defining the forecast split (e.g., "5-2d")
        batch_size: Batch size for evaluation
        include_mask: Whether to include masks
        feature_indices: Optional list of feature indices to select
        feature_names: Optional list of feature names for reporting
        feature_stats: Optional dictionary for feature standardization
        device: Device to run evaluation on
    """
    print(f"Running forecast evaluation with split: {split}")
    
    # Run evaluation
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


# Usage example:
if __name__ == "__main__":
    import argparse
    from models.lstm import load_checkpoint, RevInAutoencoderLSTM
    
    parser = argparse.ArgumentParser(description='Evaluate LSTM model forecasting')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of MHC data')
    parser.add_argument('--split', type=str, default='5-2d', help='Forecast split (e.g., "5-2d")')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Load the CSV file
    try:
        df = pd.read_csv(args.data_csv)
        print(f"Loaded data CSV with {len(df)} entries")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)
    
    # Create model (will be filled with checkpoint weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example model initialization - parameters will be overwritten by checkpoint
    model = RevInAutoencoderLSTM(
        hidden_size=128,
        encoding_dim=100,
        num_layers=2
    )
    
    # Load checkpoint
    try:
        model, _ = load_checkpoint(args.checkpoint, model, device)
        print(f"Successfully loaded model from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
    
    # Example feature names for better reporting
    feature_names = [
        "Feature 0", "Feature 1", "Feature 2", "Feature 3",
        "Feature 4", "Feature 5", "Feature 6", "Feature 7",
        "Feature 8", "Feature 9", "Feature 10", "Feature 11",
        "Feature 12", "Feature 13", "Feature 14", "Feature 15",
        "Feature 16", "Feature 17", "Feature 18", "Feature 19",
        "Feature 20", "Feature 21", "Feature 22", "Feature 23"
    ]
    
    # Run evaluation
    run_forecast_evaluation(
        model=model,
        dataframe=df,
        root_dir=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        include_mask=True,
        feature_names=feature_names
    ) 