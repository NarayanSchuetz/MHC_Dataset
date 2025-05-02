import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from torch_dataset import ForecastingEvaluationDataset
from models.forecasting_lstm import ForecastingLSTM, RevInForecastingLSTM
from dataset_postprocessors import CustomMaskPostprocessor, HeartRateInterpolationPostprocessor


def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using a trained ForecastingLSTM model')
    
    # Data paths
    parser.add_argument('--dataset_path', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset_out/splits/test_dataset.parquet",
                        help='Path to the dataset parquet file for inference')
    parser.add_argument('--root_dir', type=str, 
                        default="/scratch/groups/euan/mhc/mhc_dataset",
                        help='Root directory containing the MHC dataset')
    parser.add_argument('--standardization_path', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset_out/standardization_params.csv",
                        help='Path to standardization parameters CSV')
    
    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--use_revin', action='store_true', 
                        help='Whether the model uses RevIN (RevInForecastingLSTM)')
    parser.add_argument('--num_features', type=int, default=6, 
                        help='Number of features per minute used in the model')
    
    # Forecasting parameters
    parser.add_argument('--sequence_len_days', type=int, default=5, 
                        help='Number of days in input sequence (context)')
    parser.add_argument('--prediction_horizon_days', type=int, default=2, 
                        help='Number of days to predict (forecast horizon)')
    parser.add_argument('--overlap_days', type=int, default=0, 
                        help='Overlap between input and prediction in days')
    
    # Output parameters
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='forecasting_results',
                        help='Directory to save visualization plots')
    
    args = parser.parse_args()
    return args


def load_model(checkpoint_path, use_revin=False, num_features=6, device='cpu'):
    """
    Load a trained ForecastingLSTM model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        use_revin: Whether the model uses RevIN
        num_features: Number of features the model was trained with
        device: Device to load the model on
        
    Returns:
        The loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model of the appropriate type
    if use_revin:
        model = RevInForecastingLSTM(num_features=num_features)
    else:
        model = ForecastingLSTM(num_features=num_features)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Training loss: {checkpoint['train_loss']:.4f}, Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


def visualize_predictions(model, dataset, sample_indices, output_dir, device='cpu'):
    """
    Make predictions and visualize them for selected samples.
    
    Args:
        model: The trained forecasting model
        dataset: ForecastingEvaluationDataset to get samples from
        sample_indices: List of indices of samples to visualize
        output_dir: Directory to save the visualization plots
        device: Device to run inference on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_idx in sample_indices:
        # Get sample
        sample = dataset[sample_idx]
        
        # Prepare batch for model
        batch = {
            'data_x': sample['data_x'].unsqueeze(0).to(device),  # Add batch dimension
            'data_y': sample['data_y'].unsqueeze(0).to(device),  # Add batch dimension
            'mask_x': sample['mask_x'].unsqueeze(0).to(device) if 'mask_x' in sample else None,
            'mask_y': sample['mask_y'].unsqueeze(0).to(device) if 'mask_y' in sample else None
        }
        
        # Generate predictions
        with torch.no_grad():
            output = model(batch)
        
        # Get predictions and ground truth
        predictions = output['sequence_output'][0].cpu().numpy()  # Remove batch dimension
        ground_truth = output['target_segments'][0].cpu().numpy()
        
        # Get sample metadata
        health_code = sample['metadata']['healthCode']
        time_range = sample['metadata']['time_range']
        
        # Create a multi-panel figure to show multiple features
        num_features = model.num_original_features
        fig, axes = plt.subplots(num_features, 1, figsize=(12, 3*num_features))
        
        if num_features == 1:
            axes = [axes]  # Make sure axes is a list for consistency
        
        for feature_idx in range(num_features):
            ax = axes[feature_idx]
            
            # Plot the first few segments for this feature
            num_segments_to_plot = min(5, predictions.shape[0])
            
            for segment_idx in range(num_segments_to_plot):
                # Extract values for this segment and feature
                # Each segment is of shape (features_per_segment,)
                segment_start = segment_idx * model.minutes_per_segment * num_features
                segment_end = segment_start + model.minutes_per_segment
                
                # Extract the values for this feature from the segment
                feature_start = feature_idx * model.minutes_per_segment
                feature_end = feature_start + model.minutes_per_segment
                
                pred_values = predictions[segment_idx, feature_start:feature_end]
                truth_values = ground_truth[segment_idx, feature_start:feature_end]
                
                # Plot this segment
                ax.plot(pred_values, label=f'Predicted Segment {segment_idx}' if feature_idx == 0 else None)
                ax.plot(truth_values, '--', label=f'Ground Truth Segment {segment_idx}' if feature_idx == 0 else None)
            
            ax.set_title(f'Feature {feature_idx}')
            ax.set_xlabel('Minutes within segment')
            ax.set_ylabel('Value')
            ax.grid(True)
        
        # Add a single legend at the top
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=num_segments_to_plot*2)
        
        # Add metadata as suptitle
        fig.suptitle(f'Sample {sample_idx}: {health_code}, {time_range}')
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        output_path = output_dir / f'sample_{sample_idx}_forecast.png'
        fig.savefig(output_path)
        plt.close(fig)
        
        print(f"Saved visualization for sample {sample_idx} to {output_path}")


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert days to minutes
    minutes_per_day = 24 * 60  # 1440 minutes per day
    sequence_len = args.sequence_len_days * minutes_per_day
    prediction_horizon = args.prediction_horizon_days * minutes_per_day
    overlap = args.overlap_days * minutes_per_day
    
    # Load standardization parameters
    standardization_df = pd.read_csv(args.standardization_path)
    scaler_stats = {}
    for f_idx, row in standardization_df.iloc[:args.num_features].iterrows():
        scaler_stats[f_idx] = (row["mean"], row["std_dev"])
    
    # Load the dataset
    dataset_df = pd.read_parquet(args.dataset_path)
    dataset_df["file_uris"] = dataset_df["file_uris"].apply(eval)
    print(f"Loaded dataset with {len(dataset_df)} samples")
    
    # Define postprocessors
    p0 = CustomMaskPostprocessor(heart_rate_original_index=5, expected_raw_features=6, consecutive_zero_threshold=30)
    p1 = HeartRateInterpolationPostprocessor(heart_rate_original_index=5, expected_raw_features=6, hr_gap_threshold=30)
    
    # Create the forecasting dataset
    forecast_dataset = ForecastingEvaluationDataset(
        dataframe=dataset_df,
        root_dir=args.root_dir,
        sequence_len=sequence_len,
        prediction_horizon=prediction_horizon,
        overlap=overlap,
        include_mask=True,
        feature_indices=list(range(args.num_features)),
        feature_stats=scaler_stats,
        postprocessors=[p0, p1]
    )
    
    print(f"Created forecasting dataset with {len(forecast_dataset)} samples")
    
    # Load model
    model = load_model(args.checkpoint_path, args.use_revin, args.num_features, device)
    
    # Generate sample indices for visualization
    num_samples = min(args.num_samples, len(forecast_dataset))
    sample_indices = list(range(num_samples))
    
    # Visualize predictions
    visualize_predictions(model, forecast_dataset, sample_indices, args.output_dir, device)
    
    # Calculate overall metrics
    print("Calculating metrics on the entire dataset...")
    metrics = model.evaluate_forecast(
        dataframe=dataset_df,
        root_dir=args.root_dir,
        sequence_len=sequence_len,
        prediction_horizon=prediction_horizon,
        overlap=overlap,
        batch_size=64,
        include_mask=True,
        feature_indices=list(range(args.num_features)),
        feature_stats=scaler_stats,
        device=device
    )
    
    print(f"Overall Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    print(f"Saved visualization plots to {args.output_dir}")


if __name__ == "__main__":
    main() 