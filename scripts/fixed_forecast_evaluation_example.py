#!/usr/bin/env python3
"""
Example script demonstrating a simple wrapper approach to handle problematic samples.

This approach uses a dataset wrapper to avoid None values that cause collation errors.
"""

import sys
import os
import argparse
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add the src directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.append(src_dir)

from models.lstm import load_checkpoint, RevInAutoencoderLSTM
from models.evaluation import run_forecast_evaluation, parse_forecast_split
from torch_dataset import ForecastingEvaluationDataset


class FilteredDatasetWrapper(Dataset):
    """Simple wrapper for datasets that filters out problematic samples."""
    
    def __init__(self, dataset):
        """Initialize with the base dataset."""
        self.dataset = dataset
        self.valid_indices = []
        
        # Pre-filter indices
        print("Filtering dataset...")
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                if sample is not None and isinstance(sample, dict) and 'data_x' in sample and 'data_y' in sample:
                    # Additional checks for tensor validity
                    if (sample['data_x'] is not None and 
                        sample['data_y'] is not None and 
                        not torch.isnan(sample['data_x']).any() and 
                        not torch.isnan(sample['data_y']).any()):
                        self.valid_indices.append(i)
            except Exception as e:
                print(f"Error with sample {i}: {e}")
                continue
        
        print(f"Filtered dataset contains {len(self.valid_indices)} valid samples out of {len(dataset)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get item from the base dataset using our filtered index mapping."""
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for FilteredDatasetWrapper with {len(self.valid_indices)} items")
        
        original_idx = self.valid_indices[idx]
        return self.dataset[original_idx]


def main():
    parser = argparse.ArgumentParser(description='Example of dataset-based forecast evaluation with filtering')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of data')
    parser.add_argument('--split', type=str, default='5-2d', help='Forecast split (e.g., "5-2d")')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--no_mask', action='store_true', help='Disable using mask during evaluation')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the CSV file and filter it
    try:
        df = pd.read_csv(args.data_csv)
        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].reset_index(drop=True)
            if len(test_df) == 0:
                print("Warning: No test samples found, using entire dataset")
                test_df = df
        else:
            test_df = df
        
        print(f"Loaded dataset with {len(test_df)} samples")
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        exit(1)
    
    # Create model and load checkpoint
    model_instance = RevInAutoencoderLSTM(
        hidden_size=128,
        encoding_dim=100,
        num_layers=2
    )
    
    try:
        model, optimizer_state, epoch, loss = load_checkpoint(args.checkpoint, model_instance, None, device)
        print(f"Loaded model from checkpoint (epoch {epoch})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
    
    # Generate feature names
    num_features_loaded = model.num_features
    feature_names = [f"Feature_{i}" for i in range(num_features_loaded)]
    
    # Parse the split string
    sequence_len, prediction_horizon, overlap = parse_forecast_split(args.split)
    print(f"Split configuration: input={sequence_len} minutes, forecast={prediction_horizon} minutes, overlap={overlap} minutes")
    
    # ---------------------------------
    # Create the dataset 
    # ---------------------------------
    forecast_dataset = ForecastingEvaluationDataset(
        dataframe=test_df,
        root_dir=args.data_root,
        sequence_len=sequence_len,
        prediction_horizon=prediction_horizon,
        overlap=overlap,
        include_mask=(not args.no_mask)
    )
    
    print(f"Created dataset with {len(forecast_dataset)} total samples")
    
    # ---------------------------------
    # Wrap the dataset with our filtered wrapper
    # ---------------------------------
    filtered_dataset = FilteredDatasetWrapper(forecast_dataset)
    
    # Check if we have any samples after filtering
    if len(filtered_dataset) == 0:
        print("ERROR: No valid samples after filtering. Check your dataset.")
        exit(1)
    
    # ---------------------------------
    # Run the evaluation using the filtered dataset
    # ---------------------------------
    print("\nRunning evaluation with filtered dataset...")
    evaluation_results = run_forecast_evaluation(
        model=model,
        batch_size=args.batch_size,
        feature_names=feature_names,
        device=device,
        dataset=filtered_dataset  # Pass the filtered dataset
    )
    
    # Print results summary 
    print("\n--- Results Summary ---")
    print(f"MAE:            {evaluation_results.get('overall_mae', float('nan')):.4f}")
    print(f"MSE:            {evaluation_results.get('overall_mse', float('nan')):.4f}")
    print(f"Pearson Corr.:  {evaluation_results.get('overall_pearson_corr', float('nan')):.4f}")
    
    print("\nNote: These results are based on the filtered dataset with valid samples only.")

if __name__ == "__main__":
    main() 