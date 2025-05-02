#!/usr/bin/env python3
"""
Example script demonstrating how to use the dataset-based forecast evaluation.

This approach allows you to:
1. Create and configure your dataset once
2. Use the same dataset for multiple evaluations
3. Create custom datasets or transforms before evaluation
4. Decouple dataset creation from evaluation logic

Usage:
    python forecast_evaluation_example.py --checkpoint /path/to/model.pt --data_csv /path/to/metadata.csv --data_root /path/to/data
"""

import sys
import os
import argparse
import pandas as pd
import torch
from pathlib import Path

# Add the src directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.append(src_dir)

from models.lstm import load_checkpoint, RevInAutoencoderLSTM
from models.evaluation import run_forecast_evaluation, parse_forecast_split
from torch_dataset import ForecastingEvaluationDataset

def main():
    parser = argparse.ArgumentParser(description='Example of dataset-based forecast evaluation')
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
    # Create the dataset (this is the key part that's now decoupled)
    # ---------------------------------
    forecast_dataset = ForecastingEvaluationDataset(
        dataframe=test_df,
        root_dir=args.data_root,
        sequence_len=sequence_len,
        prediction_horizon=prediction_horizon,
        overlap=overlap,
        include_mask=(not args.no_mask)
    )
    
    print(f"Created dataset with {len(forecast_dataset)} samples")
    
    # ---------------------------------
    # Run the evaluation using the dataset-based API
    # ---------------------------------
    print("\nRunning evaluation...")
    evaluation_results = run_forecast_evaluation(
        model=model,
        batch_size=args.batch_size,
        feature_names=feature_names,
        device=device,
        dataset=forecast_dataset  # Pass the pre-configured dataset
    )
    
    # Example of how you could process the results
    print("\nProcessing results...")
    mae = evaluation_results['overall_mae']
    mse = evaluation_results['overall_mse']
    corr = evaluation_results['overall_pearson_corr']
    
    # You could log these to a file, send to a monitoring system, etc.
    print(f"Summary: MAE={mae:.4f}, MSE={mse:.4f}, Correlation={corr:.4f}")
    
    # Example of getting per-feature results
    for i, feature_name in enumerate(feature_names):
        if i < len(evaluation_results['channel_mae']):
            feature_mae = evaluation_results['channel_mae'][i]
            print(f"  {feature_name} MAE: {feature_mae:.4f}")

if __name__ == "__main__":
    main() 