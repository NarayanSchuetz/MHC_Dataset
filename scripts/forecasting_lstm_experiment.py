import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import wandb
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F

# Import forecasting models and datasets
from models.forecasting_lstm import ForecastingLSTM, RevInForecastingLSTM
from torch_dataset import ForecastingEvaluationDataset # Using flattened for potentially easier sequence handling
from dataset_postprocessors import CustomMaskPostprocessor, HeartRateInterpolationPostprocessor # Keep postprocessors if needed


def parse_args():
    parser = argparse.ArgumentParser(description='Train ForecastingLSTM on MHC dataset with wandb tracking')
    
    # Data paths
    parser.add_argument('--dataset_path', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset_out/splits/train_final_dataset.parquet",
                        help='Path to the training dataset parquet file')
    parser.add_argument('--val_dataset_path', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset_out/splits/validation_dataset.parquet",
                        help='Path to the validation dataset parquet file')
    parser.add_argument('--root_dir', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset",
                        help='Root directory containing the MHC dataset')
    parser.add_argument('--standardization_path', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset_out/standardization_params.csv",
                        help='Path to standardization parameters CSV')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training') # Smaller default may be needed
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Learning rate scheduler parameters
    parser.add_argument('--use_lr_scheduler', action='store_true', 
                        help='Use cosine annealing LR scheduler with warmup')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of epochs for linear warmup')
    parser.add_argument('--lr_cycles', type=int, default=1, 
                        help='Number of cosine annealing cycles over the total epochs')
    
    # Forecasting Dataset parameters
    parser.add_argument('--input_seq_len_days', type=int, default=7, help='Length of the input sequence in days')
    parser.add_argument('--output_seq_len_days', type=int, default=1, help='Length of the output sequence (forecast horizon) in days')
    parser.add_argument('--time_points_per_day', type=int, default=1440, help='Number of time points (minutes) per day')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=360, help='Hidden size of LSTM')
    parser.add_argument('--encoding_dim', type=int, default=180, help='Dimension of encoded segments')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--num_features', type=int, default=6, # Adjusted default based on lstm_experiment.py
                        help='Number of features per minute to use (must match dataset)')
    parser.add_argument('--l2_weight', type=float, default=0.0,
                        help='Weight for L2 regularization (0.0 means disabled)')
    
    # RevIN parameters
    parser.add_argument('--use_revin', action='store_true', 
                        help='Use RevInForecastingLSTM instead of standard ForecastingLSTM')
    parser.add_argument('--revin_affine', action='store_true', 
                        help='Use learnable affine parameters in RevIN')
    parser.add_argument('--revin_subtract_last', action='store_true', 
                        help='Subtract last element instead of mean in RevIN')
    
    # Loss parameters
    parser.add_argument('--use_masked_loss', action='store_true',
                        help='Use mask to ignore missing values in loss calculation')

    # Experiment tracking
    parser.add_argument('--wandb_project', type=str, default='mhc-forecasting-lstm', 
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help='WandB entity name')
    parser.add_argument('--run_name', type=str, default=None, 
                        help='Name for this run (default: auto-generated based on timestamp)')
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/forecasting', 
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Auto-generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = "revin_forecast" if args.use_revin else "forecast_lstm"
        args.run_name = f"{model_type}_{args.num_layers}layer_h{args.hidden_size}_in{args.input_seq_len_days}d_out{args.output_seq_len_days}d_{timestamp}"
    
    # Calculate sequence lengths in time points
    args.input_sequence_len = args.input_seq_len_days * args.time_points_per_day
    args.output_sequence_len = args.output_seq_len_days * args.time_points_per_day
    
    return args

# Note: log_sample_prediction needs significant adaptation for forecasting
# It should plot input context, target forecast, and predicted forecast
# This is a simplified placeholder version
def log_sample_prediction(model, val_dataset, device, args, sample_idx=0, feature_idx=0):
    """Log prediction visualization to wandb (simplified for forecasting)"""
    with torch.no_grad():
        model.eval()
        # Get a sample
        sample = val_dataset[sample_idx]
        
        # Prepare batch format for model's forward pass
        batch = {
            'data_x': sample['data_x'].unsqueeze(0).to(device),
            'data_y': sample['data_y'].unsqueeze(0).to(device),
        }
        if args.use_masked_loss and 'mask_x' in sample:
             batch['mask_x'] = sample['mask_x'].unsqueeze(0).to(device)
             batch['mask_y'] = sample['mask_y'].unsqueeze(0).to(device)

        # Forward pass
        output = model(batch, return_predictions=False) # No label predictions here
        
        # Get predictions and targets (already in segment format)
        # Shape: (1, num_target_segments, features_per_segment)
        predicted_segments = output['sequence_output'][0].cpu() 
        target_segments = output['target_segments'][0].cpu()

        # Reshape segments back to time series for plotting
        # (num_target_segments, features * minutes_per_segment) -> (num_target_segments * minutes_per_segment, features)
        minutes_per_segment = model.minutes_per_segment
        num_target_segments = target_segments.shape[0]
        
        pred_ts = predicted_segments.view(num_target_segments, args.num_features, minutes_per_segment)
        pred_ts = pred_ts.permute(0, 2, 1).reshape(-1, args.num_features).numpy()
        
        target_ts = target_segments.view(num_target_segments, args.num_features, minutes_per_segment)
        target_ts = target_ts.permute(0, 2, 1).reshape(-1, args.num_features).numpy()

        # Plot a specific feature
        plt.figure(figsize=(12, 5))
        time_axis = np.arange(len(pred_ts))
        plt.plot(time_axis, pred_ts[:, feature_idx], label=f'Predicted Feature {feature_idx}', marker='.', linestyle='-')
        plt.plot(time_axis, target_ts[:, feature_idx], label=f'Target Feature {feature_idx}', marker='x', linestyle='--')
        
        plt.xlabel('Time index within forecast horizon')
        plt.ylabel('Value')
        plt.title(f'Forecast vs Target for Feature {feature_idx} (Sample {sample_idx})')
        plt.legend()
        plt.grid(True)
        
        # Log to wandb
        wandb.log({"sample_forecast_comparison": wandb.Image(plt)})
        plt.close()


def train_epoch(model, dataloader, optimizer, device, use_masked_loss):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Move data to device
        batch_data_x = batch['data_x'].to(device)
        batch_data_y = batch['data_y'].to(device)
        
        model_batch = {
            'data_x': batch_data_x,
            'data_y': batch_data_y
        }
        
        # Add masks if needed
        if use_masked_loss:
            if 'mask_x' in batch: model_batch['mask_x'] = batch['mask_x'].to(device)
            if 'mask_y' in batch: model_batch['mask_y'] = batch['mask_y'].to(device)

        # Forward pass
        output = model(model_batch, return_predictions=False) # Assuming no aux labels for now
        
        # Compute loss
        loss = model.compute_loss(output, model_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model, dataloader, device, use_masked_loss):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            batch_data_x = batch['data_x'].to(device)
            batch_data_y = batch['data_y'].to(device)
            
            model_batch = {
                'data_x': batch_data_x,
                'data_y': batch_data_y
            }
            
            # Add masks if needed
            if use_masked_loss:
                 if 'mask_x' in batch: model_batch['mask_x'] = batch['mask_x'].to(device)
                 if 'mask_y' in batch: model_batch['mask_y'] = batch['mask_y'].to(device)

            # Forward pass
            output = model(model_batch, return_predictions=False)
            
            # Compute loss
            loss = model.compute_loss(output, model_batch)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args)
    )
    
    print(f"Loading training dataset from {args.dataset_path}")
    print(f"Loading validation dataset from {args.val_dataset_path}")
    print(f"Using root directory: {args.root_dir}")
    
    # Load standardization parameters
    standardization_df = pd.read_csv(args.standardization_path)
    scaler_stats = {}
    # Ensure we use the correct number of features specified
    selected_features = list(range(args.num_features))
    for f_idx in selected_features:
         row = standardization_df.iloc[f_idx]
         scaler_stats[f_idx] = (row["mean"], row["std_dev"])
    print(f"Using {args.num_features} features with indices: {selected_features}")

    # Load the training dataset from parquet
    train_df = pd.read_parquet(args.dataset_path)
    train_df["file_uris"] = train_df["file_uris"].apply(eval)
    print(f"Loaded training dataset manifest with {len(train_df)} samples")
    
    # Load the validation dataset from parquet
    val_df = pd.read_parquet(args.val_dataset_path)
    val_df["file_uris"] = val_df["file_uris"].apply(eval)
    print(f"Loaded validation dataset manifest with {len(val_df)} samples")
    
    # Define postprocessors if needed (example uses HR interpolation)
    # Adjust indices based on the actual selected features if necessary
    original_hr_index = 5 

    p0 = CustomMaskPostprocessor(heart_rate_original_index=original_hr_index, expected_raw_features=args.num_features, consecutive_zero_threshold=30)
    p1 = HeartRateInterpolationPostprocessor(heart_rate_original_index=original_hr_index, expected_raw_features=args.num_features, hr_gap_threshold=30)
    postprocessors = [p0, p1]



    # Create the FlattenedForecastingDataset datasets
    train_dataset = ForecastingEvaluationDataset(
        dataframe=train_df,
        root_dir=args.root_dir,
        sequence_len=args.input_sequence_len,
        prediction_horizon=args.output_sequence_len,
        include_mask=args.use_masked_loss, # Include mask only if needed for loss
        feature_indices=selected_features,
        feature_stats=scaler_stats,
        postprocessors=postprocessors,
    )
    
    val_dataset = ForecastingEvaluationDataset(
        dataframe=val_df,
        root_dir=args.root_dir,
        sequence_len=args.input_sequence_len,
        prediction_horizon=args.output_sequence_len,
        include_mask=args.use_masked_loss, # Include mask only if needed for loss
        feature_indices=selected_features,
        feature_stats=scaler_stats,
        postprocessors=postprocessors
    )
    
    print(f"Created datasets with {len(train_dataset)} training and {len(val_dataset)} validation samples")
    print(f"Input sequence length: {args.input_sequence_len} time points ({args.input_seq_len_days} days)")
    print(f"Output sequence length: {args.output_sequence_len} time points ({args.output_seq_len_days} days)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True # Recommended for GPU training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    target_labels = [] # ForecastingLSTM doesn't predict separate labels by default
    
    # Choose model type based on use_revin flag
    if args.use_revin:
        print("Using RevInForecastingLSTM model")
        model = RevInForecastingLSTM(
            num_features=args.num_features, # This should be the number of features used in the dataset
            hidden_size=args.hidden_size,
            encoding_dim=args.encoding_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            target_labels=target_labels, # No separate labels
            use_masked_loss=args.use_masked_loss,
            rev_in_affine=args.revin_affine,
            rev_in_subtract_last=args.revin_subtract_last,
            l2_weight=args.l2_weight  # Added L2 regularization weight
        )
    else:
        print("Using standard ForecastingLSTM model")
        model = ForecastingLSTM(
            num_features=args.num_features, # Match dataset features
            hidden_size=args.hidden_size,
            encoding_dim=args.encoding_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            target_labels=target_labels, # No separate labels
            use_masked_loss=args.use_masked_loss,
            l2_weight=args.l2_weight  # Added L2 regularization weight
        )
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log model architecture to wandb
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_type": "RevInForecastingLSTM" if args.use_revin else "ForecastingLSTM"
    })
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Set up learning rate scheduler if requested
    scheduler = None
    if args.use_lr_scheduler:
        print(f"Using cosine annealing LR scheduler with {args.warmup_epochs} warmup epochs")
        
        # Calculate the total steps for warmup and cosine annealing
        num_train_steps = len(train_loader) * args.num_epochs
        warmup_steps = len(train_loader) * args.warmup_epochs
        
        # Calculate the cycle length in steps
        cycle_steps = (num_train_steps - warmup_steps) // args.lr_cycles
        
        # Create a custom learning rate scheduler with linear warmup and cosine annealing
        def lr_lambda(current_step):
            # Linear warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine annealing phase
            progress = float(current_step - warmup_steps) / float(max(1, cycle_steps))
            cycle_idx = (current_step - warmup_steps) // cycle_steps
            
            # Adjust progress for current cycle
            progress = progress - cycle_idx
            
            # Cosine decay from 1.0 to 0.0 (or some minimum value like 0.1)
            min_lr_factor = 0.1  # Minimum LR will be 10% of max LR
            return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1.0 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Create checkpoint directory if needed
    if args.save_model:
        checkpoint_dir = Path(args.checkpoint_dir) / args.run_name # Save checkpoints in run-specific folder
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Train the model
    best_val_loss = float('inf')
    
    print(f"Training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, args.use_masked_loss)
        
        # Step scheduler after each epoch if it exists
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate
        val_loss = validate(model, val_loader, device, args.use_masked_loss)
        
        # Step scheduler if it exists
        if scheduler is not None:
            # For epoch-based schedulers
            if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                scheduler.step()
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        })
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Save checkpoint if validation loss improved
        if args.save_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args # Save args for reloading
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Update learning rate for batch-based schedulers
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
            for _ in range(len(train_loader)):
                scheduler.step()
        
        # Visualize predictions and log to wandb (e.g., every 5 epochs)
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
             if len(val_dataset) > 0:
                 log_sample_prediction(model, val_dataset, device, args, sample_idx=0, feature_idx=0) # Log first feature
             else:
                 print("Validation dataset is empty, skipping prediction visualization.")

    # Save final model checkpoint
    if args.save_model:
        final_checkpoint_path = checkpoint_dir / f"final_model_epoch{args.num_epochs}.pt"
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_loss': train_loss, # Last epoch train loss
            'val_loss': val_loss,     # Last epoch val loss
            'args': args
        }, final_checkpoint_path)
        print(f"Saved final model checkpoint to {final_checkpoint_path}")
    
    # Finish wandb run
    wandb.finish()
    
    print("Training completed!")

if __name__ == "__main__":
    main() 