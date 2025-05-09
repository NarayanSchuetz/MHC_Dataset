import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import wandb
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Import the MHC dataset and LSTM models
from torch_dataset import BaseMhcDataset
from models.lstm import AutoencoderLSTM, RevInAutoencoderLSTM, LSTMTrainer
from dataset_postprocessors import CustomMaskPostprocessor, HeartRateInterpolationPostprocessor


def parse_args():
    parser = argparse.ArgumentParser(description='Train AutoencoderLSTM on MHC dataset with wandb tracking')
    
    # Data paths
    parser.add_argument('--dataset_path', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset_out/splits/train_dataset.parquet",
                        help='Path to the training dataset parquet file')
    parser.add_argument('--val_dataset_path', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset_out/splits/val_dataset.parquet",
                        help='Path to the validation dataset parquet file')
    parser.add_argument('--root_dir', type=str, 
                        default="/scratch/groups/euan/mhc/mhc_dataset",
                        help='Root directory containing the MHC dataset')
    parser.add_argument('--standardization_path', type=str, 
                        default="/scratch/users/schuetzn/data/mhc_dataset_out/standardization_params.csv",
                        help='Path to standardization parameters CSV')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of LSTM')
    parser.add_argument('--encoding_dim', type=int, default=256, help='Dimension of encoded segments')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--prediction_horizon', type=int, default=1, 
                        help='Number of future 30-min segments to predict')
    parser.add_argument('--num_features', type=int, default=6, 
                        help='Number of features per minute to use')
    
    # Teacher forcing parameters
    parser.add_argument('--initial_tf', type=float, default=1.0, 
                        help='Initial teacher forcing ratio')
    parser.add_argument('--final_tf', type=float, default=1.0, 
                        help='Final teacher forcing ratio')
    parser.add_argument('--decay_epochs', type=int, default=15, 
                        help='Number of epochs to decay teacher forcing')
    
    # RevIN parameters
    parser.add_argument('--use_revin', action='store_true', 
                        help='Use RevInLSTM instead of standard LSTM')
    parser.add_argument('--revin_affine', action='store_true', 
                        help='Use learnable affine parameters in RevIN')
    parser.add_argument('--revin_subtract_last', action='store_true', 
                        help='Subtract last element instead of mean in RevIN')
    
    # Experiment tracking
    parser.add_argument('--wandb_project', type=str, default='mhc-lstm', 
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help='WandB entity name')
    parser.add_argument('--run_name', type=str, default=None, 
                        help='Name for this run (default: auto-generated based on timestamp)')
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Auto-generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = "revin" if args.use_revin else "lstm"
        args.run_name = f"{model_type}_{args.num_layers}layer_h{args.hidden_size}_{timestamp}"
    
    return args


def log_sample_prediction(model, val_dataset, device, sample_idx=0, feature_idx=0, segment_idx=0):
    """Log prediction visualization to wandb"""
    with torch.no_grad():
        # Get a sample from validation set
        sample = val_dataset[sample_idx]
        
        # Prepare batch format
        batch = {
            'data': sample['data'].unsqueeze(0).to(device),  # Add batch dimension
            'mask': sample['mask'].unsqueeze(0).to(device) if 'mask' in sample else None
        }
        
        # Forward pass
        model.eval()
        output = model(batch)
        
        # Get predictions
        predicted_segments = output['sequence_output'][0].cpu().numpy()  # Remove batch dimension
        target_segments = output['target_segments'][0].cpu().numpy()
        
        # Extract the first 30 values for visualization
        predicted_values = predicted_segments[segment_idx, :30]
        target_values = target_segments[segment_idx, :30]
        
        plt.figure(figsize=(10, 4))
        plt.plot(predicted_values, label='Predicted', marker='o')
        plt.plot(target_values, label='Target', marker='x')
        plt.xlabel('Time index')
        plt.ylabel('Value')
        plt.title(f'Prediction vs Target for Feature {feature_idx}, Segment {segment_idx}')
        plt.legend()
        plt.grid(True)
        
        # Log to wandb
        wandb.log({"sample_prediction": wandb.Image(plt)})
        plt.close()


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
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
    for f_idx, row in standardization_df.iloc[:args.num_features].iterrows():
        scaler_stats[f_idx] = (row["mean"], row["std_dev"])
    
    # Load the training dataset from parquet
    train_df = pd.read_parquet(args.dataset_path)
    train_df["file_uris"] = train_df["file_uris"].apply(eval)
    print(f"Loaded training dataset with {len(train_df)} samples")
    
    # Load the validation dataset from parquet
    val_df = pd.read_parquet(args.val_dataset_path)
    val_df["file_uris"] = val_df["file_uris"].apply(eval)
    print(f"Loaded validation dataset with {len(val_df)} samples")
    
    # Print available label columns
    label_cols = [col for col in train_df.columns if col.endswith('_value')]
    print(f"Available label columns: {label_cols}")
    
    p0 = CustomMaskPostprocessor(heart_rate_original_index=5, expected_raw_features=6, consecutive_zero_threshold=30)
    p1 = HeartRateInterpolationPostprocessor(heart_rate_original_index=5, expected_raw_features=6, hr_gap_threshold=30)

    # Create the datasets with mask
    train_dataset = BaseMhcDataset(
        train_df, 
        args.root_dir, 
        include_mask=True, 
        feature_stats=scaler_stats, 
        feature_indices=list(range(args.num_features)),
        postprocessors=[p0, p1]
    )
    
    val_dataset = BaseMhcDataset(
        val_df, 
        args.root_dir, 
        include_mask=True, 
        feature_stats=scaler_stats, 
        feature_indices=list(range(args.num_features)),
        postprocessors=[p0, p1]
    )
    
    print(f"Created datasets with {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Initialize model
    # Set target_labels to [] if there are no labels to predict
    target_labels = []  # Can be modified to include labels if needed
    
    # Choose model type based on use_revin flag
    if args.use_revin:
        print("Using RevInLSTM model")
        model = RevInAutoencoderLSTM(
            num_features=args.num_features,
            hidden_size=args.hidden_size,
            encoding_dim=args.encoding_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            target_labels=target_labels,
            prediction_horizon=args.prediction_horizon,
            use_masked_loss=False,
            teacher_forcing_ratio=args.initial_tf,
            rev_in_affine=args.revin_affine,
            rev_in_subtract_last=args.revin_subtract_last
        )
    else:
        print("Using standard LSTM model")
        model = AutoencoderLSTM(
            num_features=args.num_features,
            hidden_size=args.hidden_size,
            encoding_dim=args.encoding_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            target_labels=target_labels,
            prediction_horizon=args.prediction_horizon,
            use_masked_loss=False,
            teacher_forcing_ratio=args.initial_tf
        )
    
    print(f"Initialized model with target labels: {target_labels}")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log model architecture to wandb
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_type": "RevInLSTM" if args.use_revin else "LSTM"
    })
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Set up trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    trainer = LSTMTrainer(model, optimizer, device)
    
    # Calculate teacher forcing decay step
    tf_decay_step = (args.initial_tf - args.final_tf) / max(1, args.decay_epochs)
    
    print(f"Training for {args.num_epochs} epochs...")
    print(f"Teacher forcing decay: {args.initial_tf*100:.0f}% -> {args.final_tf*100:.0f}% over {args.decay_epochs} epochs.")
    print(f"Teacher forcing decay step per epoch: {tf_decay_step:.6f}")
    
    # Create checkpoint directory if needed
    if args.save_model:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    best_val_loss = float('inf')
    
    # Create a figure to visualize teacher forcing decay
    plt.figure(figsize=(12, 6))
    tf_ratios = []
    epochs_x = []
    
    for epoch in range(args.num_epochs):
        # Calculate and set teacher forcing ratio for this epoch
        if epoch < args.decay_epochs:
            current_tf_ratio = args.initial_tf - tf_decay_step * epoch
            # Ensure it doesn't go below the final ratio
            trainer.model.teacher_forcing_ratio = max(args.final_tf, current_tf_ratio)
        else:
            # After decay period, keep it at the final ratio
            trainer.model.teacher_forcing_ratio = args.final_tf
        
        # Store teacher forcing ratio for visualization
        tf_ratios.append(trainer.model.teacher_forcing_ratio)
        epochs_x.append(epoch + 1)
        
        # Print current teacher forcing ratio for debugging
        print(f"Epoch {epoch+1}: Teacher forcing ratio = {trainer.model.teacher_forcing_ratio:.6f}")
        
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate with teacher forcing explicitly enabled
        val_loss = trainer.validate(val_loader, use_teacher_forcing=True)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "teacher_forcing_ratio": trainer.model.teacher_forcing_ratio,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if args.save_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"{args.run_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            wandb.save(str(checkpoint_path))
        
        # Visualize predictions and log to wandb (every 5 epochs)
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            log_sample_prediction(model, val_dataset, device)
    
    # Plot teacher forcing decay and log to wandb
    plt.plot(epochs_x, tf_ratios, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Teacher Forcing Ratio')
    plt.title('Teacher Forcing Ratio Decay')
    plt.grid(True)
    plt.tight_layout()
    wandb.log({"teacher_forcing_decay": wandb.Image(plt)})
    plt.close()
    
    # Save final model checkpoint
    if args.save_model:
        final_checkpoint_path = checkpoint_dir / f"{args.run_name}_final.pt"
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, final_checkpoint_path)
        print(f"Saved final checkpoint to {final_checkpoint_path}")
        wandb.save(str(final_checkpoint_path))
    
    # Finish wandb run
    wandb.finish()
    
    print("Training completed!")
    return model, trainer


if __name__ == "__main__":
    main()
