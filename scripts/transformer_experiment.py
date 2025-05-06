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

from torch_dataset import BaseMhcDataset
from models.transformer import (
    ForecastingTransformer,
    RevInForecastingTransformer,
    TransformerTrainer
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ForecastingTransformer on MHC dataset with wandb tracking')

    # -------------------------------------------------------------------------
    # Data paths
    # -------------------------------------------------------------------------
    parser.add_argument('--dataset_path', type=str,
                        default="/scratch/groups/euan/mhc/mhc_dataset_out/splits/train_dataset.parquet",
                        help='Path to the dataset parquet file')
    parser.add_argument('--root_dir', type=str,
                        default="/scratch/groups/euan/mhc/mhc_dataset",
                        help='Root directory containing the MHC dataset')
    parser.add_argument('--standardization_path', type=str,
                        default="/scratch/groups/euan/mhc/mhc_dataset_out/standardization_params.csv",
                        help='Path to standardization parameters CSV')

    # -------------------------------------------------------------------------
    # Training parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # -------------------------------------------------------------------------
    # Transformer model parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--d_model', type=int, default=256, help='Dimensionality of Transformer embeddings')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of Transformer blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--ffn_ratio', type=float, default=4.0, help='Feed-forward expansion ratio')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                        help='Number of future 30-min segments to predict')
    parser.add_argument('--num_features', type=int, default=6,
                        help='Number of features per minute to use')
    parser.add_argument('--max_seq_len', type=int, default=4096,
                        help='Maximum sequence length for Rotary Pos Emb')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base theta')

    # -------------------------------------------------------------------------
    # RevIN parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--use_revin', action='store_true',
                        help='Use RevInForecastingTransformer instead of standard Transformer')
    parser.add_argument('--revin_affine', action='store_true',
                        help='Use learnable affine parameters in RevIN')
    parser.add_argument('--revin_subtract_last', action='store_true',
                        help='Subtract last element instead of mean in RevIN')

    # -------------------------------------------------------------------------
    # Experiment tracking / saving
    # -------------------------------------------------------------------------
    parser.add_argument('--wandb_project', type=str, default='mhc-transformer',
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
        model_type = "transformer_revin" if args.use_revin else "transformer"
        args.run_name = f"{model_type}_{args.num_layers}layer_d{args.d_model}_{timestamp}"

    return args


def log_sample_prediction(model, val_dataset, device, sample_idx=0, feature_idx=0, segment_idx=0):
    """
    Log a sample prediction vs target to wandb. Similar to the LSTM script.
    We fetch one item from val_dataset and do a forward pass.
    """
    with torch.no_grad():
        sample = val_dataset[sample_idx]
        batch = {
            'data': sample['data'].unsqueeze(0).to(device),
            'mask': sample['mask'].unsqueeze(0).to(device) if 'mask' in sample else None
        }
        model.eval()
        output = model(batch)

        predicted_segments = output['sequence_output'][0].cpu().numpy()  # (num_segments - horizon, features_per_segment)
        target_segments = output['target_segments'][0].cpu().numpy()

        # Just plot the first 30 values of the selected segment
        if segment_idx < predicted_segments.shape[0]:
            predicted_values = predicted_segments[segment_idx, :30]
            target_values = target_segments[segment_idx, :30]
        else:
            predicted_values = []
            target_values = []

        plt.figure(figsize=(10, 4))
        plt.plot(predicted_values, label='Predicted', marker='o')
        plt.plot(target_values, label='Target', marker='x')
        plt.xlabel('Time index')
        plt.ylabel('Value')
        plt.title(f'Prediction vs Target (Feature {feature_idx}, Segment {segment_idx})')
        plt.legend()
        plt.grid(True)

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

    print(f"Loading dataset from {args.dataset_path}")
    print(f"Using root directory: {args.root_dir}")

    # Load standardization parameters
    standardization_df = pd.read_csv(args.standardization_path)
    scaler_stats = {}
    for f_idx, row in standardization_df.iloc[:args.num_features].iterrows():
        scaler_stats[f_idx] = (row["mean"], row["std_dev"])

    # Load the dataset from parquet
    df = pd.read_parquet(args.dataset_path)
    df["file_uris"] = df["file_uris"].apply(eval)
    print(f"Loaded dataset with {len(df)} samples")

    # Print available label columns
    label_cols = [col for col in df.columns if col.endswith('_value')]
    print(f"Available label columns: {label_cols}")

    # Create the dataset with mask
    dataset = BaseMhcDataset(
        df,
        args.root_dir,
        include_mask=True,
        feature_stats=scaler_stats,
        feature_indices=list(range(args.num_features))
    )

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Split dataset into {train_size} training and {val_size} validation samples")

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
    target_labels = []  # or specify if you have label columns
    if args.use_revin:
        print("Using RevInForecastingTransformer")
        model = RevInForecastingTransformer(
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            ffn_ratio=args.ffn_ratio,
            target_labels=target_labels,
            prediction_horizon=args.prediction_horizon,
            use_masked_loss=True,  # Typically use mask
            num_features=args.num_features,
            max_seq_len=args.max_seq_len,
            rope_base=args.rope_base,
            rev_in_affine=args.revin_affine,
            rev_in_subtract_last=args.revin_subtract_last
        )
    else:
        print("Using standard ForecastingTransformer")
        model = ForecastingTransformer(
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            ffn_ratio=args.ffn_ratio,
            target_labels=target_labels,
            prediction_horizon=args.prediction_horizon,
            use_masked_loss=True,  # Typically use mask
            num_features=args.num_features,
            max_seq_len=args.max_seq_len,
            rope_base=args.rope_base
        )

    print(f"Initialized model with target labels: {target_labels}")
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Log model info to wandb
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_class": "RevInForecastingTransformer" if args.use_revin else "ForecastingTransformer"
    })

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Trainer
    trainer = TransformerTrainer(model, optimizer, device)

    if args.save_model:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training for {args.num_epochs} epochs...")

    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        print(f"Epoch {epoch+1}/{args.num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save checkpoint if validation improves
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

        # Log sample predictions every few epochs
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            log_sample_prediction(model, val_dataset, device)

    # Save final checkpoint
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

    # Finish wandb
    wandb.finish()
    print("Transformer training completed!")
    return model, trainer


if __name__ == "__main__":
    main()
