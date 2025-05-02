#!/bin/bash

## SLURM directives
#SBATCH --partition=gpu
#SBATCH --job-name=forecast_lstm_exp
#SBATCH --output=/scratch/users/schuetzn/logs/pytorch_runs/%j_forecast_lstm_exp.log
#SBATCH --time=24:00:00 # Increased time potentially needed for longer sequences
#SBATCH --mem=64G        # Increased memory potentially needed
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=schuetzn@stanford.edu
#SBATCH -G 1

# Load required modules
ml reset
ml devel
ml python/3.9.0
ml py-pytorch/2.0.0_py39

# Define base directory for checkpoints
BASE_CHECKPOINT_DIR="/scratch/users/schuetzn/data/mhc_dataset_out/forecasting_lstm_checkpoints"
mkdir -p $BASE_CHECKPOINT_DIR

# Run the Forecasting LSTM experiment
python3 scripts/forecasting_lstm_experiment.py \
  --wandb_project MHC_Dataset_Forecasting \
  --wandb_entity schuetzn \
  --batch_size 128 \
  --num_epochs 100 \
  --lr 0.0001 \
  --hidden_size 520 \
  --encoding_dim 180 \
  --num_layers 5 \
  --dropout 0.1 \
  --num_features 6 \
  --input_seq_len_days 5\
  --output_seq_len_days 2 \
  --use_masked_loss \
  --save_model \
  --checkpoint_dir $BASE_CHECKPOINT_DIR \
  --num_workers 16 \
  --dataset_path /scratch/users/schuetzn/data/mhc_dataset_out/splits/train_final_dataset.parquet \
  --val_dataset_path /scratch/users/schuetzn/data/mhc_dataset_out/splits/validation_dataset.parquet \
  --run_name "forecast_lstm_5d_in_2d_out_h720_$(date +%Y%m%d_%H%M%S)" #\
  # --use_revin 
  # --revin_affine

echo "Forecasting experiment script finished." 