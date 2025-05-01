#!/bin/bash

## SLURM directives
#SBATCH --partition=gpu
#SBATCH --job-name=lstm_experiment
#SBATCH --output=/scratch/users/schuetzn/logs/pytorch_runs/%j_lstm_experiment.log
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=schuetzn@stanford.edu
#SBATCH -G 1

# Load required modules
ml devel
ml python/3.9.0
ml py-pytorch/2.0.0_py39

# Create a directory for checkpoints if it doesn't exist
mkdir -p /scratch/users/schuetzn/data/mhc_dataset_out/lstm

# Run the LSTM experiment
python3 scripts/lstm_experiment.py \
  --wandb_project MHC_Dataset \
  --wandb_entity schuetzn \
  #--use_revin \
  --batch_size 128 \
  --num_epochs 100 \
  --lr 0.0001 \
  --hidden_size 720 \
  --encoding_dim 180 \
  --num_layers 5 \
  --dropout 0.1 \
  --prediction_horizon 1 \
  --num_features 6 \
  --initial_tf 1.0 \
  --final_tf 1.0 \
  --decay_epochs 100 \
  --save_model \
  --checkpoint_dir /scratch/users/schuetzn/data/mhc_dataset_out/lstm \
  --num_workers 16 \
  --dataset_path /scratch/users/schuetzn/data/mhc_dataset_out/splits/train_final_dataset.parquet \
  --val_dataset_path /scratch/users/schuetzn/data/mhc_dataset_out/splits/validation_dataset.parquet \
  --run_name "lstim_experiment_tf=1_hidden-dim=720_$(date +%Y%m%d_%H%M%S)"
