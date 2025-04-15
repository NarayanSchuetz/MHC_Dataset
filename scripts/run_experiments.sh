#!/bin/bash

## SLURM directives
#SBATCH --partition=gpu
#SBATCH --job-name=lstm_experiment
#SBATCH --output=/scratch/users/schuetzn/logs/pytorch_runs/%j_lstm_experiment.log
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=schuetzn@stanford.edu
#SBATCH -G 1

# Load required modules
ml devel
ml python/3.9.0
ml py-pytorch/2.0.0_py39

# Create a directory for checkpoints if it doesn't exist
mkdir -p checkpoints

# Run the LSTM experiment
python scripts/lstm_experiment.py \
  --wandb_project MHC_Dataset \
  --wandb_entity schuetzn \
  --batch_size 128 \
  --num_epochs 200 \
  --lr 0.0001 \
  --hidden_size 256 \
  --encoding_dim 256 \
  --num_layers 5 \
  --dropout 0.1 \
  --prediction_horizon 1 \
  --num_features 6 \
  --initial_tf 1.0 \
  --final_tf 0.0 \
  --decay_epochs 100 \
  --save_model \
  --checkpoint_dir checkpoints \
  --num_workers 20 \
  --run_name "lstm_experiment_$(date +%Y%m%d_%H%M%S)"

