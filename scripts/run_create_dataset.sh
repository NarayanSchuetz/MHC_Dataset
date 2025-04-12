#!/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --output=create_dataset_%j.out
#SBATCH --error=create_dataset_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=euan

# Load any necessary modules
module load python/3.9

# Set the path to your project directory
PROJECT_DIR="$HOME/MHC_Dataset"

# Change to the project directory
cd $PROJECT_DIR

# Run the dataset creation script
bash scripts/create_dataset.sh

echo "Job completed at $(date)"
