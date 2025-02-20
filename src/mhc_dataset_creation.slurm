#!/bin/bash

## add additional SBATCH directives here
#SBATCH --partition=owners
#SBATCH --job-name=mhc_dataset_creation
#SBATCH --output=/scratch/users/schuetzn/logs/mhc_dataset_creation/%j.log
#SBATCH --time=16:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=schuetzn@stanford.edu
#SBATCH --array=0-499

# load modules, adjust to your needs
ml devel
ml python/3.9.0

# execute the Python code, this should stay as is.
python3 tmp.py ${SLURM_ARRAY_TASK_ID}
