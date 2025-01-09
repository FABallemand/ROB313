#!/bin/bash

#SBATCH --output=train_res/%j/%j_train.out
#SBATCH --error=train_res/%j/%j_train.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DATE=$(date +'%Y%m%d_%H%M%S')
echo $DATE

mkdir train_res/$SLURM_JOB_ID

eval "$(conda shell.bash hook)"

conda activate prim_env

set -x
srun python3 -u train.py train_res/$SLURM_JOB_ID