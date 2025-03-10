#!/bin/bash

#SBATCH --output=train_res/%j/%j_train.out
#SBATCH --error=train_res/%j/%j_train.err
#SBATCH --time=99:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4

DATE=$(date +'%Y%m%d_%H%M%S')
echo $DATE

eval "$(conda shell.bash hook)"

conda activate rob_env

set -x
srun python3 -u train_ssl.py $SLURM_JOB_ID