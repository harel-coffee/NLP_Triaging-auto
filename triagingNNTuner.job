#!/bin/bash

#SBATCH --job-name=TriagingTuner
#SBATCH --mem=8G
#SBATCH --partition=mpcg.p
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=triaging_tuner.%j.out
#SBATCH --error=triaging_tuner.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

module load hpc-env/8.3
module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow

python nn.py