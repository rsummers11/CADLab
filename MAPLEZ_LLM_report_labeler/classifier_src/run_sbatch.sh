#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --mincpus=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -o ./dgx_log/slurm-%j.out-%N
#SBATCH -e ./dgx_log/slurm-%j.err-%N
#SBATCH --mem=60G

. ./miniconda3/etc/profile.d/conda.sh
conda activate
conda activate mimic_classifier
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 src/train_pytorch.py "$@"