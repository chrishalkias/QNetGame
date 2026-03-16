#!/bin/bash
#SBATCH --job-name=qrn_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

mkdir -p logs
mkdir -p checkpoints

module purge
module load Python/3.9.6
module load CUDA/13.2

source ~/my_python_env/bin/activate

python train.py \
    --lr 5e-4 \
    --hidden 64 \
    --episodes 10_000 \
    --batch_size 64 \
    --max_steps 30 \
    --n_lo 6 \
    --n_hi 15\
    --p_gen 0.8 \
    --p_swap 0.85 \
    --cutoff 15