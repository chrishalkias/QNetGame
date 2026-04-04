#!/bin/bash -l
#SBATCH --job-name=qrn_batch_val
#SBATCH --output=slurm_logs/batch_val_%j.out
#SBATCH --error=slurm_logs/batch_val_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs
mkdir -p results/batch_validate

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
eval "$(/usr/bin/modulecmd bash load CUDA/12.4.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u train-test/batch_validate.py \
    --model checkpoints/cluster_004/policy.pth \
    --episodes 200 \
    --seed 42 \
    --save_dir results/batch_validate \
    --sweep both \
    --sweep2_nodes 8

echo "Job completed at $(date)"
