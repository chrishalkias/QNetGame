#!/bin/bash -l
#SBATCH --job-name=qrn_train
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs
mkdir -p checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
eval "$(/usr/bin/modulecmd bash load CUDA/12.4.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u train-test/train.py \
    --run_id cluster_006 \
    --lr 5e-4 \
    --hidden 64 \
    --episodes 50000 \
    --batch_size 64 \
    --max_steps 30 \
    --n_lo 5 \
    --n_hi 10 \
    --topology chain \
    --dt_seconds 1e-4 \
    --channel_loss 0.01 \
    --F0 0.99 \
    --p_gen 0.5 \
    --p_swap 0.85 \
    --cutoff 15 \
    --save_base_dir checkpoints

echo "Job completed at $(date)"
