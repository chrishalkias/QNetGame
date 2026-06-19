#!/bin/bash -l
#SBATCH --job-name=qrn_inhomo
#SBATCH --output=slurm_logs/inhomo_%j.out
#SBATCH --error=slurm_logs/inhomo_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#
# ─────────────────────────────────────────────────────────────────────────────
# INHOMOGENEOUS REPEATERS: train on chains with per-repeater p_gen / p_swap drawn
# from (mean, std). Means/stds below match the requested run:
#     p_gen = 0.5 ± 0.2     p_swap = 0.8 ± 0.1
# std=0 would recover the homogeneous case. policy.pth = best checkpoint.
#
#   (cluster) cd ~/QNetGame && sbatch --account=liacs scripts/submit_inhomo.sh
# ─────────────────────────────────────────────────────────────────────────────

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

python -u experiments/train.py \
    --run_id inhomo_001 \
    --lr 5e-4 \
    --hidden 64 \
    --episodes 30000 \
    --batch_size 64 \
    --max_steps 50 \
    --n_lo 4 \
    --n_hi 10 \
    --topology chain \
    --dt_seconds 0.0 \
    --channel_loss 0.0 \
    --F0 1.0 \
    --p_gen 0.5 \
    --p_gen_std 0.2 \
    --p_swap 0.8 \
    --p_swap_std 0.1 \
    --cutoff 15 \
    --save_base_dir checkpoints

echo "Job completed at $(date)"
echo "Checkpoint: checkpoints/inhomo_001/policy.pth"
