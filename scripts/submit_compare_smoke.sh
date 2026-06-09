#!/bin/bash -l
#SBATCH --job-name=qrn_cmp_smoke
#SBATCH --output=slurm_logs/cmp_smoke_%j.out
#SBATCH --error=slurm_logs/cmp_smoke_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#
# ─────────────────────────────────────────────────────────────────────────────
# COMPARE SMOKE: verify training is working as intended. Homogeneous 6-repeater
# chain, PERFECT operations (p_gen = p_swap = 1, F0 = 1, no loss, no CC delay) to
# make learning easy. --compare logs, EACH episode, the greedy agent vs swap-asap
# vs random returns on one shared seeded network → training_compare.png shows the
# phases where the agent overtakes (a) random, then (b) swap-asap.
#
#   (cluster) cd ~/QNetGame && sbatch --account=liacs scripts/submit_compare_smoke.sh
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

python -u train-test/train.py \
    --run_id compare_smoke \
    --lr 5e-4 \
    --hidden 64 \
    --episodes 10000 \
    --batch_size 64 \
    --max_steps 30 \
    --n_lo 6 \
    --n_hi 6 \
    --topology chain \
    --dt_seconds 0.0 \
    --channel_loss 0.0 \
    --F0 1.0 \
    --p_gen 1.0 \
    --p_swap 1.0 \
    --p_gen_std 0.0 \
    --p_swap_std 0.0 \
    --cutoff 15 \
    --compare \
    --save_base_dir checkpoints

echo "Job completed at $(date)"
echo "Checkpoint:  checkpoints/compare_smoke/policy.pth"
echo "Crossover:   checkpoints/compare_smoke/training_compare.png"
