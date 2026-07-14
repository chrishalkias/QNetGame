#!/bin/bash -l
#SBATCH --job-name=qrn_train
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs
mkdir -p checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"

python -u experiments/training/train.py \
    --run_id cluster_008 \
    --lr 5e-4 \
    --hidden 64 \
    --episodes 50000 \
    --batch_size 64 \
    --max_steps 50 \
    --n_lo 4 \
    --n_hi 10 \
    --topology chain \
    --dt_seconds 0.0 \
    --channel_loss 0.0 \
    --F0 1.0 \
    --p_gen 0.5 \
    --p_swap 0.85 \
    --cutoff 15 \
    --save_base_dir checkpoints

echo "Job completed at $(date)"
