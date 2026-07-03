#!/bin/bash -l
#SBATCH --job-name=qrn_seed
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# Seed / training-length sweep: same config as omni_nopen_15k (NO CC delays,
# inhomogeneities sigma=0.15, same ranges + curriculum + reward). Only episodes
# and seed vary. Outputs to checkpoints/different_seeds/<run_id>/ and records the
# seed in seed.txt for later head-to-head comparison.
#
#   sbatch --job-name=ds_ep15k --time=06:00:00 submit_seed_run.sh <episodes> <seed> <run_id>

set -euo pipefail
EP="$1"; SEED="$2"; RID="$3"
cd "$SLURM_SUBMIT_DIR"
OUT="checkpoints/different_seeds/$RID"
mkdir -p slurm_logs "$OUT"
echo "run_id=$RID episodes=$EP seed=$SEED  (omni15k config, no CC, sigma=0.15)" > "$OUT/seed.txt"

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  run_id=$RID episodes=$EP seed=$SEED"

python -u experiments/training/train.py \
    --run_id "$RID" --seed "$SEED" \
    --episodes "$EP" --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 200 --gamma 0.995 \
    --n_lo 4 --n_hi 12 --n_ch 2 3 4 \
    --p_gen 0.4 0.9 --p_swap 0.4 0.9 \
    --p_gen_std 0.15 --p_swap_std 0.15 \
    --cutoff_lo 10 --cutoff_hi 40 \
    --prune_unwinnable \
    --topology chain --dt_seconds 0.0 --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints/different_seeds

echo "done $(date)"
