#!/bin/bash -l
#SBATCH --job-name=qrn_omni_v2
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# Post-physics-fix retrain (omni_v2): identical canonical omni_nopen_15k recipe
# (15k episodes, N 4-12, n_ch 2 3 4, rates 0.4-0.9, sigma=0.15, cutoff 10-40,
# prune_unwinnable, lr 5e-4, hidden 64, idealized physics) but on the corrected
# simulator (canonical BBPSSW purify success, decoherence single-count fix) with
# the new deterministic-seed contract. Each run passes an explicit --seed; with
# episodes=15000 eval-probe checkpointing engages by default (do NOT pass
# --no_eval_ckpt). This retrain produces the new SOTA; every pre-fix checkpoint
# is scientifically superseded.
#
# Usage (one seed per submission):
#   sbatch submit_omni_v2_15k.sh <seed>
# The run_id is derived as omni_v2_15k_s<seed>; outputs to checkpoints/<run_id>/.
# Launch the three-seed sweep with:
#   for s in 1 2 3; do sbatch submit_omni_v2_15k.sh $s; done

set -euo pipefail
SEED="$1"
RID="omni_v2_15k_s${SEED}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  run_id=$RID seed=$SEED"

python -u experiments/training/train.py \
    --run_id "$RID" --seed "$SEED" \
    --episodes 15000 --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 200 --gamma 0.995 \
    --n_lo 4 --n_hi 12 --n_ch 2 3 4 \
    --p_gen 0.4 0.9 --p_swap 0.4 0.9 \
    --p_gen_std 0.15 --p_swap_std 0.15 \
    --cutoff_lo 10 --cutoff_hi 40 \
    --prune_unwinnable \
    --topology chain --dt_seconds 0.0 --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints

echo "done $(date)"
