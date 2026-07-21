#!/bin/bash -l
#SBATCH --job-name=qrn_v3_20k
#SBATCH --output=slurm_logs/v3_20k_%A_s%a.out
#SBATCH --error=slurm_logs/v3_20k_%A_s%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4
#SBATCH --array=1-3

# omni_v3: first retrain on cutoff-invariant physics (2026-07-12 fix: swap
# viability gate, born-dead resolution guard, Eq.(4) purify ages). 20k
# episodes (user decision), cutoff 10-50, 3 seeds via array. ~5.5h at the
# measured ~64 ep/min plus probe-calibration overhead; 12h has margin.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  seed: ${SLURM_ARRAY_TASK_ID}"

python -u experiments/training/train.py \
    --run_id "omni_v3_20k_s${SLURM_ARRAY_TASK_ID}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --episodes 20000 --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 200 --gamma 0.995 \
    --n_lo 4 --n_hi 12 --n_ch 2 3 4 \
    --p_gen 0.4 0.9 --p_swap 0.4 0.9 \
    --p_gen_std 0.15 --p_swap_std 0.15 \
    --cutoff_lo 10 --cutoff_hi 50 \
    --prune_unwinnable \
    --topology chain --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints

echo "done $(date)"
