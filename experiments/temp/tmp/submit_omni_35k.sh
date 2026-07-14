#!/bin/bash -l
#SBATCH --job-name=qrn_omni35k
#SBATCH --output=slurm_logs/omni35k_%j.out
#SBATCH --error=slurm_logs/omni35k_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# 35k-episode run of the swap-shy-fixed omni config (FAILED_ACTION=0), same
# physics/curriculum/reward as submit_omni_3k.sh. ckpt window opens at ~ep 31699
# (reached ~8.3h at the measured ~64 ep/min); ~9h total, walltime 12h leaves margin.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)"

python -u experiments/training/train.py \
    --run_id omni_nopen_35k \
    --episodes 35000 --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 200 --gamma 0.995 \
    --n_lo 4 --n_hi 12 --n_ch 2 3 4 \
    --p_gen 0.4 0.9 --p_swap 0.4 0.9 \
    --p_gen_std 0.15 --p_swap_std 0.15 \
    --cutoff_lo 10 --cutoff_hi 40 \
    --prune_unwinnable \
    --topology chain --dt_seconds 0.0 --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints

echo "done $(date)"
