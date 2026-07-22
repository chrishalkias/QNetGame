#!/bin/bash -l
#SBATCH --job-name=qrn_pp5k
#SBATCH --output=slurm_logs/portpurify_5k_%j.out
#SBATCH --error=slurm_logs/portpurify_5k_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-short

# Smoke run after the purify-cascade + left/right-ports change (branch
# fix/purification): 5k episodes, HOMOGENEOUS chain (std=0), curriculum N 7-10,
# n_ch=2 PER SIDE (=> 4 physical qubits interior). Purpose: inspect how training
# dynamics shifted vs the pre-change omni_v3 runs. ~80 min at ~64 ep/min plus
# winnability/probe overhead; 2h has margin (cpu-short cap is 4h).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  seed: 1"

python -u experiments/training/train.py \
    --run_id "portpurify_5k_s1" \
    --seed 1 \
    --episodes 5000 --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 200 --gamma 0.995 \
    --n_lo 7 --n_hi 10 --n_ch 2 \
    --p_gen 0.4 0.9 --p_swap 0.4 0.9 \
    --p_gen_std 0.0 --p_swap_std 0.0 \
    --cutoff_lo 10 --cutoff_hi 50 \
    --prune_unwinnable \
    --topology chain --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints

echo "done $(date)"
