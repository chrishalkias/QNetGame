#!/bin/bash -l
#SBATCH --job-name=qrn_dvN
#SBATCH --output=slurm_logs/dvN_%j.out
#SBATCH --error=slurm_logs/dvN_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-zen4

# Delivery time T vs chain size N at fixed (p_gen=0.4, p_swap=0.8), comparing a
# trained agent against swap-ASAP and purify-then-swap. N: 10..15, dotted line at
# N=12 (training ceiling -> N>12 is zero-shot OOD).
#   sbatch experiments/scripts/submit_delivery_vs_N.sh [run_id]   (default omni_nopen_15k)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons
RUN="${1:-sota}"

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  run=$RUN"

python -u experiments/comparisons/policy_vs_agent/delivery_vs_N.py \
    --ckpt "checkpoints/$RUN/policy.pth" \
    --p_gen 0.4 --p_swap 0.8 \
    --n_lo 10 --n_hi 15 --n_train_max 12 \
    --n_ch 4 --cutoff 20 --horizon 300 --mc_eps 2000 \
    --out "results/comparisons/delivery_vs_N_$RUN.json"

echo "done $(date)"
