#!/bin/bash -l
#SBATCH --job-name=dN15k_fid
#SBATCH --partition=cpu-zen4
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/dN15k_fid_%j.out
#SBATCH --error=slurm_logs/dN15k_fid_%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
# Match submit_seed_run.sh EXACTLY: module loads must precede the venv or
# torch_geometric hits a circular-import ("no attribute 'typing'") at agent load.
eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# Delivery time T (agent / swap-asap / purify-then-swap) + end-to-end fidelity F
# (agent + purify-then-swap) vs N, 15k SOTA agent, idealized (no-CC) physics.
# Regenerates results/figures/delivery_vs_N_15k.pdf with the F twin axis.
python -u experiments/comparisons/delivery_vs_N.py \
    --ckpt checkpoints/sota/policy.pth \
    --hidden 64 \
    --p_gen 0.4 --p_swap 0.8 --n_ch 4 --cutoff 20 \
    --n_lo 10 --n_hi 15 --n_train_max 12 \
    --horizon 300 --mc_eps 2000 \
    --fidelity \
    --out results/comparisons/delivery_vs_N_15k_fid.json \
    --fig results/figures/delivery_vs_N_15k

echo "eval done -> results/comparisons/delivery_vs_N_15k_fid.json"
