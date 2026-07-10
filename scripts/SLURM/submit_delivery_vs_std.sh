#!/bin/bash -l
#SBATCH --job-name=qrn_dvstd
#SBATCH --output=slurm_logs/dvstd_%j.out
#SBATCH --error=slurm_logs/dvstd_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-zen4

# Delivery time T vs inhomogeneity sigma, uncensored re-run: big horizon so the
# censoring wall (H) stops dominating the mean (see delivery-rate diagnosis:
# swap-ASAP delivered only ~21% within H=300). Params: p_swap=0.75, sigma up to
# 0.25, H=1500.
#   sbatch scripts/SLURM/submit_delivery_vs_std.sh [run_id]   (default omni_nopen_15k)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons
RUN="${1:-omni_nopen_15k}"

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  run=$RUN  $(date)"

python -u experiments/comparisons/delivery_vs_std.py \
    --ckpt "checkpoints/omni_initial/$RUN/policy.pth" \
    --N 10 --p_gen 0.5 --p_swap 0.75 --n_ch 4 --cutoff 20 \
    --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 \
    --horizon 1500 --mc_eps 2000 \
    --out "results/comparisons/delivery_vs_std_pswap075_H1500.json"

echo "done $(date)"
