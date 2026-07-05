#!/bin/bash -l
#SBATCH --job-name=cc_dN
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-zen4

# TEMP: delivery_vs_N with classical-communication delays (1 step/hop).
#   sbatch experiments/temp/submit_ccdelay.sh [extra args -> the python script]
# Writes results/comparisons/delivery_vs_N_ccdelay.json; plot locally with --plot.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)"

python -u experiments/temp/delivery_vs_N_ccdelay.py \
    --ckpt checkpoints/omni_initial/omni_nopen_15k/policy.pth --mc_eps 500 "$@"

echo "done $(date)"
