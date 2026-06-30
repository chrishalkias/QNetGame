#!/bin/bash -l
#SBATCH --job-name=qrn_cmp
#SBATCH --output=slurm_logs/cmp_%j.out
#SBATCH --error=slurm_logs/cmp_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-zen4

# Comparison experiments #3-#6 vs the trained agent (omni_nopen_3k). Each script's
# defaults already encode the agreed config; eval mode writes results/comparisons/*.json
# (plots are rendered locally with --plot after download).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)"

CK=checkpoints/omni_nopen_3k/policy.pth

echo "### #3 delivery_vs_cutoff"
python -u experiments/comparisons/delivery_vs_cutoff.py --ckpt "$CK"
echo "### #4 delivery_vs_pswap"
python -u experiments/comparisons/delivery_vs_pswap.py  --ckpt "$CK"
echo "### #5 action_composition"
python -u experiments/comparisons/action_composition.py --ckpt "$CK"
echo "### #6 delivery_vs_nch"
python -u experiments/comparisons/delivery_vs_nch.py    --ckpt "$CK"

echo "done $(date)"
