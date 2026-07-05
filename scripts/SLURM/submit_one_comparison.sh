#!/bin/bash -l
#SBATCH --job-name=qrn_cmp
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-zen4

# Run ONE comparison experiment so #3-#6 can run as parallel jobs.
#   sbatch --job-name=cmp_cut submit_one_comparison.sh delivery_vs_cutoff.py [extra args]
# mc_eps defaults to 1000 here (half of the script default) for speed; the
# standard errors stay ~2-3 steps, fine for these plots.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  module=$1"

python -u experiments/comparisons/"$1" \
    --ckpt checkpoints/omni_initial/omni_nopen_15k/policy.pth --mc_eps 1000 "${@:2}"

echo "done $(date)"
