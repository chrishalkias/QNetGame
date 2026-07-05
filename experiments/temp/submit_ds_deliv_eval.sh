#!/bin/bash -l
#SBATCH --job-name=ds_eval
#SBATCH --partition=cpu-zen4
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/ds_eval_%j.out
#SBATCH --error=slurm_logs/ds_eval_%j.err
set -euo pipefail

# Delivery-time-vs-N eval (agent only) for one seed-sweep checkpoint, matching
# the config of the original single-seed ds_ep*.json curves so the seeds are
# directly comparable. Usage: sbatch submit_ds_deliv_eval.sh <run_id>
RID="$1"
cd "$SLURM_SUBMIT_DIR"
# Match submit_seed_run.sh EXACTLY: the module loads must precede the venv or
# torch_geometric hits a circular-import ("no attribute 'typing'") at agent load.
eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

python -u experiments/comparisons/delivery_vs_N.py \
    --ckpt "checkpoints/different_seeds/$RID/policy.pth" --hidden 64 \
    --agent_only \
    --p_gen 0.4 --p_swap 0.8 --n_ch 4 --cutoff 20 \
    --n_lo 10 --n_hi 15 --horizon 300 --mc_eps 2000 \
    --out "results/comparisons/delivery_vs_N_different_seeds/${RID}.json"

echo "done $RID"
