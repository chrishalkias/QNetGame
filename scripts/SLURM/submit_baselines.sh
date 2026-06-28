#!/bin/bash -l
#SBATCH --job-name=qrn_baselines
#SBATCH --output=slurm_logs/baselines_%A_%a.out
#SBATCH --error=slurm_logs/baselines_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-23
#
# ─────────────────────────────────────────────────────────────────────────────
# GATHER-ONCE HEURISTIC BASELINE TABLE for the homogeneous QRN chain.
#
# Monte-Carlo statistics (delivery-rate, steps-to-e2e, e2e fidelity; mean+std)
# for random / swap-asap / purify-then-swap, so agent-validation runs compare
# against a FIXED precomputed table instead of re-evaluating the heuristics.
#
# The PHYSICS sweep (p_gen x p_swap x cutoff at N=5, n_ch in {2,4}) is the heavy
# one and is split across a SLURM array: NCHUNKS must equal the array width
# (here 24, i.e. --array=0-23). Each task writes its own CSV shard
#   results/baselines/heuristics_physics.chunkKKofMM.csv
# (one file per chunk -> no write races). Merge afterwards with, e.g.:
#   awk 'FNR==1 && NR!=1{next}{print}' results/baselines/heuristics_physics.chunk*.csv \
#       > results/baselines/heuristics_physics.csv
#
# The tiny N-SCALING sweep (36 cells) runs in one shot on the array's task 0.
#
# PURE CPU + NUMPY (legacy engine). No GPU, no NetSquid, no torch.
#
# ── BEFORE SUBMITTING ───────────────────────────────────────────────────────
#   (cluster) git fetch && git checkout <branch> && git pull
#             ls experiments/baselines/sweep_heuristics.py
#   Submit:   sbatch scripts/SLURM/submit_baselines.sh
#   Adjust --partition / --time / --array to your cluster. Keep NCHUNKS == array
#   width below.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

NCHUNKS=24            # MUST equal the --array width above
EPISODES=300
HORIZON=20000         # common max_steps for all 3 heuristics
PILOT=50              # abort a cell after this many 0-delivery episodes (degenerate corner)

mkdir -p slurm_logs results/baselines

# `module` is not a shell function in the batch environment -- use modulecmd
# directly (same as submit.sh / submit_optimal_baseline.sh). No CUDA needed:
# this is a pure-numpy CPU job (the GNN/torch stack is never imported).
eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID started at $(date) on $(hostname)"

python - <<'PY'
import sys
try:
    import numpy
except ImportError as e:
    sys.exit(f"[BASELINES ABORT] remote venv missing numpy: {e}")
print(f"numpy {numpy.__version__}")
PY

# Self-check first (fast); abort the task if it fails.
python -u experiments/baselines/sweep_heuristics.py --smoke

# Heavy physics sweep: this task's shard of the grid.
python -u experiments/baselines/sweep_heuristics.py \
    --sweep physics \
    --episodes "$EPISODES" \
    --horizon "$HORIZON" \
    --pilot "$PILOT" \
    --chunk "$SLURM_ARRAY_TASK_ID" \
    --nchunks "$NCHUNKS" \
    --out results/baselines/heuristics_physics.csv

# Tiny N-scaling sweep: run once, on task 0 only.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -u experiments/baselines/sweep_heuristics.py \
        --sweep nscaling \
        --episodes "$EPISODES" \
        --horizon "$HORIZON" \
        --pilot "$PILOT" \
        --out results/baselines/heuristics_nscaling.csv
fi

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID completed at $(date)"
