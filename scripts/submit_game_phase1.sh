#!/bin/bash -l
#SBATCH --job-name=qrn_game_p1
#SBATCH --output=slurm_logs/game_p1_%j.out
#SBATCH --error=slurm_logs/game_p1_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#
# ─────────────────────────────────────────────────────────────────────────────
# GAME — PHASE 1: curriculum-train the agent on small chains (legacy backend),
# then report gap-to-optimal vs the exact optimal policy and swap-asap.
#
# Engine: legacy (numpy) for fast training — NetSquid is NOT needed here. The
# optimal-comparison step reuses results/optimal_policies/*.pkl produced by
# scripts/submit_optimal_baseline.sh; if those pickles are absent the report
# degrades to swap-asap-only (a warning, not an error).
#
# ── BEFORE SUBMITTING ───────────────────────────────────────────────────────
#   1. Branch feat/game-phase1 (or merged) must be on the cluster:
#         (cluster) cd ~/QNetGame && git fetch && git checkout feat/game-phase1 && git pull
#   2. For the optimal column, run scripts/submit_optimal_baseline.sh first so
#      results/optimal_policies/ contains the N=3,4 n_ch=2 cutoff=5 pg0.90 ps0.90
#      pickles. Otherwise Phase 1 still trains; the report is swap-asap-only.
#   3. Submit:  sbatch --account=liacs scripts/submit_game_phase1.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs
mkdir -p checkpoints/cluster/game

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
eval "$(/usr/bin/modulecmd bash load CUDA/12.4.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"

python - <<'PY'
import sys
try:
    import torch, numpy
except ImportError as e:
    sys.exit(f"[GAME P1 ABORT] remote venv missing a dependency: {e}")
print(f"torch {torch.__version__} | numpy {numpy.__version__} | cuda {torch.cuda.is_available()}")
PY

python -u game/run_phase1.py \
    --save_dir checkpoints/cluster/game/phase1 \
    --policy_dir results/optimal_policies \
    --mc_eps 5000 \
    --seed 0

echo "Job completed at $(date)"
echo "Checkpoint: checkpoints/cluster/game/phase1/policy.pth"
echo "Report:     checkpoints/cluster/game/phase1/optimal_comparison.json"
