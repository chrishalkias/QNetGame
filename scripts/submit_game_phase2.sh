#!/bin/bash -l
#SBATCH --job-name=qrn_game_p2
#SBATCH --output=slurm_logs/game_p2_%j.out
#SBATCH --error=slurm_logs/game_p2_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#
# ─────────────────────────────────────────────────────────────────────────────
# GAME — PHASE 2 (GRID): train the agent on grids with the topology-general PBRS
# reward, then evaluate vs swap-asap. Legacy backend (NetSquid grid unsupported).
#
# gpu-short for short queue waits. TIMEOUT-TOLERANT: best-checkpoint (greedy grid
# probe) keeps policy.pth = best agent, so a wall-clock timeout still yields a
# usable result. Re-submit to make another attempt if desired.
#
#   (cluster) cd ~/QNetGame && git fetch && git checkout feat/game-phase2-grid && git pull
#   sbatch --account=liacs scripts/submit_game_phase2.sh
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
    sys.exit(f"[GAME P2 ABORT] remote venv missing a dependency: {e}")
print(f"torch {torch.__version__} | numpy {numpy.__version__} | cuda {torch.cuda.is_available()}")
PY

python -u -m game.run_phase2 \
    --save_dir checkpoints/cluster/game/phase2 \
    --eval_episodes 500 \
    --seed 0

echo "Job completed at $(date)"
echo "Checkpoint: checkpoints/cluster/game/phase2/policy.pth"
echo "Grid eval:  checkpoints/cluster/game/phase2/grid_eval.json"
