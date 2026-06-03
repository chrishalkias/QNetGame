#!/bin/bash -l
#SBATCH --job-name=qrn_smoke_nsq
#SBATCH --output=slurm_logs/smoke_nsq_%j.out
#SBATCH --error=slurm_logs/smoke_nsq_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#
# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST: train the GNN agent on the NEW NetSquid engine (analytic mode).
#
# Goal: confirm the agent still LEARNS to solve a small imperfect chain on the
# NetSquid backend (success rate climbs, reward improves, loss finite) and that
# the network is physical/learnable — NOT a production run.
#
# Geometry: fixed 5-node chain, imperfect operations (p_gen=0.6, p_swap=0.85),
# finite memory cutoff (10), slightly imperfect generation fidelity (F0=0.95),
# homogeneous (domain randomization OFF for a clean learnability signal),
# dt_seconds=0.0 (training mode: classical-comm resolves next step).
#
# Engine: --backend netsquid --fidelity_mode analytic  (full_dm is real
# density-matrix simulation — far too slow for training; validation only.)
#
# ── BEFORE SUBMITTING (one-time cluster prep) ───────────────────────────────
#   1. NEW FILE STRUCTURE must be on the cluster. This run needs the new
#      quantum_repeater_sim/backends/ package (M0+M1+M3), which lives on branch
#      `refactor/netsquid-m1`. On your laptop push it, then on the cluster pull:
#         (laptop)  git push -u origin refactor/netsquid-m1
#         (cluster) cd <repo> && git fetch && git checkout refactor/netsquid-m1 && git pull
#      Verify on the cluster:  ls quantum_repeater_sim/backends/netsquid/
#
#   2. NETSQUID must exist in the remote venv ~/.venvs/qnetgame. It is NOT a
#      plain pip package — it comes from a private index and needs numpy 1.x:
#         source ~/.venvs/qnetgame/bin/activate
#         pip install "numpy==1.26.4"                       # NetSquid breaks on numpy 2.x
#         pip install --extra-index-url https://pypi.netsquid.org netsquid
#         #   (prompts for your netsquid.org username + password)
#      Verify:  python -c "import netsquid; print(netsquid.__version__)"
#
#   3. Submit:  sbatch scripts/submit_smoke_netsquid.sh
#
#   NOTE: adjust --partition above if your cluster's short GPU queue has a
#   different name (the production scripts use gpu-l4-24g).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs
mkdir -p checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
eval "$(/usr/bin/modulecmd bash load CUDA/12.4.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Fail fast with a clear message if the remote venv is not set up (see prep above).
python - <<'PY'
import sys
try:
    import netsquid, numpy
except ImportError as e:
    sys.exit(f"[SMOKE ABORT] remote venv missing a dependency: {e}\n"
             "  -> see the cluster-prep notes at the top of this script "
             "(install netsquid into ~/.venvs/qnetgame, pin numpy==1.26.4).")
print(f"netsquid {netsquid.__version__} | numpy {numpy.__version__}")
assert numpy.__version__.startswith("1."), \
    f"numpy {numpy.__version__} is 2.x; NetSquid needs 1.26.4 (pip install 'numpy==1.26.4')"
from quantum_repeater_sim.backends import make_backend
make_backend("netsquid", topology="chain", n_repeaters=5, fidelity_mode="analytic")
print("netsquid backend constructs OK")
PY

python -u train-test/train.py \
    --run_id smoke_netsquid_001 \
    --backend netsquid \
    --fidelity_mode analytic \
    --lr 5e-4 \
    --hidden 64 \
    --batch_size 64 \
    --episodes 3000 \
    --max_steps 30 \
    --n_lo 5 \
    --n_hi 5 \
    --heterogeneous \
    --topology chain \
    --p_gen 0.6 \
    --p_swap 0.85 \
    --cutoff 10 \
    --F0 0.95 \
    --channel_loss 0.0 \
    --dt_seconds 0.0 \
    --save_base_dir checkpoints

echo "Job completed at $(date)"
echo "Checkpoint + training curve: checkpoints/smoke_netsquid_001/"
