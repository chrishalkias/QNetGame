#!/bin/bash -l
#SBATCH --job-name=qrn_optimal
#SBATCH --output=slurm_logs/optimal_%j.out
#SBATCH --error=slurm_logs/optimal_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4
#
# ─────────────────────────────────────────────────────────────────────────────
# EXACT OPTIMAL-POLICY BASELINE for the QRN chain (n_ch = 2).
#
# Solves the entanglement-distribution MDP exactly by finite-horizon DP
# (= value iteration) on the reachable state space, for each (p_gen, p_swap)
# point and each chain length N. Persists the optimal policy per point so it
# can later be loaded as a validation baseline, and reports T_opt vs swap-asap
# (and vs the trained agent, if a checkpoint is present).
#
# This is the baseline we anchor the learned policy against on SMALL chains,
# where the optimal policy is computable; on larger chains it is intractable and
# we fall back to swap-asap. The cluster's value here is CPU time + walltime:
# it lets us push N past what's comfortable on a laptop (N=5 is the stretch
# goal; ~thousands of states, can take ~1h+).
#
# PURE CPU + NUMPY. No GPU, no NetSquid — this uses the legacy numpy engine
# only (the MDP enumeration drives RepeaterNetwork directly). torch is needed
# ONLY for the optional agent column.
#
# ── BEFORE SUBMITTING (one-time cluster prep) ───────────────────────────────
#   1. NEW FILE STRUCTURE on the cluster (branch refactor/netsquid-m1):
#         (laptop)  git push -u origin refactor/netsquid-m1
#         (cluster) cd <repo> && git fetch && git checkout refactor/netsquid-m1 && git pull
#      Verify:  ls experiments/heatmap/optimal_baseline.py
#
#   2. The remote venv ~/.venvs/qnetgame just needs numpy (1.26.4) + torch,
#      already present from the training runs. NetSquid is NOT required here.
#
#   3. Submit:  sbatch scripts/submit_optimal_baseline.sh
#
#   NOTE: adjust --partition / --time above to match your cluster's CPU queues.
#         To skip the agent column entirely, point --ckpt at a missing file.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs
mkdir -p results/optimal/optimal_policies

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"

# Fail fast with a clear message if the venv is missing numpy.
python - <<'PY'
import sys
try:
    import numpy
except ImportError as e:
    sys.exit(f"[OPTIMAL ABORT] remote venv missing numpy: {e}")
print(f"numpy {numpy.__version__}")
PY

# n_ch=2 is the only tractable channel count for the exact DP.
# N is capped at 4: N=5 measured ~5.5h build PER (p_gen,p_swap) point (6741
# states), i.e. ~38h across the 7-point sweep — not worth it. N=4 already shows
# the optimal policy beating swap-asap (T_opt 19.88 vs 20.90). --save_policy
# writes one pickle per (N, p_gen, p_swap) into results/optimal/optimal_policies/ for
# reuse as a baseline. The JSON is saved incrementally (per point).
python -u experiments/heatmap/optimal_baseline.py \
    --n_list 3,4 \
    --n_ch 2 \
    --cutoff 5 \
    --horizon 30 \
    --mc_eps 5000 \
    --mc_eps_opt 4000 \
    --ckpt checkpoints/cluster/cluster_004/policy.pth \
    --out_json results/optimal/optimal_baseline.json \
    --policy_dir results/optimal/optimal_policies \
    --save_policy

echo "Job completed at $(date)"
echo "Metrics: results/optimal/optimal_baseline.json"
echo "Policies: results/optimal/optimal_policies/"
