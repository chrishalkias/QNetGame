#!/bin/bash -l
#SBATCH --job-name=qrn_cmp_opt
#SBATCH --output=slurm_logs/cmp_opt_%j.out
#SBATCH --error=slurm_logs/cmp_opt_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#
# ─────────────────────────────────────────────────────────────────────────────
# COMPARE-vs-OPTIMAL: SWAP-ONLY agent on a homogeneous N=4 / n_ch=2 chain, with
# per-episode (p_gen, p_swap) drawn from the discrete grid {0.3,0.5,0.7,0.9}^2
# (cutoff=5). Logs, each episode on the SAME seeded net, the greedy agent vs
# swap-asap vs random vs the EXACT DP-optimal (swap-only) policy -> the agent
# should climb toward the Optimal line in training_compare.png.
#
# REQUIRES the precomputed optimal pickles at results/optimal_policies/ — these
# are EXCLUDED by scripts/upload.sh, so push them once:
#   rsync -avz results/optimal_policies/ alice-gw:~/QNetGame/results/optimal_policies/
#
#   (cluster) cd ~/QNetGame && sbatch --account=liacs scripts/submit_compare_optimal.sh
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

python -u train-test/compare_optimal_smoke.py \
    --episodes 5000 \
    --run_id compare_optimal \
    --hidden 64 \
    --lr 5e-4 \
    --save_base_dir checkpoints

echo "Job completed at $(date)"
echo "Crossover plot: checkpoints/compare_optimal/training_compare.png"
