#!/bin/bash -l
#SBATCH --job-name=qrn_heat
#SBATCH --output=slurm_logs/heat_%j.out
#SBATCH --error=slurm_logs/heat_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-zen4

# Reproduce the gap-to-optimal + vs-heuristics heatmaps with the swap-shy-fixed
# agent (omni_nopen_15k). SAME config as the original figure (N=4, n_ch=2,
# cutoff=5, 9x9 p_gen/p_swap grid, horizon=30) so it's a clean A/B; only the
# checkpoint changes. Outputs to *_nopen.json (originals untouched). Plot locally
# with experiments/heatmap/plot_heatmap_gap.py after download.
#
# Inputs needed on the cluster (both present): checkpoints/omni_nopen_15k/policy.pth
# and results/heatmaps/heatmap_Topt_N4_9x9.json (agent-independent DP optimum).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/heatmaps

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)"

CK=checkpoints/omni_nopen_15k/policy.pth
MC=${MC:-2000}
GAP=results/heatmaps/heatmap_gap_N4_9x9_nopen.json
HEUR=results/heatmaps/heatmap_heur_N4_9x9_nopen.json

echo "### stage 1: agent (purify + swap-only) vs swap-only DP optimum   (mc_eps=$MC)"
python -u experiments/heatmap/eval_heatmap_gap.py \
    --ckpt_purify "$CK" --ckpt_swaponly "$CK" \
    --mc_eps "$MC" --out "$GAP"

echo "### stage 2: agent vs swap-ASAP / purify-then-swap heuristics      (mc_eps=$MC)"
python -u experiments/heatmap/eval_vs_heuristics.py \
    --agent_json "$GAP" --mc_eps "$MC" --out "$HEUR"

echo "done $(date)"
