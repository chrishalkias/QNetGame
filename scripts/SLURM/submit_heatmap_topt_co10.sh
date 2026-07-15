#!/bin/bash -l
#SBATCH --job-name=qrn_topt_co10
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-26

# Stage 1 of the gap-to-optimal heatmap (cutoff-invariant physics): exact DP
# T_opt + optimum delivery rate over the 9x9 (p_gen,p_swap) grid at N=4, n_ch=2,
# cutoff=10, H=3000. 81 points round-robin across 27 array tasks (~3 pts/task;
# measured ~5-10 min/point after the Repeater.__deepcopy__ speedup). No agent
# column here (--ckpt none -> skipped); the agent eval is Stage 2
# (eval_heatmap_gap.py). Per-chunk JSONs are merged locally with
#   python experiments/comparisons/merge_json.py \
#     'results/heatmaps/topt_co10_9x9/chunk*.json' \
#     -o results/heatmaps/heatmap_Topt_N4_9x9_co10.json

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/heatmaps/topt_co10_9x9

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

NCHUNKS=27
i=$SLURM_ARRAY_TASK_ID
echo "Node: $(hostname)  chunk $i / $NCHUNKS"

python -u experiments/heatmap/optimal_baseline.py \
    --n_list 4 --n_ch 2 --cutoff 10 --horizon 3000 \
    --grid 0.1:0.9:9 --mc_eps_opt 500 \
    --ckpt none \
    --chunk "$i" --nchunks "$NCHUNKS" \
    --out_json "results/heatmaps/topt_co10_9x9/chunk${i}.json"

echo "done $(date)"
