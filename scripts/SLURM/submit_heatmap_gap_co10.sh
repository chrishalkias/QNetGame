#!/bin/bash -l
#SBATCH --job-name=qrn_gap_co10
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-39

# Stage 2 of the gap-to-optimal heatmap: MC-evaluate the SOTA agent
# omni_v3_20k_s3 with purify (panel B) and with PURIFY masked (panel A) against
# the exact T_opt grid from Stage 1, at each cell's own horizon. Cells the
# optimum can't deliver >= min_deliver_rate are skipped (masked in the plot),
# so only the ~71 deliverable cells run the (torch, per-step) agent MC. 81 grid
# points round-robin across 40 array tasks. Merge locally with
#   python experiments/comparisons/merge_json.py \
#     'results/heatmaps/gap_co10_9x9/chunk*.json' \
#     -o results/heatmaps/heatmap_gap_N4_9x9_co10.json

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/heatmaps/gap_co10_9x9

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CKPT="checkpoints/omni_v3_20k_s3/policy.pth"
TOPT="results/heatmaps/heatmap_Topt_N4_9x9_co10.json"
NCHUNKS=40
i=$SLURM_ARRAY_TASK_ID
echo "Node: $(hostname)  chunk $i / $NCHUNKS"

python -u experiments/heatmap/eval_heatmap_gap.py \
    --topt_json "$TOPT" --N 4 --n_ch 2 --cutoff 10 \
    --ckpt_purify "$CKPT" --ckpt_swaponly "$CKPT" \
    --mc_eps 2000 --min_deliver_rate 0.9 \
    --chunk "$i" --nchunks "$NCHUNKS" \
    --out "results/heatmaps/gap_co10_9x9/chunk${i}.json"

echo "done $(date)"
