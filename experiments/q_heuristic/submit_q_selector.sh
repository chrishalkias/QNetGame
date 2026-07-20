#!/bin/bash -l
#SBATCH --job-name=qrn_qsel
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-55

# The q-selector question (follow-up to the qcond run, job 4441077): the constant
# coin q=0.215 matched the trained agents at ONE cell (n_ch=4, pg=0.4, ps=0.8,
# co=30). Is a single fixed q enough across the whole training family, or does
# the optimal q drift with the cell, with the agent tracking it implicitly (an
# amortized rate controller)? Roster per cell: purify_then_swap + constant coins
# q in {0.1, 0.215, 0.369, 0.55} + agents s1, s3 = 7 policies. Per-cell argmin
# over the q grid locates q*(cell); fixed-q vs agent per cell answers the
# selector question.
#
# 8 cell configs x 7 chunks = 56 array tasks; each invocation handles one
# (n_ch, p_gen, p_swap, cutoff) config at N in {6, 9, 12}, its (N x policy)
# work list (21 items) split over 7 chunks of 3. Own JSON per config x chunk,
# resume-safe. Note n_ch=2 is deliberately absent: with two qubits per node,
# can_swap (2 distinct partners) and can_purify (2 same-partner) are mutually
# exclusive, both-legal states do not exist, and every q is action-identical.
#
# Merge + plot locally after download (per config):
#   PYTHONPATH=src:. python experiments/q_heuristic/eval_q_heuristic.py --plot \
#     --out 'results/comparisons/q_heuristic/qselector/eval_nch3_pg0.4_ps0.8_co30_chunk*.json' \
#     --fig results/figures/q_heuristic/qsel_nch3_pg0.4_ps0.8_co30 --logy

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons/q_heuristic/qselector

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# n_ch  p_gen  p_swap  cutoff   (axes: memory 3/6, rates, cutoff 10/50;
# the reference cell nch4 pg0.4 ps0.8 co30 is already covered by job 4441077)
CONFIGS=(
  "3 0.4 0.8 30"
  "6 0.4 0.8 30"
  "4 0.4 0.4 30"
  "4 0.6 0.6 30"
  "4 0.8 0.8 30"
  "4 0.8 0.4 30"
  "4 0.4 0.8 10"
  "4 0.4 0.8 50"
)
NCHUNKS=7
cfg_id=$((SLURM_ARRAY_TASK_ID / NCHUNKS))
chunk=$((SLURM_ARRAY_TASK_ID % NCHUNKS))
read -r nch pg ps co <<< "${CONFIGS[$cfg_id]}"
echo "Node: $(hostname)  config $cfg_id (nch=$nch pg=$pg ps=$ps co=$co)  chunk $chunk / $NCHUNKS"

python -u experiments/q_heuristic/eval_q_heuristic.py \
    --N 6 9 12 \
    --q 0.1 0.215 0.369 0.55 \
    --agents checkpoints/omni_v3_20k_s1/policy.pth \
             checkpoints/omni_v3_20k_s3/policy.pth \
    --n_ch "$nch" --p_gen "$pg" --p_swap "$ps" --cutoff "$co" \
    --horizon 10000 --episodes 300 --seed 42 \
    --out "results/comparisons/q_heuristic/qselector/eval_nch${nch}_pg${pg}_ps${ps}_co${co}.json" \
    --chunk "$chunk" --nchunks "$NCHUNKS"

echo "config $cfg_id chunk $chunk done"
