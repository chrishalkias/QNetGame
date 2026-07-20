#!/bin/bash -l
#SBATCH --job-name=qrn_qcond
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-55

# State-conditioned q-heuristic cluster evaluation: does closing the loop on
# state (q_i = sigmoid(coef . state)) recover more of the trained agent's
# purify-selectivity edge than the constant-q hybrids do? Roster adds
# qcond_s1 / qcond_s3 (from q_conditional_{s1,s3}.json) alongside the
# constant-q hybrids and the two trained agents, same purify_then_swap
# skeleton throughout.
#
# Work list = (N x policy) pairs, round-robin over NCHUNKS array tasks, each
# task writing its own _chunk{i} JSON (never a shared file). Resume-safe: a
# requeued task skips (N, policy) entries already in its JSON.
#
# Roster per N: purify_then_swap + hybrids q in {0.215, 0.369} + qcond_s1 +
# qcond_s3 + agents s1, s3 = 7 policies x 8 N = 56 work items, ONE per array
# task.
#
# Merge + plot locally after download:
#   PYTHONPATH=src:. python experiments/q_heuristic/eval_q_heuristic.py --plot \
#     --out 'results/comparisons/q_heuristic/qcond_nch4_pg0.4_ps0.8_co30_chunk*.json' \
#     --logy

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons/q_heuristic

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

NCHUNKS=56
i=$SLURM_ARRAY_TASK_ID
echo "Node: $(hostname)  chunk $i / $NCHUNKS"

python -u experiments/q_heuristic/eval_q_heuristic.py \
    --N 5 6 7 8 9 10 11 12 \
    --q 0.215 0.369 \
    --qcond experiments/q_heuristic/q_conditional_s1.json \
            experiments/q_heuristic/q_conditional_s3.json \
    --agents checkpoints/omni_v3_20k_s1/policy.pth \
             checkpoints/omni_v3_20k_s3/policy.pth \
    --n_ch 4 --p_gen 0.4 --p_swap 0.8 --cutoff 30 \
    --horizon 5000 --episodes 500 --seed 42 \
    --out results/comparisons/q_heuristic/qcond_nch4_pg0.4_ps0.8_co30.json \
    --chunk "$i" --nchunks "$NCHUNKS"

echo "chunk $i done"
