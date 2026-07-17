#!/bin/bash -l
#SBATCH --job-name=qrn_qheur
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-55

# q-heuristic cluster evaluation: does a single scalar q (both-legal purify
# probability inside the purify_then_swap skeleton) recover the trained agent's
# edge, and where does the optimal q sit as a function of N?
#
# Work list = (N x policy) pairs, round-robin over NCHUNKS array tasks, each
# task writing its own _chunk{i} JSON (never a shared file). Resume-safe: a
# requeued task skips (N, policy) entries already in its JSON.
#
# Roster per N: purify_then_swap + hybrids q in {0.15, 0.215, 0.3, 0.369, 0.5}
# + agents s1, s3 = 8 policies x 7 N = 56 work items, ONE per array task
# (censored agent cells at H=20000 x 500 eps are the walltime risk; keep them
# isolated so a requeue resumes exactly one item).
#
# Merge + plot locally after download:
#   PYTHONPATH=. python experiments/q_heuristic/eval_q_heuristic.py --plot \
#     --out 'results/comparisons/q_heuristic/eval_nch4_pg0.4_ps0.8_co30_chunk*.json' \
#     --logy

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons/q_heuristic

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

NCHUNKS=56
i=$SLURM_ARRAY_TASK_ID
echo "Node: $(hostname)  chunk $i / $NCHUNKS"

python -u experiments/q_heuristic/eval_q_heuristic.py \
    --N 10 11 12 13 14 15 16 \
    --q 0.15 0.215 0.3 0.369 0.5 \
    --agents checkpoints/omni_v3_20k_s1/policy.pth \
             checkpoints/omni_v3_20k_s3/policy.pth \
    --n_ch 4 --p_gen 0.4 --p_swap 0.8 --cutoff 30 \
    --horizon 20000 --episodes 500 --seed 42 \
    --chunk "$i" --nchunks "$NCHUNKS"

echo "chunk $i done"
