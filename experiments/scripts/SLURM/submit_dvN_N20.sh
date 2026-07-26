#!/bin/bash -l
#SBATCH --job-name=qrn_dvN20
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-21

# Paper figure: delivery time vs chain size N=10..20 at (p_gen=0.4,
# p_swap=0.8), cutoff=30, H=40000 (omni_v3 agent vs purify-then-swap only;
# swap-ASAP dropped 2026-07-13). One array task per (N, policy) so wall time =
# the worst censored point (~4.8 h at 150 eps x 40k steps, measured ~350
# steps/s at N=20). Each task writes its own JSON (+ .meta.json provenance
# sidecar); merge locally with
#   python experiments/comparisons/merge_json.py \
#     'results/comparisons/dvN_c30_H40000/*.json' \
#     -o results/comparisons/delivery_vs_N_c30_H40000.json

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons/dvN_c30_H40000

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CKPT="checkpoints/omni_v3_20k_s3/policy.pth"
POLS=(agent purify_swap)

i=$SLURM_ARRAY_TASK_ID
pol=${POLS[$((i % 2))]}
N=$((10 + i / 2))
echo "Node: $(hostname)  task $i -> N=$N policy=$pol"

python -u experiments/comparisons/policy_vs_agent/delivery_vs_N.py \
    --ckpt "$CKPT" --policies "$pol" \
    --n_lo "$N" --n_hi "$N" --n_ch 4 \
    --p_gen 0.4 --p_swap 0.8 --cutoff 30 --horizon 40000 --mc_eps 150 \
    --out "results/comparisons/dvN_c30_H40000/N${N}_${pol}.json"

echo "done $(date)"
