#!/bin/bash -l
#SBATCH --job-name=qrn_dvNinfstd3
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-21

# Perfect-memory delivery_vs_N (cutoff=1e8, N=10..20, p_gen=0.4, p_swap=0.8,
# n_ch=4, H=40000, 5000 eps/point) on INHOMOGENEOUS chains: per-repeater
# rates spread with sigma = 0.3 on both p_gen and p_swap (variance-matched
# uniform, clipped to [0.05,1]). Companion to submit_dvN_infmem.sh (sigma=0).
# Merge locally with
#   python experiments/comparisons/merge_json.py \
#     'results/comparisons/dvN_cinf_std0.3_H40000/*.json' \
#     -o results/comparisons/delivery_vs_N_cinf_std0.3_H40000.json

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons/dvN_cinf_std0.3_H40000

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
echo "Node: $(hostname)  task $i -> N=$N policy=$pol sigma=0.3"

python -u experiments/comparisons/delivery_vs_N.py \
    --ckpt "$CKPT" --policies "$pol" \
    --n_lo "$N" --n_hi "$N" --n_ch 4 \
    --p_gen 0.4 --p_swap 0.8 --p_gen_std 0.3 --p_swap_std 0.3 \
    --cutoff 100000000 --horizon 40000 --mc_eps 5000 \
    --out "results/comparisons/dvN_cinf_std0.3_H40000/N${N}_${pol}.json"

echo "done $(date)"
