#!/bin/bash -l
#SBATCH --job-name=qrn_pswap15
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-69

# Paper figure: delivery time vs p_swap at N=15, cutoff=40, H=20000 (omni_v3
# agent vs purify-then-swap only; swap-ASAP dropped 2026-07-13). One array task
# per (p_gen, p_swap, policy) cell so wall time = the single worst censored
# cell (~2.5 h at 200 eps x 20k steps, measured ~450 steps/s). Each task writes
# its own JSON (+ .meta.json provenance sidecar); merge locally with
#   python experiments/comparisons/merge_json.py \
#     'results/comparisons/pswap_N15_c40/*.json' \
#     -o results/comparisons/delivery_vs_pswap_N15_c40_H20000.json

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons/pswap_N15_c40

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CKPT="checkpoints/omni_v3_20k_s3/policy.pth"
PGS=(0.4 0.5 0.6 0.7 0.8)
PSS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
POLS=(agent purify_swap)

i=$SLURM_ARRAY_TASK_ID
pol=${POLS[$((i % 2))]}
cell=$((i / 2))
pg=${PGS[$((cell / 7))]}
ps=${PSS[$((cell % 7))]}
echo "Node: $(hostname)  task $i -> p_gen=$pg p_swap=$ps policy=$pol"

python -u experiments/comparisons/delivery_vs_pswap.py \
    --ckpt "$CKPT" --policies "$pol" \
    --N 15 --n_ch 4 --cutoff 40 --horizon 20000 --mc_eps 200 \
    --p_gens "$pg" --p_swaps "$ps" \
    --out "results/comparisons/pswap_N15_c40/pg${pg}_ps${ps}_${pol}.json"

echo "done $(date)"
