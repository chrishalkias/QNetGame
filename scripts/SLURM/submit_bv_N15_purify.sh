#!/bin/bash -l
#SBATCH --job-name=qrn_bv_n15_pur
#SBATCH --output=slurm_logs/bv_n15_pur_%j.out
#SBATCH --error=slurm_logs/bv_n15_pur_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# Regenerate the generalization heatmap, N=15 panel only, infinite memory
# coherence (cutoff=1e9: links never expire, no decoherence), SOTA
# omni_nopen_15k agent vs purify-then-swap. (Old figure: legacy cluster_004
# vs swap-ASAP with per-cell adaptive cutoff.)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_logs
mkdir -p results/batch_validate/N15_purify_infmem

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"

python -u experiments/training/batch_validate.py \
    --model checkpoints/omni_initial/omni_nopen_15k/policy.pth \
    --episodes 200 \
    --seed 42 \
    --save_dir results/batch_validate/N15_purify_infmem \
    --sweep pgen_pswap \
    --node_counts 15 \
    --baseline purify_swap \
    --fixed_cutoff 1000000000

echo "Job completed at $(date)"
