#!/bin/bash -l
#SBATCH --job-name=qrn_obs8
#SBATCH --output=slurm_logs/obs8_soft_smoke_%j.out
#SBATCH --error=slurm_logs/obs8_soft_smoke_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# First cluster run of the SEQUENTIAL-SWEEP agent on the reduced 8-feature
# observation (dropped mean_fidelity/in_endnode/frac_available). TINY smoke:
# small chains (N 4-8), soft per-repeater impurities (std 0.10, means 0.7/0.6),
# 100 episodes, single seed. Goal: prove the new step model + 8-feature obs train
# end-to-end on the cluster and produce a checkpoint to eval against
# purify-then-swap and run the policy probes on. NOT a publication agent.
#
# "soft impurities": std 0.10 keeps within-chain rate variation gentle and nearly
# unclipped at these means (per-node uniform mean +/- sqrt(3)*0.10 ~= [0.53, 0.87]
# for p_gen). --prune_unwinnable may resample a bit more with inhomogeneity.
# cpu-zen4 (beats GPU for this tiny GNN); 100 eps finishes in minutes.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  seed: 1"

python -u experiments/training/train.py \
    --run_id "obs8_soft_smoke_s2" \
    --seed 1 \
    --episodes 1000 --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 200 --gamma 0.995 \
    --n_lo 4 --n_hi 8 --n_ch 2 3 \
    --p_gen 0.7 --p_swap 0.6 \
    --p_gen_std 0.20 --p_swap_std 0.20 \
    --cutoff_lo 5 --cutoff_hi 10 \
    --prune_unwinnable \
    --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints

echo "done $(date)"
