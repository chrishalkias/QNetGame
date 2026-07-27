#!/bin/bash -l
#SBATCH --job-name=qrn_inh15k
#SBATCH --output=slurm_logs/inhomog_15k_%j.out
#SBATCH --error=slurm_logs/inhomog_15k_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# High-inhomogeneity smoke run (ports+purify physics, branch fix/purification):
# 15k episodes, single seed. Goal: force the agent to learn a pe-ps-DEPENDENT
# policy by making per-repeater rates vary WIDELY within a chain, so obs
# features 6/7 (per-node p_gen/p_swap) carry real signal instead of being
# bottom-ranked as in the homogeneous runs.
#
# Rate spread: means FIXED centrally (0.5/0.5) with std 0.25. A single value on
# --p_gen/--p_swap is a fixed mean; the per-node uniform spans mean +/-
# sqrt(3)*std = [0.07, 0.93], almost unclipped -> near-uniform coverage of the
# legal [0.05, 1.0] band. (A wide mean RANGE would NOT do this: it only shifts a
# whole chain's average, creating zero within-chain contrast. And a mean near a
# bound + large std would clip one tail onto the bound, shrinking the realized
# std. mean/std are NOMINAL -- report realized per-node stats from the run.)
#
# --prune_unwinnable resamples more with high std (a low-p_swap bottleneck node
# can make a cell unwinnable under the swap-asap oracle): expect slower ep/min.
# cpu-zen4 (7-day cap) beats GPU for this tiny GNN; 15k @ ~28 ep/min ~= 9h, so
# the 20h wall has margin. metrics.json is written only at run end.
# This is a SMOKE run (one seed) -- scale to --array=1-3 if features 6/7 rise.

source experiments/scripts/_setup.sh
mkdir -p checkpoints
echo "Node: $(hostname)  seed: 1"

python -u experiments/training/train.py \
    --run_id "inhomog_15k_s1" \
    --seed 1 \
    --episodes 15000 --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 200 --gamma 0.995 \
    --n_lo 4 --n_hi 12 --n_ch 2 3 4 \
    --p_gen 0.5 --p_swap 0.5 \
    --p_gen_std 0.25 --p_swap_std 0.25 \
    --cutoff_lo 10 --cutoff_hi 50 \
    --prune_unwinnable \
    --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints

echo "done $(date)"
