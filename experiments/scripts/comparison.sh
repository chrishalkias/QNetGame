#!/bin/bash -l
#SBATCH --job-name=qrn_cmp
#SBATCH --output=slurm_logs/cmp_%j.out
#SBATCH --error=slurm_logs/cmp_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-zen4
#
# Generic runner for any experiments/ python entry point.
#   sbatch [--job-name=X] [--time=H:MM:SS] experiments/scripts/comparison.sh <path-under-experiments/> [args...]
#
# $1 is the script path RELATIVE TO experiments/; everything after is forwarded
# verbatim. This replaces the 8 per-figure wrappers deleted 2026-07-26, whose
# only content was the argument list below.
#
# ---------------------------------------------------------------------------
# PAPER FIGURE RECIPES (this block IS the reproduction record; keep it current)
# ---------------------------------------------------------------------------
#
# Fig: delivery time T vs chain size N, with the zero-shot boundary at N=12
#   sbatch --job-name=dvN experiments/scripts/comparison.sh \
#       comparisons/policy_vs_agent/delivery_vs_N.py \
#       --ckpt checkpoints/sota/policy.pth \
#       --p_gen 0.4 --p_swap 0.8 --n_lo 10 --n_hi 15 --n_train_max 12 \
#       --n_ch 4 --cutoff 20 --horizon 300 --mc_eps 2000 \
#       --out results/comparisons/delivery_vs_N_sota.json
#
# Fig: the same, as the headline % reduction vs swap-ASAP
#   ...as above, plus --policies agent swap_asap ; then locally:
#   PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_N.py \
#       --plot --metric delta --out results/comparisons/delivery_vs_N_sota.json
#
# Fig variant: infinite memory (cutoff far beyond any horizon; the plot labels
# it tau=inf). The original run swept per-repeater rate spread sigma across
# 0, 0.1, 0.3 as three companion figures, one array task per (N, policy),
# N=10..20, H=40000, 5000 eps/point:
#   sbatch --job-name=dvNinf --time=10:00:00 experiments/scripts/comparison.sh \
#       comparisons/policy_vs_agent/delivery_vs_N.py \
#       --ckpt checkpoints/sota/policy.pth --policies agent purify_swap \
#       --n_lo 10 --n_hi 20 --n_ch 4 --p_gen 0.4 --p_swap 0.8 \
#       --p_gen_std 0.15 --p_swap_std 0.15 \
#       --cutoff 1000000000 --horizon 40000 --mc_eps 5000 \
#       --out results/comparisons/delivery_vs_N_cinf.json
#
# Fig variant: extended zero-shot range, N up to 20, cutoff=30, H=40000
#   sbatch --job-name=dvN20 --time=10:00:00 experiments/scripts/comparison.sh \
#       comparisons/policy_vs_agent/delivery_vs_N.py \
#       --ckpt checkpoints/sota/policy.pth --policies agent purify_swap \
#       --n_lo 10 --n_hi 20 --n_ch 4 \
#       --p_gen 0.4 --p_swap 0.8 --cutoff 30 --horizon 40000 --mc_eps 150 \
#       --out results/comparisons/delivery_vs_N_c30_H40000.json
#
# Fig: T vs memory cutoff at N=12 (memory pressure, with the fidelity axis)
#   sbatch --job-name=dvcut experiments/scripts/comparison.sh \
#       comparisons/policy_vs_agent/delivery_vs_cutoff.py \
#       --ckpt checkpoints/sota/policy.pth --N 12 --n_ch 4 \
#       --horizon 400 --mc_eps 2000 \
#       --out results/comparisons/delivery_vs_cutoff_N12.json
#
# Fig: T vs p_swap, one line per p_gen, at N=15
#   sbatch --job-name=pswap experiments/scripts/comparison.sh \
#       comparisons/policy_vs_agent/delivery_vs_pswap.py \
#       --ckpt checkpoints/sota/policy.pth --N 15 --n_ch 4 --cutoff 20 \
#       --horizon 300 --mc_eps 2000 \
#       --out results/comparisons/delivery_vs_pswap_N15.json
#
# Fig: T vs inhomogeneity sigma (agents are trained at sigma=0.15)
#   sbatch --job-name=dvstd experiments/scripts/comparison.sh \
#       comparisons/policy_vs_agent/delivery_vs_std.py \
#       --ckpt checkpoints/sota/policy.pth --N 12 --n_ch 4 --cutoff 20 \
#       --horizon 300 --mc_eps 2000 \
#       --out results/comparisons/delivery_vs_std_N12.json
#
# Fig: T vs n_ch (zero-shot transfer in memory size)
#   sbatch --job-name=dvnch experiments/scripts/comparison.sh \
#       comparisons/policy_vs_agent/delivery_vs_nch.py \
#       --ckpt checkpoints/sota/policy.pth --N 12 --cutoff 20 \
#       --horizon 300 --mc_eps 2000 \
#       --out results/comparisons/delivery_vs_nch_N12.json
#
# Fig (appendix): paired 3-seed delivery-time + fidelity barplots
#   sbatch --job-name=seedbar experiments/scripts/comparison.sh \
#       comparisons/agent_vs_agent/agents_seed_barplot.py \
#       --ckpts checkpoints/<s1>/policy.pth checkpoints/<s2>/policy.pth checkpoints/<s3>/policy.pth \
#       --episodes 3000 --horizon 2000
#
# Probe: decision map over (occupancy, normalized age), rows n_ch = 1,2,3
#   sbatch --job-name=decmap --time=08:00:00 experiments/scripts/comparison.sh \
#       policy_probes/decision_map.py --ckpt checkpoints/sota/policy.pth
#
# Probe: quality map over (p_e, p_s), the inhomogeneity readout
#   sbatch --job-name=qualmap --time=08:00:00 experiments/scripts/comparison.sh \
#       policy_probes/quality_map.py --ckpt checkpoints/sota/policy.pth --p_lo 0.15
#
# Probe: permutation feature importance
#   sbatch --job-name=featimp experiments/scripts/comparison.sh \
#       policy_probes/feature_importance.py --ckpt checkpoints/sota/policy.pth
#
# Chunked SLURM arrays: pass --chunk $SLURM_ARRAY_TASK_ID --nchunks N, then
#   PYTHONPATH=src:. python experiments/comparisons/merge_json.py \
#       'results/comparisons/<dir>/*.json' -o results/comparisons/<merged>.json
# ---------------------------------------------------------------------------

source experiments/scripts/_setup.sh

SCRIPT="$1"; shift
echo "running experiments/$SCRIPT $*"
python -u "experiments/$SCRIPT" "$@"
echo "done $(date)"
