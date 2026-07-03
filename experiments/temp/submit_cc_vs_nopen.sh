#!/bin/bash -l
#SBATCH --job-name=cc_vs_nopen
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-zen4

# TEMP: delivery_vs_N in a CC-delay env (1 step/hop) comparing the two agents ---
# omni_nopen_15k (trained WITHOUT CC delays, zero-shot) vs omni_cc_15k (trained
# WITH CC delays from scratch) --- plus swap-ASAP / purify-then-swap references.
# N=5..12, mc_eps=2000. Writes two JSONs + overlay PDF; prints the delta table.
#   sbatch experiments/temp/submit_cc_vs_nopen.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons results/figures/temp

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)"

BASE=results/comparisons/cc_vs_nopen_base.json    # nopen15k: agent + purify-then-swap
CC=results/comparisons/cc_vs_nopen_cc.json        # cc15k: agent only

# nopen15k (zero-shot in CC env) + purify-then-swap reference (swap-ASAP dropped:
# near-fully censored at large N in the CC env, so it dominated runtime).
python -u experiments/temp/delivery_vs_N_ccdelay.py \
    --ckpt checkpoints/omni_nopen_15k/policy.pth --drop_swap_asap \
    --n_lo 5 --n_hi 12 --mc_eps 2000 --out "$BASE"

# cc15k (trained with CC delays)
python -u experiments/temp/delivery_vs_N_ccdelay.py \
    --ckpt checkpoints/omni_cc_15k/policy.pth --agent_only \
    --n_lo 5 --n_hi 12 --mc_eps 2000 --out "$CC"

# overlay + delta table
python -u experiments/temp/compare_ft_ccdelay.py \
    --base "$BASE" --ft "$CC" \
    --fig results/figures/temp/delivery_vs_N_cc_vs_nopen

echo "done $(date)"
