#!/bin/bash -l
#SBATCH --job-name=qrn_omniCC15k
#SBATCH --output=slurm_logs/omniCC15k_%j.out
#SBATCH --error=slurm_logs/omniCC15k_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# Same config as submit_omni_15k.sh (ranges + curriculum + reward), but trained
# WITH classical-communication delays from scratch -> the omni_cc_15k agent.
#   dt_seconds = spacing/c_fiber = 50/200000 = 2.5e-4  => exactly 1 step per hop.
# Horizon raised 200 -> 600 because CC delays make delivery much slower (an N=12
# chain needs ~600+ steps under CC), so large-N episodes can still complete.
# best-ckpt window opens at ~ep 13500 (0.9*episodes), same as omni_nopen_15k.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)"

python -u experiments/training/train.py \
    --run_id omni_cc_15k \
    --episodes 15000 --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 600 --gamma 0.995 \
    --n_lo 4 --n_hi 12 --n_ch 2 3 4 \
    --p_gen 0.4 0.9 --p_swap 0.4 0.9 \
    --p_gen_std 0.15 --p_swap_std 0.15 \
    --cutoff_lo 10 --cutoff_hi 40 \
    --prune_unwinnable \
    --topology chain --dt_seconds 2.5e-4 --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints

echo "done $(date)"
