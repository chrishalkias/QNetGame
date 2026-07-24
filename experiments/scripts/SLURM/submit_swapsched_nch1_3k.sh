#!/bin/bash -l
#SBATCH --job-name=qrn_swsch
#SBATCH --output=slurm_logs/swapsched_nch1_3k_%j.out
#SBATCH --error=slurm_logs/swapsched_nch1_3k_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-short

# Swap-scheduling test run: PURIFY masked (pure swap-scheduler), n_ch=1 (=> 2
# physical qubits interior, 1 per side, the tightest memory). Inhomogeneous
# chain sigma=0.2, fixed p_gen=0.4 / p_swap=0.6 / cutoff=30, curriculum N 7-10,
# 3k episodes. Purpose: does the learned swap schedule differ substantially from
# swap-asap? metrics.json is written only at run end, so wall must exceed the
# full run; 3k eps at n_ch=1 purify-off is well under the 2h wall (cpu-short cap 4h).

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
    --run_id "swapsched_nch1_3k_s1" \
    --seed 1 \
    --episodes 3000 --batch_size 64 --hidden 64 --lr 5e-4 \
    --max_steps 200 --gamma 0.995 \
    --n_lo 7 --n_hi 10 --n_ch 1 \
    --p_gen 0.4 --p_swap 0.6 \
    --p_gen_std 0.2 --p_swap_std 0.2 \
    --cutoff 30 \
    --disable_purify \
    --prune_unwinnable \
    --topology chain --channel_loss 0.0 --F0 1.0 \
    --save_base_dir checkpoints

echo "done $(date)"
