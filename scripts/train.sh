#!/bin/bash
# Local training run
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -u train-test/train.py \
    --run_id local_run \
    --lr 5e-4 \
    --hidden 64 \
    --episodes 300 \
    --batch_size 64 \
    --max_steps 20 \
    --n_lo 5 \
    --n_hi 8 \
    --topology chain \
    --p_gen 0.60 \
    --p_swap 0.85 \
    --cutoff 6 \
    --save_base_dir checkpoints/local \
    "$@"
