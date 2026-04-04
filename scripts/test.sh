#!/bin/bash
# Local validation run
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python3 -u train-test/validation.py \
    --run_id v006 \
    --episodes 200 \
    --steps 100 \
    --nodes 10 \
    --n_ch 4 \
    --p_gen 0.85 \
    --p_swap 0.95 \
    --cutoff 15 \
    --topology chain \
    --path checkpoints/cluster_004/ \
    --dict policy.pth \
    --verbose 0 \
    "$@"
