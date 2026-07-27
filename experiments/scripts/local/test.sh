#!/bin/bash
# Local validation run
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"

python3 -u experiments/training/validation.py \
    --episodes 200 \
    --steps 100 \
    --nodes 7 \
    --n_ch 4 \
    --p_gen 0.5 \
    --p_swap 0.85 \
    --cutoff 15 \
    --path checkpoints/legacy/cluster/smoke_netsquid_001/ \
    --dict policy.pth \
    --verbose 0 \
    "$@"