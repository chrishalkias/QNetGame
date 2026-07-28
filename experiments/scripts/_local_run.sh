#!/bin/bash
# Local smoke runs. NOT the cluster path (see comparison.sh / train_*.sh).
#   ./experiments/scripts/_local_run.sh train [extra train.py args...]
#   ./experiments/scripts/_local_run.sh test  [extra validation.py args...]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"

MODE="${1:?usage: _local_run.sh train|test [args...]}"; shift

case "$MODE" in
  train)
    "$PY" -u experiments/training/train.py \
        --run_id local_run \
        --lr 5e-4 --hidden 64 --episodes 300 --batch_size 64 \
        --max_steps 20 --n_lo 5 --n_hi 10 \
        --p_gen 0.60 --p_swap 0.85 --cutoff 9 \
        --save_base_dir checkpoints/local \
        "$@"
    ;;
  test)
    "$PY" -u experiments/training/validation.py \
        --path checkpoints/local/local_run/ --dict policy.pth \
        --episodes 100 --steps 100 --nodes 6 --n_ch 4 \
        --p_gen 0.5 --p_swap 0.85 --cutoff 20 \
        "$@"
    ;;
  *) echo "unknown mode '$MODE' (want: train|test)" >&2; exit 2 ;;
esac
