#!/bin/bash
# Sourced (not executed) by every SLURM script in this directory.
# Loads the ALICE modules, activates the project venv, and puts src/ on the path.
# Usage, from a script whose CWD is $SLURM_SUBMIT_DIR:
#     source experiments/scripts/_setup.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
mkdir -p slurm_logs results/comparisons
echo "Node: $(hostname)  repo: $PWD  python: $(python -V 2>&1)"
