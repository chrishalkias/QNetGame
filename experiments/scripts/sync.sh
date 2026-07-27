#!/bin/bash
# Cluster sync. Never hand-roll rsync; use this.
#   ./experiments/scripts/sync.sh up   [extra rsync args, e.g. --exclude '.local']
#   ./experiments/scripts/sync.sh down [extra rsync args]
#
# up   = code -> cluster  (never touches checkpoints/ or results/ on the remote)
# down = checkpoints + slurm_logs -> local
#
# Override host/path/dirs via env, e.g.:
#   REMOTE_HOST=chalkiasc1@alice-gw ./experiments/scripts/sync.sh up
#   PULL_DIRS="checkpoints/legacy/cluster/inhomo_001" ./experiments/scripts/sync.sh down
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-alice-gw}"
REMOTE_PATH="${REMOTE_PATH:-~/QNetGame}"
LOCAL_DIR="$PWD"

MODE="${1:?usage: sync.sh up|down [rsync args...]}"; shift

case "$MODE" in
  up)
    # Sync the local working tree UP to the ALICE cluster. Pushes CODE only,
    # excludes venvs, caches and generated artifacts (checkpoints/results/
    # slurm_logs) so cluster outputs are never clobbered, and no commit/push
    # is needed (ships the working tree as-is).
    echo "Uploading $LOCAL_DIR/ -> $REMOTE_HOST:$REMOTE_PATH/"
    rsync -avz --delete \
      --exclude '.git' \
      --exclude '.venv' --exclude '.venv311' \
      --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
      --exclude 'checkpoints' --exclude 'results' --exclude 'slurm_logs' \
      "$@" \
      "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_PATH/"
    ;;
  down)
    # Pull cluster-generated artifacts DOWN. Pulls checkpoints/ and
    # slurm_logs/ only (NO --delete) so local code and other results are
    # never clobbered by the remote.
    PULL_DIRS="${PULL_DIRS:-checkpoints slurm_logs}"
    for sub in $PULL_DIRS; do
      echo "Downloading $REMOTE_HOST:$REMOTE_PATH/$sub/ -> $LOCAL_DIR/$sub/"
      mkdir -p "$LOCAL_DIR/$sub"
      rsync -avz "$@" \
        "$REMOTE_HOST:$REMOTE_PATH/$sub/" "$LOCAL_DIR/$sub/"
    done
    ;;
  *) echo "unknown mode '$MODE' (want: up|down)" >&2; exit 2 ;;
esac
