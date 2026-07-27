#!/bin/bash
# -----------------------------------------------------------------------------
# Sync cluster-generated artifacts DOWN from the ALICE cluster (alice-gw).
# Pulls checkpoints/ and slurm_logs/ only (NO --delete) so local code and other
# results are never clobbered by the remote.
#
#   ./experiments/scripts/sync_download.sh             # pull checkpoints + slurm_logs
#   ./experiments/scripts/sync_download.sh --dry-run   # preview (extra args pass through to rsync)
#
# Override host/path/dirs via env, e.g.:
#   REMOTE_HOST=chalkiasc1@alice-gw ./experiments/scripts/sync_download.sh
#   PULL_DIRS="checkpoints/legacy/cluster/inhomo_001" ./experiments/scripts/sync_download.sh
# -----------------------------------------------------------------------------
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-alice-gw}"
REMOTE_PATH="${REMOTE_PATH:-~/QNetGame}"
PULL_DIRS="${PULL_DIRS:-checkpoints slurm_logs}"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"  # experiments/scripts/ -> repo root

for sub in $PULL_DIRS; do
  echo "Downloading $REMOTE_HOST:$REMOTE_PATH/$sub/ -> $LOCAL_DIR/$sub/"
  mkdir -p "$LOCAL_DIR/$sub"
  rsync -avz "$@" \
    "$REMOTE_HOST:$REMOTE_PATH/$sub/" "$LOCAL_DIR/$sub/"
done
