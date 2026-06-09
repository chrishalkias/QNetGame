#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Sync the local working tree UP to the ALICE cluster (alice-gw).
# Pushes CODE only — excludes venvs, caches and generated artifacts
# (checkpoints/results/slurm_logs) so cluster outputs are never clobbered, and
# no commit/push is needed (ships the working tree as-is).
#
#   ./scripts/upload.sh             # sync up
#   ./scripts/upload.sh --dry-run   # preview (extra args pass through to rsync)
#
# Override host/path via env, e.g.:
#   REMOTE_HOST=chalkiasc1@alice-gw ./scripts/upload.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-alice-gw}"
REMOTE_PATH="${REMOTE_PATH:-~/QNetGame}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uploading $LOCAL_DIR/ -> $REMOTE_HOST:$REMOTE_PATH/"
rsync -avz --delete \
  --exclude '.git' \
  --exclude '.venv' --exclude '.venv311' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude 'checkpoints' --exclude 'results' --exclude 'slurm_logs' \
  "$@" \
  "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_PATH/"
