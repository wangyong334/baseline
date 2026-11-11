#!/usr/bin/env bash
# Linux server only. The existing gpu_pool/gpu_queue remain the scheduler.
# Explicit --recover-reboot archives stale queue jobs/control files, never resumes them.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p run/queue/pool
exec 9>run/queue/pool/bst_final_dispatch.lock
if ! flock -n 9; then
  echo "This project's final pool wrapper is already running; no duplicate started."
  exit 0
fi
set +e
python tools/bst/queue_final.py pool-preflight "$@"
status=$?
set -e
if [ "$status" -eq 3 ]; then exit 0; fi
if [ "$status" -ne 0 ]; then exit "$status"; fi
export GPUS=1 HOLD_GPUS=1 HOLD_GB=20 KEEP=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
# QNAME would redirect workers to an unexpected queue while the dispatcher
# still watches gpu1/. Force the original scheduler's standard queue layout.
unset NEED_GB QNAME
# The wrapper holds the lock, but children MUST NOT inherit it: otherwise an
# idle gpu_queue child could retain the dispatch lock after the pool exits.
bash tools/final/gpu_pool.sh 9>&- &
dispatcher=$!
trap 'kill "$dispatcher" 2>/dev/null || true; wait "$dispatcher" 2>/dev/null || true' INT TERM
echo "Final pool dispatcher PID=$dispatcher; GPU=1 only; idle hold=20 GiB."
wait "$dispatcher"
