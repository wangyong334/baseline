#!/usr/bin/env bash
# Job queue on one GPU without a memory holder (11-15). One worker runs the lines of the queue in order, one shell
# command per line ('#' comments and blank lines skipped; a line runs once: it is recorded in taken, delete it there
# to rerun); with no new line it looks again every minute, so later experiments are just appended. Nothing occupies the
# card between jobs (unlike tools/v18/pool_hold.sh, whose holder keeps a CUDA context even at HOLD_GB=0).
# Logs: $D/queue.log (start / end of every job), $D/job_<time>.log (its output).
# Usage (V20 code dir):  GPU=1 nohup bash tools/v20/queue_gpu.sh > run/queue_gpu1.out 2>&1 < /dev/null &
# Add a job:  echo '<command>' >> run/queue_gpu1/queue.txt          Stop:  touch run/queue_gpu1/STOP (after the job)
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:?GPU=<physical id>}
D=run/queue_gpu$GPU
QUEUE=$D/queue.txt
mkdir -p "$D"
touch "$QUEUE" "$D/taken"
rm -f "$D/STOP"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
echo "$(date '+%m-%d %H:%M:%S') queue on GPU $GPU (no holder): $QUEUE" >> "$D/queue.log"
while [ ! -f "$D/STOP" ]; do
  job=""
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    if ! grep -qxF -- "$line" "$D/taken"; then job=$line; break; fi
  done < "$QUEUE"
  if [ -z "$job" ]; then
    sleep 60
    continue
  fi
  echo "$job" >> "$D/taken"
  jid=$(date +%m%d_%H%M%S)
  echo "$(date '+%m-%d %H:%M:%S') start: $job" >> "$D/queue.log"
  bash -c "$job" > "$D/job_$jid.log" 2>&1
  rc=$?
  echo "$(date '+%m-%d %H:%M:%S') end (exit $rc): $job" >> "$D/queue.log"
done
echo "$(date '+%m-%d %H:%M:%S') QUEUE GPU $GPU STOPPED" >> "$D/queue.log"
