#!/usr/bin/env bash
# Job pool that keeps one GPU occupied (11-14). A holder (tools/v18/hold_gpu.py) keeps HOLD_GB reserved while the card
# is idle and gives back JOB_GB for every running job. WORKERS workers run the lines of QUEUE, one shell command per
# line, from this code dir with only this GPU visible (blank lines and '#' comments skipped; a line runs once, rerun
# it by deleting it from $D/taken); with no new line they wait and look again every minute -- they never exit by
# themselves, so later experiments are just appended to the queue. Logs: $D/pool.log (start / end of every job),
# $D/job_<time>_w<k>.log (its output), $D/holder.log.
# Usage (code dir, after `conda activate evuav`):
#   GPU=2 HOLD_GB=20 JOB_GB=4 WORKERS=3 nohup bash tools/v18/pool_hold.sh > run/pool_gpu2.out 2>&1 < /dev/null &
# Add a job:  echo 'TAG=v5s37 SEED=0 NAME=x ARGS="..." bash tools/v18/run_sdu.sh' >> run/pool_gpu2/queue.txt
# Stop:       touch run/pool_gpu2/STOP   (running jobs finish; the holder frees the card at once)
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:?GPU=<physical id>}
HOLD_GB=${HOLD_GB:-20}
JOB_GB=${JOB_GB:-4}
WORKERS=${WORKERS:-3}
D=run/pool_gpu$GPU
QUEUE=${QUEUE:-$D/queue.txt}
mkdir -p "$D/jobs"
touch "$QUEUE" "$D/taken"
rm -f "$D/STOP" "$D"/jobs/*
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

set_hold() {                                       # reserve HOLD_GB minus JOB_GB per running job
  local r h
  r=$(ls "$D/jobs" | wc -l)
  h=$((HOLD_GB - JOB_GB * r))
  [ "$h" -lt 0 ] && h=0
  echo "$h" > "$D/hold_gb.$$" && mv -f "$D/hold_gb.$$" "$D/hold_gb"
}

free_gb() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$GPU" |
    awk -F', ' '{print int(($1 - $2) / 1024)}'
}

next_job() {                                       # first line not run yet; marks it taken (under the lock)
  (
    flock 9
    while IFS= read -r line; do
      case "$line" in ''|\#*) continue ;; esac
      if ! grep -qxF -- "$line" "$D/taken"; then
        echo "$line" >> "$D/taken"
        echo "$line"
        break
      fi
    done < "$QUEUE"
  ) 9> "$D/lock"
}

worker() {
  local id=$1 job jid rc i
  while [ ! -f "$D/STOP" ]; do
    job=$(next_job)
    if [ -z "$job" ]; then
      sleep 60
      continue
    fi
    jid=$(date +%m%d_%H%M%S)_w$id
    touch "$D/jobs/$jid"
    set_hold
    for i in $(seq 1 60); do                       # the holder frees its share within a few seconds
      [ "$(free_gb)" -ge "$JOB_GB" ] && break
      sleep 5
    done
    echo "$(date '+%m-%d %H:%M:%S') worker $id start: $job" >> "$D/pool.log"
    bash -c "$job" > "$D/job_$jid.log" 2>&1
    rc=$?
    echo "$(date '+%m-%d %H:%M:%S') worker $id end (exit $rc): $job | $(tail -n 1 "$D/job_$jid.log" | cut -c1-150)" \
      >> "$D/pool.log"
    rm -f "$D/jobs/$jid"
    set_hold
    sleep 10
  done
}

set_hold
python -u tools/v18/hold_gpu.py "$D/hold_gb" "$D/STOP" > "$D/holder.log" 2>&1 &
echo "$(date '+%m-%d %H:%M:%S') pool on GPU $GPU: hold $HOLD_GB GB, $JOB_GB GB per job, $WORKERS workers, queue $QUEUE" \
  >> "$D/pool.log"
for i in $(seq 1 "$WORKERS"); do
  worker "$i" &
  sleep 30
done
wait
echo "$(date '+%m-%d %H:%M:%S') POOL GPU $GPU STOPPED" >> "$D/pool.log"
