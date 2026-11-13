#!/usr/bin/env bash
# V18 job pool on one GPU (11-13 night; the user gave GPU 1). WORKERS workers take the jobs of QUEUE in order, one
# line per job: "TAG SEED NAME STAGE1 [ARGS...]" (STAGE1 = V17 run dir of the first population, or '-'; blank lines
# and '#' comments are skipped), each through tools/v18/run_sdu.sh. A worker starts a job only when the GPU has at
# least NEED_GB free; a worker exits when no job is left. Jobs appended to QUEUE later are taken by running workers;
# run the pool again to add workers. Every finished or failed job is logged in run/v18/pool.log. A failed job is not
# retried automatically: delete its line from run/v18/queue.taken and start the pool again (run_sdu.sh resumes it from
# the step that failed).
# Usage (V18 code dir, after `conda activate evuav`):
#   GPU=1 WORKERS=3 nohup bash tools/v18/pool_v18.sh > run/v18/pool_main.log 2>&1 < /dev/null &
# Stop after the current jobs: touch run/v18/STOP_POOL. Last line: "POOL V18 FINISHED".
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:?GPU=<physical id>}
WORKERS=${WORKERS:-3}
NEED_GB=${NEED_GB:-4}
QUEUE=${QUEUE:-run/v18/queue.txt}
mkdir -p run/v18
LOCK=run/v18/queue.lock
TAKEN=run/v18/queue.taken
touch "$QUEUE" "$TAKEN"
rm -f run/v18/STOP_POOL
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

next_job() {                                   # first job line not taken yet; marks it taken (under the lock)
  (
    flock 9
    while IFS= read -r line; do
      case "$line" in ''|\#*) continue ;; esac
      if ! grep -qxF -- "$line" "$TAKEN"; then
        echo "$line" >> "$TAKEN"
        echo "$line"
        break
      fi
    done < "$QUEUE"
  ) 9> "$LOCK"
}

free_gb() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$GPU" |
    awk -F', ' '{print int(($1 - $2) / 1024)}'
}

worker() {
  local id=$1 job tag seed name st1 last
  while [ ! -f run/v18/STOP_POOL ]; do
    job=$(next_job)
    [ -z "$job" ] && break
    until [ "$(free_gb)" -ge "$NEED_GB" ]; do sleep 30; done
    set -- $job
    tag=$1 seed=$2 name=$3 st1=$4
    shift 4
    [ "$st1" = "-" ] && st1=""
    echo "$(date '+%m-%d %H:%M:%S') worker $id: start $tag $name s$seed stage1='$st1' ARGS='$*'"
    last=$(TAG=$tag SEED=$seed NAME=$name STAGE1=$st1 ARGS="$*" bash tools/v18/run_sdu.sh 2>&1 | tail -n 1)
    echo "$(date '+%m-%d %H:%M:%S') worker $id: $last" | tee -a run/v18/pool.log
    sleep 20
  done
  echo "$(date '+%m-%d %H:%M:%S') worker $id exits"
}

for i in $(seq 1 "$WORKERS"); do
  worker "$i" &
  sleep 90                                     # staggered: the next worker sees the memory the last one took
done
wait
echo "POOL V18 FINISHED"
