#!/usr/bin/env bash
# Hold one GPU and run queued jobs one after another (11-06).
# Start (code dir, after `conda activate evuav`):
#   GPU=<id> HOLD_GB=20 nohup bash tools/final/gpu_queue.sh > run/queue/gpu<id>.log 2>&1 &
# Queue a job: put a shell script into run/queue/gpu<id>/pending/ ; jobs run in file-name order (010_a.sh, 020_b.sh, ...).
#   A job runs from the code dir with GPU=<id>, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<id>: plain python jobs
#   use --device cuda:0, and tools/final/*.sh scripts that read $GPU work unchanged.
# While the queue is empty and HOLD_GB is set, a placeholder process keeps HOLD_GB GB allocated on the GPU, so the card
# stays ours; it is stopped right before the next job starts.
# HOLD_GB=elastic (11-10): instead, tools/final/gpu_hold.py runs for the runner's whole life (also during jobs) and keeps
# only a small margin free (1 GB idle, 4 GB during a job, 8 GB for 5 min after a job or a new process of ours starts);
# it is restarted if it dies and released when the runner stops (pid in $Q/hold.pid, log $Q/hold_elastic.log).
# Stop: touch run/queue/gpu<id>/STOP (the runner exits after the running job). Logs: run/queue/gpu<id>.log (runner),
# run/queue/gpu<id>/logs/<job>.log (each job); finished jobs move to done/ or failed/.
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:?GPU=<physical id>}
Q=run/queue/${QNAME:-gpu$GPU}
mkdir -p "$Q/pending" "$Q/running" "$Q/done" "$Q/failed" "$Q/logs"
export GPU CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
hold_pid=""
ts() { date '+%m-%d %H:%M:%S'; }
elastic_alive() {                      # the elastic placeholder of this queue is running
  local p
  [ -f "$Q/hold.pid" ] || return 1
  p=$(cat "$Q/hold.pid")
  kill -0 "$p" 2>/dev/null && { tr '\0' ' ' < "/proc/$p/cmdline"; } 2>/dev/null | grep -q "gpu_hold.py"
}
start_hold() {
  if [ "${HOLD_GB:-}" = elastic ]; then
    if ! elastic_alive; then
      rm -f "$Q/HOLD_STOP"
      nohup python -u tools/final/gpu_hold.py --gpu "$GPU" --queue "$Q" >> "$Q/hold_elastic.log" 2>&1 < /dev/null &
      echo "[$(ts)] elastic hold started on GPU $GPU (pid $!, log $Q/hold_elastic.log)"
    fi
    return 0
  fi
  if [ -z "${HOLD_GB:-}" ] || [ -n "$hold_pid" ]; then return 0; fi
  python -c "import torch, time; x = torch.empty(int(float('$HOLD_GB') * 2 ** 30) // 4, device='cuda:0'); print('holding', flush=True); time.sleep(10 ** 9)" \
    > "$Q/hold.log" 2>&1 &
  hold_pid=$!
  echo "[$(ts)] queue empty: holding $HOLD_GB GB on GPU $GPU (pid $hold_pid)"
}
stop_hold() {                          # the elastic placeholder is not stopped between jobs
  if [ -z "$hold_pid" ]; then return 0; fi
  kill "$hold_pid" 2>/dev/null
  wait "$hold_pid" 2>/dev/null
  hold_pid=""
  sleep 5
}
release_elastic() { [ "${HOLD_GB:-}" = elastic ] && elastic_alive && touch "$Q/HOLD_STOP"; return 0; }
trap 'stop_hold; release_elastic; exit 0' INT TERM
echo "[$(ts)] GPU $GPU queue started in $(pwd); python $(command -v python)"
[ "${HOLD_GB:-}" = elastic ] && start_hold   # elastic: hold from the start, also during the first job
while true; do
  if [ -f "$Q/STOP" ]; then
    stop_hold
    release_elastic
    rm -f "$Q/STOP"
    echo "[$(ts)] STOP found: runner exits"
    exit 0
  fi
  job=$(ls "$Q"/pending/*.sh 2>/dev/null | sort | head -n 1)
  if [ -z "$job" ]; then
    start_hold
    sleep 30
    continue
  fi
  [ "${HOLD_GB:-}" = elastic ] && start_hold   # restart the elastic placeholder if it died
  stop_hold
  name=$(basename "$job" .sh)
  mv "$job" "$Q/running/"
  sed -i 's/\r$//' "$Q/running/$name.sh"
  echo "[$(ts)] start $name"
  if bash "$Q/running/$name.sh" > "$Q/logs/$name.log" 2>&1; then
    mv "$Q/running/$name.sh" "$Q/done/"
    echo "[$(ts)] done $name: $(tail -n 1 "$Q/logs/$name.log")"
  else
    mv "$Q/running/$name.sh" "$Q/failed/"
    echo "[$(ts)] FAILED $name: $(tail -n 1 "$Q/logs/$name.log")"
  fi
done
