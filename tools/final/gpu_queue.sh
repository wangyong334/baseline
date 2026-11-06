#!/usr/bin/env bash
# Hold one GPU and run queued jobs one after another (11-06).
# Start (code dir, after `conda activate evuav`):
#   GPU=<id> HOLD_GB=20 nohup bash tools/final/gpu_queue.sh > run/queue/gpu<id>.log 2>&1 &
# Queue a job: put a shell script into run/queue/gpu<id>/pending/ ; jobs run in file-name order (010_a.sh, 020_b.sh, ...).
#   A job runs from the code dir with GPU=<id>, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<id>: plain python jobs
#   use --device cuda:0, and tools/final/*.sh scripts that read $GPU work unchanged.
# While the queue is empty and HOLD_GB is set, a placeholder process keeps HOLD_GB GB allocated on the GPU, so the card
# stays ours; it is stopped right before the next job starts.
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
start_hold() {
  if [ -z "${HOLD_GB:-}" ] || [ -n "$hold_pid" ]; then return 0; fi
  python -c "import torch, time; x = torch.empty(int(float('$HOLD_GB') * 2 ** 30) // 4, device='cuda:0'); print('holding', flush=True); time.sleep(10 ** 9)" \
    > "$Q/hold.log" 2>&1 &
  hold_pid=$!
  echo "[$(ts)] queue empty: holding $HOLD_GB GB on GPU $GPU (pid $hold_pid)"
}
stop_hold() {
  if [ -z "$hold_pid" ]; then return 0; fi
  kill "$hold_pid" 2>/dev/null
  wait "$hold_pid" 2>/dev/null
  hold_pid=""
  sleep 5
}
trap 'stop_hold; exit 0' INT TERM
echo "[$(ts)] GPU $GPU queue started in $(pwd); python $(command -v python)"
while true; do
  if [ -f "$Q/STOP" ]; then
    stop_hold
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
