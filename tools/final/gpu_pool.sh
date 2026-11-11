#!/usr/bin/env bash
# Shared job pool over several GPUs (11-07). Jobs in run/queue/pool/pending/*.sh run in file-name order on whichever
# managed GPU is ready; a job named <name>@gpu<id>.sh runs only on that GPU. Each GPU runs its jobs through its own
# runner (tools/final/gpu_queue.sh, logs in run/queue/gpu<id>/logs). The pool hands out one job at a time:
#   - a GPU with our runner gets its next job when the runner has nothing pending or running and no other program uses
#     the GPU (our runner's placeholder that keeps HOLD_GB allocated does not count);
#   - a GPU without our runner is taken once nothing at all has run on it for FREE_POLLS polls in a row: the job goes
#     into its queue first, then a runner is started. Only GPUs in HOLD_GPUS (default 3) keep HOLD_GB GB (default 20)
#     allocated once their queue is empty; runners started on the other GPUs hold nothing, so other users can take
#     those GPUs between our jobs, and the next job waits until such a GPU is free again.
# Runners are recognised by run/queue/gpu<id>/runner.pid (written here, or by tools/deco/pool_round23.sh for runners
# started earlier) or by the GPU= / QNAME= entry in their environment.
# When the pool is empty the dispatcher exits and asks the runners it started on GPUs outside HOLD_GPUS to stop after
# their current job (runners on HOLD_GPUS keep their GPU; release one with touch run/queue/gpu<id>/STOP).
# Stop the dispatcher early: touch run/queue/pool/STOP.
# KEEP=1 (11-08): long-running pool. The dispatcher never exits when the pool is empty and never stops runners; new
# jobs dropped into run/queue/pool/pending/ (tools/final/pool_add.sh) start on the next free GPU. HOLD_GPUS=none: no
# GPU keeps a placeholder (idle runners hold no GPU memory).
# 11-10: default GPUS=1 only (the user's rule: the other cards are shared, pool jobs wait for GPU 1).
# 11-10 evening (SHELVED by the user, opt-in only): HOLD_GB=elastic makes runners on HOLD_GPUS keep the card with tools/final/gpu_hold.py, also during
# jobs. Our own processes (placeholder, anything whose cwd is this code dir) never count as "another program". A GPU
# with our runner also gets its next job while another program is on it, provided the memory not used by others is at
# least NEED_GB (unset by default = old rule; our jobs use about 5 GB), because the elastic placeholder makes room.
# Usage (code dir, after `conda activate evuav`):
#   GPUS="0 1 2 3" HOLD_GPUS="3" nohup bash tools/final/gpu_pool.sh >> run/queue/pool.log 2>&1 &
#   GPUS="1 2 3" HOLD_GPUS=none KEEP=1 nohup bash tools/final/gpu_pool.sh >> run/queue/pool.log 2>&1 < /dev/null &
#   POLL=30 FREE_POLLS=2 HOLD_GB=20 ...   poll interval (s), idle polls before taking a free GPU, placeholder size
set -u
cd "$(dirname "$0")/../.."
read -r -a GS <<< "${GPUS:-1}"
POOL=run/queue/pool
POLL=${POLL:-30}
FREE_POLLS=${FREE_POLLS:-2}
HOLD_GB=${HOLD_GB-20}
NEED_GB=${NEED_GB:-}
HERE=$(pwd -P)
HOLD_GPUS=" ${HOLD_GPUS:-3} "
KEEP=${KEEP:-0}
NVSMI=${NVSMI:-nvidia-smi}
mkdir -p "$POOL/pending" "$POOL/given"
ts() { date '+%m-%d %H:%M:%S'; }
pool_size() { ls "$POOL"/pending/*.sh 2>/dev/null | wc -l; }

job_for() {                            # first pool job (file-name order) that GPU $1 may run
  local f b
  for f in $(ls "$POOL"/pending/*.sh 2>/dev/null | sort); do
    b=$(basename "$f" .sh)
    case "$b" in
      *@gpu*) [ "${b##*@gpu}" = "$1" ] && { echo "$f"; return; } ;;
      *) echo "$f"; return ;;
    esac
  done
}

runner_alive() {                       # our gpu_queue.sh runner for GPU $1: pid file, or its environment
  local f=run/queue/gpu$1/runner.pid p env
  if [ -f "$f" ]; then                 # the pid must still be a gpu_queue.sh process (pids get reused)
    p=$(cat "$f")
    if kill -0 "$p" 2>/dev/null && { tr '\0' ' ' < "/proc/$p/cmdline"; } 2>/dev/null | grep -q "gpu_queue.sh"; then
      return 0
    fi
  fi
  for p in $(pgrep -f "tools/final/gpu_queue.sh" 2>/dev/null); do
    env=$( { tr '\0' '\n' < "/proc/$p/environ"; } 2>/dev/null ) || continue
    if echo "$env" | grep -qx "QNAME=gpu$1"; then return 0; fi
    if ! echo "$env" | grep -q "^QNAME=" && echo "$env" | grep -qx "GPU=$1"; then return 0; fi
  done
  return 1
}

foreign_apps() {                       # "pid MiB" of programs on GPU $1 that are not ours; "ERR" if unreadable
  local bus apps p m cwd                # ours = a placeholder (fixed or elastic) or a process whose cwd is this dir
  bus=$($NVSMI --query-gpu=pci.bus_id --format=csv,noheader -i "$1" 2>/dev/null | tr -d ' ')
  [ -n "$bus" ] || { echo ERR; return; }
  apps=$($NVSMI --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
         | tr -d ' ') || { echo ERR; return; }
  echo "$apps" | awk -F, -v b="$bus" '$1 == b {print $2, $3}' | while read -r p m; do
    { tr '\0' ' ' < "/proc/$p/cmdline"; } 2>/dev/null | grep -q "print('holding'\|gpu_hold.py" && continue
    cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null)
    case "$cwd" in "$HERE"|"$HERE"/*) continue ;; esac
    echo "$p $m"
  done
}

others_on() {                          # PIDs of programs on GPU $1 that are not ours; "ERR" if unreadable
  foreign_apps "$1" | awk '{print $1}'
}

room_for_job() {                       # memory on GPU $1 not used by other programs >= NEED_GB
  local total used
  total=$($NVSMI --query-gpu=memory.total --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ')
  [ -n "$total" ] || return 1
  used=$(foreign_apps "$1" | awk '$1 == "ERR" {e = 1} {s += $2} END {if (e) print "ERR"; else print s + 0}')
  [ "$used" != ERR ] && [ $(( total - used )) -ge $(( NEED_GB * 1024 )) ]
}

give() {                               # move the next pool job for GPU $1 into its queue (tag dropped)
  local job b q=run/queue/gpu$1
  job=$(job_for "$1")
  [ -n "$job" ] || return 1
  b=$(basename "$job" .sh)
  mkdir -p "$q/pending"
  cp "$job" "$POOL/given/${b%@gpu*}__gpu$1.sh"
  mv "$job" "$q/pending/${b%@gpu*}.sh"
  echo "[$(ts)] ${b%@gpu*} -> GPU $1"
}

declare -A idle
started=()
echo "[$(ts)] pool started: GPUs ${GS[*]}, poll ${POLL} s, take a free GPU after ${FREE_POLLS} idle polls," \
     "GPUs${HOLD_GPUS% } hold: ${HOLD_GB:-none} (GB when idle, or elastic), the others hold nothing;" \
     "a GPU with our runner takes jobs beside other programs if ${NEED_GB} GB are not theirs; $(pool_size) jobs"
while true; do
  if [ -f "$POOL/STOP" ]; then
    rm -f "$POOL/STOP"
    echo "[$(ts)] STOP found: dispatcher exits ($(pool_size) jobs left in the pool)"
    exit 0
  fi
  if [ "$(pool_size)" -eq 0 ] && [ "$KEEP" = 1 ]; then
    sleep "$POLL"
    continue
  fi
  if [ "$(pool_size)" -eq 0 ]; then
    for g in ${started[@]+"${started[@]}"}; do
      case "$HOLD_GPUS" in *" $g "*) ;; *) touch "run/queue/gpu$g/STOP"; echo "[$(ts)] GPU $g: runner asked to stop after its job" ;; esac
    done
    echo "[$(ts)] pool empty: dispatcher exits"
    exit 0
  fi
  for g in "${GS[@]}"; do
    [ -n "$(job_for "$g")" ] || continue
    q=run/queue/gpu$g
    others=$(others_on "$g")
    if runner_alive "$g"; then
      idle[$g]=0
      if [ -z "$(ls "$q"/pending/*.sh 2>/dev/null)" ] && [ -z "$(ls "$q"/running/*.sh 2>/dev/null)" ] \
         && { [ -z "$others" ] || { [ -n "$NEED_GB" ] && room_for_job "$g"; }; }; then
        give "$g"
      fi
    elif [ -z "$others" ]; then
      idle[$g]=$(( ${idle[$g]:-0} + 1 ))
      if [ "${idle[$g]}" -ge "$FREE_POLLS" ]; then
        give "$g" || continue
        rm -f "$q/STOP"
        case "$HOLD_GPUS" in *" $g "*) hold=$HOLD_GB ;; *) hold= ;; esac
        GPU=$g HOLD_GB=$hold nohup bash tools/final/gpu_queue.sh >> "run/queue/gpu$g.log" 2>&1 < /dev/null &
        echo $! > "$q/runner.pid"
        started+=("$g")
        echo "[$(ts)] GPU $g idle for ${idle[$g]} polls: runner started (pid $!, hold ${hold:-no} GB)"
        idle[$g]=0
      fi
    else
      idle[$g]=0
    fi
  done
  sleep "$POLL"
done
