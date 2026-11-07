#!/usr/bin/env bash
# One-time switch of rounds 2 / 3 to the shared GPU pool (tools/final/gpu_pool.sh; 11-07 evening, user):
#   - the job running now on GPU 3 (EV-Flying V10) keeps running; GPU 3 keeps holding 20 GB when idle;
#   - EV-Flying V9 runs on GPU 0 as soon as GPU 0 is free (pool job 520_evf_v9@gpu0);
#   - every other job that has not started (round 2: alpha sweep, LIF 50 ms; round 3: 25 / 100 ms backbones) goes into
#     the pool in priority order and runs on whichever of GPUS is ready: GPU 3 after its current job, GPUs 0 / 1 / 2 as
#     soon as nothing else runs on them; GPUs 0 / 1 / 2 hold nothing between jobs.
# Steps: register the runner of GPU 3 (run/queue/gpu3/runner.pid, so no second runner is ever started there); old
# runners of ours on other GPUs are asked to stop (they may still hold 20 GB; the user wants GPUs 0 / 1 / 2 free
# between our jobs) -> move the pending jobs of GPU 3 into the pool -> queue round 3 into the pool (no duplicates)
# -> start the dispatcher (log run/queue/pool.log).
# Usage (code dir, after `conda activate evuav`): [GPUS="0 1 2 3"] bash tools/deco/pool_round23.sh
set -u
cd "$(dirname "$0")/../.."
GPUS=${GPUS:-"0 1 2 3"}
P=run/queue/pool/pending
die() { echo "POOL SETUP STOPPED: $*"; exit 1; }
[ -f tools/final/gpu_pool.sh ] || die "tools/final/gpu_pool.sh missing (upload deco_round3_20261107.tgz)"
grep -q 'QUEUES="pool"' tools/deco/queue_round3.sh || die "tools/deco/queue_round3.sh is the old version"
if pgrep -f "tools/final/gpu_pool.sh" > /dev/null; then die "a pool dispatcher is running already"; fi

found=" "
old=""
for p in $(pgrep -f "tools/final/gpu_queue.sh"); do
  g=$( { tr '\0' '\n' < "/proc/$p/environ"; } 2>/dev/null | sed -n 's/^GPU=//p' | head -n 1)
  [ -n "$g" ] || continue
  mkdir -p "run/queue/gpu$g"
  if [ "$g" = 3 ]; then
    echo "$p" > "run/queue/gpu$g/runner.pid"
    found="$found$g "
    echo "runner on GPU 3: pid $p (registered, keeps 20 GB when idle)"
  else
    touch "run/queue/gpu$g/STOP"
    rm -f "run/queue/gpu$g/runner.pid"
    old="$old $p"
    echo "old runner on GPU $g: pid $p asked to stop (frees its placeholder)"
  fi
done
alive=""
for k in $(seq 1 12); do                                      # wait up to 2 min for the old runners to exit
  alive=""
  for p in $old; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
  [ -z "$alive" ] && break
  sleep 10
done
[ -z "$alive" ] || echo "note: old runner(s)$alive still busy; they stop after their current job"
case "$found" in *" 3 "*) ;; *) die "no runner found for GPU 3 (check: ps -ef | grep gpu_queue)" ;; esac

mkdir -p "$P"
if [ -e run/queue/gpu3/pending/520_evf_v9.sh ]; then
  mv run/queue/gpu3/pending/520_evf_v9.sh "$P/520_evf_v9@gpu0.sh"
  echo "pool <- 520_evf_v9 (GPU 0 only)"
fi
for f in run/queue/gpu3/pending/[56][0-9][0-9]_*.sh; do
  [ -e "$f" ] || continue
  mv "$f" "$P/"
  echo "pool <- $(basename "$f" .sh)"
done
QUEUES=pool bash tools/deco/queue_round3.sh || die "round 3 was not queued (see above)"

GPUS="$GPUS" HOLD_GPUS="3" nohup bash tools/final/gpu_pool.sh >> run/queue/pool.log 2>&1 < /dev/null &
echo "dispatcher started (pid $!): GPUs $GPUS, only GPU 3 holds 20 GB"
sleep 3
tail -n 4 run/queue/pool.log
echo "pool: $(ls "$P" | tr '\n' ' ')"
echo "POOL SETUP DONE"
