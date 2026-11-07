#!/usr/bin/env bash
# Queue round 3 (11-07 evening; streaming advantage of BIR, user decision: no mechanism change): complete the
# window-length comparison BIR vs LIF to three seeds per point. Missing backbones (50 / 200 / 400 ms have three seeds
# already; LIF 50 ms seeds 38 / 39 are in round 2): LIF 25 ms seeds 38 / 39, BIR 25 ms seed 39, BIR and LIF 100 ms
# seeds 38 / 39. Each job trains the backbone (tools/final/backbone.sh, guarded selection, val / test evaluation, no
# noise evaluation) and then runs the V10 decision layer on it. Order: pairs with existing seeds first, 25 ms first.
# Jobs are named 6xx, so on a queue that still holds round 2 (5xx) they run after it; they wait for the round-2 smoke
# marker (run/deco/SMOKE_R2_OK) like the round-2 jobs. A runner is started for every queue that has none (HOLD_GB
# default 20 once the queue is empty; HOLD_GB= for none).
# Usage (code dir, after `conda activate evuav`): QUEUES="gpu3" bash tools/deco/queue_round3.sh
#   QUEUES="pool": put the jobs into the shared pool of tools/final/gpu_pool.sh (no runner is started here).
#   A job that is already pending or running in any queue or the pool is not queued again.
#   DRY=1 QROOT=<dir>: only write the job files under <dir> (no gates, no runners).
# Results: log/final/bb_v5{,lif}_{25,100}_seed3x/, run/deco/{v5w,lifw}{25,100}s3x/dnm_a0.01_s0_nr_v10/;
# pack: bash tools/deco/pack_round3.sh
set -u
cd "$(dirname "$0")/../.."
read -r -a QS <<< "${QUEUES:-gpu3}"
QROOT=${QROOT:-run/queue}
DRY=${DRY:-0}
HOLD_GB=${HOLD_GB-20}
die() { echo "QUEUE ROUND3 NOT STARTED: $*"; exit 1; }
JOBS=(                                                       # name config seed tag (priority order)
  "lif25_s38 bb_v5lif_25 38 lifw25s38"
  "bir25_s39 bb_v5_25 39 v5w25s39"
  "lif25_s39 bb_v5lif_25 39 lifw25s39"
  "bir100_s38 bb_v5_100 38 v5w100s38"
  "lif100_s38 bb_v5lif_100 38 lifw100s38"
  "bir100_s39 bb_v5_100 39 v5w100s39"
  "lif100_s39 bb_v5lif_100 39 lifw100s39"
)
for q in "${QS[@]}"; do [[ $q =~ ^(gpu[0-9]+|pool)$ ]] || die "queue names are gpu<id> or pool, got '$q'"; done
if [ "$DRY" != 1 ]; then
  grep -q 'gpu-evidence-gb' tools/deco/train_dnm.py || die "upload deco_round2_20261107.tgz first (new train_dnm.py)"
  for j in "${JOBS[@]}"; do
    set -- $j
    [ -f "configs/final/$2.yaml" ] || die "missing configs/final/$2.yaml"
    [ ! -e "log/final/$2_seed$3" ] || die "log/final/$2_seed$3 exists already (check it, move it away)"
  done
fi
for q in "${QS[@]}"; do mkdir -p "$QROOT/$q/pending"; done
i=0
for j in "${JOBS[@]}"; do
  set -- $j
  q=${QS[$((i % ${#QS[@]}))]}
  n=$((610 + 10 * (i / ${#QS[@]})))
  dup=$(ls "$QROOT"/*/pending/[0-9]*_"$1"*.sh "$QROOT"/*/running/[0-9]*_"$1"*.sh 2>/dev/null | head -n 1)
  if [ -n "$dup" ]; then
    echo "skip $1: already queued ($dup)"
    i=$((i + 1))
    continue
  fi
  cat > "$QROOT/$q/pending/${n}_$1.sh" <<EOF
for k in \$(seq 1 240); do [ -f run/deco/SMOKE_R2_OK ] && break; [ -f run/deco/SMOKE_R2_FAILED ] && break; sleep 30; done
[ -f run/deco/SMOKE_R2_OK ] || { echo "smoke r2 not OK: $1 skipped"; exit 1; }
NAME=$2 SEED=$3 HZ="" bash tools/final/backbone.sh && CKPT=log/final/$2_seed$3/selected_seed$3.pt TAG=$4 DESIGN=v10 WORKERS=8 bash tools/deco/run_dnm.sh
EOF
  echo "queued $QROOT/$q/pending/${n}_$1.sh"
  i=$((i + 1))
done

runner_alive() {                                             # a gpu_queue.sh runner serving queue $1 (gpu<id>)?
  local p env
  for p in $(pgrep -f "tools/final/gpu_queue.sh" 2>/dev/null); do
    env=$( { tr '\0' '\n' < "/proc/$p/environ"; } 2>/dev/null ) || continue
    if echo "$env" | grep -qx "QNAME=$1"; then return 0; fi
    if ! echo "$env" | grep -q "^QNAME=" && echo "$env" | grep -qx "GPU=${1#gpu}"; then return 0; fi
  done
  return 1
}
if [ "$DRY" != 1 ]; then
  for q in "${QS[@]}"; do
    [ "$q" = pool ] && continue
    if runner_alive "$q"; then
      echo "runner for $q is alive: the new jobs run after what it already has"
    else
      rm -f "run/queue/$q/STOP"
      GPU=${q#gpu} HOLD_GB=$HOLD_GB nohup bash tools/final/gpu_queue.sh >> "run/queue/$q.log" 2>&1 < /dev/null &
      echo "started runner for $q (pid $!, hold ${HOLD_GB:-none} GB when empty, log run/queue/$q.log)"
    fi
  done
fi
for q in "${QS[@]}"; do echo "$q pending: $(ls "$QROOT/$q/pending" | tr '\n' ' ')"; done
echo "QUEUE ROUND3 DONE: ${#JOBS[@]} jobs over ${QS[*]}"
