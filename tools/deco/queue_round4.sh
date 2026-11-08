#!/usr/bin/env bash
# Round 4 (11-08): why does single-step LIF tie BIR at 400 ms? Two checks, each = backbone + V10 decision layer, put into
# the long-running GPU pool (tools/final/gpu_pool.sh, KEEP=1) in priority order:
#   (1) main: BIR without the message budget (configs/final/bb_v5nb_400.yaml), seeds 37 / 38 / 39.
#       Compare with BIR 400 ms with budget (0.8608 / 0.8470 / 0.8497) and LIF (0.8554 / 0.8435 / 0.8612).
#   (2) narrow network, all channels 1/4: BIR (budget scaled 1/4, bb_v5n4_400) and LIF (bb_v5lifn4_400), seed 37.
#       If information per message is the bottleneck once neurons are few, LIF should lose more than BIR.
# Gates: the three configs are uploaded, their output dirs do not exist yet, the pool scripts are the 11-08 version.
# Starts the dispatcher (GPUS default "1 2 3", no placeholder, KEEP=1) if none is running.
# Usage (code dir, after `conda activate evuav`): [GPUS="1 2 3"] bash tools/deco/queue_round4.sh
# Results: log/final/bb_v5nb_400_seed3x, bb_v5n4_400_seed37, bb_v5lifn4_400_seed37 and run/deco/<TAG>/dnm_a0.01_s0_nr_v10;
# pack: bash tools/deco/pack_round4.sh
set -u
cd "$(dirname "$0")/../.."
GPUS=${GPUS:-"1 2 3"}
die() { echo "QUEUE ROUND4 NOT STARTED: $*"; exit 1; }
grep -q 'KEEP=${KEEP:-0}' tools/final/gpu_pool.sh || die "tools/final/gpu_pool.sh is the old version (upload the 11-08 one)"
[ -f tools/final/pool_add.sh ] || die "tools/final/pool_add.sh missing"
for c in bb_v5nb_400 bb_v5n4_400 bb_v5lifn4_400; do [ -f "configs/final/$c.yaml" ] || die "missing configs/final/$c.yaml"; done
for d in bb_v5nb_400_seed37 bb_v5nb_400_seed38 bb_v5nb_400_seed39 bb_v5n4_400_seed37 bb_v5lifn4_400_seed37; do
  [ ! -e "log/final/$d" ] || die "log/final/$d exists already (check it, move it away)"
done
job() {                                      # config seed tag
  echo "NAME=$1 SEED=$2 HZ=\"\" bash tools/final/backbone.sh && CKPT=log/final/$1_seed$2/selected_seed$2.pt TAG=$3 DESIGN=v10 WORKERS=8 bash tools/deco/run_dnm.sh"
}
bash tools/final/pool_add.sh bir400nb_s37 "$(job bb_v5nb_400 37 v5nbw400s37)" > /dev/null
bash tools/final/pool_add.sh bir400nb_s38 "$(job bb_v5nb_400 38 v5nbw400s38)" > /dev/null
bash tools/final/pool_add.sh bir400nb_s39 "$(job bb_v5nb_400 39 v5nbw400s39)" > /dev/null
bash tools/final/pool_add.sh bir400n4_s37 "$(job bb_v5n4_400 37 v5n4w400s37)" > /dev/null
bash tools/final/pool_add.sh lif400n4_s37 "$(job bb_v5lifn4_400 37 lifn4w400s37)" > /dev/null
echo "pool: $(ls run/queue/pool/pending | tr '\n' ' ')"
if pgrep -f "tools/final/gpu_pool.sh" > /dev/null; then
  echo "dispatcher already running: jobs start on the next free GPU"
else
  rm -f run/queue/pool/STOP
  GPUS="$GPUS" HOLD_GPUS=none KEEP=1 nohup bash tools/final/gpu_pool.sh >> run/queue/pool.log 2>&1 < /dev/null &
  echo "dispatcher started (pid $!): GPUs $GPUS, no placeholder, stays on when the pool is empty"
fi
echo "QUEUE ROUND4 DONE"
