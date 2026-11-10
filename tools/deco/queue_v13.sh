#!/usr/bin/env bash
# V13 round 1 (11-10): tools/deco/run_v13.sh on the three 50 ms BIR backbones, one job per tag in the shared GPU pool
# (tools/final/gpu_pool.sh, KEEP=1), in the order of TAGS.
# Gates: the new files are uploaded, the V11 versions of evidence.py / train_dnm.py are in place, z0 of every tag exists,
# no V13 result dir exists yet, the pool scripts are the 11-08 version.
# Starts the dispatcher (GPUS default "1 2 3", no placeholder, KEEP=1) if none is running.
# Usage (code dir, after `conda activate evuav`): [GPUS="1 2 3"] [TAGS="v5s37 v5s38 v5s39"] bash tools/deco/queue_v13.sh
# Results: run/deco/<TAG>/dnm_m0.001_s0_nr_v13/{summary.json, vow/vow.json}; pack: bash tools/deco/pack_v13.sh
set -u
cd "$(dirname "$0")/../.."
GPUS=${GPUS:-"1 2 3"}
TAGS=${TAGS:-"v5s37 v5s38 v5s39"}
die() { echo "QUEUE V13 NOT STARTED: $*"; exit 1; }
grep -q 'KEEP=${KEEP:-0}' tools/final/gpu_pool.sh || die "tools/final/gpu_pool.sh is the old version (upload the 11-08 one)"
[ -f tools/final/pool_add.sh ] || die "tools/final/pool_add.sh missing"
for f in tools/deco/vow.py tools/deco/fit_vow.py tools/deco/run_v13.sh tools/deco/sdn_v11.py tools/deco/fit_range.py; do
  [ -f "$f" ] || die "missing $f (upload it)"
done
grep -q '"martingale"' tools/deco/evidence.py || die "tools/deco/evidence.py has no --band martingale (upload the V11 one)"
grep -q 'v11-rho' tools/deco/train_dnm.py || die "tools/deco/train_dnm.py has no V11 options (upload the V11 one)"
for t in $TAGS; do
  for s in train val test; do [ -f "run/deco/$t/z0_$s/DONE" ] || die "run/deco/$t/z0_$s/DONE missing (wrong TAGS?)"; done
  [ ! -e "run/deco/$t/dnm_m0.001_s0_nr_v13" ] || die "run/deco/$t/dnm_m0.001_s0_nr_v13 exists already (check, move away)"
done
for t in $TAGS; do
  bash tools/final/pool_add.sh v13_$t "TAG=$t WORKERS=8 bash tools/deco/run_v13.sh" || die "pool_add v13_$t"
done
echo "pool: $(ls run/queue/pool/pending | tr '\n' ' ')"
if pgrep -f "tools/final/gpu_pool.sh" > /dev/null; then
  echo "dispatcher already running: the jobs start on the next free GPU"
else
  rm -f run/queue/pool/STOP
  GPUS="$GPUS" HOLD_GPUS=none KEEP=1 nohup bash tools/final/gpu_pool.sh >> run/queue/pool.log 2>&1 < /dev/null &
  echo "dispatcher started (pid $!): GPUs $GPUS, no placeholder, stays on when the pool is empty"
fi
echo "QUEUE V13 DONE"
