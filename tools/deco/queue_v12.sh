#!/usr/bin/env bash
# DeCo-SNN V12 final round (11-09): the complete end-to-end network, seeds 37 / 38 / 39, plus three ablations at seed 37
# (no feedback loop; gradient coupling = the neurons' loss also reaches the priors; the V10 / V11 birth band), all into
# the long-running GPU pool (tools/final/gpu_pool.sh, KEEP=1), in this order.
# Gates before anything is queued: the uploaded files match Res/v12/V12_SERVER_SHA256SUMS, the unit tests pass
# (29 tests), the output dirs are new. The first pool job is a GPU smoke test (2 short epochs, 2 recordings per split,
# runtime on 1 recording, ~10 min); the other jobs wait for its marker run/deco/v12/SMOKE_OK and stop if it fails.
# The dispatcher is NOT restarted here (see the commands that came with this script for opening GPUs 0-3).
# Usage (code dir, after `conda activate evuav`): bash tools/deco/queue_v12.sh
#   ABLATIONS=0 bash tools/deco/queue_v12.sh       only the three seeds
# Results: run/deco/v12/<name>_seed<seed>/ ; pack: bash tools/deco/pack_v12.sh
set -u
cd "$(dirname "$0")/../.."
die() { echo "QUEUE V12 NOT STARTED: $*"; exit 1; }
[ -f Res/v12/V12_SERVER_SHA256SUMS ] || die "Res/v12/V12_SERVER_SHA256SUMS missing"
sha256sum -c --quiet Res/v12/V12_SERVER_SHA256SUMS || die "uploaded files differ from Res/v12/V12_SERVER_SHA256SUMS"
grep -q 'KEEP=${KEEP:-0}' tools/final/gpu_pool.sh || die "tools/final/gpu_pool.sh is the old version"
[ -f tools/final/pool_add.sh ] || die "tools/final/pool_add.sh missing"
out=$(python -m unittest tests.test_deco_v12 tests.test_deco_v11 tests.test_nested_v11 2>&1 | tail -n 3)
echo "$out" | grep -q "^Ran 29 tests" && echo "$out" | grep -q "^OK" || die "unit tests: $out"
echo "unit tests: 29 OK"
ABLATIONS=${ABLATIONS:-1}
runs="v12_full_seed37 v12_full_seed38 v12_full_seed39"
[ "$ABLATIONS" = 1 ] && runs="$runs v12_nofb_seed37 v12_coupled_seed37 v12_band_seed37"
for r in smoke_seed37 $runs; do
  [ ! -e "run/deco/v12/$r" ] || die "run/deco/v12/$r exists already (check it, move it away)"
done
mkdir -p run/deco/v12
rm -f run/deco/v12/SMOKE_OK run/deco/v12/SMOKE_FAILED
SMOKE='NAME=smoke SEED=37 ARGS="--epochs 2 --max-chunks 3 --limit 2 --runtime 1 --log-every 1 --profile" bash tools/deco/run_v12.sh && touch run/deco/v12/SMOKE_OK || { touch run/deco/v12/SMOKE_FAILED; exit 1; }'
bash tools/final/pool_add.sh v12_smoke "$SMOKE" || die "pool_add smoke"
job() {                                      # name seed stagger(s) args
  echo "WAIT_SMOKE=1 STAGGER=$3 NAME=$1 SEED=$2 ARGS=\"$4\" bash tools/deco/run_v12.sh"
}
bash tools/final/pool_add.sh v12_full_s37 "$(job v12_full 37 0 "")" > /dev/null
bash tools/final/pool_add.sh v12_full_s38 "$(job v12_full 38 60 "")" > /dev/null
bash tools/final/pool_add.sh v12_full_s39 "$(job v12_full 39 120 "")" > /dev/null
if [ "$ABLATIONS" = 1 ]; then
  bash tools/final/pool_add.sh v12_nofb_s37 "$(job v12_nofb 37 180 "--feedback none")" > /dev/null
  bash tools/final/pool_add.sh v12_coupled_s37 "$(job v12_coupled 37 240 "--prior-grad on")" > /dev/null
  bash tools/final/pool_add.sh v12_band_s37 "$(job v12_band 37 300 "--birth band")" > /dev/null
fi
echo "pool: $(ls run/queue/pool/pending | tr '\n' ' ')"
if pgrep -f "tools/final/gpu_pool.sh" > /dev/null; then
  echo "dispatcher running: jobs start on the next free GPU of its GPU list"
else
  echo "NOTE: no dispatcher is running (start it with the commands that came with this script)"
fi
echo "QUEUE V12 DONE"
