#!/usr/bin/env bash
# V18 one job (11-13 night): train the decision units on the priors of backbone TAG (with STAGE1: as the second
# population, reading the decision spikes of that trained V17 run), calibrate the stopping rule and evaluate (official
# IoU / ACC / Pd / Fa, waits, unit load, energy), analyse, decompose the backbone's errors by what waiting removes (A1),
# and replay the test split window by window (stream == offline check). Steps already finished are skipped, so
# re-running a failed job resumes it. With STAGE1 the job first waits (up to WAIT_H hours) until that run is evaluated.
# Usage (V18 code dir, after `conda activate evuav`; the pool sets CUDA_VISIBLE_DEVICES):
#   TAG=v5s37 SEED=0 NAME=coop STAGE1=<V17 run dir> ARGS="" bash tools/v18/run_sdu.sh
#   DATA_ROOT=<root> Z0_ROOT=<dir with z0_*> DEVICE=cpu BACKBONE=<V5 checkpoint, optional: time it in the replay>
# Output: run/v18/<TAG>/<NAME>_s<SEED>/ (sdu.pt, summary.json, traj_*.npz, messages_*.npz, eval/*.json|png, *.log)
# Last line: "RUN V18 DONE <dir>" or "RUN V18 FAILED: ...".
set -u
cd "$(dirname "$0")/../.."
TAG=${TAG:?TAG=v5s37|v5s38|v5s39}
SEED=${SEED:-0}
NAME=${NAME:-full}
ARGS=${ARGS:-}
STAGE1=${STAGE1:-}
WAIT_H=${WAIT_H:-6}
DATA_ROOT=${DATA_ROOT:-/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset}
Z0_ROOT=${Z0_ROOT:-run/deco/$TAG}
DEVICE=${DEVICE:-cuda:0}
BACKBONE=${BACKBONE:-}
OUT=run/v18/$TAG/${NAME}_s$SEED
mkdir -p "$OUT"
LOG=$OUT/run.log
die() { echo "RUN V18 FAILED: $*" | tee -a "$LOG"; exit 1; }
for s in train val test; do
  ls "$Z0_ROOT/z0_$s"/*.npz > /dev/null 2>&1 || die "no priors in $Z0_ROOT/z0_$s (tools/v18/setup_v18_dir.sh)"
done
[ -d "$DATA_ROOT/test" ] || die "no data in $DATA_ROOT"
if [ -n "$STAGE1" ]; then                    # the first population must be trained and calibrated
  waited=0
  until [ -f "$STAGE1/sdu.pt" ] && grep -q "SDU EVAL FINISHED" "$STAGE1/eval.log" 2>/dev/null; do
    [ "$waited" -ge $((WAIT_H * 3600)) ] && die "first population $STAGE1 not evaluated after $WAIT_H h"
    [ "$waited" -eq 0 ] && echo "$(date '+%m-%d %H:%M:%S') waiting for $STAGE1" >> "$LOG"
    sleep 60
    waited=$((waited + 60))
  done
  ARGS="--stage1 $STAGE1 $ARGS"
fi
echo "$(date '+%m-%d %H:%M:%S') start $OUT on GPU ${CUDA_VISIBLE_DEVICES:-?} ($DEVICE), ARGS='$ARGS'" >> "$LOG"
step() {                                     # name marker command...
  local name=$1 marker=$2
  shift 2
  if grep -q "$marker" "$OUT/$name.log" 2>/dev/null; then
    echo "  $name: done before" >> "$LOG"
    return 0
  fi
  local t0=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1 || die "$name (see $OUT/$name.log)"
  grep -q "$marker" "$OUT/$name.log" || die "$name finished without '$marker'"
  echo "  $name: $(( $(date +%s) - t0 )) s | $(tail -n 1 "$OUT/$name.log")" >> "$LOG"
}
step train "SDU TRAIN FINISHED" python -u tools/v18/train_sdu.py --z0-root "$Z0_ROOT" --data-root "$DATA_ROOT" \
  --out "$OUT" --seed "$SEED" --device "$DEVICE" $ARGS
step eval "SDU EVAL FINISHED" python -u tools/v18/eval_sdu.py --run "$OUT"
step analyze "SDU ANALYSIS FINISHED" python -u tools/v18/analyze_sdu.py --run "$OUT" --device "$DEVICE"
step reducible "REDUCIBLE FINISHED" python -u tools/v18/reducible.py --run "$OUT"
step stream "SDU STREAM FINISHED" python -u tools/v18/stream_sdu.py --run "$OUT" --device "$DEVICE" \
  ${BACKBONE:+--backbone "$BACKBONE"}
echo "RUN V18 DONE $OUT" | tee -a "$LOG"
