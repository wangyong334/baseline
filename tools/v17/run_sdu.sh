#!/usr/bin/env bash
# V17 one job (11-13): train the decision units on the priors of backbone TAG, calibrate the stopping rule and evaluate
# (official IoU / ACC / Pd / Fa, waits, unit load), analyse, and replay the test split window by window (timing and the
# stream == offline check). Steps already finished are skipped, so re-running a failed job resumes it.
# Usage (V17 code dir, after `conda activate evuav`; inside the GPU pool the runner sets CUDA_VISIBLE_DEVICES):
#   TAG=v5s37 SEED=0 NAME=full ARGS="" bash tools/v17/run_sdu.sh
#   DATA_ROOT=<root> Z0_ROOT=<dir with z0_*> DEVICE=cpu BACKBONE=<V5 checkpoint, optional: time it in the replay>
# Output: run/v17/<TAG>/<NAME>_s<SEED>/ (sdu.pt, summary.json, traj_*.npz, eval/*.json|png, *.log)
# Last line: "RUN V17 DONE <dir>" or "RUN V17 FAILED: ...".
set -u
cd "$(dirname "$0")/../.."
TAG=${TAG:?TAG=v5s37|v5s38|v5s39}
SEED=${SEED:-0}
NAME=${NAME:-full}
ARGS=${ARGS:-}
DATA_ROOT=${DATA_ROOT:-/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset}
Z0_ROOT=${Z0_ROOT:-run/deco/$TAG}
DEVICE=${DEVICE:-cuda:0}
BACKBONE=${BACKBONE:-}
OUT=run/v17/$TAG/${NAME}_s$SEED
mkdir -p "$OUT"
LOG=$OUT/run.log
die() { echo "RUN V17 FAILED: $*" | tee -a "$LOG"; exit 1; }
for s in train val test; do
  ls "$Z0_ROOT/z0_$s"/*.npz > /dev/null 2>&1 || die "no priors in $Z0_ROOT/z0_$s (tools/v17/setup_v17_dir.sh)"
done
[ -d "$DATA_ROOT/test" ] || die "no data in $DATA_ROOT"
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
step train "SDU TRAIN FINISHED" python -u tools/v17/train_sdu.py --z0-root "$Z0_ROOT" --data-root "$DATA_ROOT" \
  --out "$OUT" --seed "$SEED" --device "$DEVICE" $ARGS
step eval "SDU EVAL FINISHED" python -u tools/v17/eval_sdu.py --run "$OUT"
step analyze "SDU ANALYSIS FINISHED" python -u tools/v17/analyze_sdu.py --run "$OUT" --device "$DEVICE"
step stream "SDU STREAM FINISHED" python -u tools/v17/stream_sdu.py --run "$OUT" --device "$DEVICE" \
  ${BACKBONE:+--backbone "$BACKBONE"}
echo "RUN V17 DONE $OUT" | tee -a "$LOG"
