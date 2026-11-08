#!/usr/bin/env bash
# Staged V11 decision neurons on one backbone (11-08): train_dnm.py --design v11 on the V10 evidence of the tag, then the
# commit thresholds (tools/deco/fit_range.py: fixed band / change-risk head / self-predicted range / CQR / taper) with
# the official four metrics. Reuses run/deco/<TAG>/z0_* and ev_a0.01_nr_* (tools/deco/run_dnm.sh DESIGN=v10 made them).
# Usage (code dir, after `conda activate evuav`; inside the GPU pool the GPU is set by the runner):
#   TAG=v5s38 bash tools/deco/run_v11_staged.sh
#   V11_ARGS="--v11-rho mean" NAME=dnm_a0.01_s0_nr_v11_mean SEED=0 ...   (defaults)
# Last line: "V11 <TAG> FINISHED" or "V11 <TAG> FAILED at <step>".
set -u
cd "$(dirname "$0")/../.."
TAG=${TAG:?TAG=<run/deco tag>}
V11_ARGS=${V11_ARGS:-"--v11-rho mean"}
NAME=${NAME:-dnm_a0.01_s0_nr_v11_mean}
SEED=${SEED:-0}
OUT=run/deco/$TAG
EV=$OUT/ev_a0.01_nr
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] V11 $TAG FAILED at $1 (log $2)"; exit 1; }
for s in train val test; do [ -f "${EV}_$s.npz" ] || fail evidence "${EV}_$s.npz missing"; done
[ -d "$OUT/z0_val" ] || fail z0 "$OUT/z0_val missing"
ROOT=${DATA_ROOT:-/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset}
[ -d "$ROOT/val" ] || fail data_root "$ROOT/val missing (set DATA_ROOT)"
echo "[$(ts)] V11 $TAG: $NAME ($V11_ARGS, seed $SEED), data $ROOT, GPU ${CUDA_VISIBLE_DEVICES:-?}"
R=$OUT/$NAME
if [ ! -f "$R/summary.json" ]; then
  python -u tools/deco/train_dnm.py --evidence "$EV" --z0-root "$OUT" --data-root "$ROOT" --out "$R" --design v11 \
    --seed $SEED $V11_ARGS > "$R.log" 2>&1 || fail train "$R.log"
fi
echo "  train: $(tail -n 1 "$R.log")"
python tools/deco/fit_range.py --run "$R" --evidence "$EV" --z0-root "$OUT" --data-root "$ROOT" > "$R/range.log" 2>&1 \
  || fail range "$R/range.log"
echo "  range: $(tail -n 1 "$R/range.log" | cut -c1-300)"
echo "[$(ts)] V11 $TAG FINISHED"
