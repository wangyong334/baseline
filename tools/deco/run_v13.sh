#!/usr/bin/env bash
# V13 on one backbone (11-10): BIR prior -> VoW unit (wait or not) -> spiking decision unit (SDU), design
# Res/v13_design/V13_DESIGN.md.
#   1. evidence of the wide training band: martingale band at ALPHA_TRAIN (default 0.001, ~7.5% of the events),
#      no radius cap (tools/deco/evidence.py --band martingale --radius 0)
#   2. stage 2: SDU = V11 spiking decision units, stepwise BCE on the training split (train_dnm.py --design v11
#      --v11-rho mean, the V11 default)
#   3. stages 3-4: tools/deco/fit_vow.py: VoW gate (k = 0) + VoW commit (k >= 1) on val recordings A (no labels), CQR on
#      val recordings B, all policies (arrival / sym / ville / outer / vow / vow_z0 / margin_m / wait_all) on val and test
#      with the official four metrics
# Reuses run/deco/<TAG>/z0_{train,val,test} (tools/deco/run_dnm.sh made them).
# Usage (code dir, after `conda activate evuav`; inside the GPU pool the runner sets the GPU):
#   TAG=v5s37 bash tools/deco/run_v13.sh
#   SEED=0 ALPHA_TRAIN=0.001 WORKERS=8 DATA_ROOT=<root> EXTRA="--epochs 2" VOW_ARGS="--steps 200" ...   (smoke tests)
# Results: run/deco/<TAG>/dnm_m<ALPHA_TRAIN>_s<SEED>_nr_v13/{summary.json, vow/vow.json, vow.log}
# Last line: "V13 <TAG> FINISHED" or "V13 <TAG> FAILED at <step> (log ...)".
set -u
cd "$(dirname "$0")/../.."
TAG=${TAG:?TAG=<run/deco tag>}
SEED=${SEED:-0}
AT=${ALPHA_TRAIN:-0.001}
WORKERS=${WORKERS:-8}
EXTRA=${EXTRA:-}
VOW_ARGS=${VOW_ARGS:-}
OUT=${OUT_ROOT:-run/deco/$TAG}
EV=$OUT/ev_m${AT}_nr
NAME=${NAME:-dnm_m${AT}_s${SEED}_nr_v13}
R=$OUT/$NAME
ROOT=${DATA_ROOT:-/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset}
DEV=${DEVICE:-cuda:0}
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] V13 $TAG FAILED at $1 (log $2)"; exit 1; }
for s in train val test; do
  [ -f "$OUT/z0_$s/DONE" ] || fail z0 "$OUT/z0_$s/DONE missing (tools/deco/run_dnm.sh makes it)"
done
[ -d "$ROOT/val" ] || fail data_root "$ROOT/val missing (set DATA_ROOT)"
echo "[$(ts)] V13 $TAG: training band martingale alpha $AT, SDU seed $SEED, data $ROOT, GPU ${CUDA_VISIBLE_DEVICES:-?}"
for s in train val test; do
  if [ ! -f "${EV}_$s.npz" ]; then
    python -u tools/deco/evidence.py --z0-dir "$OUT/z0_$s" --split $s --data-root "$ROOT" --alpha $AT --band martingale \
      --radius 0 --out "${EV}_$s.npz" --workers $WORKERS > "${EV}_$s.log" 2>&1 || fail evidence_$s "${EV}_$s.log"
  fi
  echo "  evidence $s: $(tail -n 1 "${EV}_$s.log" | cut -c1-200)"
done
if [ ! -f "$R/summary.json" ]; then
  python -u tools/deco/train_dnm.py --evidence "$EV" --z0-root "$OUT" --data-root "$ROOT" --out "$R" --design v11 \
    --v11-rho mean --seed $SEED --device $DEV $EXTRA > "$R.log" 2>&1 || fail train "$R.log"
fi
echo "  SDU: $(tail -n 1 "$R.log" | cut -c1-300)"
python -u tools/deco/fit_vow.py --run "$R" --evidence "$EV" --z0-root "$OUT" --data-root "$ROOT" --device $DEV $VOW_ARGS \
  > "$R/vow.log" 2>&1 || fail vow "$R/vow.log"
echo "  VoW: $(tail -n 1 "$R/vow.log" | cut -c1-500)"
echo "[$(ts)] V13 $TAG FINISHED"
