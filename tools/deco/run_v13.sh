#!/usr/bin/env bash
# V13 on one backbone (11-10): BIR prior -> VoW unit (wait or not) -> spiking decision unit (SDU), design
# Res/v13_design/V13_DESIGN.md.
#   1. evidence of the wide training band: martingale band at ALPHA_TRAIN (default 0.001 = alpha / 10 for every
#      backbone; its share of the events depends on the backbone: 7.5% / 21% / 62% of the training events for
#      seeds 37 / 38 / 39, see tools/deco/pick_band.py), no radius cap (tools/deco/evidence.py --band martingale
#      --radius 0); a file counts as done only with its .done marker or a log ending in "EVIDENCE FINISHED" (a
#      half-written file is rebuilt)
#   2. stage 2: SDU = V11 spiking decision units, stepwise BCE on the training split (train_dnm.py --design v11
#      $SDU_ARGS, default "--v11-rho mean" = the V11 default); skipped when <run>/summary.json exists
#   3. stages 3-4: tools/deco/fit_vow.py: VoW gate (k = 0) + VoW commit (k >= 1), fitted on val recordings A (no labels),
#      CQR on val recordings B, all policies on val and test with the official four metrics -> <run>/$VOW_NAME/
# Reuses run/deco/<TAG>/z0_{train,val,test} (tools/deco/run_dnm.sh made them).
# Ablations without code changes: SDU_ARGS (decision-unit options, with a new NAME), VOW_ARGS (e.g. "--alpha-bg 0.001")
# with a new VOW_NAME; SEED = SDU seed (the evidence is shared by all seeds of a tag).
# Usage (code dir, after `conda activate evuav`; inside the GPU pool the runner sets the GPU):
#   TAG=v5s37 bash tools/deco/run_v13.sh
#   SEED=1 VOW_NAME=vow_best VOW_ARGS="--alpha-bg 0.001" ...                       a second SDU seed, another VoW setting
#   SDU_ARGS="--v11-rho sum" NAME=dnm_m0.001_s0_nr_v13_rhosum ...                    an SDU ablation
#   REUSE_FROM=<other code dir> ...      copy a trained SDU of the same tag / name from there instead of training
#   DATA_ROOT=<root> DEVICE=cpu EXTRA="--epochs 2" VOW_ARGS="--steps 200 --fast" OUT_ROOT=<dir> ...   smoke tests
# Results: run/deco/<TAG>/dnm_m<alpha_train>_s<SEED>_nr_v13/{summary.json, <VOW_NAME>/vow.json, <VOW_NAME>.log}
# Last line: "V13 <TAG> FINISHED" or "V13 <TAG> FAILED at <step> (log ...)".
set -u
cd "$(dirname "$0")/../.."
TAG=${TAG:?TAG=<run/deco tag>}
SEED=${SEED:-0}
AT=${ALPHA_TRAIN:-0.001}
WORKERS=${WORKERS:-8}
EXTRA=${EXTRA:-}
SDU_ARGS=${SDU_ARGS:-"--v11-rho mean"}
VOW_ARGS=${VOW_ARGS:-}
VOW_NAME=${VOW_NAME:-vow}
OUT=${OUT_ROOT:-run/deco/$TAG}
EV=$OUT/ev_m${AT}_nr
NAME=${NAME:-dnm_m${AT}_s${SEED}_nr_v13}
R=$OUT/$NAME
ROOT=${DATA_ROOT:-/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset}
DEV=${DEVICE:-cuda:0}
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] V13 $TAG FAILED at $1 (log $2)"; exit 1; }
ev_done() {                                  # evidence file of split $1 is complete
  [ -f "${EV}_$1.npz" ] && { [ -f "${EV}_$1.done" ] || tail -n 1 "${EV}_$1.log" 2>/dev/null | grep -q "EVIDENCE FINISHED"; }
}
for s in train val test; do
  [ -f "$OUT/z0_$s/DONE" ] || fail z0 "$OUT/z0_$s/DONE missing (tools/deco/run_dnm.sh makes it)"
done
[ -d "$ROOT/val" ] || fail data_root "$ROOT/val missing (set DATA_ROOT)"
echo "[$(ts)] V13 $TAG: training band martingale alpha $AT, SDU seed $SEED ($SDU_ARGS), VoW -> $VOW_NAME ($VOW_ARGS)," \
     "data $ROOT, GPU ${CUDA_VISIBLE_DEVICES:-?}"
for s in train val test; do
  if ! ev_done $s; then
    rm -f "${EV}_$s.npz" "${EV}_$s.done"
    python -u tools/deco/evidence.py --z0-dir "$OUT/z0_$s" --split $s --data-root "$ROOT" --alpha $AT --band martingale \
      --radius 0 --out "${EV}_$s.npz" --workers $WORKERS > "${EV}_$s.log" 2>&1 || fail evidence_$s "${EV}_$s.log"
    tail -n 1 "${EV}_$s.log" | grep -q "EVIDENCE FINISHED" || fail evidence_$s "${EV}_$s.log"
    touch "${EV}_$s.done"
  fi
  echo "  evidence $s: $(tail -n 1 "${EV}_$s.log" | cut -c1-200)"
done
if [ ! -f "$R/summary.json" ] && [ -n "${REUSE_FROM:-}" ] && [ -f "$REUSE_FROM/$R/summary.json" ]; then
  mkdir -p "$R"                              # a trained SDU of another code dir (same tag, same name)
  for f in decision_neurons.pt summary.json raw_val.npz raw_test.npz; do
    cp "$REUSE_FROM/$R/$f" "$R/$f" || fail reuse_sdu "$REUSE_FROM/$R/$f"
  done
  cp "$REUSE_FROM/$R.log" "$R.log" 2>/dev/null || echo "SDU reused from $REUSE_FROM/$R" > "$R.log"
  echo "  SDU reused from $REUSE_FROM/$R"
fi
if [ ! -f "$R/summary.json" ]; then
  python -u tools/deco/train_dnm.py --evidence "$EV" --z0-root "$OUT" --data-root "$ROOT" --out "$R" --design v11 \
    $SDU_ARGS --seed $SEED --device $DEV $EXTRA > "$R.log" 2>&1 || fail train "$R.log"
  [ -f "$R/summary.json" ] && [ -f "$R/raw_val.npz" ] && [ -f "$R/raw_test.npz" ] || fail train_outputs "$R.log"
fi
echo "  SDU: $(tail -n 1 "$R.log" | cut -c1-300)"
python -u tools/deco/fit_vow.py --run "$R" --evidence "$EV" --z0-root "$OUT" --data-root "$ROOT" --device $DEV \
  --out "$R/$VOW_NAME" $VOW_ARGS > "$R/$VOW_NAME.log" 2>&1 || fail vow "$R/$VOW_NAME.log"
[ -f "$R/$VOW_NAME/vow.json" ] && tail -n 1 "$R/$VOW_NAME.log" | grep -q "VOW FINISHED" || fail vow_outputs "$R/$VOW_NAME.log"
echo "  VoW: $(tail -n 1 "$R/$VOW_NAME.log" | cut -c1-500)"
echo "[$(ts)] V13 $TAG FINISHED"
