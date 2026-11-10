#!/usr/bin/env bash
# DeCo-SNN V12 on EV-Flying x4 (11-09): one seed with the EV-Flying backbone's recipe (configs/evisseg_stream_tif_comm_
# peak_evflying.yaml: 16 epochs, its pos_weight / message budget / sensor size) and every V12 default.
# EV-Flying is 7-13x larger than EV-UAV (26 M test events, bursts of 1e5 events in one window), so: no fit_range and no
# gated export (their arrays need tens of GB of RAM), official four metrics by tools/deco/official_v12.py recording by
# recording (prior / arrival / fixed-band crossing / deadline, sensor 180 x 320), runtime on 2 test recordings, smaller
# GPU blocks (--sub-batch 1024 --pick-block 1024).
# Usage (code dir, after `conda activate evuav`): SEED=37 bash tools/deco/run_v12_evf.sh   [WAIT_SMOKE=1] [STAGGER=<s>]
# Output: run/deco/v12/v12_evf_seed<SEED>/ (summary.json, runtime.json, official/official_v12.json),
#         log run/deco/v12/v12_evf_seed<SEED>.log. An unfinished run resumes from last.pt; a finished one is kept.
# Last line: "V12 EVF seed <SEED> FINISHED | ..." or "V12 EVF seed <SEED> FAILED at <step> (log ...)".
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
SEED=${SEED:-37}
CFG=configs/evisseg_stream_tif_comm_peak_evflying.yaml
OUT=run/deco/v12/v12_evf_seed${SEED}
LOG=$OUT.log
mkdir -p run/deco/v12
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] V12 EVF seed $SEED FAILED at $1 (log $2)"; exit 1; }
[ -f "$CFG" ] || fail config "$CFG missing"
ROOT=${DATA_ROOT:-$(python -c "
import sys; sys.path.insert(0, '.')
from utils.stream_common import load_flat_config
print(load_flat_config(sys.argv[1])['root'])" "$CFG")} || fail config "$CFG"
[ -d "$ROOT/train" ] || fail data "$ROOT/train missing"
if [ -f "$OUT/official/official_v12.json" ]; then
  echo "[$(ts)] V12 EVF seed $SEED FINISHED (already) | $(tail -n 1 "$OUT/official.log" | cut -c1-400)"
  exit 0
fi
if [ "${WAIT_SMOKE:-0}" = 1 ]; then
  until [ -f run/deco/v12/SMOKE_OK ] || [ -f run/deco/v12/SMOKE_FAILED ]; do sleep 30; done
  [ -f run/deco/v12/SMOKE_OK ] || fail smoke run/deco/v12/smoke_seed37.log
fi
sleep "${STAGGER:-0}"                       # staggered starts (power peaks of simultaneous calibrations, 09-24)
if ! grep -q "^V12 FINISHED" "$LOG" 2>/dev/null; then
  RESUME=""
  [ -f "$OUT/last.pt" ] && RESUME="--resume"
  echo "[$(ts)] V12 EVF seed $SEED: data $ROOT, GPU ${CUDA_VISIBLE_DEVICES:-?} $RESUME, python $(command -v python)"
  python -u tools/deco/train_v12.py --config "$CFG" --data-root "$ROOT" --out "$OUT" --seed "$SEED" --epochs 16 \
    --runtime 2 --fit-range off --gated off --sub-batch 1024 --pick-block 1024 $RESUME >> "$LOG" 2>&1 || fail train "$LOG"
fi
python -u tools/deco/official_v12.py --run "$OUT" --data-root "$ROOT" --config "$CFG" > "$OUT/official.log" 2>&1 \
  || fail official "$OUT/official.log"
echo "[$(ts)] V12 EVF seed $SEED FINISHED | $(grep '^V12 FINISHED' "$LOG" | tail -n 1 | cut -c1-200) | $(tail -n 1 "$OUT/official.log" | cut -c1-420)"
