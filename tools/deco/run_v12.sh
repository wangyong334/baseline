#!/usr/bin/env bash
# DeCo-SNN V12 (11-09): one end-to-end run from scratch with tools/deco/train_v12.py (training, streaming export of
# val / test, runtime measurement, fit_range with the official four metrics).
# Usage (code dir, after `conda activate evuav`; inside the GPU pool the runner sets the GPU):
#   NAME=v12_full SEED=37 bash tools/deco/run_v12.sh
#   NAME=v12_nofb ARGS="--feedback none" ...      ablations: any train_v12.py options
#   WAIT_SMOKE=1 ...                              wait for run/deco/v12/SMOKE_OK (written by the smoke job of
#                                                 tools/deco/queue_v12.sh); stop if run/deco/v12/SMOKE_FAILED appears
#   STAGGER=60 ...                                seconds to wait before starting (staggered launches)
#   DATA_ROOT=<root> CFG=<yaml> ...               defaults: the root of configs/evisseg_stream_tif_comm_peak.yaml
# Output: run/deco/v12/<NAME>_seed<SEED>/ (summary.json, runtime.json, run/range/range.json, deco_v12.pt, last.pt),
#         log run/deco/v12/<NAME>_seed<SEED>.log. An unfinished run (last.pt present) resumes; a finished one is kept.
# Last line: "V12 <NAME> seed <SEED> FINISHED | ..." or "V12 <NAME> seed <SEED> FAILED at <step> (log ...)".
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
NAME=${NAME:?NAME=<run name>}
SEED=${SEED:-37}
ARGS=${ARGS:-}
CFG=${CFG:-configs/evisseg_stream_tif_comm_peak.yaml}
OUT=run/deco/v12/${NAME}_seed${SEED}
LOG=$OUT.log
mkdir -p run/deco/v12
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] V12 $NAME seed $SEED FAILED at $1 (log $2)"; exit 1; }
ROOT=${DATA_ROOT:-$(python -c "
import sys; sys.path.insert(0, '.')
from utils.stream_common import load_flat_config
print(load_flat_config(sys.argv[1])['root'])" "$CFG")} || fail config "$CFG"
[ -d "$ROOT/train" ] || fail data "$ROOT/train missing"
if [ -f "$OUT/range.log" ] && grep -q "RANGE FINISHED" "$OUT/range.log"; then
  echo "[$(ts)] V12 $NAME seed $SEED FINISHED (already) | $(tail -n 1 "$OUT/range.log" | cut -c1-300)"
  exit 0
fi
if [ "${WAIT_SMOKE:-0}" = 1 ]; then
  until [ -f run/deco/v12/SMOKE_OK ] || [ -f run/deco/v12/SMOKE_FAILED ]; do sleep 30; done
  [ -f run/deco/v12/SMOKE_OK ] || fail smoke run/deco/v12/smoke_seed37.log
fi
sleep "${STAGGER:-0}"                       # staggered starts (power peaks of simultaneous calibrations, 09-24)
RESUME=""
[ -f "$OUT/last.pt" ] && RESUME="--resume"
echo "[$(ts)] V12 $NAME seed $SEED: data $ROOT, GPU ${CUDA_VISIBLE_DEVICES:-?}, args '$ARGS' $RESUME, python $(command -v python)"
python -u tools/deco/train_v12.py --config "$CFG" --data-root "$ROOT" --out "$OUT" --seed "$SEED" $ARGS $RESUME \
  >> "$LOG" 2>&1 || fail train "$LOG"
grep -q "RANGE FINISHED" "$OUT/range.log" 2>/dev/null || fail range "$OUT/range.log"
echo "[$(ts)] V12 $NAME seed $SEED FINISHED | $(grep 'V12 FINISHED' "$LOG" | tail -n 1 | cut -c1-220) | $(tail -n 1 "$OUT/range.log" | cut -c1-320)"
