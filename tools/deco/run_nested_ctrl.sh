#!/usr/bin/env bash
# Control for the nested V11 run (11-09): the same joint training as tools/deco/run_nested.sh, but without the report
# channel (--report-dim 0; the priors are detached as with the default --prior-grad off). The small network and the
# decision neurons are then trained at the same time on their own losses and never coupled:
#   joint (run_nested.sh) - this control  = what the coupling itself (the small network learning what to report) adds
#   this control - staged run             = what 4 more backbone epochs with the neurons following them add
# The staged run $OUT/$S1NAME is made by run_v11_staged.sh / run_nested.sh; this script waits for it (polling once a
# minute, at most WAIT_MIN minutes), so it can be queued next to the job that makes it.
# Usage (code dir, after `conda activate evuav`; inside the GPU pool the GPU is set by the runner):
#   CKPT=log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt TAG=v5s37 S1NAME=dnm_a0.01_s0_nr_v11_ei \
#     NAME=nested_ei_norep bash tools/deco/run_nested_ctrl.sh
#   EPOCHS=4 K=32 STREAMS=2 WORKERS=8 WAIT_MIN=120 ...   (defaults; keep EPOCHS / K / STREAMS equal to the joint run)
# Last line: "NESTED-CTRL <TAG> FINISHED" or "NESTED-CTRL <TAG> FAILED at <step>".
set -u
cd "$(dirname "$0")/../.."
CKPT=${CKPT:?CKPT=<backbone checkpoint of the staged run>}
TAG=${TAG:?TAG=<run/deco tag>}
S1NAME=${S1NAME:?S1NAME=<staged V11 run name>}
NAME=${NAME:?NAME=<output name>}
EPOCHS=${EPOCHS:-4}
K=${K:-32}
STREAMS=${STREAMS:-2}
WORKERS=${WORKERS:-8}
WAIT_MIN=${WAIT_MIN:-120}
OUT=run/deco/$TAG
EV=$OUT/ev_a0.01_nr
S1=$OUT/$S1NAME
N=$OUT/$NAME
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] NESTED-CTRL $TAG FAILED at $1 (log $2)"; exit 1; }
[ -f "$CKPT" ] || fail checkpoint "$CKPT missing"
for s in train val test; do [ -f "${EV}_$s.npz" ] || fail evidence "${EV}_$s.npz missing"; done
ROOT=$(python -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu')['config']['root'])" "$CKPT") || fail checkpoint "$CKPT"
echo "[$(ts)] NESTED-CTRL $TAG: backbone $CKPT, staged $S1NAME, control $NAME (no reports; epochs $EPOCHS, k $K, streams $STREAMS), GPU ${CUDA_VISIBLE_DEVICES:-?}"
n=0
while [ ! -f "$S1/summary.json" ]; do
  [ $n -ge "$WAIT_MIN" ] && fail wait_staged "$S1/summary.json not there after $WAIT_MIN min"
  sleep 60
  n=$((n + 1))
done
[ $n -gt 0 ] && echo "  waited $n min for $S1"
if [ ! -f "$N/summary.json" ]; then
  python -u tools/deco/train_nested.py --checkpoint "$CKPT" --stage1 "$S1" --stage1-evidence "$EV" --data-root "$ROOT" \
    --out "$N" --epochs $EPOCHS --k $K --streams $STREAMS --workers $WORKERS --report-dim 0 > "$N.log" 2>&1 \
    || fail joint "$N.log"
fi
echo "  control: $(tail -n 1 "$N.log")"
python tools/deco/fit_range.py --run "$N/run" --evidence "$N/evm" --z0-root "$N" --data-root "$ROOT" > "$N/range.log" 2>&1 \
  || fail range "$N/range.log"
echo "  control range: $(tail -n 1 "$N/range.log" | cut -c1-260)"
echo "[$(ts)] NESTED-CTRL $TAG FINISHED"
