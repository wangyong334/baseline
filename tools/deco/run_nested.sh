#!/usr/bin/env bash
# Nested V11 on one backbone (11-08): staged V11 neurons -> thresholds -> consistency gate -> joint-training smoke ->
# end-to-end joint training -> thresholds with the official four metrics. Reuses run/deco/<TAG>/z0_* and the V10
# evidence run/deco/<TAG>/ev_a0.01_nr_* (tools/deco/run_dnm.sh DESIGN=v10 made them).
# Usage (code dir, after `conda activate evuav`; inside the GPU pool the GPU is set by the runner):
#   CKPT=log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt TAG=v5s37 V11_ARGS="--v11-rho mean" \
#     bash tools/deco/run_nested.sh
#   EPOCHS=4 K=32 STREAMS=2 NAME=nested_mean ...   (defaults)
# Outputs: run/deco/<TAG>/<S1NAME> (staged), <NAME>_gate (gate), <NAME>_smoke, <NAME> (joint, with range/range.json).
# Last line: "NESTED <TAG> FINISHED" or "NESTED <TAG> FAILED at <step>".
set -u
cd "$(dirname "$0")/../.."
CKPT=${CKPT:?CKPT=<backbone checkpoint of the staged run>}
TAG=${TAG:?TAG=<run/deco tag>}
V11_ARGS=${V11_ARGS:-"--v11-rho mean"}
S1NAME=${S1NAME:-dnm_a0.01_s0_nr_v11_mean}
NAME=${NAME:-nested_mean}
EPOCHS=${EPOCHS:-4}
K=${K:-32}
STREAMS=${STREAMS:-2}
WORKERS=${WORKERS:-8}
OUT=run/deco/$TAG
EV=$OUT/ev_a0.01_nr
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] NESTED $TAG FAILED at $1 (log $2)"; exit 1; }
[ -f "$CKPT" ] || fail checkpoint "$CKPT missing"
for s in train val test; do [ -f "${EV}_$s.npz" ] || fail evidence "${EV}_$s.npz missing"; done
ROOT=$(python -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu')['config']['root'])" "$CKPT") || fail checkpoint "$CKPT"
echo "[$(ts)] NESTED $TAG: backbone $CKPT, data $ROOT, staged $S1NAME ($V11_ARGS), joint $NAME (epochs $EPOCHS, k $K, streams $STREAMS), GPU ${CUDA_VISIBLE_DEVICES:-?}"
S1=$OUT/$S1NAME
if [ ! -f "$S1/summary.json" ]; then
  python -u tools/deco/train_dnm.py --evidence "$EV" --z0-root "$OUT" --data-root "$ROOT" --out "$S1" --design v11 \
    --seed 0 $V11_ARGS > "$S1.log" 2>&1 || fail staged "$S1.log"
fi
echo "  staged: $(tail -n 1 "$S1.log")"
if [ ! -f "$S1/range/range.json" ]; then
  python tools/deco/fit_range.py --run "$S1" --evidence "$EV" --z0-root "$OUT" --data-root "$ROOT" > "$S1/range.log" 2>&1 \
    || fail staged_range "$S1/range.log"
fi
echo "  staged range: $(tail -n 1 "$S1/range.log" | cut -c1-260)"
G=$OUT/${NAME}_gate
python -u tools/deco/train_nested.py --checkpoint "$CKPT" --stage1 "$S1" --stage1-evidence "$EV" --data-root "$ROOT" \
  --out "$G" --epochs 0 --workers $WORKERS > "$G.log" 2>&1 || fail gate "$G.log"
python - "$S1" "$G" "$EV" <<'PY' || fail gate_compare "$G.log"
import sys
import numpy as np
s1, g, ev = sys.argv[1:4]
for split in ("val", "test"):
    a = np.load("%s/raw_%s.npz" % (s1, split))["lam_traj"]
    b = np.load("%s/run/raw_%s.npz" % (g, split))["lam_traj"]
    ea, eb = np.load("%s_%s.npz" % (ev, split)), np.load("%s/evm_%s_meta.npz" % (g, split))
    ia, ib = ea["idx"] + 10 ** 7 * ea["rec"], eb["idx"] + 10 ** 7 * eb["rec"]
    common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
    diff = np.nanmax(np.abs(a[ka] - b[kb])) if common.size else float("nan")
    print("  gate %s: pending staged %d nested %d common %d | max |U_nested - U_staged| on common events %.2e"
          % (split, len(ia), len(ib), common.size, diff))
PY
echo "  gate: $(tail -n 1 "$G.log")"
SM=$OUT/${NAME}_smoke
rm -rf "$SM"
python -u tools/deco/train_nested.py --checkpoint "$CKPT" --stage1 "$S1" --stage1-evidence "$EV" --data-root "$ROOT" \
  --out "$SM" --epochs 1 --max-chunks 3 --limit 2 --k $K --streams $STREAMS --workers 2 > "$SM.log" 2>&1 \
  || fail smoke "$SM.log"
echo "  smoke: $(tail -n 1 "$SM.log")"
N=$OUT/$NAME
if [ ! -f "$N/summary.json" ]; then
  python -u tools/deco/train_nested.py --checkpoint "$CKPT" --stage1 "$S1" --stage1-evidence "$EV" --data-root "$ROOT" \
    --out "$N" --epochs $EPOCHS --k $K --streams $STREAMS --workers $WORKERS > "$N.log" 2>&1 || fail joint "$N.log"
fi
echo "  joint: $(tail -n 1 "$N.log")"
python tools/deco/fit_range.py --run "$N/run" --evidence "$N/evm" --z0-root "$N" --data-root "$ROOT" > "$N/range.log" 2>&1 \
  || fail joint_range "$N/range.log"
echo "  joint range: $(tail -n 1 "$N/range.log" | cut -c1-260)"
echo "[$(ts)] NESTED $TAG FINISHED"
