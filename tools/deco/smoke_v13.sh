#!/usr/bin/env bash
# V13 GPU smoke test (11-10): the whole tools/deco/run_v13.sh on 3 / 4 / 2 recordings (train / val / test) of one
# backbone, 2 SDU epochs, 200 VoW steps, in a fresh dir that is removed again when the run finishes. About 3-5 min.
# Usage (code dir, after `conda activate evuav`): GPU=1 TAG=v5s37 bash tools/deco/smoke_v13.sh
# Last line: "SMOKE V13 OK" or "SMOKE V13 FAILED (...)".
set -u
cd "$(dirname "$0")/../.."
TAG=${TAG:-v5s37}
SM=run/deco/smoke_v13_$TAG
rm -rf "$SM" "$SM.log"
for s in train val test; do
  case $s in train) n=3 ;; val) n=4 ;; test) n=2 ;; esac
  mkdir -p "$SM/z0_$s"
  ls run/deco/$TAG/z0_$s/*.npz | head -n $n | while read -r f; do cp "$f" "$SM/z0_$s/"; done
  touch "$SM/z0_$s/DONE"
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
TAG=smoke OUT_ROOT=$SM WORKERS=4 EXTRA="--epochs 2" VOW_ARGS="--steps 200" bash tools/deco/run_v13.sh | tee "$SM.log"
if tail -n 1 "$SM.log" | grep -q "FINISHED"; then
  rm -rf "$SM"
  echo "SMOKE V13 OK"
else
  echo "SMOKE V13 FAILED (kept $SM, log $SM.log)"
fi
