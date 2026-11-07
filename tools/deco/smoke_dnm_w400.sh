#!/usr/bin/env bash
# Smoke test for the 11-06 queue (final DNM + 400 ms backbones): both 400 ms backbones train in smoke mode, and the final
# DNM pipeline runs end to end on 2 recordings per split (2 epochs). Marker: run/deco/SMOKE_DNM_W400_OK on success,
# run/deco/SMOKE_DNM_W400_FAILED on failure (tools/deco/wait_smoke.sh waits for one of them).
# Usage (code dir, after `conda activate evuav`): as a gpu_queue job, or GPU=<id> bash tools/deco/smoke_dnm_w400.sh
set -eu
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p run/deco
rm -rf log/final/smoke_bb_v5_400 log/final/smoke_bb_v5lif_400 run/deco/smoke_dnm \
  run/deco/SMOKE_DNM_W400_OK run/deco/SMOKE_DNM_W400_FAILED
trap 'touch run/deco/SMOKE_DNM_W400_FAILED; echo "SMOKE DNM W400 FAILED"' ERR
for cfg in bb_v5_400 bb_v5lif_400; do
  python -u train_stream_v1.py --config configs/final/$cfg.yaml --mode smoke --device cuda:0 \
    --save-root log/final/smoke_$cfg > run/deco/smoke_$cfg.log 2>&1
  echo "train smoke $cfg: $(tail -n 1 run/deco/smoke_$cfg.log | cut -c1-150)"
done
rm -rf log/final/smoke_bb_v5_400 log/final/smoke_bb_v5lif_400
CKPT=log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt TAG=smoke_dnm LIMIT=2 EXTRA="--epochs 2" \
  bash tools/deco/run_dnm.sh > run/deco/smoke_dnm.log 2>&1
tail -n 1 run/deco/smoke_dnm.log | grep -q "DNM smoke_dnm FINISHED"
rm -rf run/deco/smoke_dnm
touch run/deco/SMOKE_DNM_W400_OK
echo "SMOKE DNM W400 OK"
