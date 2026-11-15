#!/usr/bin/env bash
# Idle-card time per window of the V18 final model (backbone + decision layer) by the stream replay. Runs inside the
# V18 dir (its runs, priors, relative paths) with the code of this dir, whose model/sdu.py loads V18 checkpoints
# (the V18 dir's copy is the pre-fix V19 one and fails with "Missing key(s) r, box_area, ring_idx").
# Usage (V20 code dir, card otherwise idle): GPU=3 bash tools/v20/timing_v18.sh > run/v20/timing.log 2>&1
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
V18B=${V18B:-/media/stephen/nvme0n1/wy_data/EV-UAV-v18b}
BB37=${BB37:-/media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41/log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt}
cd "$V18B" && mkdir -p run/v18/timing
for run in coopw_s0 base_s0; do
  out=run/v18/timing/${run}_gpu${GPU:-x}
  python -u "$HERE/tools/v18/stream_sdu.py" --run run/v18/v5s37/$run --backbone "$BB37" --device cuda:0 \
    --out $out.json > $out.log 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] timing $run: $(tail -n 1 $out.log | cut -c1-200)"
done
