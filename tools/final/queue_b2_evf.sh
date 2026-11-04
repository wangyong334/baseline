#!/usr/bin/env bash
# Second batch (after the basic batch), one GPU:
#   1. B2 learned look-ahead head on the frozen V5, plain and with the same occupancy supervision as RCL
#      (is RCL's model-based structure necessary? pair = rcl_score_50_on_v5 from the basic batch)
#   2. EV-Flying: the RCL feature head already trained on V5-EVF, final layout, no retraining
#   3. RETRAIN_V5=1 (default): V5 seed 37 retrained under the guarded protocol (2x2 uniformity; deterministic)
# Usage: GPU=3 nohup bash tools/final/queue_b2_evf.sh > run/final/queue_b2_evf.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-3}
mkdir -p run/final log/final
V5=log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt
[ -f $V5 ] || { echo "没有冻结 V5 的权重：$V5"; exit 1; }
for b2 in b2_lookahead_50 b2_lookahead_occ_50; do
  NAME=$b2 RUN=${b2}_on_v5 BASE=$V5 HZ="" bash tools/final/head.sh || { echo "QUEUE B2 EVF FAILED at $b2"; exit 1; }
done
V5E=log/stream_tif_commpeak_msg_evflying_x4_seed37/best_val_iou_seed37.pt
SRC=log/changepoint_v41_evflying_feature_seed37
if [ -f $V5E ] && [ -f $SRC/best_val_iou_seed37.pt ]; then
  mkdir -p log/final/rcl_feature_evf_on_v5evf_seed37
  cp -n $SRC/best_val_iou_seed37.pt $SRC/metrics.jsonl $SRC/run_config.json log/final/rcl_feature_evf_on_v5evf_seed37/
  NAME=rcl_feature_evf RUN=rcl_feature_evf_on_v5evf BASE=$V5E TRAIN=0 HZ="" bash tools/final/head.sh \
    || { echo "QUEUE B2 EVF FAILED at EV-Flying"; exit 1; }
else
  echo "跳过 EV-Flying：缺 $V5E 或 $SRC/best_val_iou_seed37.pt"
fi
if [ "${RETRAIN_V5:-1}" = "1" ]; then
  NAME=bb_v5_50 bash tools/final/backbone.sh || { echo "QUEUE B2 EVF FAILED at V5 retrain"; exit 1; }
fi
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE B2 EVF FINISHED"
