#!/usr/bin/env bash
# Phase 1, GPU C (FINAL_SERVER_COMMANDS_20261104.md): module-three controls on the frozen V5 (seed 37).
#   1. the frozen V5 in the final layout (eval only: per-sequence val, noise val / test): module-two reference
#   2. B2 learned look-ahead head, plain and with the same occupancy supervision as RCL: train + full evaluation
#   3. the existing heads in the final layout, no retraining (legacy runs -> best-val checkpoint):
#      RCL feature (+ full-model noise, + B1), RCL score-only (B2's pair), same-capacity causal head
#   4. EV-Flying: the RCL feature head on V5-EVF, no retraining (needs the GPU1 run to be finished)
#   5. RETRAIN_V5=1 (default): V5 seed 37 retrained under the guarded protocol (2x2 uniformity; deterministic)
# Start only when the GPU is free and the causal-head run (log/changepoint_v41_insample_feature_causal_seed37) finished.
# Usage: GPU=3 nohup bash tools/final/queue_m3.sh > run/final/queue_m3.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-3}
mkdir -p run/final log/final
V5=log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt
[ -f $V5 ] || { echo "没有冻结 V5 的权重：$V5"; exit 1; }
mkdir -p log/final/bb_v5_frozen_seed37
cp -n log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt log/stream_tif_commpeak_msg_seed37/metrics.jsonl \
  log/final/bb_v5_frozen_seed37/
NAME=bb_v5_50 RUN=bb_v5_frozen TRAIN=0 bash tools/final/backbone.sh || { echo "QUEUE M3 FAILED at frozen V5"; exit 1; }
for b2 in b2_lookahead_50 b2_lookahead_occ_50; do
  NAME=$b2 RUN=${b2}_on_v5 BASE=$V5 HZ="" bash tools/final/head.sh || { echo "QUEUE M3 FAILED at $b2"; exit 1; }
done
# name : source run (log/<src>) : final run name : B1 : past-only eval : noise levels
for spec in "rcl_feature_50:changepoint_v41_insample_feature_seed37:rcl_feature_50_on_v5:1:1:0.8 1.6 2.4 4" \
            "rcl_score_50:changepoint_v41_insample_seed37:rcl_score_50_on_v5:0:1:" \
            "rcl_feature_causal_50:changepoint_v41_insample_feature_causal_seed37:rcl_feature_causal_50_on_v5:0:0:"; do
  IFS=: read -r name src run b1 past hz <<< "$spec"
  [ -f log/$src/best_val_iou_seed37.pt ] || { echo "QUEUE M3 FAILED：没有 log/$src/best_val_iou_seed37.pt"; exit 1; }
  mkdir -p log/final/${run}_seed37
  cp -n log/$src/best_val_iou_seed37.pt log/$src/metrics.jsonl log/$src/run_config.json log/final/${run}_seed37/
  NAME=$name RUN=$run BASE=$V5 TRAIN=0 B1=$b1 PAST=$past HZ="$hz" bash tools/final/head.sh \
    || { echo "QUEUE M3 FAILED at $run"; exit 1; }
done
V5E=log/stream_tif_commpeak_msg_evflying_x4_seed37/best_val_iou_seed37.pt
SRC=log/changepoint_v41_evflying_feature_seed37
if [ -f $V5E ] && [ -f $SRC/best_val_iou_seed37.pt ]; then
  mkdir -p log/final/rcl_feature_evf_on_v5evf_seed37
  cp -n $SRC/best_val_iou_seed37.pt $SRC/metrics.jsonl $SRC/run_config.json log/final/rcl_feature_evf_on_v5evf_seed37/
  NAME=rcl_feature_evf RUN=rcl_feature_evf_on_v5evf BASE=$V5E TRAIN=0 HZ="" bash tools/final/head.sh \
    || { echo "QUEUE M3 FAILED at EV-Flying"; exit 1; }
else
  echo "跳过 EV-Flying：缺 $V5E 或 $SRC/best_val_iou_seed37.pt（卡 1 的任务是否跑完？）"
fi
if [ "${RETRAIN_V5:-1}" = "1" ]; then
  NAME=bb_v5_50 bash tools/final/backbone.sh || { echo "QUEUE M3 FAILED at V5 retrain"; exit 1; }
fi
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE M3 FINISHED"
