#!/usr/bin/env bash
# Basic batch (verify the final model), GPU C:
#   1. the frozen V5 in the final layout (eval only: per-sequence val, val / test noise): module one + 2x2 reference
#   2. the existing heads on V5, no retraining (legacy runs -> best-val checkpoint): RCL feature (= M1 + M3, with B1
#      and noise), RCL score-only, same-capacity causal head (is waiting for the future necessary?)
#   3. E3a backbone (2x2)
#   4. once E1 / E3b are done (queue_basic_e1.sh, queue_basic_e3b.sh): the pre-registered backbone choice
#      (tools/select_backbone.py, val only) and the full model = chosen backbone + RCL feature head, linked to the
#      *_m2 names the paper manifest reads; then tools/final_summary.py -> run/final/summary_basic.md
# Start only when the GPU is free and the causal-head run (log/changepoint_v41_insample_feature_causal_seed37) finished.
# Usage: GPU=3 nohup bash tools/final/queue_basic_v5.sh > run/final/queue_basic_v5.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-3}
mkdir -p run/final log/final run/final/analysis
ROOT=$(python -c "import sys; sys.path.insert(0, '.'); from utils.stream_common import load_flat_config; print(load_flat_config('configs/final/bb_e3b_50.yaml')['root'])") \
  || { echo "读不出数据根目录（是否已 conda activate evuav？）"; exit 1; }
V5=log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt
[ -f $V5 ] || { echo "没有冻结 V5 的权重：$V5"; exit 1; }
mkdir -p log/final/bb_v5_frozen_seed37
for f in best_val_iou_seed37.pt metrics.jsonl; do      # the archived V5 may have no metrics.jsonl: then used as is
  [ -f log/stream_tif_commpeak_msg_seed37/$f ] && cp -n log/stream_tif_commpeak_msg_seed37/$f log/final/bb_v5_frozen_seed37/
done
NAME=bb_v5_50 RUN=bb_v5_frozen TRAIN=0 bash tools/final/backbone.sh || { echo "QUEUE BASIC V5 FAILED at frozen V5"; exit 1; }
# name : source run (log/<src>) : final run name : B1 : past-only eval : noise levels
for spec in "rcl_feature_50:changepoint_v41_insample_feature_seed37:rcl_feature_50_on_v5:1:1:0.8 1.6 2.4 4" \
            "rcl_score_50:changepoint_v41_insample_seed37:rcl_score_50_on_v5:0:1:" \
            "rcl_feature_causal_50:changepoint_v41_insample_feature_causal_seed37:rcl_feature_causal_50_on_v5:0:0:"; do
  IFS=: read -r name src run b1 past hz <<< "$spec"
  [ -f log/$src/best_val_iou_seed37.pt ] || { echo "QUEUE BASIC V5 FAILED：没有 log/$src/best_val_iou_seed37.pt"; exit 1; }
  mkdir -p log/final/${run}_seed37
  cp -n log/$src/best_val_iou_seed37.pt log/$src/metrics.jsonl log/$src/run_config.json log/final/${run}_seed37/
  NAME=$name RUN=$run BASE=$V5 TRAIN=0 B1=$b1 PAST=$past HZ="$hz" bash tools/final/head.sh \
    || { echo "QUEUE BASIC V5 FAILED at $run"; exit 1; }
done
NAME=bb_e3a_50 bash tools/final/backbone.sh || { echo "QUEUE BASIC V5 FAILED at E3a"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] 等 E1 / E3b 骨干（queue_basic_e1.sh / queue_basic_e3b.sh）"
while [ ! -f log/final/bb_e1_50_seed37/BACKBONE_DONE ] || [ ! -f log/final/bb_e3b_50_seed37/BACKBONE_DONE ]; do
  sleep 300
done
python tools/select_backbone.py --run v5=log/final/bb_v5_frozen_seed37 --run e1=log/final/bb_e1_50_seed37 \
  --run e3a=log/final/bb_e3a_50_seed37 --run e3b=log/final/bb_e3b_50_seed37 --data-root $ROOT --seed 37 \
  --out run/final/backbone_choice_seed37.json > run/final/backbone_choice_seed37.log 2>&1 \
  || { echo "QUEUE BASIC V5 FAILED at select_backbone（run/final/backbone_choice_seed37.log）"; exit 1; }
cat run/final/backbone_choice_seed37.log
W=$(python -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['winner'])" run/final/backbone_choice_seed37.json)
WRUN=$(python -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['winner_run'])" run/final/backbone_choice_seed37.json)
ln -sfn "$(basename "$WRUN")" log/final/bb_m2_50_seed37
HEADRUN=rcl_feature_50_on_$W
if [ "$W" = "e3a" ]; then                             # the only candidate without a head yet
  NAME=rcl_feature_50 RUN=$HEADRUN BASE=log/final/bb_e3a_50_seed37/selected_seed37.pt B1=1 bash tools/final/head.sh \
    || { echo "QUEUE BASIC V5 FAILED at head on E3a"; exit 1; }
fi
echo "[$(date '+%m-%d %H:%M:%S')] 选定骨干 $W（$WRUN），完整模型 = $HEADRUN；等它的评估完成"
while [ ! -f log/final/${HEADRUN}_seed37/HEAD_DONE ]; do sleep 300; done
ln -sfn ${HEADRUN}_seed37 log/final/rcl_feature_50_on_m2_seed37
for s in val test; do
  for e in json md; do
    ln -sfn ${HEADRUN}_seed37_$s.$e run/final/analysis/rcl_feature_50_on_m2_seed37_$s.$e
  done
done
python tools/final_summary.py --data-root $ROOT > run/final/summary_basic.md 2> run/final/summary_basic.err \
  || echo "汇总脚本出错，见 run/final/summary_basic.err"
cat run/final/summary_basic.md
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE BASIC V5 FINISHED"
