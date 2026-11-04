#!/usr/bin/env bash
# Phase 1, GPU A (FINAL_SERVER_COMMANDS_20261104.md): module-two 2x2 backbones E1, E3b, E3a (guarded, seed 37);
# then the pre-registered backbone choice (tools/select_backbone.py, val only; the reference V5 is the frozen V5
# evaluated by queue_m3.sh) and the full model = chosen backbone + RCL feature head.
# Usage: GPU=0 nohup bash tools/final/queue_m2.sh > run/final/queue_m2.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-0}
SEED=${SEED:-37}
mkdir -p run/final log/final
ROOT=$(python -c "import sys; sys.path.insert(0, '.'); from utils.stream_common import load_flat_config; print(load_flat_config('configs/final/bb_e3b_50.yaml')['root'])") \
  || { echo "读不出数据根目录（是否已 conda activate evuav？）"; exit 1; }
for n in bb_e1_50 bb_e3b_50 bb_e3a_50; do
  NAME=$n SEED=$SEED bash tools/final/backbone.sh || { echo "QUEUE M2 FAILED at $n"; exit 1; }
done
V5RUN=log/final/bb_v5_frozen_seed37
echo "[$(date '+%m-%d %H:%M:%S')] 等冻结 V5 的新口径评估（queue_m3.sh）：$V5RUN/BACKBONE_DONE"
while [ ! -f $V5RUN/BACKBONE_DONE ]; do sleep 300; done
python tools/select_backbone.py --run v5=$V5RUN --run e1=log/final/bb_e1_50_seed$SEED \
  --run e3a=log/final/bb_e3a_50_seed$SEED --run e3b=log/final/bb_e3b_50_seed$SEED --data-root $ROOT --seed $SEED \
  --out run/final/backbone_choice_seed$SEED.json > run/final/backbone_choice_seed$SEED.log 2>&1 \
  || { echo "QUEUE M2 FAILED at select_backbone（run/final/backbone_choice_seed$SEED.log）"; exit 1; }
cat run/final/backbone_choice_seed$SEED.log
WIN=$(python -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['winner_run'])" \
      run/final/backbone_choice_seed$SEED.json)
ln -sfn "$(basename "$WIN")" log/final/bb_m2_50_seed$SEED
echo "选定骨干：$WIN -> log/final/bb_m2_50_seed$SEED"
NAME=rcl_feature_50 RUN=rcl_feature_50_on_m2 SEED=$SEED BASE=log/final/bb_m2_50_seed$SEED/selected_seed$SEED.pt \
  bash tools/final/head.sh || { echo "QUEUE M2 FAILED at full model"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE M2 seed $SEED FINISHED"
