#!/usr/bin/env bash
# V9 optimisation 3: the full model at 25 ms (M1 + M2 + M3), now that module two's constants are physical (model/comod.py,
# train_stream_v1.ref_rate_per_window; bit-identical at 50 ms).
#   E3b @ 25 ms backbone (guarded selection; noise evals 0.8 and 4 Hz/px) -> RCL feature head @ 25 ms (D = 20) on it.
# Runs only when the pre-registered rule chose E3b (run/final/backbone_choice_seed37.json); otherwise the 25 ms rows
# stay V5 @ 25 + RCL (queue_25ms.sh), reported as they are.
# Usage: GPU=1 nohup bash tools/final/queue_25ms_full.sh > run/final/queue_25ms_full.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-1}
SEED=${SEED:-37}
mkdir -p run/final log/final
[ -f configs/stream_v1_train_stats_w25.json ] || { echo "缺 configs/stream_v1_train_stats_w25.json"; exit 1; }
CHOICE=run/final/backbone_choice_seed37.json
[ -f $CHOICE ] || { echo "还没有选骨干的结果：$CHOICE"; exit 1; }
W=$(python -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['winner'])" $CHOICE)
if [ "$W" != "e3b" ]; then
  echo "选定的骨干是 $W，不是 E3b：不跑 25 ms 完整模型（25 ms 只报 V5@25 + RCL）"
  echo "[$(date '+%m-%d %H:%M:%S')] QUEUE 25MS FULL FINISHED"
  exit 0
fi
NAME=bb_e3b_25 SEED=$SEED HZ="0.8 4" bash tools/final/backbone.sh || { echo "QUEUE 25MS FULL FAILED at E3b@25"; exit 1; }
NAME=rcl_feature_25 RUN=rcl_feature_25_on_e3bw25 SEED=$SEED BASE=log/final/bb_e3b_25_seed$SEED/selected_seed$SEED.pt \
  HZ="0.8 4" bash tools/final/head.sh || { echo "QUEUE 25MS FULL FAILED at RCL@25"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE 25MS FULL seed $SEED FINISHED"
