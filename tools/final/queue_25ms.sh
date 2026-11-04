#!/usr/bin/env bash
# Phase 1, GPU B (FINAL_SERVER_COMMANDS_20261104.md): the 25 ms core experiment of module three.
#   V5 @ 25 ms backbone (guarded, no noise evals) -> RCL feature head @ 25 ms (D = 20 windows = 500 ms) on it.
# Compared with RCL @ 50 ms on the frozen V5 (queue_m3.sh); judged on val (FINAL_PLAN §4.6).
# Usage: GPU=1 nohup bash tools/final/queue_25ms.sh > run/final/queue_25ms.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-1}
SEED=${SEED:-37}
mkdir -p run/final log/final
[ -f configs/stream_v1_train_stats_w25.json ] || { echo "缺 configs/stream_v1_train_stats_w25.json（随代码包上传）"; exit 1; }
NAME=bb_v5_25 SEED=$SEED HZ="" bash tools/final/backbone.sh || { echo "QUEUE 25MS FAILED at V5@25"; exit 1; }
NAME=rcl_feature_25 RUN=rcl_feature_25_on_v5w25 SEED=$SEED BASE=log/final/bb_v5_25_seed$SEED/selected_seed$SEED.pt \
  HZ="" bash tools/final/head.sh || { echo "QUEUE 25MS FAILED at RCL@25"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE 25MS seed $SEED FINISHED"
