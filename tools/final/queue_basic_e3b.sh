#!/usr/bin/env bash
# Basic batch (verify the final model), GPU A: E3b backbone under the guarded protocol (seed 37), then the RCL feature
# head on it (= the full model when E3b is chosen), with trajectories, adaptive / no-future / noise evals and B1.
# Usage: GPU=0 nohup bash tools/final/queue_basic_e3b.sh > run/final/queue_basic_e3b.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-0}
mkdir -p run/final log/final
NAME=bb_e3b_50 bash tools/final/backbone.sh || { echo "QUEUE BASIC E3B FAILED at backbone"; exit 1; }
NAME=rcl_feature_50 RUN=rcl_feature_50_on_e3b BASE=log/final/bb_e3b_50_seed37/selected_seed37.pt B1=1 \
  bash tools/final/head.sh || { echo "QUEUE BASIC E3B FAILED at head"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE BASIC E3B FINISHED"
