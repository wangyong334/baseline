#!/usr/bin/env bash
# Basic batch (verify the final model), GPU B: E1 backbone (narrow noise training, no NISI: module two's matched
# control) under the guarded protocol (seed 37), then the RCL feature head on it (the full model when E1 is chosen).
# Usage: GPU=1 nohup bash tools/final/queue_basic_e1.sh > run/final/queue_basic_e1.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-1}
mkdir -p run/final log/final
NAME=bb_e1_50 bash tools/final/backbone.sh || { echo "QUEUE BASIC E1 FAILED at backbone"; exit 1; }
NAME=rcl_feature_50 RUN=rcl_feature_50_on_e1 BASE=log/final/bb_e1_50_seed37/selected_seed37.pt B1=1 \
  bash tools/final/head.sh || { echo "QUEUE BASIC E1 FAILED at head"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE BASIC E1 FINISHED"
