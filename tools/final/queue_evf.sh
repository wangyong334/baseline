#!/usr/bin/env bash
# Phase 2 (FINAL_SERVER_COMMANDS_20261104.md): EV-Flying x4 with the chosen module-two backbone.
# The RCL feature head on V5-EVF is evaluated by queue_m3.sh; this queue adds, when the chosen module-two variant is
# E3b (M2=bb_e3b_50): E3b-EVF retrained with the mask off (D9) + the RCL feature head on it.
# EV-Flying trajectories are large (~1 GB per split); TRAJ_KEEP=0 deletes them after the analysis.
# Usage: GPU=1 M2=bb_e3b_50 nohup bash tools/final/queue_evf.sh > run/final/queue_evf.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-1}
M2=${M2:?M2=<config of the chosen backbone, e.g. bb_e3b_50>}
mkdir -p run/final log/final
if [ "$M2" != "bb_e3b_50" ]; then
  echo "选定的模块二变体是 $M2，EV-Flying 只有 E3b 的配置：不跑（如实报告 V5-EVF + 模块三）"
  echo "[$(date '+%m-%d %H:%M:%S')] QUEUE EVF FINISHED"
  exit 0
fi
NAME=bb_e3b_evf NOISE_SPLITS="test" HZ="0.8" bash tools/final/backbone.sh || { echo "QUEUE EVF FAILED at E3b-EVF"; exit 1; }
NAME=rcl_feature_evf RUN=rcl_feature_evf_on_e3bevf BASE=log/final/bb_e3b_evf_seed37/selected_seed37.pt HZ="" \
  bash tools/final/head.sh || { echo "QUEUE EVF FAILED at RCL on E3b-EVF"; exit 1; }
[ "${TRAJ_KEEP:-1}" = "1" ] || rm -rf run/final/traj/rcl_feature_evf_on_e3bevf_seed37_*
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE EVF FINISHED"
