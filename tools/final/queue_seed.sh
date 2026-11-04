#!/usr/bin/env bash
# Phase 2 (FINAL_SERVER_COMMANDS_20261104.md): one more seed of the paper's main rows, on one GPU.
#   V5 and the chosen module-two backbone (M2 = its config name, from run/final/backbone_choice_seed37.json),
#   RCL feature head on both, B2 and the causal head on V5 (B2OCC=1: also B2 + occupancy). About 20-24 h per seed.
# Usage: GPU=0 SEED=38 M2=bb_e3b_50 nohup bash tools/final/queue_seed.sh > run/final/queue_seed38.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-0}
SEED=${SEED:?SEED=38 or 39}
M2=${M2:?M2=<config of the chosen backbone, e.g. bb_e3b_50>}
mkdir -p run/final log/final
NAME=bb_v5_50 SEED=$SEED bash tools/final/backbone.sh || { echo "QUEUE SEED $SEED FAILED at V5"; exit 1; }
if [ "$M2" != "bb_v5_50" ]; then
  NAME=$M2 SEED=$SEED bash tools/final/backbone.sh || { echo "QUEUE SEED $SEED FAILED at $M2"; exit 1; }
fi
ln -sfn ${M2}_seed$SEED log/final/bb_m2_50_seed$SEED
V5=log/final/bb_v5_50_seed$SEED/selected_seed$SEED.pt
NAME=rcl_feature_50 RUN=rcl_feature_50_on_m2 SEED=$SEED BASE=log/final/bb_m2_50_seed$SEED/selected_seed$SEED.pt \
  bash tools/final/head.sh || { echo "QUEUE SEED $SEED FAILED at full model"; exit 1; }
NAME=rcl_feature_50 RUN=rcl_feature_50_on_v5 SEED=$SEED BASE=$V5 HZ="" B1=1 bash tools/final/head.sh \
  || { echo "QUEUE SEED $SEED FAILED at RCL on V5"; exit 1; }
NAME=b2_lookahead_50 RUN=b2_lookahead_50_on_v5 SEED=$SEED BASE=$V5 HZ="" bash tools/final/head.sh \
  || { echo "QUEUE SEED $SEED FAILED at B2"; exit 1; }
NAME=rcl_feature_causal_50 RUN=rcl_feature_causal_50_on_v5 SEED=$SEED BASE=$V5 HZ="" PAST=0 bash tools/final/head.sh \
  || { echo "QUEUE SEED $SEED FAILED at causal head"; exit 1; }
if [ "${B2OCC:-0}" = "1" ]; then                     # B2 + occupancy for this seed too (about +4 h)
  NAME=b2_lookahead_occ_50 RUN=b2_lookahead_occ_50_on_v5 SEED=$SEED BASE=$V5 HZ="" bash tools/final/head.sh \
    || { echo "QUEUE SEED $SEED FAILED at B2 + occupancy"; exit 1; }
fi
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE SEED $SEED FINISHED"
