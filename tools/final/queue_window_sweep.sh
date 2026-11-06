#!/usr/bin/env bash
# Slice-length sweep (11-06): BIR vs LIF backbones at 25 / 100 / 200 ms (50 ms done: bb_v5_50, bb_v5lif_50).
# Prediction: the BIR - LIF gap grows with the slice length (fewer steps per second; a binary spike carries <= 1 bit per
# step, a BIR message up to ~3.9 bits). Pre-registered rule: BIR - LIF >= +0.01 test IoU at 100 or 200 ms -> BIR is a main
# component of the paper; otherwise BIR stays an efficiency component.
# Usage (code dir, after `conda activate evuav`):
#   GPU=3 nohup bash tools/final/queue_window_sweep.sh > run/final/queue_window_sweep.log 2>&1 &
#   RUNS="bb_v5_100 bb_v5lif_100" ...   subset / order (default: all five, most decisive first)
# Last line: "QUEUE WINDOW SWEEP seed <SEED> FINISHED" (or "FAILED at <run>").
set -u
cd "$(dirname "$0")/../.."
export GPU=${GPU:-3}
SEED=${SEED:-37}
RUNS=${RUNS:-"bb_v5_100 bb_v5lif_100 bb_v5lif_25 bb_v5_200 bb_v5lif_200"}
mkdir -p run/final log/final
for f in configs/stream_v1_train_stats_w25.json configs/stream_v1_train_stats_w100.json \
         configs/stream_v1_train_stats_w200.json; do
  [ -f "$f" ] || { echo "缺 $f（随代码包上传）"; exit 1; }
done
for name in $RUNS; do
  NAME=$name SEED=$SEED HZ="" bash tools/final/backbone.sh || { echo "QUEUE WINDOW SWEEP FAILED at $name"; exit 1; }
done
echo "[$(date '+%m-%d %H:%M:%S')] QUEUE WINDOW SWEEP seed $SEED FINISHED"
