#!/usr/bin/env bash
# 模块三航迹头的完整流程：训练 -> val / test 评估（逐事件预测、逐窗航迹、误差类别、关口）-> 打包结果。
# 用法（实验室服务器，3 号卡）：
#   cd /media/stephen/nvme0n1/wy_data/EV-UAV-v7-track && conda activate evuav
#   nohup bash tools/run_track_head.sh > run/track_head_all.log 2>&1 &
# 只评估已有权重：EVAL_ONLY=1 nohup bash tools/run_track_head.sh > run/track_head_eval.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-3} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CFG=${CFG:-configs/track_head_e3a.yaml}
OUT=${OUT:-log/track_head_e3a_seed37}
mkdir -p run run/dump run/tracks
if [ "${EVAL_ONLY:-0}" != "1" ]; then
  python -u train_track_head.py --config $CFG --mode train --device cuda:0 --save-root $OUT > run/track_head_train.log 2>&1
fi
for split in val test; do
  python -u train_track_head.py --config $CFG --mode eval --device cuda:0 --split $split \
    --checkpoint $OUT/best_val_iou_seed37.pt --dump-dir run/dump/track_$split --tracks-dir run/tracks/track_$split \
    > run/track_head_$split.log 2>&1
done
pack=track_head_results_$(date +%Y%m%d_%H%M%S).tar.gz
tar -czf $pack $OUT/*.json $OUT/metrics.jsonl run/track_head_*.log run/dump/track_val run/dump/track_test \
  run/tracks/track_val run/tracks/track_test 2>/dev/null
ls -lh $pack
echo "ALL FINISHED"
