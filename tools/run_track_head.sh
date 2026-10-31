#!/usr/bin/env bash
# 一个航迹头配置的完整流程：训练 -> val / test 评估（逐事件预测、逐窗航迹、误差类别、关口）。
# 日志、预测、航迹都按配置名区分，几个配置可以在同一张卡上同时跑。
# 用法：cd 代码目录 && conda activate evuav
#   CFG=configs/track_head_v2.yaml GPU=3 nohup bash tools/run_track_head.sh > run/track_head_v2_all.log 2>&1 &
#   EVAL_ONLY=1 ...                  只评估已有权重
set -e
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
CFG=${CFG:-configs/track_head_v2.yaml}
NAME=${NAME:-$(basename "$CFG" .yaml)}
echo "[$(date '+%m-%d %H:%M:%S')] $NAME 开始：配置 $CFG，GPU $CUDA_VISIBLE_DEVICES，python $(command -v python)"
read OUT SEED < <(python -c "import sys, yaml; d = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['TRAIN']; print(d['save_root'], d['seed'])" "$CFG")   || { echo "读不出 $CFG 的 save_root / seed（是否已 conda activate evuav？）"; exit 1; }
[ -n "$OUT" ] || { echo "读不出 $CFG 的 save_root"; exit 1; }
echo "  输出目录 $OUT，种子 $SEED"
mkdir -p run run/dump run/tracks
if [ "${EVAL_ONLY:-0}" != "1" ]; then
  python -u train_track_head.py --config "$CFG" --mode train --device cuda:0 > run/${NAME}_train.log 2>&1
fi
for split in val test; do
  python -u train_track_head.py --config "$CFG" --mode eval --device cuda:0 --split $split \
    --checkpoint $OUT/best_val_iou_seed${SEED}.pt --dump-dir run/dump/${NAME}_$split \
    --tracks-dir run/tracks/${NAME}_$split > run/${NAME}_$split.log 2>&1
done
echo "[$(date '+%m-%d %H:%M:%S')] $NAME FINISHED"
