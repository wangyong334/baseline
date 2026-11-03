#!/usr/bin/env bash
# 在一张卡上依次训练若干个交叉拟合的折骨干（一个骨干约占 15.5 GB 显存，一张 24 GB 的卡只跑一个）。
# 每项写成 配置:K:折号:BLOCK:保存目录，日志 = run/<保存目录名去掉 _seed37>_train.log（与 tools/run_rcl_crossfit.sh 相同）。
# 已训练完（有 summary.json 与 best_val_iou_seed37.pt）的跳过；目录存在但没训练完的报错退出（确认后删掉再重跑）。
# WAIT_PID：先等这个进程退出再开始（例如同一张卡上正在训练的骨干）。任何一项失败即退出（返回非零）。
# 用法：cd 代码目录 && conda activate evuav
#   GPU=1 nohup bash tools/run_crossfit_queue.sh configs/x.yaml:4:1:5:log/v5_cv4_f1_seed37 ... > run/queue_gpu1.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:?需要 GPU=卡号}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p run log
stamp() { date '+%m-%d %H:%M:%S'; }
for job in "$@"; do                                  # check every item before waiting or training
  IFS=: read -r cfg k fold block out <<< "$job"
  [ -n "$out" ] && [ -f "$cfg" ] || { echo "格式应为 配置:K:折号:BLOCK:保存目录，收到 $job"; exit 1; }
done
if [ -n "$WAIT_PID" ]; then
  echo "[$(stamp)] 等进程 $WAIT_PID 退出（卡 $CUDA_VISIBLE_DEVICES）"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
for job in "$@"; do
  IFS=: read -r cfg k fold block out <<< "$job"
  if [ -f "$out/summary.json" ] && [ -f "$out/best_val_iou_seed37.pt" ]; then
    echo "[$(stamp)] 已训练完，跳过：$out"
    continue
  fi
  [ ! -d "$out" ] || { echo "$out 已存在但没训练完（没有 summary.json）：确认后删掉该目录再重跑"; exit 1; }
  log=run/$(basename "$out" _seed37)_train.log
  echo "[$(stamp)] 开始 $out（$cfg，第 $fold / $k 折，每块 $block 条，卡 $CUDA_VISIBLE_DEVICES）-> $log"
  python -u train_stream_v1.py --config "$cfg" --mode train --device cuda:0 \
    --cv-folds "$k" --cv-fold "$fold" --cv-block "$block" --save-root "$out" > "$log" 2>&1 \
    || { echo "[$(stamp)] 失败：$out，看 $log"; exit 1; }
  echo "[$(stamp)] 完成 $out"
done
echo "[$(stamp)] 队列 FINISHED（卡 $CUDA_VISIBLE_DEVICES）"
