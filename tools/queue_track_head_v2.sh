#!/usr/bin/env bash
# 排队运行航迹头 v2：先预检（单元测试 + 一次短的 GPU 冒烟），再一直等到 3 号卡上的任务全部结束
# （v1 流程 tools/run_track_head.sh 的进程不在了，并且卡上连续 3 分钟没有计算进程），
# 然后在 3 号卡上同时启动三个运行（主方案、一窗滞后判决、无航迹对照），全部结束后打包。
# 用法（在 v2 的代码目录里，先 conda activate evuav）：
#   mkdir -p run && nohup bash tools/queue_track_head_v2.sh > run/queue_v2.log 2>&1 &
#   tail -n 20 run/queue_v2.log           # 看进度
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-3}
WAIT_PATTERN=${WAIT_PATTERN:-"run_track_head[.]sh"}
CONFIGS=${CONFIGS:-"configs/track_head_v2.yaml configs/track_head_v2_lag1.yaml configs/track_head_v2_notrack.yaml"}
POLL=${POLL:-60}           # 检查间隔（秒）
STAGGER=${STAGGER:-30}     # 相邻两个运行的启动间隔（秒）
mkdir -p run
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 1. 预检：E3a 权重、单元测试（CPU，不占卡）
[ -f log/stream_tif_commpeak_inv_seed37/best_val_iou_seed37.pt ] || { log "缺 E3a 权重 log/stream_tif_commpeak_inv_seed37/best_val_iou_seed37.pt"; exit 1; }
log "单元测试..."
if ! python -m unittest discover -s tests -p "test_*.py" > run/queue_v2_tests.log 2>&1; then
  log "单元测试失败，见 run/queue_v2_tests.log"; tail -n 20 run/queue_v2_tests.log; exit 1
fi
tail -n 3 run/queue_v2_tests.log

# 2. GPU 冒烟：滞后判决版本（覆盖最多的代码路径），2 轮、每轮 5 个片段，约 4 分钟，显存约 4 GiB
log "GPU 冒烟（3 号卡，与当前任务并存几分钟）..."
if ! CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -u train_track_head.py \
    --config configs/track_head_v2_lag1.yaml --mode train --device cuda:0 --epochs 2 --max-chunks 5 \
    --save-root log/track_head_v2_smoke > run/queue_v2_smoke.log 2>&1; then
  log "冒烟失败，见 run/queue_v2_smoke.log"; tail -n 20 run/queue_v2_smoke.log; exit 1
fi
grep "^epoch" run/queue_v2_smoke.log
rm -rf log/track_head_v2_smoke

# 3. 等待 3 号卡上的任务全部结束
busy() { nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; }
log "等待：v1 流程结束且 $GPU 号卡连续 3 次检查（间隔 ${POLL} 秒）没有计算进程..."
idle=0
while [ $idle -lt 3 ]; do
  if pgrep -f "$WAIT_PATTERN" > /dev/null || busy; then idle=0; else idle=$((idle + 1)); fi
  sleep $POLL
done

# 4. 同时启动三个运行（相邻两个间隔 30 秒）
log "$GPU 号卡空闲，启动 v2"
for cfg in $CONFIGS; do
  name=$(basename "$cfg" .yaml)
  CFG=$cfg GPU=$GPU nohup bash tools/run_track_head.sh > run/${name}_all.log 2>&1 &
  log "  已启动 $name（PID $!）"
  sleep $STAGGER
done
wait
for cfg in $CONFIGS; do
  name=$(basename "$cfg" .yaml)
  tail -n 1 run/${name}_all.log
  grep -hE "Error|Traceback" run/${name}_*.log | head -3
done

# 5. 打包：结果包（JSON、日志，小）与预测包（逐事件预测、逐窗航迹，大）分开
stamp=$(date +%Y%m%d_%H%M%S)
tar -czf track_head_v2_results_${stamp}.tar.gz log/track_head_v2*_seed37/*.json log/track_head_v2*_seed37/metrics.jsonl \
  run/track_head_v2*.log run/queue_v2*.log 2>/dev/null
tar -czf track_head_v2_dumps_${stamp}.tar.gz run/dump/track_head_v2* run/tracks/track_head_v2* 2>/dev/null
ls -lh track_head_v2_results_${stamp}.tar.gz track_head_v2_dumps_${stamp}.tar.gz
log "ALL FINISHED"
