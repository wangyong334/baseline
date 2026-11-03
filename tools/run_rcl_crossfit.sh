#!/usr/bin/env bash
# RCL v4.1 的完整流程（交叉拟合 = 堆叠泛化，configs/changepoint_v41.yaml 文件头）：
#   1. K 个折骨干：与 V5 同一配方（BACKBONE），第 k 个留出第 k 折（连续 BLOCK 条录像为一块，块轮流分折），同时跑 PAR 个
#      （一个骨干训练约占 15.5 GB 显存，24 GB 的卡只能跑一个，所以默认 PAR = 1；11-03 两个并行时第二个 OOM）；
#   2. 样本外输出：第 k 个骨干推理第 k 折录像（tools/export_oof_outputs.py），写到判决层配置的 train_outputs；
#   3. 判决层：tools/run_track_head.sh（训练 50 轮 -> val / test 评估）。验证 / 测试用完整的 V5（配置里的 base_checkpoint）。
# 已完成的步骤自动跳过（折骨干目录里有 summary.json 视为训练完；样本外文件存在则不再导出），中途退出后原命令重跑即可续上。
# 用法：cd 代码目录 && conda activate evuav
#   GPU=3 nohup bash tools/run_rcl_crossfit.sh > run/rcl_crossfit_all.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
K=${K:-4}
BLOCK=${BLOCK:-5}
PAR=${PAR:-1}
BACKBONE=${BACKBONE:-configs/evisseg_stream_tif_comm_peak.yaml}
HEAD=${HEAD:-configs/changepoint_v41.yaml}
OOF=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['TRAIN']['train_outputs'])" "$HEAD") \
  || { echo "读不出 $HEAD 的 train_outputs（是否已 conda activate evuav？）"; exit 1; }
mkdir -p run log
echo "[$(date '+%m-%d %H:%M:%S')] 交叉拟合开始：K=$K BLOCK=$BLOCK PAR=$PAR，骨干配方 $BACKBONE，判决层 $HEAD，GPU $CUDA_VISIBLE_DEVICES"
pids=()
wait_all() {
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "[$(date '+%m-%d %H:%M:%S')] 折骨干训练退出（PID $pid），看 run/v5_cv${K}_f*_train.log"; exit 1; }
  done
  pids=()
}
cks=()
for k in $(seq 0 $((K - 1))); do
  out=log/v5_cv${K}_f${k}_seed37
  cks+=("$out/best_val_iou_seed37.pt")
  if [ -f "$out/summary.json" ] && [ -f "$out/best_val_iou_seed37.pt" ]; then
    echo "  第 $k 折骨干已训练完，跳过：$out"
    continue
  fi
  [ ! -d "$out" ] || { echo "$out 已存在但没训练完（没有 summary.json）：确认后删掉该目录再重跑本脚本"; exit 1; }
  echo "[$(date '+%m-%d %H:%M:%S')]   第 $k 折骨干开始 -> $out"
  python -u train_stream_v1.py --config "$BACKBONE" --mode train --device cuda:0 \
    --cv-folds "$K" --cv-fold "$k" --cv-block "$BLOCK" --save-root "$out" > run/v5_cv${K}_f${k}_train.log 2>&1 &
  pids+=($!)
  if [ "${#pids[@]}" -ge "$PAR" ]; then
    wait_all
  else
    sleep 30
  fi
done
wait_all
echo "[$(date '+%m-%d %H:%M:%S')] 折骨干全部完成"
if [ -f "$OOF" ]; then
  echo "  样本外输出已存在，跳过导出：$OOF"
else
  python -u tools/export_oof_outputs.py --config "$HEAD" --checkpoints "${cks[@]}" --out "$OOF" --device cuda:0 \
    > run/rcl_crossfit_export.log 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] 样本外输出 -> $OOF"
fi
CFG="$HEAD" GPU="$GPU" bash tools/run_track_head.sh
echo "[$(date '+%m-%d %H:%M:%S')] 交叉拟合流程 FINISHED"
