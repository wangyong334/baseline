#!/usr/bin/env bash
# 模块二统一评估 + 打包。可重复运行：已有结果的评估会跳过；正在训练 / 评估的模型会跳过（下次再运行补上）。
# 每个模型（后台并行）：
#   1. val / test 干净数据（L=0）正式评估 + 逐事件预测 -> 逐序列误差分解 run/perseq/<名字>_<划分>_L0.json
#   2. test 上鲁棒性曲线 L = $LOADS（缺哪个补哪个）；L=2 另做逐序列误差分解（预测用完即删）
# 最后把所有评估 JSON、逐序列 JSON、训练日志打包到 /hy-tmp/module2_results_<时间>.tar.gz（不含权重和预测）。
# 用法：cd /hy-tmp/EV-UAV-v1 && nohup bash tools/eval_module2.sh > run/eval_module2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
LOADS=${LOADS:-"0.3 1 2 3 5"}
SEED=37
mkdir -p run/perseq run/dump
# 名字 配置名 权重目录
MODELS="v5  evisseg_stream_tif_comm_peak                          log/stream_tif_commpeak_msg_seed37
e3a evisseg_stream_tif_comm_peak_invariant                    log/stream_tif_commpeak_inv_seed37
e3b evisseg_stream_tif_comm_peak_invariant_noiseaug           log/stream_tif_commpeak_inv_noiseaug_seed37
e3w evisseg_stream_tif_comm_peak_invariant_noiseaug_wide      log/stream_tif_commpeak_inv_noiseaug_wide_seed37
e1w evisseg_stream_tif_comm_peak_noiseaug_wide                log/stream_tif_commpeak_noiseaug_wide_seed37"

one_eval() {  # 名字 配置 目录 划分 L 预测目录(可空)
  local name=$1 cfg=$2 out=$3 split=$4 L=$5 dump=$6
  local args=(--config configs/$cfg.yaml --mode eval --device cuda:0 --split $split --overwrite
              --checkpoint $out/best_val_iou_seed${SEED}.pt)
  [ "$L" != 0 ] && args+=(--eval-noise-load $L --eval-noise-seed 0)
  [ -n "$dump" ] && { rm -rf "$dump"; args+=(--dump-dir "$dump"); }
  python -u train_stream_v1.py "${args[@]}" > run/m2eval_${name}_${split}_L${L}.log 2>&1 \
    || { echo "[FAIL] $name $split L=$L，见 run/m2eval_${name}_${split}_L${L}.log"; return 1; }
}

per_seq() {  # 名字 划分 L 预测目录
  python tools/per_sequence_metrics.py --dump-dir $1=$4 --out run/perseq/$1_$2_L$3.json > run/perseq/$1_$2_L$3.txt 2>&1 \
    || echo "[FAIL] 逐序列 $1 $2 L=$3"
}

model() {  # 名字 配置 目录
  local name=$1 cfg=$2 out=$3 split L dump
  for split in val test; do
    if [ ! -f run/perseq/${name}_${split}_L0.json ]; then
      dump=run/dump/${name}_${split}_L0
      one_eval $name $cfg $out $split 0 $dump && per_seq $name $split 0 $dump
    fi
  done
  for L in $LOADS; do
    if [ "$L" = 2 ] && [ ! -f run/perseq/${name}_test_L2.json ]; then
      dump=run/dump/${name}_test_L2
      one_eval $name $cfg $out test 2 $dump && per_seq $name test 2 $dump && rm -rf $dump
    elif [ ! -f $out/eval_test_best_val_iou_seed${SEED}_noiseL${L}_seed0.json ]; then
      one_eval $name $cfg $out test $L ""
    fi
  done
  echo "[DONE] $name"
}

while read -r name cfg out; do
  if [ ! -f $out/best_val_iou_seed${SEED}.pt ]; then echo "[SKIP] $name：还没有权重"; continue; fi
  if pgrep -f "$out( |/|$)" > /dev/null; then echo "[SKIP] $name：还在训练或评估，之后再运行本脚本补上"; continue; fi
  echo "[START] $name"
  model $name $cfg $out &
done <<< "$MODELS"
wait

pack=/hy-tmp/module2_results_$(date +%Y%m%d_%H%M%S).tar.gz
tar -czf $pack log/stream_tif_commpeak_*_seed37/eval_*.json log/stream_tif_commpeak_*_seed37/summary.json \
  run/perseq run/*_train.log run/robust_*.log run/m2eval_*.log 2>/dev/null
ls -l $pack
echo "ALL FINISHED"
