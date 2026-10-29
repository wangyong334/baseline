#!/usr/bin/env bash
# 一个模块二版本的完整流程：训练 -> val / test 正式评估 -> test 上的鲁棒性曲线（Inn 13.53 的协议）。
# 用法：bash tools/run_variant.sh 名字 配置名（不带 .yaml） 输出目录 [种子，默认 37] ["噪声强度列表"，默认 "0.3 1 2 3 5"]
# 例：  nohup bash tools/run_variant.sh e3w evisseg_stream_tif_comm_peak_invariant_noiseaug_wide \
#         log/stream_tif_commpeak_inv_noiseaug_wide_seed37 > run/e3w_all.log 2>&1 &
# 只评估已有权重（不训练）：EVAL_ONLY=1 bash tools/run_variant.sh 名字 配置名 输出目录 种子 "5"
set -e
name=$1; cfgname=$2; out=$3; seed=${4:-37}; loads=${5:-"0.3 1 2 3 5"}
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p run
CK=$out/best_val_iou_seed${seed}.pt
if [ "${EVAL_ONLY:-0}" != "1" ]; then
  python -u train_stream_v1.py --config configs/$cfgname.yaml --mode train --device cuda:0 --seed $seed \
    --save-root $out > run/${name}_train.log 2>&1
  for split in val test; do
    python -u train_stream_v1.py --config configs/$cfgname.yaml --mode eval --device cuda:0 --split $split \
      --overwrite --checkpoint $CK > run/${name}_${split}.log 2>&1
  done
fi
for L in $loads; do
  python -u train_stream_v1.py --config configs/$cfgname.yaml --mode eval --device cuda:0 --split test \
    --overwrite --checkpoint $CK --eval-noise-load $L --eval-noise-seed 0 > run/robust_${name}_L${L}.log 2>&1
done
echo "$name FINISHED"
