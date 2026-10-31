#!/usr/bin/env bash
# 整体模型从头训练（10-31）：
#   第一段 估计器（模块一 BIR + 峰值消息约束，模块二不变输入接口）从头训练 50 轮 -> val / test 评估；
#   第二段 判决器（模块三，目标级航迹头）接在新估计器的实际输出上训练 -> val / test 评估（每个候选一份）。
# 两段式：判决器必须在估计器的实际输出上训练（估计不依赖判决，见 PAPER-NOTES 12.5）；端到端联合训练不在本轮范围。
# 断点续跑：每段做完写 .done 标记，重跑时跳过；估计器中断后自动 --resume；判决器中断后删掉重训（每个几小时）。
# 用法：cd 代码目录 && conda activate evuav
#   冒烟（前台，约 10 分钟；通过后再开训）：SMOKE=1 GPU=2 bash tools/run_full_model.sh
#   正式（后台）：GPU=2 nohup bash tools/run_full_model.sh > run/full_seed38_all.log 2>&1 &
#   可选环境变量：SEED（默认 38；37 是 E3a 的种子，配置是确定性的，等于原样重跑 E3a）
#                HEADS（默认 "track_head_v4d track_head_v4e"）、SMOKE_HEADS（冒烟只测新代码路径，默认 v4e）、BB_CFG
set -e
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
SEED=${SEED:-38}
BB_CFG=${BB_CFG:-configs/evisseg_stream_tif_comm_peak_invariant.yaml}
HEADS=${HEADS:-"track_head_v4d track_head_v4e"}
SMOKE_HEADS=${SMOKE_HEADS:-"track_head_v4e"}
E3A=log/stream_tif_commpeak_inv_seed37/best_val_iou_seed37.pt     # 冒烟时判决器接的现成骨干
ROOT=log/full_seed${SEED}
BB=$ROOT/backbone
CK=$BB/best_val_iou_seed${SEED}.pt
stamp() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(stamp)] 失败：$1"; if [ -n "$2" ]; then tail -n 30 "$2"; fi; exit 1; }

python -c "import torch, yaml" 2>/dev/null || fail "python 环境不对（先 conda activate evuav）"
[ -f configs/stream_v1_train_stats.json ] || fail "缺 configs/stream_v1_train_stats.json（上传本地那份）"
[ -f "$BB_CFG" ] || fail "缺 $BB_CFG"
for h in $HEADS $SMOKE_HEADS; do [ -f configs/$h.yaml ] || fail "缺 configs/$h.yaml"; done
mkdir -p run run/dump run/tracks log
# 同一种子同时只能有一份流水线（重复启动会让两个进程续训同一个目录）；冒烟另用一把锁
if [ "${SMOKE:-0}" = "1" ]; then LOCK=log/.full_seed${SEED}_smoke.lock; else LOCK=log/.full_seed${SEED}.lock; fi
exec 9> "$LOCK"
flock -n 9 || fail "已有一份种子 $SEED 的流水线在跑（$LOCK），不要重复启动"
echo "[$(stamp)] 整体模型：种子 $SEED，GPU $CUDA_VISIBLE_DEVICES，估计器 $BB_CFG，判决器 $HEADS，python $(command -v python)"

if [ "${SMOKE:-0}" = "1" ]; then
  S=log/smoke_full_seed${SEED}
  rm -rf "$S"
  mkdir -p "$S"
  echo "[$(stamp)] 冒烟 1/3：航迹头单元测试（含小骨干的端到端训练与评估，CPU，约 3 分钟）"
  python -m unittest discover -s tests -p "test_track_head.py" > $S/tests.log 2>&1 || fail "单元测试" $S/tests.log
  tail -n 3 $S/tests.log
  echo "[$(stamp)] 冒烟 2/3：估计器在 GPU 上校准 + 一次参数更新"
  python -u train_stream_v1.py --config $BB_CFG --mode smoke --device cuda:0 --seed $SEED --save-root $S/backbone \
    > $S/backbone.log 2>&1 || fail "估计器冒烟" $S/backbone.log
  grep -E '"peak_memory_gib"|SMOKE TEST FINISHED' $S/backbone.log || true
  [ -f "$E3A" ] || fail "缺 $E3A（判决器冒烟要接一个现成骨干）"
  for h in $SMOKE_HEADS; do
    echo "[$(stamp)] 冒烟 3/3：判决器 $h 接 E3a 骨干训练 2 轮（每轮 3 个片段）+ 两次 val 评估（约 5～8 分钟）"
    python -u train_track_head.py --config configs/$h.yaml --mode train --device cuda:0 --seed $SEED \
      --base-checkpoint $E3A --epochs 2 --max-chunks 3 --save-root $S/$h > $S/$h.log 2>&1 || fail "判决器 $h" $S/$h.log
    grep -E "^epoch" $S/$h.log | tail -n 1
  done
  echo "[$(stamp)] 冒烟通过（输出在 $S，可删）"
  exit 0
fi

# 第一段：估计器
if [ ! -f $ROOT/.done_backbone ]; then
  RESUME=""
  if [ -f $BB/last.pt ]; then
    RESUME="--resume"
    echo "[$(stamp)] 估计器从 $BB/last.pt 续训"
  elif [ -d $BB ]; then
    fail "$BB 已存在但没有 last.pt（上次没跑完第一轮），确认后删掉这个目录再跑"
  fi
  echo "[$(stamp)] 估计器训练（50 轮），日志 run/full_seed${SEED}_backbone_train.log"
  python -u train_stream_v1.py --config $BB_CFG --mode train --device cuda:0 --seed $SEED --save-root $BB $RESUME \
    >> run/full_seed${SEED}_backbone_train.log 2>&1 || fail "估计器训练" run/full_seed${SEED}_backbone_train.log
  for split in val test; do
    python -u train_stream_v1.py --config $BB_CFG --mode eval --device cuda:0 --seed $SEED --split $split \
      --overwrite --checkpoint $CK > run/full_seed${SEED}_backbone_$split.log 2>&1 \
      || fail "估计器 $split 评估" run/full_seed${SEED}_backbone_$split.log
  done
  touch $ROOT/.done_backbone
  echo "[$(stamp)] 估计器完成"
else
  echo "[$(stamp)] 估计器已完成，跳过（$CK）"
fi

# 第二段：判决器（每个候选一份；挑选规则在 val 上预先定好，test 只报一次）
for h in $HEADS; do
  OUT=$ROOT/$h
  if [ -f $ROOT/.done_$h ]; then
    echo "[$(stamp)] $h 已完成，跳过"
    continue
  fi
  rm -rf $OUT
  echo "[$(stamp)] 判决器 $h 训练，日志 run/full_seed${SEED}_${h}_train.log"
  python -u train_track_head.py --config configs/$h.yaml --mode train --device cuda:0 --seed $SEED \
    --base-checkpoint $CK --save-root $OUT > run/full_seed${SEED}_${h}_train.log 2>&1 \
    || fail "判决器 $h 训练" run/full_seed${SEED}_${h}_train.log
  for split in val test; do
    python -u train_track_head.py --config configs/$h.yaml --mode eval --device cuda:0 --seed $SEED --split $split \
      --base-checkpoint $CK --save-root $OUT --checkpoint $OUT/best_val_iou_seed${SEED}.pt \
      --dump-dir run/dump/full_seed${SEED}_${h}_$split --tracks-dir run/tracks/full_seed${SEED}_${h}_$split \
      > run/full_seed${SEED}_${h}_$split.log 2>&1 || fail "判决器 $h $split 评估" run/full_seed${SEED}_${h}_$split.log
  done
  touch $ROOT/.done_$h
  echo "[$(stamp)] 判决器 $h 完成"
done
echo "[$(stamp)] FINISHED"
