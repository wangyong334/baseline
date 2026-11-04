#!/usr/bin/env bash
# V9-final backbone pipeline (FINAL_PLAN_20261104.md §5): train -> guarded selection on val -> val / test clean eval
# -> controlled-noise eval (Hz/px, carry only; the same noise as the full-model eval). One backbone per GPU (~15.5 GB).
# Usage (code dir, after `conda activate evuav`):
#   GPU=0 NAME=bb_e3b_50 nohup bash tools/final/backbone.sh > run/final/bb_e3b_50_seed37.log 2>&1 &
#   SEED=38 ...                       another seed (output log/final/<RUN>_seed38)
#   RUN=bb_v5_frozen TRAIN=0 ...      evaluate an existing run copied to log/final/<RUN>_seed<SEED> (legacy runs without
#                                     per-sequence val counts fall back to their best-val checkpoint)
#   HZ="0.8 1.6 2.4 4" NOISE_SPLITS="val test" ...   noise levels / splits (default: both splits, 4 levels; HZ="" = none)
# Last line: "BACKBONE <RUN> seed <SEED> FINISHED" (or "FAILED at <step>"); marker file <OUT>/BACKBONE_DONE.
set -u
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
NAME=${NAME:?NAME=configs/final/<NAME>.yaml}
SEED=${SEED:-37}
RUN=${RUN:-$NAME}
CFG=configs/final/$NAME.yaml
OUT=log/final/${RUN}_seed${SEED}
HZ=${HZ-"0.8 1.6 2.4 4"}
NOISE_SPLITS=${NOISE_SPLITS:-"val test"}
LOG=run/final/${RUN}_seed${SEED}
mkdir -p run/final log/final
read ROOT HEIGHT WIDTH < <(python -c "
import sys; sys.path.insert(0, '.')
from utils.stream_common import load_flat_config
c = load_flat_config(sys.argv[1]); print(c['root'], c['height'], c['width_px'])" "$CFG") \
  || { echo "读不出 $CFG（是否已 conda activate evuav？）"; exit 1; }
fail() { echo "[$(date '+%m-%d %H:%M:%S')] BACKBONE $RUN seed $SEED FAILED at $1（日志 $2）"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] $RUN seed $SEED 开始：配置 $CFG，GPU $CUDA_VISIBLE_DEVICES，输出 $OUT，数据 $ROOT，python $(command -v python)"
rm -f $OUT/BACKBONE_DONE
if [ "${TRAIN:-1}" = "1" ]; then
  python -u train_stream_v1.py --config $CFG --mode train --device ${DEVICE:-cuda:0} --seed $SEED --save-root $OUT \
    > ${LOG}_train.log 2>&1 || fail train ${LOG}_train.log
  echo "  训练完成：$(tail -n 1 ${LOG}_train.log)"
fi
python tools/select_checkpoint.py --run $OUT --data-root $ROOT --height $HEIGHT --width $WIDTH --copy \
  > ${LOG}_select.log 2>&1 || fail select ${LOG}_select.log
sed -n '1,4p' ${LOG}_select.log
CK=$OUT/selected_seed${SEED}.pt
for split in val test; do
  python -u train_stream_v1.py --config $CFG --mode eval --device ${DEVICE:-cuda:0} --split $split --overwrite --checkpoint $CK \
    > ${LOG}_eval_$split.log 2>&1 || fail eval_$split ${LOG}_eval_$split.log
  echo "  $split：$(grep '^\[carry\]' ${LOG}_eval_$split.log | cut -c1-90)"
done
for split in $NOISE_SPLITS; do
  for hz in $HZ; do
    python -u train_stream_v1.py --config $CFG --mode eval --device ${DEVICE:-cuda:0} --split $split --overwrite --checkpoint $CK \
      --eval-noise-hz $hz --eval-noise-seed 0 --carry-only > ${LOG}_noise_${split}_${hz}hz.log 2>&1 \
      || fail noise_${split}_${hz} ${LOG}_noise_${split}_${hz}hz.log
    echo "  $split 噪声 $hz Hz/px：$(grep '^\[carry\]' ${LOG}_noise_${split}_${hz}hz.log | cut -c1-70)"
  done
done
touch $OUT/BACKBONE_DONE
echo "[$(date '+%m-%d %H:%M:%S')] BACKBONE $RUN seed $SEED FINISHED"
