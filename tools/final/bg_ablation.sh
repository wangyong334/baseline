#!/usr/bin/env bash
# V9 optimisation 2: module two's background-estimator ablation on a trained backbone (evaluation only, no retraining).
# Waits for <run>/BACKBONE_DONE, then evaluates its selected checkpoint with the centre component off (no_center), the
# abrupt-change restart off (no_forget) and both off (ring_only): val / test, clean and at the HZ noise levels (carry).
# The reference (all on) is the backbone's own eval: eval_<split>_selected_seed<S>[_noise<hz>hz_seed0].json.
# Results next to the checkpoint: eval_<split>_selected_seed<S>_bg_<mode>[_noise<hz>hz_seed0].json.
# Usage: GPU=1 nohup bash tools/final/bg_ablation.sh > run/final/bg_ablation_e3b.log 2>&1 &
#        NAME=bb_e3b_50 (default)  SEED=37  MODES="no_center no_forget ring_only"  HZ="4"
set -u
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
NAME=${NAME:-bb_e3b_50}
SEED=${SEED:-37}
MODES=${MODES:-"no_center no_forget ring_only"}
HZ=${HZ-"4"}
CFG=configs/final/$NAME.yaml
OUT=log/final/${NAME}_seed${SEED}
CK=$OUT/selected_seed${SEED}.pt
LOG=run/final/${NAME}_seed${SEED}_bg
mkdir -p run/final
echo "[$(date '+%m-%d %H:%M:%S')] 背景估计器消融 $NAME seed $SEED：等 $OUT/BACKBONE_DONE，GPU $CUDA_VISIBLE_DEVICES"
until [ -f $OUT/BACKBONE_DONE ]; do sleep 300; done
for m in $MODES; do
  for split in val test; do
    python -u train_stream_v1.py --config $CFG --mode eval --device cuda:0 --split $split --overwrite --checkpoint $CK \
      --carry-only --bg-ablation $m > ${LOG}_${m}_${split}.log 2>&1 || { echo "BG ABLATION FAILED at $m $split"; exit 1; }
    echo "  $m $split 干净：$(grep '^\[carry\]' ${LOG}_${m}_${split}.log | cut -c1-80)"
    for hz in $HZ; do
      python -u train_stream_v1.py --config $CFG --mode eval --device cuda:0 --split $split --overwrite --checkpoint $CK \
        --carry-only --bg-ablation $m --eval-noise-hz $hz --eval-noise-seed 0 > ${LOG}_${m}_${split}_${hz}hz.log 2>&1 \
        || { echo "BG ABLATION FAILED at $m $split ${hz}Hz"; exit 1; }
      echo "  $m $split 噪声 $hz Hz/px：$(grep '^\[carry\]' ${LOG}_${m}_${split}_${hz}hz.log | cut -c1-80)"
    done
  done
done
echo "[$(date '+%m-%d %H:%M:%S')] BG ABLATION $NAME seed $SEED FINISHED"
