#!/usr/bin/env bash
# V9-final decision-head pipeline (module three and its controls; FINAL_PLAN_20261104.md §7):
# train on the frozen backbone's in-sample outputs -> guarded selection on val -> val / test eval at fixed D with
# per-lag trajectories -> official adaptive eval -> same head without future (past) -> full-model noise eval ->
# tools/final_analysis.py (all commit policies, information-arrival curve, waits) [-> B1 when B1=1].
# Usage (code dir, after `conda activate evuav`):
#   GPU=3 NAME=rcl_feature_50 BASE=log/final/bb_e3b_50_seed37/selected_seed37.pt RUN=rcl_feature_50_on_e3b \
#     nohup bash tools/final/head.sh > run/final/rcl_feature_50_on_e3b_seed37.log 2>&1 &
#   RUN       run name (default NAME); output log/final/<RUN>_seed<SEED>, trajectories run/final/traj/<RUN>_seed<SEED>_<split>
#   SEED=38   another seed;  TRAIN=0 skip training;  HZ="" skip the noise evals;  PAST=0 skip the no-future eval
#   B1=1      also run the non-learned B1 control on this backbone (reads the lag-0 column = backbone output)
# Last line: "HEAD <RUN> seed <SEED> FINISHED" (or "FAILED at <step>").
set -u
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
NAME=${NAME:?NAME=configs/final/<NAME>.yaml}
BASE=${BASE:?BASE=<backbone checkpoint>}
RUN=${RUN:-$NAME}
SEED=${SEED:-37}
CFG=configs/final/$NAME.yaml
OUT=log/final/${RUN}_seed${SEED}
HZ=${HZ-"0.8 1.6 2.4 4"}
LOG=run/final/${RUN}_seed${SEED}
TRAJ=run/final/traj/${RUN}_seed${SEED}
ANA=run/final/analysis
mkdir -p run/final/traj $ANA log/final
[ -f "$BASE" ] || { echo "骨干权重不存在：$BASE"; exit 1; }
read ROOT WMS HEIGHT WIDTH < <(python -c "
import sys, torch; sys.path.insert(0, '.')
from utils.stream_common import load_flat_config
c = load_flat_config(sys.argv[1]); b = torch.load(sys.argv[2], map_location='cpu')['config']
print(c['root'], b['window_ms'], b['height'], b['width_px'])" "$CFG" "$BASE") \
  || { echo "读不出 $CFG / $BASE（是否已 conda activate evuav？）"; exit 1; }
fail() { echo "[$(date '+%m-%d %H:%M:%S')] HEAD $RUN seed $SEED FAILED at $1（日志 $2）"; exit 1; }
COMMON=(--config $CFG --device ${DEVICE:-cuda:0} --base-checkpoint $BASE --seed $SEED)
echo "[$(date '+%m-%d %H:%M:%S')] $RUN seed $SEED 开始：GPU $CUDA_VISIBLE_DEVICES，骨干 $BASE（$WMS ms），输出 $OUT"
rm -f $OUT/HEAD_DONE
if [ "${TRAIN:-1}" = "1" ]; then
  python -u train_track_head.py "${COMMON[@]}" --mode train --save-root $OUT > ${LOG}_train.log 2>&1 \
    || fail train ${LOG}_train.log
  echo "  训练完成：$(tail -n 1 ${LOG}_train.log)"
fi
python tools/select_checkpoint.py --run $OUT --data-root $ROOT --height $HEIGHT --width $WIDTH --copy \
  > ${LOG}_select.log 2>&1 || fail select ${LOG}_select.log
sed -n '1,4p' ${LOG}_select.log
CK=$OUT/selected_seed${SEED}.pt
for split in val test; do
  rm -rf ${TRAJ}_$split
  python -u train_track_head.py "${COMMON[@]}" --mode eval --split $split --checkpoint $CK \
    --trajectory-dir ${TRAJ}_$split > ${LOG}_eval_$split.log 2>&1 || fail eval_$split ${LOG}_eval_$split.log
  grep -E "^$split (base|track) " ${LOG}_eval_$split.log | cut -c1-60
  python -u train_track_head.py "${COMMON[@]}" --mode eval --split $split --checkpoint $CK --commit adaptive \
    --tag adaptive > ${LOG}_adaptive_$split.log 2>&1 || fail adaptive_$split ${LOG}_adaptive_$split.log
  if [ "${PAST:-1}" = "1" ]; then
    python -u train_track_head.py "${COMMON[@]}" --mode eval --split $split --checkpoint $CK --future-stats past \
      --tag past > ${LOG}_past_$split.log 2>&1 || fail past_$split ${LOG}_past_$split.log
  fi
  python tools/final_analysis.py --traj-dir ${TRAJ}_$split --split $split --data-root $ROOT --window-ms $WMS \
    --height $HEIGHT --width $WIDTH --out $ANA/${RUN}_seed${SEED}_$split.json --title "${RUN} seed ${SEED} $split" \
    > ${LOG}_analysis_$split.log 2>&1 || fail analysis_$split ${LOG}_analysis_$split.log
done
for hz in $HZ; do
  python -u train_track_head.py "${COMMON[@]}" --mode eval --split test --checkpoint $CK --commit adaptive \
    --tag adaptive --eval-noise-hz $hz --eval-noise-seed 0 > ${LOG}_noise_${hz}hz.log 2>&1 \
    || fail noise_$hz ${LOG}_noise_${hz}hz.log
  echo "  test 噪声 $hz Hz/px：$(grep -E '^test track ' ${LOG}_noise_${hz}hz.log | cut -c1-60)"
done
if [ "${B1:-0}" = "1" ]; then
  python tools/b1_delayed_confirm.py --val-traj ${TRAJ}_val --test-traj ${TRAJ}_test --data-root $ROOT --window-ms $WMS \
    --height $HEIGHT --width $WIDTH --out $ANA/b1_on_${RUN}_seed${SEED}.json > ${LOG}_b1.log 2>&1 || fail b1 ${LOG}_b1.log
  grep "^B1" ${LOG}_b1.log | cut -c1-120
fi
touch $OUT/HEAD_DONE
echo "[$(date '+%m-%d %H:%M:%S')] HEAD $RUN seed $SEED FINISHED"
