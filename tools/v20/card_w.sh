#!/usr/bin/env bash
# V20 slicing experiment (11-15): the final decision layer (V18 single population, seed 0) on BIR backbones with 25 /
# 100 / 200 / 400 ms windows (50 ms = p20_bir_carry, p20_bir_carry_bb38 / bb39). The backbones of the 11-06 slicing
# experiment (V8-rcl41 dir, configs/final/bb_v5_<ms>.yaml, the same code path) are reused; missing seeds are trained
# here. The decision layer reads the window length from the priors: steps = round(500 ms / window) + 1, i.e. a 500 ms
# deadline (400 ms at 200 / 400 ms windows).
#   W1: decision layer on the existing backbones, seed 37 at every length first, then seeds 38 and 39
#   W2: train BIR 100 ms seeds 38 / 39 and 25 ms seed 39 here (val + test eval), each followed by its decision layer
# Every step is skipped when its result exists, so rerunning resumes; a missing backbone is reported and skipped.
# Usage (V20 code dir): GPU=3 bash tools/v20/card_w.sh W1 > run/v20/cardW1.log 2>&1
set -u
cd "$(dirname "$0")/../.."
CARD=${1:?card W1 or W2}
GPU=${GPU:?GPU=<physical id>}
export GPU CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
RCL=${RCL:-/media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41}
mkdir -p run/v20 run/final log/final run/deco
ts() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

sdu() {                      # backbone checkpoint, tag
  local ck=$1 tag=$2 s
  [ -f $ck ] || { say "$tag skipped: no backbone $ck"; return 1; }
  for s in train val test; do
    grep -q "DUMP Z0 FINISHED" run/v20/${tag}_z0_$s.log 2>/dev/null && continue
    python -u tools/deco/dump_z0.py --checkpoint $ck --split $s --out run/deco/$tag/z0_$s > run/v20/${tag}_z0_$s.log 2>&1 \
      || { say "$tag priors $s FAILED: $(tail -n 1 run/v20/${tag}_z0_$s.log)"; return 1; }
  done
  TAG=$tag SEED=0 NAME=base ARGS="" bash tools/v18/run_sdu.sh > run/v20/${tag}_sdu.log 2>&1 \
    || { say "$tag decision layer FAILED: $(tail -n 1 run/v20/${tag}_sdu.log)"; return 1; }
  say "$tag: $(grep -o 'test calibrated IoU [0-9.]* ACC [0-9.]* Pd [0-9.]* Fa [0-9.e+-]*' run/v18/$tag/base_s0/eval.log | tail -n 1)"
}

bb() {                       # window ms, seed: train here, eval val + test
  local name=bb_v5_$1 seed=$2 out=log/final/bb_v5_$1_seed$2
  if [ -f $out/BACKBONE_DONE ]; then say "$name seed $seed: done before"; return 0; fi
  say "$name seed $seed: start"
  NAME=$name SEED=$seed HZ="" EVAL_SPLITS="val test" bash tools/final/backbone.sh > run/v20/${name}_s$seed.log 2>&1 \
    || { say "$name seed $seed FAILED: $(tail -n 1 run/v20/${name}_s$seed.log)"; return 1; }
  say "$name seed $seed: $(grep -h '^\[carry\]' run/final/${name}_seed${seed}_eval_test.log | cut -c1-70)"
}

say "V20 card $CARD on GPU $GPU: start"
case "$CARD" in
  W1)
    for ws in 25:37 100:37 200:37 400:37 25:38 200:38 400:38 200:39 400:39; do
      w=${ws%:*}; s=${ws#*:}
      sdu $RCL/log/final/bb_v5_${w}_seed$s/selected_seed$s.pt p20_w${w}_bb$s
    done
    ;;
  W2)
    for ws in 100:38 100:39 25:39; do
      w=${ws%:*}; s=${ws#*:}
      ck=$RCL/log/final/bb_v5_${w}_seed$s/selected_seed$s.pt
      if [ -f $ck ]; then say "bb_v5_$w seed $s exists in $RCL: no training"; sdu $ck p20_w${w}_bb$s; continue; fi
      bb $w $s && sdu log/final/bb_v5_${w}_seed$s/selected_seed$s.pt p20_w${w}_bb$s
    done
    ;;
  *) echo "card must be W1 or W2"; exit 1 ;;
esac
say "V20 CARD $CARD FINISHED"
