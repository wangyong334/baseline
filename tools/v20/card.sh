#!/usr/bin/env bash
# V20 past-information ablation (11-14 night): the jobs of one GPU, run one after another (a backbone needs ~15.5 GB,
# so nothing else may run on the card meanwhile). Every step is skipped when its result exists, so rerunning the same
# command resumes; a failed step is reported and the next one still runs (later steps that need it fail on their own).
#   A (GPU 1): BIR carry (+ noise, long stream) -> BIR reset each window -> decision layer on both (2x2 past x future)
#              -> memory-length diagnostic on the carry weights -> BIR segments of 4 / 16 windows -> decision layer on LIF
#   B (GPU 2): LIF carry (+ noise, long stream) -> BIR without the message budget (+ noise, long stream)
#              -> LIF reset each window -> conventional SNN (LIF, reset each window, T = 4 steps per window)
# Usage (V20 code dir, after `conda activate evuav`): GPU=1 bash tools/v20/card.sh A > run/v20/cardA.log 2>&1
# Progress: bash tools/v20/status.sh
set -u
cd "$(dirname "$0")/../.."
CARD=${1:?card A or B}
GPU=${GPU:?GPU=<physical id>}
export GPU CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
SEED=37
mkdir -p run/v20 run/final log/final run/deco
ts() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

bb() {                       # name, noise levels (Hz/px, test only; "" = none), eval splits, extra eval option
  local name=$1 hz=$2 splits=$3 extra=${4:-} out=log/final/${1}_seed$SEED
  if [ -f $out/BACKBONE_DONE ]; then say "$name: done before"; return 0; fi
  say "$name: start (training + selection + eval $splits${hz:+ + noise $hz})"
  NAME=$name SEED=$SEED HZ="$hz" NOISE_SPLITS=test EVAL_SPLITS="$splits" EVAL_EXTRA="$extra" \
    bash tools/final/backbone.sh > run/v20/$name.log 2>&1 || { say "$name FAILED: $(tail -n 1 run/v20/$name.log)"; return 1; }
  say "$name: $(grep -h '^\[' run/final/${name}_seed${SEED}_eval_test.log | cut -c1-70 | tr '\n' ' ')"
}

long() {                     # long-stream evaluation of a carry backbone (test, played back to back twice)
  local name=$1 dir=log/final/${1}_seed$SEED
  [ -f $dir/long_stream_test.json ] && { say "$name long stream: done before"; return 0; }
  [ -f $dir/BACKBONE_DONE ] || { say "$name long stream skipped: no backbone"; return 1; }
  python -u tools/v20/long_stream.py --checkpoint $dir/selected_seed$SEED.pt --split test --repeats 2 \
    > run/v20/${name}_long.log 2>&1 || { say "$name long stream FAILED: $(tail -n 1 run/v20/${name}_long.log)"; return 1; }
  say "$name long stream: $(grep -E '^(fresh|long)' run/v20/${name}_long.log | cut -c1-48 | tr '\n' '|')"
}

diag() {                     # memory-length diagnostic on the same carry weights: clear the state every M windows
  local name=$1 m dir=log/final/${1}_seed$SEED
  shift
  for m in "$@"; do
    ls $dir/eval_test_selected_seed${SEED}_reset_every$m.json > /dev/null 2>&1 && continue
    python -u train_stream_v1.py --config configs/final/$name.yaml --mode eval --device cuda:0 --split test --overwrite \
      --checkpoint $dir/selected_seed$SEED.pt --reset-every $m > run/v20/${name}_diag$m.log 2>&1 \
      || { say "$name diag $m FAILED"; continue; }
    say "$name cleared every $m windows: $(grep -h '^\[carry\]' run/v20/${name}_diag$m.log | cut -c1-60)"
  done
}

sdu() {                      # priors of a backbone -> the V18 single-population decision layer (seed 0) on them
  local name=$1 tag=$2 s ck=log/final/${1}_seed$SEED/selected_seed$SEED.pt
  [ -f $ck ] || { say "$tag decision layer skipped: no backbone $name"; return 1; }
  for s in train val test; do
    grep -q "DUMP Z0 FINISHED" run/v20/${tag}_z0_$s.log 2>/dev/null && continue
    python -u tools/deco/dump_z0.py --checkpoint $ck --split $s --out run/deco/$tag/z0_$s > run/v20/${tag}_z0_$s.log 2>&1 \
      || { say "$tag priors $s FAILED: $(tail -n 1 run/v20/${tag}_z0_$s.log)"; return 1; }
  done
  TAG=$tag SEED=0 NAME=base ARGS="" bash tools/v18/run_sdu.sh > run/v20/${tag}_sdu.log 2>&1 \
    || { say "$tag decision layer FAILED: $(tail -n 1 run/v20/${tag}_sdu.log)"; return 1; }
  say "$tag decision layer: $(grep -o 'test calibrated IoU [0-9.]* ACC [0-9.]* Pd [0-9.]* Fa [0-9.e+-]*' run/v18/$tag/base_s0/eval.log | tail -n 1)"
}

say "V20 card $CARD on GPU $GPU: start"
case "$CARD" in
  A)
    bb v20_bir_carry "0.8 1.6 4" "val test"
    long v20_bir_carry
    bb v20_bir_reset1 "" "val test"
    sdu v20_bir_carry p20_bir_carry
    sdu v20_bir_reset1 p20_bir_reset1
    diag v20_bir_carry 4 16
    bb v20_bir_reset4 "" "test" --carry-only
    bb v20_bir_reset16 "" "test" --carry-only
    sdu v20_lif_carry p20_lif_carry        # LIF backbone comes from card B (usually done by then)
    ;;
  B)
    bb v20_lif_carry "0.8 1.6 4" "val test"
    long v20_lif_carry
    bb v20_bir_nobudget "0.8 1.6 4" "val test"
    long v20_bir_nobudget
    bb v20_lif_reset1 "" "test"
    bb v20_lif_reset1_T4 "" "test" --carry-only
    ;;
  *) echo "card must be A or B"; exit 1 ;;
esac
say "V20 CARD $CARD FINISHED"
