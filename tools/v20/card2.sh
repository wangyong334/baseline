#!/usr/bin/env bash
# V20 second round (11-15), one chain per GPU appended after the first round (card.sh A / B, card_c.sh) in the same
# pools. A new file on purpose: the first-round scripts are still running and bash reads a script while it runs.
#   D (GPU 1): message budget 0.25x / 0.5x / 2x per window (1x = v20_bir_carry, none = v20_bir_nobudget)
#              -> accuracy vs messages, hyperparameter table
#   E (GPU 2): TBPTT segment k = 4 / 8 / 32 windows (16 = v20_bir_carry) -> how far back training lets memory reach
#   F (GPU 3): decision layer (V18 single population, seed 0) on BIR reset each window / carry of backbone seeds 38
#              and 39 (card C) -> the past x future 2x2 on three backbone seeds
#   G (11-15 rerun): BIR reset each window seed 37 + its decision layer (card A's run died at the first backward with
#              a cuDNN error, card memory) -> the last cell of the 2x2
#   H (11-15 rerun): conventional SNN (LIF, reset each window, T = 4 steps per window; card B's run died the same way)
# Every step is skipped when its result exists, so rerunning resumes; a failed step is reported and the next one runs.
# Usage (V20 code dir): GPU=1 bash tools/v20/card2.sh D > run/v20/cardD.log 2>&1
set -u
cd "$(dirname "$0")/../.."
CARD=${1:?card D, E, F, G or H}
GPU=${GPU:?GPU=<physical id>}
export GPU CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p run/v20 run/final log/final run/deco
ts() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

bb() {                       # name, seed, eval splits, extra eval option
  local name=$1 seed=$2 splits=$3 extra=${4:-} out=log/final/${1}_seed$2
  if [ -f $out/BACKBONE_DONE ]; then say "$name seed $seed: done before"; return 0; fi
  say "$name seed $seed: start"
  NAME=$name SEED=$seed HZ="" EVAL_SPLITS="$splits" EVAL_EXTRA="$extra" bash tools/final/backbone.sh \
    > run/v20/${name}_s$seed.log 2>&1 || { say "$name seed $seed FAILED: $(tail -n 1 run/v20/${name}_s$seed.log)"; return 1; }
  say "$name seed $seed: $(grep -h '^\[' run/final/${name}_seed${seed}_eval_test.log | cut -c1-70 | tr '\n' ' ')"
}

sdu() {                      # backbone name, backbone seed, tag
  local name=$1 seed=$2 tag=$3 s ck=log/final/${1}_seed$2/selected_seed$2.pt
  [ -f log/final/${name}_seed$seed/BACKBONE_DONE ] || { say "$tag decision layer skipped: no backbone $name seed $seed"; return 1; }
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
  D)
    bb v20_bir_budget05 37 "val test"
    bb v20_bir_budget2 37 "val test"
    bb v20_bir_budget025 37 "val test"
    ;;
  E)
    bb v20_bir_k8 37 "val test"
    bb v20_bir_k4 37 "val test"
    bb v20_bir_k32 37 "val test"           # stream_groups 2 keeps it at the memory of k = 16
    ;;
  F)
    sdu v20_bir_carry 38 p20_bir_carry_bb38
    sdu v20_bir_reset1 38 p20_bir_reset1_bb38
    sdu v20_bir_carry 39 p20_bir_carry_bb39
    sdu v20_bir_reset1 39 p20_bir_reset1_bb39
    ;;
  G)
    bb v20_bir_reset1 37 "val test"
    sdu v20_bir_reset1 37 p20_bir_reset1
    ;;
  H)
    bb v20_lif_reset1_T4 37 "test" --carry-only   # window_steps runs on the carry path, which clears every window itself
    ;;
  *) echo "card must be D, E, F, G or H"; exit 1 ;;
esac
say "V20 CARD $CARD FINISHED"
