#!/usr/bin/env bash
# V20 extra jobs for GPU 3 (free on the night of 11-14/15), one after another; finished steps are skipped on a rerun.
#   1. time per window on the idle card, backbone + decision layer, by the stream replay (V18 dir): the final model
#      (two populations, v5s37 coopw_s0) and the single population (v5s37 base_s0) -> complexity table
#   2. more seeds for the memory rows of the ablation (BIR each window reset / BIR carry, seeds 38 and 39; test eval)
#   3. the decision layer on the BIR backbone without the message budget (waits for card B's backbone): completes the
#      component table (memory / neuron / budget / decision layer)
#   4. two populations (decision spikes) on the V20 BIR carry backbone (waits for card A's single population): the
#      last row of the component table on the same backbone
# Usage (V20 code dir): GPU=3 bash tools/v20/card_c.sh > run/v20/cardC.log 2>&1
#   BB37=<V5 seed-37 checkpoint the v5s37 priors came from> (default below; the replay reports max |dz0| against them)
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:?GPU=<physical id>}
export GPU CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
V18B=${V18B:-/media/stephen/nvme0n1/wy_data/EV-UAV-v18b}
BB37=${BB37:-/media/stephen/nvme0n1/wy_data/EV-UAV-v8-rcl41/log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt}
HERE=$(pwd)
mkdir -p run/v20 run/final log/final run/deco
ts() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

timing() {                   # run name under V18B run/v18/v5s37
  local run=$1 out=run/v18/timing/${1}_gpu${GPU}.json
  ( cd "$V18B" && mkdir -p run/v18/timing && [ ! -f $out ] && \
    python -u tools/v18/stream_sdu.py --run run/v18/v5s37/$run --backbone "$BB37" --device cuda:0 --out $out \
      > run/v18/timing/${run}_gpu${GPU}.log 2>&1 ) ; \
  say "timing $run: $(tail -n 1 "$V18B/run/v18/timing/${run}_gpu${GPU}.log" | cut -c1-200)"
}

bb() {                       # name, seed, eval splits, extra eval option
  local name=$1 seed=$2 splits=$3 extra=${4:-} out=log/final/${1}_seed$2
  if [ -f $out/BACKBONE_DONE ]; then say "$name seed $seed: done before"; return 0; fi
  say "$name seed $seed: start"
  NAME=$name SEED=$seed HZ="" EVAL_SPLITS="$splits" EVAL_EXTRA="$extra" bash tools/final/backbone.sh \
    > run/v20/${name}_s$seed.log 2>&1 || { say "$name seed $seed FAILED: $(tail -n 1 run/v20/${name}_s$seed.log)"; return 1; }
  say "$name seed $seed: $(grep -h '^\[' run/final/${name}_seed${seed}_eval_test.log | cut -c1-70 | tr '\n' ' ')"
}

waitfor() {                  # file, pattern (empty = the file exists), what, hours
  local f=$1 pat=$2 what=$3 h=${4:-10} waited=0
  until { [ -z "$pat" ] && [ -e "$f" ]; } || { [ -n "$pat" ] && grep -q "$pat" "$f" 2>/dev/null; }; do
    [ $waited -ge $((h * 3600)) ] && { say "gave up waiting for $what"; return 1; }
    [ $waited -eq 0 ] && say "waiting for $what"
    sleep 300
    waited=$((waited + 300))
  done
}

sdu() {                      # backbone name, tag
  local name=$1 tag=$2 s ck=log/final/${1}_seed37/selected_seed37.pt
  for s in train val test; do
    grep -q "DUMP Z0 FINISHED" run/v20/${tag}_z0_$s.log 2>/dev/null && continue
    python -u tools/deco/dump_z0.py --checkpoint $ck --split $s --out run/deco/$tag/z0_$s > run/v20/${tag}_z0_$s.log 2>&1 \
      || { say "$tag priors $s FAILED"; return 1; }
  done
  TAG=$tag SEED=0 NAME=base ARGS="" bash tools/v18/run_sdu.sh > run/v20/${tag}_sdu.log 2>&1 \
    || { say "$tag decision layer FAILED: $(tail -n 1 run/v20/${tag}_sdu.log)"; return 1; }
  say "$tag decision layer: $(grep -o 'test calibrated IoU [0-9.]* ACC [0-9.]* Pd [0-9.]* Fa [0-9.e+-]*' run/v18/$tag/base_s0/eval.log | tail -n 1)"
}

say "V20 card C on GPU $GPU: start"
timing coopw_s0
timing base_s0
bb v20_bir_reset1 38 "test"
bb v20_bir_carry 38 "test"
bb v20_bir_reset1 39 "test"
bb v20_bir_carry 39 "test"
waitfor log/final/v20_bir_nobudget_seed37/BACKBONE_DONE "" "card B: BIR without budget" 10 && sdu v20_bir_nobudget p20_bir_nobudget
if waitfor run/v18/p20_bir_carry/base_s0/run.log "RUN V18 DONE" "card A: decision layer on BIR carry" 10; then
  if ! grep -q "RUN V18 DONE" run/v18/p20_bir_carry/coop_s0/run.log 2>/dev/null; then
    TAG=p20_bir_carry SEED=0 NAME=coop STAGE1=run/v18/p20_bir_carry/base_s0 ARGS="--spikes feature" \
      bash tools/v18/run_sdu.sh > run/v20/p20_bir_carry_coop.log 2>&1 || say "coop FAILED: $(tail -n 1 run/v20/p20_bir_carry_coop.log)"
  fi
  say "p20_bir_carry two populations: $(grep -o 'test calibrated IoU [0-9.]* ACC [0-9.]* Pd [0-9.]* Fa [0-9.e+-]*' run/v18/p20_bir_carry/coop_s0/eval.log 2>/dev/null | tail -n 1)"
fi
say "V20 CARD C FINISHED"
