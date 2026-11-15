#!/usr/bin/env bash
# V20 "where BIR should beat LIF" (11-15), 50 ms, seed 37 unless noted. One shared list of backbones: every card that
# runs this script takes the next item nobody has claimed yet (atomic mkdir under run/v20/claims), so the same command
# can be queued on several GPUs and on a card that frees up later. Each item = train + selection + val / test eval.
#   A  same message budget: LIF with the BIR peak constraint (messages = spikes of the sending layers) at 1 / 0.5 /
#      0.25 x 157700 per window; BIR points = v20_bir_carry / v20_bir_budget05 / v20_bir_budget025
#   B  narrow networks: channels x 0.5 and x 0.25 for BIR (budget scaled with the sending neurons) and LIF
#   +  BIR without budget, seeds 38 / 39: is "budget >= no budget" more than seed noise?
# A failed item keeps its claim (no automatic retry): rm -r run/v20/claims/<name>_s<seed> and queue again.
# Usage (V20 code dir): GPU=2 bash tools/v20/card_x.sh > run/v20/cardX_gpu2.log 2>&1
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:?GPU=<physical id>}
export GPU CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p run/v20/claims run/final log/final
ts() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
ITEMS="v20_lif_budget05:37 v20_lif_w025:37 v20_bir_w025:37 v20_lif_budget025:37 v20_lif_w05:37 v20_bir_w05:37
       v20_lif_budget1:37 v20_bir_nobudget:38 v20_bir_nobudget:39"

say "V20 card X on GPU $GPU: start"
for item in $ITEMS; do
  name=${item%:*}; seed=${item#*:}; out=log/final/${name}_seed$seed
  [ -f $out/BACKBONE_DONE ] && continue
  mkdir run/v20/claims/${name}_s$seed 2> /dev/null || continue           # another card has it
  say "$name seed $seed: start"
  NAME=$name SEED=$seed HZ="" EVAL_SPLITS="val test" bash tools/final/backbone.sh > run/v20/${name}_s$seed.log 2>&1 \
    || { say "$name seed $seed FAILED: $(tail -n 1 run/v20/${name}_s$seed.log)"; continue; }
  say "$name seed $seed: $(grep -h '^\[carry\]' run/final/${name}_seed${seed}_eval_test.log | cut -c1-70)"
done
say "V20 CARD X FINISHED (GPU $GPU)"
