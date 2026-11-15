#!/usr/bin/env bash
# V20 progress (V20 code dir): GPUs, the last lines of every card log, per backbone run its state and test numbers,
# and the decision layers.
cd "$(dirname "$0")/../.."
date '+%m-%d %H:%M:%S'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,2,3
for c in A B C D E F G H W1 W2; do [ -f run/v20/card$c.log ] && { echo "== card $c"; tail -n 3 run/v20/card$c.log; }; done
echo "== backbones"
for cfg in configs/final/v20_*.yaml; do
  n=$(basename $cfg .yaml)
  row=carry                                       # a model trained with reset each window is read on that row
  grep -q '^  state_mode: reset_each_window' $cfg && row=reset_each_window
  for d in log/final/${n}_seed37 log/final/${n}_seed38 log/final/${n}_seed39; do
    s=${d##*_seed}
    if [ -f $d/BACKBONE_DONE ]; then
      printf "  done     %-22s s%s %s\n" $n $s "$(grep -h "^\[$row\]" run/final/${n}_seed${s}_eval_test.log 2>/dev/null | cut -c1-72)"
    elif [ -f $d/metrics.jsonl ]; then
      printf "  training %-22s s%s epoch %s\n" $n $s "$(wc -l < $d/metrics.jsonl)"
    elif [ $s = 37 ]; then
      printf "  waiting  %-22s\n" $n
    fi
  done
done
echo "== decision layers (V18 single population, seed 0; coop = two populations)"
for f in run/v18/p20_*/*_s0/eval.log; do
  [ -f "$f" ] || continue
  printf "  %-32s %s\n" "$(dirname ${f#run/v18/})" "$(grep -o 'test calibrated IoU [0-9.]* ACC [0-9.]* Pd [0-9.]*' $f | tail -n 1)"
done
