#!/usr/bin/env bash
# V18 pool status (11-14): GPU use, the pool's last lines, and per job of run/v18/<TAG>/<NAME>_s<SEED>: done (test
# calibrated IoU, val honest IoU, energy), failed (reason), waiting for its first population, or running (step and
# last epoch). Usage (V18 code dir): bash tools/v18/status_v18.sh [GPU id, default 1]
cd "$(dirname "$0")/../.."
R=run/v18
date '+%m-%d %H:%M:%S'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "${1:-1}"
tail -n 3 $R/pool_main.log 2>/dev/null
for r in $R/v5s3*/*_s*; do
  [ -d "$r" ] || continue
  n=${r#$R/}
  if grep -q "RUN V18 DONE" $r/run.log 2>/dev/null; then
    t=$(grep -o "test calibrated IoU [0-9.]* ACC [0-9.]* Pd [0-9.]* Fa [0-9.e+-]*" $r/eval.log | tail -n 1 | cut -d' ' -f4-)
    v=$(grep -o "val (honest) IoU [0-9.]*" $r/eval.log | tail -n 1 | awk '{print $4}')
    e=$(grep -o "[0-9.]* mJ / 8 s" $r/eval.log | tail -n 1)
    printf "  done    %-30s test %s | val %s | %s\n" "$n" "$t" "$v" "$e"
  elif grep -q "RUN V18 FAILED" $r/run.log 2>/dev/null; then
    printf "  FAILED  %-30s %s\n" "$n" "$(grep 'RUN V18 FAILED' $r/run.log | tail -n 1 | cut -c1-100)"
  elif [ ! -f $r/train.log ]; then
    printf "  waiting %-30s for its first population\n" "$n"
  else
    st=train
    grep -q "SDU TRAIN FINISHED" $r/train.log 2>/dev/null && st=eval
    grep -q "SDU EVAL FINISHED" $r/eval.log 2>/dev/null && st=analyze
    grep -q "SDU ANALYSIS FINISHED" $r/analyze.log 2>/dev/null && st=reducible
    grep -q "REDUCIBLE FINISHED" $r/reducible.log 2>/dev/null && st=stream
    ep=$(grep -h "^ep " $r/train.log 2>/dev/null | tail -n 1 | sed 's/ loss.*val IoU/ | val IoU/' | cut -c1-90)
    printf "  running %-30s [%s] %s\n" "$n" "$st" "${ep:-no epoch yet}"
  fi
done
