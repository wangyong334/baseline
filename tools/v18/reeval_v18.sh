#!/usr/bin/env bash
# V18 (11-14): re-calibrate finished runs (V17 or V18, any code dir; their dirs are only read) with this code's stopping
# rule, without training: tools/v18/eval_sdu.py on their saved trajectories, outputs under OUT/<tag>__<run name>/.
# Prints per run the old line (the rule it was evaluated with) and the new one: test four metrics, born waits, energy,
# val (honest). Default rule "window" (V18); STOP=isotonic reproduces V17.
# Usage (V18 code dir, after `conda activate evuav`):
#   RUNS="<run dir> ..." [OUT=run/v18/reeval] [STOP=window] bash tools/v18/reeval_v18.sh
# Last line: "REEVAL V18 DONE <n> runs".
set -u
cd "$(dirname "$0")/../.."
OUT=${OUT:-run/v18/reeval}
STOP=${STOP:-window}
RUNS=${RUNS:?RUNS="<run dir> ..."}
mkdir -p "$OUT"
n=0
pick() { grep -o "test calibrated IoU .*" "$1" 2>/dev/null | tail -n 1 | sed -E 's/ [0-9.]+ MMAC\/window, (first population [0-9.]+ mJ, )?/ /' | cut -c21-; }
for r in $RUNS; do
  if [ ! -f "$r/sdu.pt" ] || [ ! -f "$r/traj_val.npz" ]; then
    echo "skip $r (no sdu.pt / traj_val.npz)"
    continue
  fi
  o="$OUT/$(basename "$(dirname "$r")")__$(basename "$r")"
  if ! python -u tools/v18/eval_sdu.py --run "$r" --stop "$STOP" --out "$o" > "$o.log" 2>&1; then
    echo "FAILED $r (see $o.log)"
    continue
  fi
  echo "$(basename "$(dirname "$r")")/$(basename "$r")"
  echo "  before: $(pick "$r/eval.log")"
  echo "  $STOP: $(pick "$o.log")"
  n=$((n + 1))
done
echo "REEVAL V18 DONE $n runs"
