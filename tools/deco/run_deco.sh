#!/usr/bin/env bash
# DeCo probe pipeline on one frozen backbone (11-06): arrival evidence z0 -> pending events + hourglass tokens ->
# decision neurons -> official evaluation (tools/final_analysis.py). One GPU; CPU workers for the tokens.
# Usage (code dir, after `conda activate evuav`):
#   CKPT=<backbone .pt> TAG=v5s37 ALPHAS="0.01 0.003" SEEDS="0" GPU=1 bash tools/deco/run_deco.sh
#   EXTRA="--step0 1" NAME_SUFFIX=_s0 ...    probe options (tools/deco/train_probe.py) and a result-dir suffix
#   CHECK_TRAJ=run/final/traj/b2_lookahead_50_on_v5_seed37 ...   (optional) compare z0 with that head's column 0
#   DATA_ROOT=<root> (default: from the checkpoint)  DEVICE=cpu  LIMIT=2 (first recordings only)   for smoke tests
# Results: run/deco/<TAG>/a<ALPHA>_s<SEED><SUFFIX>/{summary.json, official_val.md, official_test.md}
# Last line: "DECO <TAG> FINISHED" (or "DECO <TAG> FAILED at <step>").
set -u
cd "$(dirname "$0")/../.."
CKPT=${CKPT:?CKPT=<backbone checkpoint>}
TAG=${TAG:?TAG=<name>}
ALPHAS=${ALPHAS:-0.01}
SEEDS=${SEEDS:-0}
EXTRA=${EXTRA:-}
NAME_SUFFIX=${NAME_SUFFIX:-}
WORKERS=${WORKERS:-16}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
OUT=run/deco/$TAG
mkdir -p "$OUT"
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] DECO $TAG FAILED at $1 (log $2)"; exit 1; }
read ROOT WMS < <(python -c "
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu')['config']; print(c['root'], c['window_ms'])" "$CKPT") \
  || fail checkpoint "$CKPT"
ROOT=${DATA_ROOT:-$ROOT}
DEV=${DEVICE:-cuda:0}
LIM=${LIMIT:-0}
echo "[$(ts)] DECO $TAG: backbone $CKPT, data $ROOT, window $WMS ms, alphas $ALPHAS, seeds $SEEDS, GPU $CUDA_VISIBLE_DEVICES"
for split in train val test; do
  if [ ! -f "$OUT/z0_$split/DONE" ]; then
    chk=""
    if [ -n "${CHECK_TRAJ:-}" ] && [ "$split" != "train" ] && [ -d "${CHECK_TRAJ}_$split" ]; then
      chk="--check-traj ${CHECK_TRAJ}_$split"
    fi
    python -u tools/deco/dump_z0.py --checkpoint "$CKPT" --split $split --out "$OUT/z0_$split" $chk \
      --data-root "$ROOT" --device $DEV --limit $LIM \
      > "$OUT/dump_$split.log" 2>&1 || fail dump_$split "$OUT/dump_$split.log"
    touch "$OUT/z0_$split/DONE"
  fi
  echo "  z0 $split: $(tail -n 1 "$OUT/dump_$split.log")"
done
for a in $ALPHAS; do
  for split in train val test; do
    f="$OUT/tok_a${a}_$split.npz"
    if [ ! -f "$f" ]; then
      python -u tools/deco/tokens.py --z0-dir "$OUT/z0_$split" --split $split --data-root "$ROOT" --alpha $a --out "$f" \
        --workers $WORKERS > "$OUT/tok_a${a}_$split.log" 2>&1 || fail tokens_a${a}_$split "$OUT/tok_a${a}_$split.log"
    fi
    echo "  tokens alpha $a $split: $(tail -n 1 "$OUT/tok_a${a}_$split.log" | cut -c1-170)"
  done
  for seed in $SEEDS; do
    name=a${a}_s${seed}${NAME_SUFFIX}
    python -u tools/deco/train_probe.py --tokens "$OUT/tok_a${a}" --z0-root "$OUT" --data-root "$ROOT" --out "$OUT/$name" \
      --seed $seed --device $DEV $EXTRA > "$OUT/$name.log" 2>&1 || fail train_$name "$OUT/$name.log"
    echo "  $name: $(tail -n 1 "$OUT/$name.log")"
    for split in val test; do
      python tools/final_analysis.py --traj-dir "$OUT/$name/traj_$split" --split $split --data-root "$ROOT" \
        --window-ms $WMS --out "$OUT/$name/official_$split.json" --title "${TAG}_${name}_$split" \
        > "$OUT/$name/official_$split.log" 2>&1 || fail official_${name}_$split "$OUT/$name/official_$split.log"
      echo "    $split official (zero / fixed D): $(grep -E '^\| (zero|fixed_[0-9]+) ' "$OUT/$name/official_$split.md" \
        | sed -n '1p;$p' | cut -d'|' -f2-6 | tr '\n' ' ')"
    done
  done
done
echo "[$(ts)] DECO $TAG FINISHED"
