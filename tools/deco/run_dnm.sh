#!/usr/bin/env bash
# DeCo final pipeline (11-06) on one frozen backbone: arrival evidence z0 -> pending events + neighbourhood evidence
# (tools/deco/evidence.py) -> decision neurons (tools/deco/train_dnm.py) -> official evaluation (tools/final_analysis.py).
# Usage (code dir, after `conda activate evuav`):
#   CKPT=<backbone .pt> TAG=v5s37 ALPHAS="0.01 0.003" SEEDS="0" GPU=1 bash tools/deco/run_dnm.sh
#   EXTRA="--epochs 2" ...                       training options (tools/deco/train_dnm.py)
#   EV_ARGS="--radius 12" EV_SUFFIX=_r12 ...    evidence options; EV_SUFFIX (required with EV_ARGS) names their files
#   NAME_SUFFIX=_x ...                          result-dir suffix
#   DESIGN=v10 ...                              V10 decision neurons (11-07): evidence without a radius cap
#                                               (EV_ARGS "--radius 0", EV_SUFFIX _nr), train_dnm.py --design v10,
#                                               result dir dnm_a<ALPHA>_s<SEED>_nr_v10; each alpha of ALPHAS is then
#                                               the one risk level of the run: band, crossing commit (--commit-alpha)
#                                               and change-risk head (v9 keeps 0.01 for the last two)
#   HEAD=0 ...                                  skip the change-risk head (tools/deco/fit_head.py; default on)
#   DATA_ROOT=<root> (default: from the checkpoint)  DEVICE=cpu  LIMIT=2 (first recordings only)   for smoke tests
# z0 is dumped once per backbone into run/deco/<TAG>/z0_<split> (shared with tools/deco/run_deco.sh).
# Results: run/deco/<TAG>/dnm_a<ALPHA>_s<SEED><EV_SUFFIX><NAME_SUFFIX>/{summary.json, official_val.md, official_test.md,
#          head/head.json (fixed crossing vs change-risk head, official four metrics)}
# Last line: "DNM <TAG> FINISHED" (or "DNM <TAG> FAILED at <step>").
set -u
cd "$(dirname "$0")/../.."
CKPT=${CKPT:?CKPT=<backbone checkpoint>}
TAG=${TAG:?TAG=<name>}
ALPHAS=${ALPHAS:-0.01}
SEEDS=${SEEDS:-0}
EXTRA=${EXTRA:-}
DESIGN=${DESIGN:-v9}
HEAD=${HEAD:-1}
if [ "$DESIGN" = v10 ]; then
  EV_ARGS=${EV_ARGS:---radius 0}
  EV_SUFFIX=${EV_SUFFIX:-_nr}
  NAME_SUFFIX=${NAME_SUFFIX:-_v10}
  EXTRA="--design v10 $EXTRA"
fi
EV_ARGS=${EV_ARGS:-}
EV_SUFFIX=${EV_SUFFIX:-}
NAME_SUFFIX=${NAME_SUFFIX:-}
WORKERS=${WORKERS:-16}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
OUT=run/deco/$TAG
mkdir -p "$OUT"
ts() { date '+%m-%d %H:%M:%S'; }
fail() { echo "[$(ts)] DNM $TAG FAILED at $1 (log $2)"; exit 1; }
if [ -n "$EV_ARGS" ] && [ -z "$EV_SUFFIX" ]; then
  echo "EV_ARGS needs EV_SUFFIX (evidence files are cached by name)"; exit 1
fi
[ -f "$CKPT" ] || fail checkpoint "$CKPT missing"
read ROOT WMS < <(python -c "
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu')['config']; print(c['root'], c['window_ms'])" "$CKPT") \
  || fail checkpoint "$CKPT"
ROOT=${DATA_ROOT:-$ROOT}
DEV=${DEVICE:-cuda:0}
LIM=${LIMIT:-0}
echo "[$(ts)] DNM $TAG: backbone $CKPT, data $ROOT, window $WMS ms, alphas $ALPHAS, seeds $SEEDS, GPU $CUDA_VISIBLE_DEVICES"
for split in train val test; do
  if [ ! -f "$OUT/z0_$split/DONE" ]; then
    python -u tools/deco/dump_z0.py --checkpoint "$CKPT" --split $split --out "$OUT/z0_$split" \
      --data-root "$ROOT" --device $DEV --limit $LIM \
      > "$OUT/dump_$split.log" 2>&1 || fail dump_$split "$OUT/dump_$split.log"
    touch "$OUT/z0_$split/DONE"
  fi
  echo "  z0 $split: $(tail -n 1 "$OUT/dump_$split.log")"
done
for a in $ALPHAS; do
  ev=$OUT/ev_a${a}${EV_SUFFIX}
  for split in train val test; do
    if [ ! -f "${ev}_$split.npz" ]; then
      python -u tools/deco/evidence.py --z0-dir "$OUT/z0_$split" --split $split --data-root "$ROOT" --alpha $a \
        --out "${ev}_$split.npz" --workers $WORKERS $EV_ARGS > "${ev}_$split.log" 2>&1 \
        || fail evidence_a${a}_$split "${ev}_$split.log"
    fi
    echo "  evidence alpha $a $split: $(tail -n 1 "${ev}_$split.log" | cut -c1-200)"
  done
  if [ "$DESIGN" = v10 ]; then ca=$a; else ca=0.01; fi     # commit / head risk level (one alpha in V10)
  for seed in $SEEDS; do
    name=dnm_a${a}_s${seed}${EV_SUFFIX}${NAME_SUFFIX}
    python -u tools/deco/train_dnm.py --evidence "$ev" --z0-root "$OUT" --data-root "$ROOT" --out "$OUT/$name" \
      --seed $seed --device $DEV --commit-alpha $ca $EXTRA > "$OUT/$name.log" 2>&1 || fail train_$name "$OUT/$name.log"
    echo "  $name: $(tail -n 1 "$OUT/$name.log")"
    for split in val test; do
      python tools/final_analysis.py --traj-dir "$OUT/$name/traj_$split" --split $split --data-root "$ROOT" \
        --window-ms $WMS --out "$OUT/$name/official_$split.json" --title "${TAG}_${name}_$split" \
        > "$OUT/$name/official_$split.log" 2>&1 || fail official_${name}_$split "$OUT/$name/official_$split.log"
      echo "    $split official (zero / fixed D): $(grep -E '^\| (zero|fixed_[0-9]+) ' "$OUT/$name/official_$split.md" \
        | sed -n '1p;$p' | cut -d'|' -f2-6 | tr '\n' ' ')"
    done
    if [ "$HEAD" = 1 ]; then
      python -u tools/deco/fit_head.py --run "$OUT/$name" --evidence "$ev" --z0-root "$OUT" --data-root "$ROOT" \
        --alpha $ca > "$OUT/$name/head.log" 2>&1 || fail head_$name "$OUT/$name/head.log"
      echo "    $(tail -n 1 "$OUT/$name/head.log" | cut -c1-200)"
    fi
  done
done
echo "[$(ts)] DNM $TAG FINISHED"
