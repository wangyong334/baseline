#!/usr/bin/env bash
# Frozen BIR -> shared SDU -> frozen-policy VoW -> real stopped-query evaluation.
# Use a new OUT for every architecture/configuration; never overwrite legacy runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the raw dataset root}
Z0_ROOT=${Z0_ROOT:?Set Z0_ROOT to a directory containing z0_train/z0_val/z0_test}
OUT=${OUT:?Set OUT to a NEW run directory}
CONFIG=${CONFIG:-configs/bst_shared.json}
DEVICE=${DEVICE:-cuda:0}
BACKBONE=${BACKBONE:-}

if [ -e "$OUT/model.pt" ] || [ -e "$OUT/policy.pt" ]; then
  echo "Preserving existing checkpoints. Choose a new OUT or run individual commands."
  exit 1
fi
mkdir -p "$OUT"
if [ -n "$BACKBONE" ]; then
  for split in train val test; do
    mkdir -p "$Z0_ROOT/z0_$split"
    python -u tools/bst/export_priors.py --checkpoint "$BACKBONE" --data-root "$DATA_ROOT" \
      --split "$split" --out "$Z0_ROOT/z0_$split" --device "$DEVICE"
  done
fi
python -u tools/bst/train.py --config "$CONFIG" --data-root "$DATA_ROOT" --z0-root "$Z0_ROOT" \
  --out "$OUT" --device "$DEVICE" 2>&1 | tee "$OUT/train.log"
python -u tools/bst/fit_policy.py --checkpoint "$OUT/model.pt" --data-root "$DATA_ROOT" --z0-root "$Z0_ROOT" \
  --out "$OUT/policy.pt" --device "$DEVICE" 2>&1 | tee "$OUT/policy.log"
# These policies are declared comparisons, not test-selected winners.
for split in val test; do
  for commit in fixed learned; do
    python -u tools/bst/evaluate.py --checkpoint "$OUT/model.pt" --policy "$OUT/policy.pt" \
      --data-root "$DATA_ROOT" --z0-root "$Z0_ROOT" --split "$split" --birth band --commit "$commit" \
      --out "$OUT/${split}_band_$commit" --device "$DEVICE" 2>&1 | tee "$OUT/${split}_band_$commit.log"
  done
  python -u tools/bst/evaluate.py --checkpoint "$OUT/model.pt" --policy "$OUT/policy.pt" \
    --data-root "$DATA_ROOT" --z0-root "$Z0_ROOT" --split "$split" --birth learned --commit learned \
    --out "$OUT/${split}_learned_learned" --device "$DEVICE" 2>&1 | tee "$OUT/${split}_learned_learned.log"
done
echo "BST PIPELINE FINISHED: $OUT"
