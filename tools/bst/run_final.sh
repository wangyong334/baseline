#!/usr/bin/env bash
# Immutable source snapshot job, invoked only by the GPU 1 queue.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATA_ROOT:?}" "${BACKBONE:?}" "${OUT:?}" "${Z0_ROOT:?}" "${CONFIG:?}" "${SEED:?}" "${RELEASE_MANIFEST:?}" "${BACKBONE_SHA256:?}" "${BST_PYTHON:?}"
[ "${GPU:-}" = 1 ] || { echo "Refusing to run outside the GPU 1 queue"; exit 1; }
[ "${CUDA_VISIBLE_DEVICES:-}" = 1 ] || { echo "CUDA_VISIBLE_DEVICES must be physical GPU 1"; exit 1; }
# Use the interpreter selected when publishing, even if an older worker has a
# different conda PATH. Local modules must come from this frozen source tree.
unset PYTHONPATH
"$BST_PYTHON" - "$CONFIG" "$SEED" "$BACKBONE" "$BACKBONE_SHA256" <<'PY'
import json,sys
from tools.bst.queue_final import backbone_seed,digest
config,seed,backbone,expected=sys.argv[1:]
if json.load(open(config))["train"]["seed"]!=int(seed):
    raise ValueError("SDU seed mismatch")
if digest(backbone)!=expected:
    raise ValueError("Backbone changed after queue publication")
if backbone_seed(backbone)!=int(seed):
    raise ValueError("BIR checkpoint.config.seed does not match requested seed")
PY
# Atomic claim; incomplete old runs are never overwritten or silently resumed.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
cp "$CONFIG" "$OUT/release_config.json"
cp "$RELEASE_MANIFEST" "$OUT/release.json"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "failed exit=%s at %s\n" "$rc" "$(date -Is)" > "$OUT/FAILED"; fi' EXIT
"$BST_PYTHON" - "$OUT/environment.json" <<'PY'
import json,os,platform,subprocess,sys
import numpy as np
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise RuntimeError("Expected exactly one CUDA device (physical GPU 1) in the training environment")
try:
    driver=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,driver_version","--format=csv,noheader"],timeout=10).decode()
except (OSError,subprocess.SubprocessError) as error:
    driver=str(error)
record={"python_executable":sys.executable,"python_prefix":sys.prefix,"python_version":sys.version,"platform":platform.platform(),
        "torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version(),
        "numpy_version":np.__version__,"visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_order":os.environ.get("CUDA_DEVICE_ORDER"),"physical_gpu":1,"logical_device":"cuda:0",
        "device_name":torch.cuda.get_device_name(0),"driver_inventory":driver,"cwd":os.getcwd(),
        "seed":int(os.environ["SEED"]),"conda_prefix":os.environ.get("CONDA_PREFIX")}
with open(sys.argv[1],"w") as out:json.dump(record,out,indent=2)
PY
for split in train val test; do
  "$BST_PYTHON" -u tools/bst/export_priors.py --checkpoint "$BACKBONE" --data-root "$DATA_ROOT" \
    --split "$split" --out "$Z0_ROOT/z0_$split" --device cuda:0 2>&1 | tee "$OUT/export_$split.log"
done
"$BST_PYTHON" -u tools/bst/train.py --config "$CONFIG" --data-root "$DATA_ROOT" --z0-root "$Z0_ROOT" \
  --out "$OUT" --device cuda:0 --seed "$SEED" --require-complete-priors 2>&1 | tee "$OUT/train.log"
"$BST_PYTHON" -u tools/bst/fit_policy.py --checkpoint "$OUT/model.pt" --data-root "$DATA_ROOT" --z0-root "$Z0_ROOT" \
  --out "$OUT/policy.pt" --device cuda:0 --require-complete-priors 2>&1 | tee "$OUT/policy.log"
# All policies are declared ahead of test evaluation; no test-picked winner.
for split in val test; do
  for commit in arrival fixed learned; do
    "$BST_PYTHON" -u tools/bst/evaluate.py --checkpoint "$OUT/model.pt" --policy "$OUT/policy.pt" \
      --data-root "$DATA_ROOT" --z0-root "$Z0_ROOT" --split "$split" --birth band --commit "$commit" \
      --out "$OUT/${split}_band_$commit" --device cuda:0 2>&1 | tee "$OUT/${split}_band_$commit.log"
  done
  "$BST_PYTHON" -u tools/bst/evaluate.py --checkpoint "$OUT/model.pt" --policy "$OUT/policy.pt" \
    --data-root "$DATA_ROOT" --z0-root "$Z0_ROOT" --split "$split" --birth learned --commit learned \
    --out "$OUT/${split}_learned_learned" --device cuda:0 2>&1 | tee "$OUT/${split}_learned_learned.log"
done
"$BST_PYTHON" -u tools/bst/final_metrics.py --checkpoint "$OUT/model.pt" --policy "$OUT/policy.pt" \
  --backbone "$BACKBONE" --data-root "$DATA_ROOT" --out "$OUT/final_metrics" --split test \
  --physical-gpu 1 2>&1 | tee "$OUT/final_metrics.log"
"$BST_PYTHON" -u tools/bst/audit_reconsideration.py --checkpoint "$OUT/model.pt" --priors "$Z0_ROOT/z0_test" \
  --data-root "$DATA_ROOT" --out "$OUT/reconsideration" --max-queries 8 --device cpu --threads 2 \
  2>&1 | tee "$OUT/reconsideration.log"
printf 'seed=%s completed=%s\n' "$SEED" "$(date -Is)" > "$OUT/COMPLETE"
echo "BST FINAL FINISHED seed=$SEED OUT=$OUT"
