#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
: "${EVUAV_CLIP_EXPERIMENT:?Set a new experiment directory first}"
test -d "$EVUAV_CLIP_EXPERIMENT"
unset EVUAV_RESUME EVUAV_SAMPLE EVUAV_CHECKPOINT EVUAV_RUN_DIR EVUAV_OVERFIT_STEPS
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export EVUAV_VARIANT=clip
export EVUAV_MP_SPLIT=2
trap 'echo "CLIP QUEUE FAILED: inspect unit_tests.log, smoke.log or the active seed training log."' ERR

python - <<'PY'
import copy
import os
from pathlib import Path
import yaml
root = Path(os.environ['EVUAV_CLIP_EXPERIMENT'])
base = yaml.safe_load(Path('log/snn_v0_localrate_T4_seed37_retry1/config.yaml').read_text())
assert base['TRAIN']['epochs'] == 50
assert base['TRAIN']['lr_schedule'] == 'linear'
assert base['TRAIN']['lr'] == .001 and base['TRAIN']['lr_end'] == .0001
assert base['SNN']['snn_stages'] == [3, 4]
assert base['SNN']['snn_threshold'] == 1.
assert base['SNN']['validation_start'] == 40
for seed in (37, 38, 39):
    if (root / ('seed%d' % seed)).exists() or (root / ('seed%d.yaml' % seed)).exists():
        raise SystemExit('Existing run/config; do not reuse experiment directory')
for seed in (37, 38, 39):
    config = copy.deepcopy(base)
    config['SNN']['seed'] = seed
    config['TRAIN']['model_save_root'] = str(root / ('seed%d' % seed))
    config['TEST']['model_path'] = str(root / ('seed%d' % seed) / ('best_iou_seed%d.pt' % seed))
    with (root / ('seed%d.yaml' % seed)).open('x') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
print('CONFIGS_READY: continuous clipped-ReLU; no SNN simulation')
PY

echo 'UNIT TESTS START'
python -c 'import torch; print(torch.__version__)'
python -m unittest discover -s tests -p test_clip_v0.py -v > "$EVUAV_CLIP_EXPERIMENT/unit_tests.log" 2>&1
echo 'UNIT TESTS OK; SMOKE START'
EVUAV_MODE=smoke python -u train_snn_v0.py \
  --config "$EVUAV_CLIP_EXPERIMENT/seed37.yaml" \
  > "$EVUAV_CLIP_EXPERIMENT/smoke.log" 2>&1
echo 'SMOKE OK'

for seed in 37 38 39; do
    echo "TRAIN START seed=$seed"
    date
    EVUAV_MODE=train EVUAV_RUN_DIR="$EVUAV_CLIP_EXPERIMENT/seed$seed" \
      python -u train_snn_v0.py --config "$EVUAV_CLIP_EXPERIMENT/seed$seed.yaml" \
      > "$EVUAV_CLIP_EXPERIMENT/seed${seed}_train.log" 2>&1
    test -s "$EVUAV_CLIP_EXPERIMENT/seed$seed/best_iou_seed$seed.pt"
    echo "TRAIN DONE seed=$seed"
done
echo 'ALL CLIP TRAINING DONE; TEST SET NOT EVALUATED'
