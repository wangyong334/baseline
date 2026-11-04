#!/usr/bin/env bash
# V9 optimisation 1: calibrated commit interface (tools/commit_calibration.py; CPU only, evaluation only).
# For each head run (in order), waits for its HEAD_DONE, fits the threshold-anchored temperature on the val trajectories
# and reports calibrated / uncalibrated adaptive commits -> run/final/analysis/commit_cal_<run>_seed<S>.{json,md}.
# Data root, window length and sensor size come from the head config and its backbone checkpoint, as in head.sh.
# Usage: nohup bash tools/final/commit_cal.sh > run/final/commit_cal.log 2>&1 &
#        RUNS="<run>:<head config name> ..."  SEED=37
set -u
cd "$(dirname "$0")/../.."
SEED=${SEED:-37}
RUNS=${RUNS:-"rcl_feature_50_on_v5:rcl_feature_50 rcl_feature_50_on_e1:rcl_feature_50 rcl_feature_50_on_e3b:rcl_feature_50 rcl_feature_evf_on_v5evf:rcl_feature_evf rcl_feature_25_on_v5w25:rcl_feature_25"}
mkdir -p run/final/analysis
for spec in $RUNS; do
  run=${spec%%:*}; name=${spec##*:}
  OUT=log/final/${run}_seed${SEED}
  TRAJ=run/final/traj/${run}_seed${SEED}
  echo "[$(date '+%m-%d %H:%M:%S')] 等 $OUT/HEAD_DONE"
  until [ -f $OUT/HEAD_DONE ]; do sleep 300; done
  read ROOT WMS HEIGHT WIDTH < <(python -c "
import json, sys, torch; sys.path.insert(0, '.')
from utils.stream_common import load_flat_config
c = load_flat_config(sys.argv[1])
base = json.load(open(sys.argv[2], encoding='utf-8'))['base_checkpoint']
b = torch.load(base, map_location='cpu')['config']
print(c['root'], b['window_ms'], b['height'], b['width_px'])" configs/final/$name.yaml $OUT/eval_val_selected_seed${SEED}.json) \
    || { echo "读不出 $run 的配置（是否已 conda activate evuav？）"; continue; }
  python -u tools/commit_calibration.py --val-traj ${TRAJ}_val --test-traj ${TRAJ}_test --data-root $ROOT \
    --window-ms $WMS --height $HEIGHT --width $WIDTH --out run/final/analysis/commit_cal_${run}_seed${SEED}.json \
    > run/final/commit_cal_${run}_seed${SEED}.log 2>&1 || { echo "  $run：FAILED（run/final/commit_cal_${run}_seed${SEED}.log）"; continue; }
  echo "  $run：$(grep '温度 T' run/final/commit_cal_${run}_seed${SEED}.log | head -n 1)"
done
echo "[$(date '+%m-%d %H:%M:%S')] COMMIT CAL FINISHED"
