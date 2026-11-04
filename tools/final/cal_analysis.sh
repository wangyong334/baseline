#!/usr/bin/env bash
# V9 optimisation (alpha with its calibrated meaning): for each head run, waits for the temperature fitted on val by
# commit_cal.sh (run/final/analysis/commit_cal_<run>_seed<S>.json), then reruns tools/final_analysis.py on the val and
# test trajectories with --temperature, writing the usual analysis files run/final/analysis/<run>_seed<S>_{val,test}.json.
# The rerun is a superset: every uncalibrated policy is bit-identical, plus adaptive_cal_* / slice_cal_* / simple_cal_*
# and the wait analysis at calibrated alpha 0.05 (tools/paper_tables.py and final_summary.py prefer them when present).
# CPU only. Usage: nohup bash tools/final/cal_analysis.sh > run/final/cal_analysis.log 2>&1 &
#        RUNS="<run>:<head config name> ..."  SEED=37
set -u
cd "$(dirname "$0")/../.."
SEED=${SEED:-37}
RUNS=${RUNS:-"rcl_feature_50_on_v5:rcl_feature_50 rcl_feature_50_on_e1:rcl_feature_50 rcl_feature_50_on_e3b:rcl_feature_50 rcl_feature_evf_on_v5evf:rcl_feature_evf rcl_feature_25_on_v5w25:rcl_feature_25"}
ANA=run/final/analysis
for spec in $RUNS; do
  run=${spec%%:*}; name=${spec##*:}
  CAL=$ANA/commit_cal_${run}_seed${SEED}.json
  TRAJ=run/final/traj/${run}_seed${SEED}
  OUT=log/final/${run}_seed${SEED}
  echo "[$(date '+%m-%d %H:%M:%S')] 等 $CAL"
  until [ -f $CAL ]; do sleep 300; done
  read T ROOT WMS HEIGHT WIDTH < <(python -c "
import json, sys, torch; sys.path.insert(0, '.')
from utils.stream_common import load_flat_config
t = json.load(open(sys.argv[1], encoding='utf-8'))['fit']['temperature']
c = load_flat_config(sys.argv[2])
b = torch.load(json.load(open(sys.argv[3], encoding='utf-8'))['base_checkpoint'], map_location='cpu')['config']
print(t, c['root'], b['window_ms'], b['height'], b['width_px'])" $CAL configs/final/$name.yaml $OUT/eval_val_selected_seed${SEED}.json) \
    || { echo "  $run：读不出温度或配置（是否已 conda activate evuav？）"; continue; }
  for split in val test; do
    python -u tools/final_analysis.py --traj-dir ${TRAJ}_$split --split $split --data-root $ROOT --window-ms $WMS \
      --height $HEIGHT --width $WIDTH --temperature $T --out $ANA/${run}_seed${SEED}_$split.json \
      --title "${run} seed ${SEED} $split（校准 T = $T）" > run/final/cal_analysis_${run}_seed${SEED}_$split.log 2>&1 \
      || { echo "  $run $split：FAILED（run/final/cal_analysis_${run}_seed${SEED}_$split.log）"; continue; }
    echo "  $run $split（T = $T）：$(grep '^| adaptive_cal_0.05 ' run/final/cal_analysis_${run}_seed${SEED}_$split.log | cut -c1-110)"
  done
done
echo "[$(date '+%m-%d %H:%M:%S')] CAL ANALYSIS FINISHED"
