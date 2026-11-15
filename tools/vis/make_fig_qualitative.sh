#!/usr/bin/env bash
# Paper figure 4.2 (qualitative comparison, 11-15): four 2 s clips chosen by tools/vis/scene_stats.py (SDNet IoU minus
# the best rival, one scene type per row): (a) dense background noise, (b) static flicker, (c) missed targets,
# (d) several crossing targets. Columns: Input | [RVT] | EV-SpSegNet | PACT | SDNet (ours) | GT.
# Inputs (local, unpacked under Res/vis_data): run/vis/{EV-Seg,PACT} (dump_competitor.py), run/vis/sdnet_v5s37_coopw_s0
# (export_preds.py, paper model); optional RVT=<plug-in prior dir with host_meta.json and z0_test/> adds the RVT column.
# Usage (EV-UAV code dir, local):  bash tools/vis/make_fig_qualitative.sh [RVT=Res/vis_data/run/deco/p20_rvt] [OUT=...]
set -eu
cd "$(dirname "$0")/../.."
DATA_ROOT=${DATA_ROOT:-D:/桌面-D/EV-UAV/dataset/EV-UAV-dataset}
V=${V:-Res/vis_data/run/vis}
OUT=${OUT:-Res/fig_paper/fig_qualitative_v1.pdf}
RVT=${RVT:-}
cols=()
[ -n "$RVT" ] && cols+=(--column "RVT=host:$RVT")
cols+=(--column "EV-SpSegNet=prob:$V/EV-Seg" --column "PACT=prob:$V/PACT"
       --column "SDNet (ours)=final:$V/sdnet_v5s37_coopw_s0")
python tools/vis/fig_qualitative.py --data-root "$DATA_ROOT" --out "$OUT" \
  --scene test_023:5000-7000 --scene test_008:500-2500 --scene test_013:4500-6500 --scene test_010:3500-5500 \
  "${cols[@]}"
