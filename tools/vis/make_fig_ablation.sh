#!/usr/bin/env bash
# Paper figure 4.3.1 (ablation visualisation, 11-16): the four rows of Table 2 on 2 s test clips, backbone seed 37, two
# populations, decision seed 0 (export_preds.py of p20_bir_reset1 and p20_bir_carry). Clips were ranked by the smallest
# of the four IoU gaps per-slice < (+ SDM, streaming) < SDNet, among clips not used in the qualitative comparison:
# (a) dense background noise (false alarms fall), (b), (c) targets the per-slice SNN misses (misses fall).
# Columns: Input | Per-slice SNN | Per-slice SNN + SDM | Streaming SNN | SDNet (ours) | GT.
# Usage (EV-UAV code dir, local):  bash tools/vis/make_fig_ablation.sh [OUT=...]
set -eu
cd "$(dirname "$0")/../.."
DATA_ROOT=${DATA_ROOT:-D:/桌面-D/EV-UAV/dataset/EV-UAV-dataset}
V=${V:-Res/vis_data/run/vis}
OUT=${OUT:-Res/fig_paper/fig_ablation_v1.pdf}
python tools/vis/fig_qualitative.py --data-root "$DATA_ROOT" --out "$OUT" \
  --scene test_022:2500-4500 --scene test_012:5500-7500 --scene test_014:2500-4500 \
  --column "Per-slice SNN=prior:$V/abl_reset_coop_s37" --column "Per-slice SNN + SDM=final:$V/abl_reset_coop_s37" \
  --column "Streaming SNN=prior:$V/abl_carry_coop_s37" --column "SDNet (ours)=final:$V/abl_carry_coop_s37"
