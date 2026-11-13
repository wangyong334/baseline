#!/usr/bin/env bash
# Pack the V17 results for download (11-13): the runs under run/v17 (or only RUNS) without the trajectories
# (traj_*.npz, ~150 MB per run; add TRAJ=1 to include them), plus the V17 code that produced them.
# Usage (V17 code dir): [TRAJ=1] [RUNS="run/v17/v5s37/r2_boxes_s0 ..."] bash tools/v17/pack_v17.sh
set -u
cd "$(dirname "$0")/../.."
stamp=$(date +%Y%m%d_%H%M%S)
archive="v17_results_${stamp}.tar.gz"
ex=(--exclude='__pycache__')
[ "${TRAJ:-0}" = 1 ] || ex+=(--exclude='traj_*.npz')
read -r -a runs <<< "${RUNS:-run/v17}"
items=("${runs[@]}" run/v17/*.log model/sdu.py tools/v17 tests/test_sdu_v17.py docs/V17_METHOD.md)
existing=()
for p in "${items[@]}"; do [ -e "$p" ] && existing+=("$p"); done
tar "${ex[@]}" -czf "$archive" "${existing[@]}"
ls -lh "$archive"
tar -tzf "$archive" | grep -E "(run|eval)\.(log|json)$|summary.json" | sed -n '1,80p'
echo "PACK V17 DONE $archive"
