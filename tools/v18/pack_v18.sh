#!/usr/bin/env bash
# Pack the V18 results for download (11-13): the runs under run/v18 (or only RUNS) without the trajectories
# (traj_*.npz, ~150 MB per run; add TRAJ=1 to include them), plus the V18 code that produced them.
# Usage (V18 code dir): [TRAJ=1] [RUNS="run/v18/v5s37/r2_boxes_s0 ..."] bash tools/v18/pack_v18.sh
set -u
cd "$(dirname "$0")/../.."
stamp=$(date +%Y%m%d_%H%M%S)
archive="v18_results_${stamp}.tar.gz"
ex=(--exclude='__pycache__')
[ "${TRAJ:-0}" = 1 ] || ex+=(--exclude='traj_*.npz')
read -r -a runs <<< "${RUNS:-run/v18}"
items=("${runs[@]}" run/v18/*.log model/sdu.py tools/v18 tests/test_sdu_v18.py docs/V18_METHOD.md)
existing=()
for p in "${items[@]}"; do [ -e "$p" ] && existing+=("$p"); done
tar "${ex[@]}" -czf "$archive" "${existing[@]}"
ls -lh "$archive"
tar -tzf "$archive" | grep -E "(run|eval)\.(log|json)$|summary.json" | sed -n '1,80p'
echo "PACK V18 DONE $archive"
