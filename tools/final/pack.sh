#!/usr/bin/env bash
# Pack V9-final results for download: log/final (no *.pt), run/final/*.log, run/final/*.json (backbone choice), run/final/analysis
# (+ run/final/traj when TRAJ=1: per-lag trajectories, ~100 MB per EV-UAV split and head).
# Usage: bash tools/final/pack.sh            -> final_results_<stamp>.tgz in the code dir (.tgz: sftp skips *.tar.gz)
set -u
cd "$(dirname "$0")/../.."
stamp=$(date +%Y%m%d_%H%M%S); archive="final_results_${stamp}.tgz"; shopt -s nullglob
items=( log/final run/final/*.log run/final/*.json run/final/analysis )
[ "${TRAJ:-0}" = "1" ] && items+=( run/final/traj )
existing=(); for path in "${items[@]}"; do [ -e "$path" ] && existing+=("$path"); done
tar --exclude='*.pt' --exclude='__pycache__' -czf "$archive" "${existing[@]}"
ls -lh "$archive"; echo "文件数 $(tar -tzf "$archive" | wc -l)"
echo "PACK FINISHED: $archive"
