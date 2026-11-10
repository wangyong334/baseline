#!/usr/bin/env bash
# Pack the V13 results (json / md / log only, no trajectories) for download -> v13_results_<MMDD_HHMM>.tgz in the code dir.
# Usage (code dir, after `conda activate evuav`): [TAGS="v5s37 v5s38 v5s39"] bash tools/deco/pack_v13.sh
set -u
cd "$(dirname "$0")/../.."
TAGS=${TAGS:-"v5s37 v5s38 v5s39"}
out=v13_results_$(date +%m%d_%H%M).tgz
list=$(mktemp)
for t in $TAGS; do
  d=run/deco/$t/dnm_m0.001_s0_nr_v13
  [ -d "$d" ] && find "$d" -maxdepth 2 \( -name '*.json' -o -name '*.md' -o -name '*.log' \) >> "$list"
  ls run/deco/$t/dnm_m0.001_s0_nr_v13.log run/deco/$t/ev_m0.001_nr_*.log 2>/dev/null >> "$list"
done
ls run/queue/gpu*/logs/*v13_* 2>/dev/null >> "$list"
tar czf "$out" -T "$list" && echo "packed $out ($(du -h "$out" | cut -f1)), $(wc -l < "$list") files"
rm -f "$list"
