#!/usr/bin/env bash
# Pack the V13 results for download -> v13_results_<MMDD_HHMM>.tgz in the code dir: json / md / log of every SDU seed
# and VoW stage (vow, vow_best, ...) plus the small per-event test arrays (vow_test_events.npz) for the figures; no
# trajectories, no evidence.
# Usage (code dir, after `conda activate evuav`): [TAGS="v5s37 v5s38 v5s39"] bash tools/deco/pack_v13.sh
set -u
cd "$(dirname "$0")/../.."
TAGS=${TAGS:-"v5s37 v5s38 v5s39"}
out=v13_results_$(date +%m%d_%H%M).tgz
list=$(mktemp)
for t in $TAGS; do
  for d in run/deco/$t/dnm_m0.001_s*_nr_v13; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 2 \( -name '*.json' -o -name '*.md' -o -name '*.log' -o -name 'vow_test_events.npz' \) >> "$list"
    ls "$d.log" 2>/dev/null >> "$list"
  done
  ls run/deco/$t/ev_m0.001_nr_*.log 2>/dev/null >> "$list"
done
ls run/queue/gpu*/logs/*v13* 2>/dev/null >> "$list"
tar czf "$out" -T "$list" && echo "packed $out ($(du -h "$out" | cut -f1)), $(wc -l < "$list") files"
rm -f "$list"
