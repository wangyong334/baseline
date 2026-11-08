#!/usr/bin/env bash
# Pack the results of tools/deco/queue_round4.sh (BIR without budget at 400 ms seeds 37-39; narrow BIR / LIF at 400 ms
# seed 37; each with V10): backbone evaluations and selection, training logs, V10 summaries / official tables /
# change-risk head results, job logs (no trajectories, no weights). Usage (code dir): bash tools/deco/pack_round4.sh
set -u
cd "$(dirname "$0")/../.."
out=round4_results_$(date +%m%d_%H%M).tgz
{
  for r in bb_v5nb_400_seed37 bb_v5nb_400_seed38 bb_v5nb_400_seed39 bb_v5n4_400_seed37 bb_v5lifn4_400_seed37; do
    [ -d "log/final/$r" ] && find "log/final/$r" -maxdepth 1 \( -name '*.json' -o -name '*.jsonl' -o -name BACKBONE_DONE \) -print0
    find run/final -maxdepth 1 -name "${r}_*.log" -print0 2>/dev/null
  done
  for t in v5nbw400s37 v5nbw400s38 v5nbw400s39 v5n4w400s37 lifn4w400s37; do
    [ -d "run/deco/$t" ] || continue
    find "run/deco/$t" -maxdepth 3 -path '*dnm_a*' \( -name summary.json -o -name 'official_*.json' \
      -o -name 'official_*.md' -o -name head.json -o -name '*.log' \) -print0
  done
  find run/queue -path '*/logs/*' \( -name '*_bir400nb_*' -o -name '*_bir400n4_*' -o -name '*_lif400n4_*' \) -print0 2>/dev/null
  printf 'run/queue/pool.log\0'
} | tar --null -czf "$out" --files-from=-
echo "PACK ROUND4: $out ($(tar -tzf "$out" | wc -l) files, $(du -h "$out" | cut -f1))"
