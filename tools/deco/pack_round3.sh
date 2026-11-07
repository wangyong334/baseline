#!/usr/bin/env bash
# Pack the results of tools/deco/queue_round3.sh (BIR / LIF backbones at 25 and 100 ms + V10 decision layers):
# backbone evaluations and selection, training logs, V10 summaries / official tables / change-risk head results
# (no trajectories, no weights). Usage (code dir): bash tools/deco/pack_round3.sh
set -u
cd "$(dirname "$0")/../.."
stamp=$(date +%m%d_%H%M)
out=round3_results_$stamp.tgz
{
  for r in bb_v5lif_25_seed38 bb_v5_25_seed39 bb_v5lif_25_seed39 bb_v5_100_seed38 bb_v5lif_100_seed38 \
           bb_v5_100_seed39 bb_v5lif_100_seed39; do
    [ -d "log/final/$r" ] && find "log/final/$r" -maxdepth 1 \( -name '*.json' -o -name '*.jsonl' -o -name BACKBONE_DONE \) -print0
    find run/final -maxdepth 1 -name "${r}_*.log" -print0 2>/dev/null
  done
  for t in lifw25s38 v5w25s39 lifw25s39 v5w100s38 lifw100s38 v5w100s39 lifw100s39; do
    [ -d "run/deco/$t" ] || continue
    find "run/deco/$t" -maxdepth 3 -path '*dnm_a*' \( -name summary.json -o -name 'official_*.json' \
      -o -name 'official_*.md' -o -name head.json -o -name '*.log' \) -print0
  done
  find run/queue -path '*/logs/6[0-9][0-9]_*.log' -print0 2>/dev/null
} | tar --null -czf "$out" --files-from=-
echo "PACK ROUND3: $out ($(tar -tzf "$out" | wc -l) files, $(du -h "$out" | cut -f1))"
