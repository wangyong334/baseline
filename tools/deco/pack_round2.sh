#!/usr/bin/env bash
# Pack the results of tools/deco/queue_round2.sh (EV-Flying, alpha sweep, LIF 50 ms seeds 38 / 39 + V10) for local
# analysis: summaries, official tables, change-risk head results, backbone evaluations, logs (no trajectories, no
# weights). Usage (code dir): bash tools/deco/pack_round2.sh
set -u
cd "$(dirname "$0")/../.."
stamp=$(date +%m%d_%H%M)
out=round2_results_$stamp.tgz
{
  for t in v5evfs37 v5s37 v5s38 v5s39 lifw50s38 lifw50s39; do
    [ -d "run/deco/$t" ] || continue
    find "run/deco/$t" -maxdepth 3 -path '*dnm_a*' \( -name summary.json -o -name 'official_*.json' \
      -o -name 'official_*.md' -o -name head.json -o -name '*.log' \) -print0
  done
  find run/deco/v5evfs37 -maxdepth 1 \( -name 'ev_*.log' -o -name 'dump_*.log' \) -print0 2>/dev/null
  find run/deco/smoke_r2 -maxdepth 3 -name '*.log' -print0 2>/dev/null
  for s in 38 39; do
    d=log/final/bb_v5lif_50_seed$s
    [ -d "$d" ] && find "$d" -maxdepth 1 \( -name '*.json' -o -name '*.jsonl' -o -name BACKBONE_DONE \) -print0
    find run/final -maxdepth 1 -name "bb_v5lif_50_seed${s}_*.log" -print0 2>/dev/null
  done
  find run/queue -path '*/logs/5[0-9][0-9]_*.log' -print0 2>/dev/null
} | tar --null -czf "$out" --files-from=-
echo "PACK ROUND2: $out ($(tar -tzf "$out" | wc -l) files, $(du -h "$out" | cut -f1))"
