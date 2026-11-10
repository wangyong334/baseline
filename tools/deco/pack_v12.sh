#!/usr/bin/env bash
# Pack the results of tools/deco/queue_v12.sh (+ the EV-Flying run of run_v12_evf.sh): summaries (per-epoch history),
# runtime, range / gated range / official tables, logs,
# the final weights deco_v12.pt (small), job logs; no trajectories, no exports, no last.pt.
# Usage (code dir): bash tools/deco/pack_v12.sh
set -u
cd "$(dirname "$0")/../.."
out=v12_results_$(date +%m%d_%H%M).tgz
{
  find run/deco/v12 -maxdepth 1 \( -name '*.log' -o -name 'SMOKE_*' \) -print0 2>/dev/null
  for d in run/deco/v12/*_seed*/; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 1 \( -name summary.json -o -name runtime.json -o -name range.log -o -name deco_v12.pt \
      -o -name 'calib*' \) -print0
    [ -d "$d/run/range" ] && find "$d/run/range" -maxdepth 1 \( -name '*.json' -o -name '*.md' -o -name '*.log' \) -print0
    [ -d "$d/gated/run/range" ] && find "$d/gated/run/range" -maxdepth 1 \( -name '*.json' -o -name '*.md' \) -print0
    [ -d "$d/official" ] && find "$d/official" -maxdepth 1 \( -name '*.json' -o -name '*.md' \) -print0
    find "$d" -maxdepth 1 \( -name 'official.log' -o -name 'range_gated.log' \) -print0
  done
  find run/queue -path '*/logs/*' -name '*_v12_*' -print0 2>/dev/null
  printf 'run/queue/pool.log\0'
} | tar --null -czf "$out" --files-from=-
echo "PACK V12: $out ($(tar -tzf "$out" | wc -l) files, $(du -h "$out" | cut -f1))"
