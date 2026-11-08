#!/usr/bin/env bash
# Drop one job into the shared GPU pool (tools/final/gpu_pool.sh, 11-08).
# Usage (code dir, after `conda activate evuav`):
#   bash tools/final/pool_add.sh <name> '<shell command>' [gpu id]
#   e.g. bash tools/final/pool_add.sh bir50_s40 'NAME=bb_v5_50 SEED=40 HZ="" bash tools/final/backbone.sh'
# The job file is run/queue/pool/pending/<NNN>_<name>[@gpu<id>].sh; NNN = largest number used so far + 10, so jobs
# start in the order they were added. With a gpu id the job runs only on that GPU. Job logs: run/queue/gpu<id>/logs/.
set -u
cd "$(dirname "$0")/../.."
[ $# -ge 2 ] || { echo "usage: bash tools/final/pool_add.sh <name> '<command>' [gpu id]"; exit 1; }
name=$1
cmd=$2
gpu=${3:-}
[[ $name =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "name may contain letters, digits, _ . - only"; exit 1; }
[ -z "$gpu" ] || [[ $gpu =~ ^[0-9]+$ ]] || { echo "gpu id must be a number"; exit 1; }
P=run/queue/pool
mkdir -p "$P/pending" "$P/given"
last=$(ls "$P/pending" "$P/given" run/queue/gpu*/pending run/queue/gpu*/running run/queue/gpu*/done \
       run/queue/gpu*/failed 2>/dev/null | sed -n 's/^\([0-9][0-9]*\)_.*/\1/p' | sort -n | tail -n 1)
n=$(( ${last:-690} + 10 ))
file="$P/pending/${n}_${name}${gpu:+@gpu$gpu}.sh"
printf '%s\n' "$cmd" > "$file"
echo "added $file"
if pgrep -f "tools/final/gpu_pool.sh" > /dev/null; then
  echo "dispatcher is running: the job starts on the next free GPU"
else
  echo "NOTE: no dispatcher is running. Start it with:"
  echo '  rm -f run/queue/pool/STOP; GPUS="1 2 3" HOLD_GPUS=none KEEP=1 nohup bash tools/final/gpu_pool.sh >> run/queue/pool.log 2>&1 < /dev/null &'
fi
