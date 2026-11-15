#!/usr/bin/env bash
# Qualitative figure (11-15): per-event test predictions of EV-SpSegNet and PACT from their official code. For each
# repository: train with its own train.py and our config copy (only paths changed) unless best_iou_seed37.pt exists,
# then dump the predictions with tools/vis/dump_competitor.py. A failed step is reported and the next one runs.
# Usage (V20 code dir, after `conda activate evuav`; the repositories unpacked under $CMP):
#   GPU=1 bash tools/vis/run_competitors.sh > run/vis/competitors.log 2>&1
# Output: run/vis/<repo>/<name>.npz (prob) and run/vis/<repo>_{train,dump}.log; last line "COMPETITORS FINISHED".
set -u
cd "$(dirname "$0")/../.."
HERE=$(pwd)
CMP=${CMP:-/media/stephen/nvme0n1/wy_data/competitors}
GPU=${GPU:?GPU=<physical id>}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p run/vis
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "competitors on GPU $GPU: start"
for spec in EV-Seg:evspsegnet:evspsegnet_evuav.yaml PACT:PACT:pact_evuav.yaml; do
  repo=${spec%%:*}; rest=${spec#*:}; model=${rest%%:*}; cfg=$HERE/tools/vis/competitors/${rest#*:}
  save=$(grep -m1 '^  model_save_root:' "$cfg" | awk '{print $2}')
  mkdir -p "$save"
  if [ ! -f "$save/best_iou_seed37.pt" ]; then
    say "$repo: training (official train.py, 50 epochs)"
    (cd "$CMP/$repo" && python -u train.py --config "$cfg") > run/vis/${repo}_train.log 2>&1 \
      || { say "$repo training FAILED: $(tail -n 2 run/vis/${repo}_train.log | tr '\n' ' ')"; continue; }
  fi
  python -u tools/vis/dump_competitor.py --repo "$CMP/$repo" --model $model --config "$cfg" \
    --ckpt "$save/best_iou_seed37.pt" --out run/vis/$repo > run/vis/${repo}_dump.log 2>&1 \
    || { say "$repo dump FAILED: $(tail -n 2 run/vis/${repo}_dump.log | tr '\n' ' ')"; continue; }
  say "$repo: $(tail -n 1 run/vis/${repo}_dump.log)"
done
say "COMPETITORS FINISHED"
