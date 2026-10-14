#!/usr/bin/env bash
# Run from project root. One worker per GPU; safe to resume completed outputs.
set -euo pipefail
mkdir -p run
variants=("${@}")
if [ ${#variants[@]} -eq 0 ]; then variants=(enc4 enc34 enc234); fi
for variant in "${variants[@]}"; do
  dir="log/stream_dual_${variant}_seed37"
  checkpoint="$dir/best_val_iou_seed37.pt"
  if [ ! -f "$dir/eval_test_best_val_iou_seed37_memory_off.json" ]; then
    python -u train_stream_v1.py --config "configs/evisseg_stream_dual_${variant}.yaml" \
      --mode eval --device cuda:0 --split test --checkpoint "$checkpoint" --memory-ablation off
  fi
  for intervention in zero delay spatial_mean; do
    output="$dir/diagnostic_test_${intervention}_delay4.json"
    if [ ! -f "$output" ]; then
      python -u tools/diagnose_stream_memory.py --checkpoint "$checkpoint" \
        --root /media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset \
        --split test --device cuda:0 --intervention "$intervention" --delay-windows 4 --output "$output"
    fi
  done
done
