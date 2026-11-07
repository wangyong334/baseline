#!/usr/bin/env bash
# Queue the V10 decision-layer runs (11-07; DECO_PROBE_LOG 7.10) on the GPU queues of tools/final/gpu_queue.sh.
# QUEUES (default "gpu2 gpu1"): the first queue runs the smoke (2 recordings, 2 epochs) first; the 10 backbones below are
# dealt round-robin over QUEUES in priority order (50 ms seeds first). Every real job waits for the smoke marker and
# exits if the smoke failed. Each job = tools/deco/run_dnm.sh with DESIGN=v10: evidence without a radius cap, V10
# decision neurons, official evaluation, change-risk head (results in run/deco/<TAG>/dnm_a0.01_s0_nr_v10/).
# Usage (code dir, after `conda activate evuav`): [QUEUES="gpu2 gpu1"] bash tools/deco/queue_v10.sh
set -u
cd "$(dirname "$0")/../.."
read -r -a QS <<< "${QUEUES:-gpu2 gpu1}"
JOBS=(
  "v5s37 log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt"
  "v5s38 log/final/bb_v5_50_seed38/selected_seed38.pt"
  "v5s39 log/final/bb_v5_50_seed39/selected_seed39.pt"
  "v5w25s37 log/final/bb_v5_25_seed37/selected_seed37.pt"
  "v5w25s38 log/final/bb_v5_25_seed38/selected_seed38.pt"
  "v5w100s37 log/final/bb_v5_100_seed37/selected_seed37.pt"
  "v5w200s37 log/final/bb_v5_200_seed37/selected_seed37.pt"
  "v5w400s37 log/final/bb_v5_400_seed37/selected_seed37.pt"
  "lifw50s37 log/final/bb_v5lif_50_seed37/selected_seed37.pt"
  "lifw400s37 log/final/bb_v5lif_400_seed37/selected_seed37.pt"
)
for j in "${JOBS[@]}"; do
  set -- $j
  [ -f "$2" ] || { echo "checkpoint missing for $1: $2"; exit 1; }
done
for q in "${QS[@]}"; do mkdir -p run/queue/$q/pending; done
rm -f run/deco/SMOKE_V10_OK run/deco/SMOKE_V10_FAILED
cat > run/queue/${QS[0]}/pending/400_smoke_v10.sh <<'EOF'
rm -rf run/deco/smoke_v10
if CKPT=log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt TAG=smoke_v10 DESIGN=v10 LIMIT=2 EXTRA="--epochs 2" \
   WORKERS=8 bash tools/deco/run_dnm.sh; then
  touch run/deco/SMOKE_V10_OK; echo "SMOKE V10 OK"
else
  touch run/deco/SMOKE_V10_FAILED; echo "SMOKE V10 FAILED"; exit 1
fi
EOF
echo "queued run/queue/${QS[0]}/pending/400_smoke_v10.sh (smoke)"
i=0
for j in "${JOBS[@]}"; do
  set -- $j
  q=${QS[$((i % ${#QS[@]}))]}
  n=$((410 + 10 * (i / ${#QS[@]})))
  cat > run/queue/$q/pending/${n}_v10_$1.sh <<EOF
for k in \$(seq 1 240); do [ -f run/deco/SMOKE_V10_OK ] && break; [ -f run/deco/SMOKE_V10_FAILED ] && break; sleep 30; done
[ -f run/deco/SMOKE_V10_OK ] || { echo "smoke v10 not OK: $1 skipped"; exit 1; }
CKPT=$2 TAG=$1 DESIGN=v10 WORKERS=8 bash tools/deco/run_dnm.sh
EOF
  echo "queued run/queue/$q/pending/${n}_v10_$1.sh"
  i=$((i + 1))
done
echo "QUEUE V10 DONE: smoke on ${QS[0]} + ${#JOBS[@]} jobs over ${QS[*]}"
