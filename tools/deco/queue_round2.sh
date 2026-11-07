#!/usr/bin/env bash
# Queue round 2 (11-07 evening) on the GPU queues of tools/final/gpu_queue.sh; every GPU runs its jobs one after
# another, in this priority order (user, 11-07 20:05):
#   1. decision layer on EV-Flying (V5 backbone evflying_x4 seed 37): V10, then V9 (hand radius 32). The training
#      evidence is ~30 GiB of tokens, so train_dnm.py keeps it memory-mapped in host memory (--gpu-evidence-gb);
#   2. alpha sweep of the V10 decision layer (one risk level alpha for band, crossing commit and change-risk head;
#      alpha in SWEEP, default 0.05 0.02 0.003 0.001; 0.01 exists already), 50 ms backbones seeds 37 / 38 / 39;
#   3. BIR vs LIF: plain LIF backbones at 50 ms (configs/final/bb_v5lif_50.yaml, no communication budget, no noise
#      evaluation) seeds 38 / 39, each followed by the V10 decision layer (TAG lifw50s38 / lifw50s39).
# Gates: new code uploaded (tied alpha in run_dnm.sh, host evidence in train_dnm.py), unit tests (10), checkpoints, the
# EV-Flying data root, the LIF config, LIF output dirs not present yet. The first queue runs a smoke first (2 recordings,
# 2 epochs, alpha 0.05, host evidence forced); every real job waits for the smoke marker and exits if it failed. Jobs
# are dealt round-robin over QUEUES in priority order (one queue = all in order). A runner (tools/final/gpu_queue.sh)
# is started for every queue that has none; it keeps HOLD_GB GB on the GPU once its queue is empty (default 20;
# HOLD_GB= for none).
# Usage (code dir, after `conda activate evuav`): QUEUES="gpu3" bash tools/deco/queue_round2.sh
#   DRY=1 QROOT=<dir>: only write the job files under <dir> (no gates, no runners) -- for checking the script.
# Results: run/deco/v5evfs37/dnm_a0.01_s0{_nr_v10,}/, run/deco/v5s3{7,8,9}/dnm_a<alpha>_s0_nr_v10/,
# log/final/bb_v5lif_50_seed3{8,9}/, run/deco/lifw50s3{8,9}/dnm_a0.01_s0_nr_v10/; pack: bash tools/deco/pack_round2.sh
set -u
cd "$(dirname "$0")/../.."
read -r -a QS <<< "${QUEUES:-gpu3}"
QROOT=${QROOT:-run/queue}
DRY=${DRY:-0}
HOLD_GB=${HOLD_GB-20}
SWEEP=${SWEEP:-"0.05 0.02 0.003 0.001"}
EVF_CKPT=log/stream_tif_commpeak_msg_evflying_x4_seed37/best_val_iou_seed37.pt
EVF_ROOT=${EVF_ROOT:-/media/stephen/nvme0n1/wy_data/datasets/evflying_x4}
S37=log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt
S38=log/final/bb_v5_50_seed38/selected_seed38.pt
S39=log/final/bb_v5_50_seed39/selected_seed39.pt
die() { echo "QUEUE ROUND2 NOT STARTED: $*"; exit 1; }

for q in "${QS[@]}"; do [[ $q =~ ^gpu[0-9]+$ ]] || die "queue names are gpu<id>, got '$q'"; done
grep -q 'commit-alpha \$ca' tools/deco/run_dnm.sh || die "tools/deco/run_dnm.sh is the old version (upload the new one)"
grep -q 'gpu-evidence-gb' tools/deco/train_dnm.py || die "tools/deco/train_dnm.py is the old version (upload the new one)"
if [ "$DRY" != 1 ]; then
  tests=$(python -m unittest tests.test_deco_v10 2>&1 | tail -n 3)
  { echo "$tests" | grep -q "Ran 10 tests" && echo "$tests" | grep -q "^OK"; } || die "unit tests: $tests"
  echo "unit tests: 10 OK"
  for c in "$S37" "$S38" "$S39" "$EVF_CKPT" configs/final/bb_v5lif_50.yaml; do [ -f "$c" ] || die "missing: $c"; done
  [ -d "$EVF_ROOT" ] || die "EV-Flying data root missing: $EVF_ROOT"
  for s in 38 39; do
    [ ! -e "log/final/bb_v5lif_50_seed$s" ] || die "log/final/bb_v5lif_50_seed$s exists already (check it, move it away)"
  done
fi

lif() {                                                      # LIF backbone at 50 ms, then the V10 decision layer
  echo "NAME=bb_v5lif_50 SEED=$1 HZ=\"\" bash tools/final/backbone.sh && CKPT=log/final/bb_v5lif_50_seed$1/selected_seed$1.pt TAG=lifw50s$1 DESIGN=v10 WORKERS=8 bash tools/deco/run_dnm.sh"
}
JOBS=(                                                       # priority order
  "evf_v10|CKPT=$EVF_CKPT TAG=v5evfs37 DATA_ROOT=$EVF_ROOT DESIGN=v10 WORKERS=8 bash tools/deco/run_dnm.sh"
  "evf_v9|CKPT=$EVF_CKPT TAG=v5evfs37 DATA_ROOT=$EVF_ROOT WORKERS=8 bash tools/deco/run_dnm.sh"
  "alpha_v5s37|CKPT=$S37 TAG=v5s37 DESIGN=v10 ALPHAS=\"$SWEEP\" WORKERS=8 bash tools/deco/run_dnm.sh"
  "alpha_v5s38|CKPT=$S38 TAG=v5s38 DESIGN=v10 ALPHAS=\"$SWEEP\" WORKERS=8 bash tools/deco/run_dnm.sh"
  "alpha_v5s39|CKPT=$S39 TAG=v5s39 DESIGN=v10 ALPHAS=\"$SWEEP\" WORKERS=8 bash tools/deco/run_dnm.sh"
  "lif50_s38|$(lif 38)"
  "lif50_s39|$(lif 39)"
)
for q in "${QS[@]}"; do mkdir -p "$QROOT/$q/pending"; done
[ "$DRY" = 1 ] || rm -f run/deco/SMOKE_R2_OK run/deco/SMOKE_R2_FAILED
cat > "$QROOT/${QS[0]}/pending/500_smoke_r2.sh" <<EOF
rm -rf run/deco/smoke_r2
if CKPT=$S37 TAG=smoke_r2 DESIGN=v10 LIMIT=2 ALPHAS="0.05" EXTRA="--epochs 2 --gpu-evidence-gb 0" WORKERS=8 bash tools/deco/run_dnm.sh; then
  touch run/deco/SMOKE_R2_OK; echo "SMOKE R2 OK"
else
  touch run/deco/SMOKE_R2_FAILED; echo "SMOKE R2 FAILED"; exit 1
fi
EOF
echo "queued $QROOT/${QS[0]}/pending/500_smoke_r2.sh (smoke)"
i=0
for j in "${JOBS[@]}"; do
  name=${j%%|*}
  cmd=${j#*|}
  q=${QS[$((i % ${#QS[@]}))]}
  n=$((510 + 10 * (i / ${#QS[@]})))
  cat > "$QROOT/$q/pending/${n}_$name.sh" <<EOF
for k in \$(seq 1 240); do [ -f run/deco/SMOKE_R2_OK ] && break; [ -f run/deco/SMOKE_R2_FAILED ] && break; sleep 30; done
[ -f run/deco/SMOKE_R2_OK ] || { echo "smoke r2 not OK: $name skipped"; exit 1; }
$cmd
EOF
  echo "queued $QROOT/$q/pending/${n}_$name.sh"
  i=$((i + 1))
done

runner_alive() {                                             # a gpu_queue.sh runner serving queue $1 (gpu<id>)?
  local p env
  for p in $(pgrep -f "tools/final/gpu_queue.sh" 2>/dev/null); do
    env=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null) || continue
    if echo "$env" | grep -qx "QNAME=$1"; then return 0; fi
    if ! echo "$env" | grep -q "^QNAME=" && echo "$env" | grep -qx "GPU=${1#gpu}"; then return 0; fi
  done
  return 1
}
if [ "$DRY" != 1 ]; then
  for q in "${QS[@]}"; do
    if runner_alive "$q"; then
      echo "runner for $q is alive: it takes the new jobs within 30 s (its own HOLD_GB applies)"
    else
      rm -f "run/queue/$q/STOP"
      GPU=${q#gpu} HOLD_GB=$HOLD_GB nohup bash tools/final/gpu_queue.sh >> "run/queue/$q.log" 2>&1 < /dev/null &
      echo "started runner for $q (pid $!, hold ${HOLD_GB:-none} GB when empty, log run/queue/$q.log)"
    fi
  done
fi
for q in "${QS[@]}"; do echo "$q pending: $(ls "$QROOT/$q/pending" | tr '\n' ' ')"; done
echo "QUEUE ROUND2 DONE: smoke on ${QS[0]} + ${#JOBS[@]} jobs over ${QS[*]}"
