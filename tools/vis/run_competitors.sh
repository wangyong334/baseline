#!/usr/bin/env bash
# Competitors (11-15): EV-SpSegNet, then PACT, from their official code on one GPU. For each repository: train with
# its own train.py and our config copy (only paths changed), then dump per-event probabilities of test (metrics with
# the baseline evaluator), val and train with tools/vis/dump_competitor.py (val / train: for a later SDM plug-in).
# GPU memory (the 8 s voxel input ran out of 24 GB before): the card must be free. If training still runs out of
# memory it restarts with a lower cap on the events of a training recording -- the repositories' own random
# subsampling (max_events_num: EV-SpSegNet 700k official -> 450k -> 300k); PACT trains on 100k events already, its
# fallback validates one recording at a time (eval mode: same predictions). Test / val / train predictions always use
# every event; dump_competitor.py splits a recording in time only if it does not fit whole (counted in its log).
# PACT imports HAIS_OP from lib/hais_ops (meant to be built in place); its voxelize sources equal those of the
# installed HAIS_OP (PACT only dropped a THC include), so without an in-place build a one-line shim imports that one.
# Usage (V20 code dir, after `conda activate evuav`):
#   GPU=1 nohup bash tools/vis/run_competitors.sh > run/vis/competitors.log 2>&1 < /dev/null &
# Output: run/vis/<repo>/ (test), run/vis/<repo>_val/, run/vis/<repo>_train/ (<name>.npz, key prob); logs
# run/vis/<repo>_train.log, run/vis/<repo>_dump_<split>.log; last line "COMPETITORS FINISHED".
set -u
cd "$(dirname "$0")/../.."
HERE=$(pwd)
CMP=${CMP:-/media/stephen/nvme0n1/wy_data/competitors}
GPU=${GPU:?GPU=<physical id>}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p run/vis
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [ ! -f "$CMP/EV-Seg/train.py" ] || [ ! -f "$CMP/PACT/train.py" ]; then
  mkdir -p "$CMP" && tar -xzf tools/vis/competitors/competitor_code_20261115.tgz -C "$CMP" \
    || { say "cannot unpack the official code into $CMP"; exit 1; }
  say "official code unpacked into $CMP"
fi
if ! (cd "$CMP/PACT" && python -c "from lib.hais_ops import HAIS_OP" > /dev/null 2>&1); then
  printf '%s\n' "# Shim (11-15): the HAIS ops installed in this environment (same voxelize sources; PACT dropped a THC include)." \
    "from HAIS_OP import *  # noqa: F401,F403" > "$CMP/PACT/lib/hais_ops/HAIS_OP.py"
  (cd "$CMP/PACT" && python -c "from lib.hais_ops import HAIS_OP; HAIS_OP.voxelize_idx") \
    || { say "PACT: HAIS_OP not importable"; exit 1; }
  say "PACT: lib/hais_ops/HAIS_OP.py shim -> installed HAIS_OP"
fi
if [ ! -f "$CMP/PACT/train_valbs1.py" ]; then               # fallback variant: validation one recording at a time
  python - "$CMP/PACT/train.py" "$CMP/PACT/train_valbs1.py" <<'EOF' || { say "PACT: cannot write train_valbs1.py"; exit 1; }
import sys
src = open(sys.argv[1], encoding="utf-8").read()
old = "        val_dataset,\n        batch_size=cfg.batch_size,"
assert src.count(old) == 1, "PACT val_loader block not found"
open(sys.argv[2], "w", encoding="utf-8").write(src.replace(old, "        val_dataset,\n        batch_size=1,"))
EOF
fi

train_repo() {                      # repo, config, attempts ("<cap>" or "<cap>/v1" = PACT validation batch 1)
  local repo=$1 cfg=$2 save attempt cap script c rc log=run/vis/${1}_train.log
  shift 2
  save=$(grep -m1 '^  model_save_root:' "$cfg" | awk '{print $2}')
  if [ -f "$save/TRAIN_DONE" ]; then say "$repo: trained before ($(cat "$save/TRAIN_DONE"))"; return 0; fi
  for attempt in "$@"; do
    cap=${attempt%/v1}
    script=train.py
    [ "$attempt" != "$cap" ] && script=train_valbs1.py
    if [ -d "$save" ] && [ -n "$(ls -A "$save" 2>/dev/null)" ]; then  # keep files of an earlier attempt aside
      mv "$save" "${save}_old_$(date +%m%d_%H%M%S)"
    fi
    mkdir -p "$save"
    c=run/vis/${repo}_cap$cap.yaml
    sed "s/^  max_events_num *:.*/  max_events_num : $cap/" "$cfg" > "$c"
    say "$repo: training ($script, max_events_num $cap; log $log)"
    (cd "$CMP/$repo" && python -u "$script" --config "$HERE/$c") > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$save/best_iou_seed37.pt" ]; then
      echo "$script max_events_num $cap" > "$save/TRAIN_DONE"
      say "$repo: trained ($script, max_events_num $cap)"
      return 0
    fi
    if grep -qa "out of memory" "$log"; then
      say "$repo: out of GPU memory ($script, max_events_num $cap), next attempt"
      continue
    fi
    say "$repo training FAILED: $(tail -n 3 "$log" | tr '\n' ' ')"
    return 1
  done
  say "$repo training FAILED: out of GPU memory in every attempt"
  return 1
}

dump_repo() {                       # repo, model, config
  local repo=$1 model=$2 cfg=$3 save split out log
  save=$(grep -m1 '^  model_save_root:' "$cfg" | awk '{print $2}')
  for split in test val train; do
    out=run/vis/$repo
    [ "$split" != test ] && out=run/vis/${repo}_$split
    log=run/vis/${repo}_dump_$split.log
    if grep -q "DUMP COMPETITOR FINISHED" "$log" 2>/dev/null; then say "$repo $split: dumped before"; continue; fi
    python -u tools/vis/dump_competitor.py --repo "$CMP/$repo" --model "$model" --config "$cfg" \
      --ckpt "$save/best_iou_seed37.pt" --split "$split" --out "$out" --eval-repo "$CMP/EV-Seg" > "$log" 2>&1 \
      || { say "$repo $split dump FAILED: $(tail -n 2 "$log" | tr '\n' ' ')"; return 1; }
    say "$repo $split: $(tail -n 1 "$log")"
  done
}

say "competitors on GPU $GPU: start"
C=tools/vis/competitors
train_repo EV-Seg $C/evspsegnet_evuav.yaml 700000 450000 300000 && dump_repo EV-Seg evspsegnet $C/evspsegnet_evuav.yaml
train_repo PACT $C/pact_evuav.yaml 100000 100000/v1 60000/v1 && dump_repo PACT PACT $C/pact_evuav.yaml
say "COMPETITORS FINISHED"
