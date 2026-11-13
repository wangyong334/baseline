#!/usr/bin/env bash
# V17 round 3 on one GPU without the pool (11-13: the user gave GPU 2). A lane runs its jobs one after another through
# tools/v17/run_sdu.sh; three lanes share the card (about 7 GB each). Nothing is held between jobs, the lane exits
# when its list is done. Failed jobs are reported and skipped; running the lane again resumes them.
# Main model = defaults of tools/v17/train_sdu.py (window pooling, events as observations); its v5s37 seed-0 run is
# run/v17/v5s37/r2_events_s0 (round 2, same configuration).
#   main  v5s37 decision seeds 1 and 2 (noise of the decision layer), then the priors of v5s38 and v5s39
#   abl1  v5s37, one component changed: gate distance (no attribution query), pool mean (one mean over all windows),
#         pool sum (every token independent), context off (no state update)
#   abl2  v5s37: audit ledger (each window read once), value signed (one signed channel), levels 3 (reading range
#         81 px instead of 243), dim 32 (half width: cost)
#   cost variants (user 11-13: at similar accuracy take the cheaper one): lean = dim 32 + levels 3 (abl1),
#         cap20k = sampling cap 20000 instead of 50000 per margin bin, about half the training time (abl2)
#   (evidence boxes = run/v17/v5s37/r2_boxes_s0, round 2)
# Round 4 (11-13 evening, decision-layer energy: the main model costs about 10.6 mJ per 8 s, 3-4 x the backbone), one
# job per lane, v5s37 seed 0, all with distance or feature gates, the context without feed-forward and d = 32:
#   e_glm       boxes read by a context-modulated linear model of their statistics, distance gate (~6 k MAC/step)
#   e_glm_attr  the same with the attribution gate reading the box statistics (~8 k MAC/step)
#   e_linear    embedding linear in the features (no LayerNorm), distance gate (~17 k MAC/step)
# Round 5 (11-13 evening): integrate-to-bound units -- each window is read once and its evidence added to the membrane
# (audit ledger), constant cost per step (~3.5 k MAC) and a state of a few numbers instead of a token buffer:
#   l_glm        e_glm with the ledger; the context = a linear map of the accumulated window statistics
#   l_glm_noctx  the same without context (membrane only): does the reading need what the unit has seen?
# Usage (V17 code dir, after `conda activate evuav`):
#   GPU=2 LANE=main nohup bash tools/v17/round_v17.sh > run/v17/lane_main.log 2>&1 < /dev/null &
# Stop a lane after its current job: touch run/v17/STOP_<LANE>. Last line: "ROUND V17 LANE <LANE> FINISHED ...".
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:?GPU=<physical id>}
LANE=${LANE:?LANE=main|abl1|abl2|e_glm|e_glm_attr|e_linear|l_glm|l_glm_noctx}
LEAN="--context-ff off --dim 32"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case $LANE in
  main) JOBS=("v5s37 1 main" "v5s37 2 main" "v5s38 0 main" "v5s39 0 main") ;;
  abl1) JOBS=("v5s37 0 gate_distance --gate distance" "v5s37 0 pool_mean --pool mean"
              "v5s37 0 pool_sum --pool sum" "v5s37 0 context_off --context off" "v5s37 0 lean --dim 32 --levels 3") ;;
  abl2) JOBS=("v5s37 0 audit_ledger --audit ledger" "v5s37 0 value_signed --value signed"
              "v5s37 0 levels3 --levels 3" "v5s37 0 dim32 --dim 32" "v5s37 0 cap20k --sample-cap 20000") ;;
  e_glm) JOBS=("v5s37 0 e_glm --tokens glm --gate distance $LEAN") ;;
  e_glm_attr) JOBS=("v5s37 0 e_glm_attr --tokens glm --gate attribution $LEAN") ;;
  e_linear) JOBS=("v5s37 0 e_linear --tokens linear --gate distance $LEAN") ;;
  l_glm) JOBS=("v5s37 0 l_glm --tokens glm --gate distance --audit ledger $LEAN") ;;
  l_glm_noctx) JOBS=("v5s37 0 l_glm_noctx --tokens glm --gate distance --audit ledger --context off $LEAN") ;;
  *) echo "ROUND V17 LANE $LANE NOT STARTED: LANE=main|abl1|abl2|e_glm|e_glm_attr|e_linear|l_glm|l_glm_noctx"; exit 1 ;;
esac
mkdir -p run/v17
rm -f "run/v17/STOP_$LANE"
ok=0
bad=0
for j in "${JOBS[@]}"; do
  if [ -f "run/v17/STOP_$LANE" ]; then
    echo "$(date '+%m-%d %H:%M:%S') STOP_$LANE found, lane stops"
    break
  fi
  set -- $j
  tag=$1 seed=$2 name=$3
  shift 3
  echo "$(date '+%m-%d %H:%M:%S') job $tag $name s$seed ARGS='$*' on GPU $GPU"
  last=$(TAG=$tag SEED=$seed NAME=$name ARGS="$*" bash tools/v17/run_sdu.sh 2>&1 | tail -n 1)
  echo "$(date '+%m-%d %H:%M:%S')   $last"
  case $last in RUN\ V17\ DONE*) ok=$((ok + 1)) ;; *) bad=$((bad + 1)) ;; esac
done
echo "ROUND V17 LANE $LANE FINISHED: $ok done, $bad failed"
