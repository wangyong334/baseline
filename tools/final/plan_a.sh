#!/usr/bin/env bash
# Decision A (11-04 evening): main model = V5 (module one) + decision head (module three); module two leaves this paper
# (kept for later). Replaces the E3b parts of chains.sh / queue_seed.sh / queue_b2_evf.sh, which are still running on the
# server and are not edited; `stop` ends only the parts listed below.
#   stop    preview (DRY=0: do it): chain gpu0 + queue_seed.sh + its backbone.sh + the E3b training it started (the
#           finished V5 seed 38 stays); chain gpu3 + queue_b2_evf.sh (the running B2 + occupancy head keeps running; the
#           V5 retrain is dropped). GPU 1's 25 ms queue is left alone.
#   stop25  only together with SELECT_RULE=plain: also end GPU 1's old 25 ms queue (it uses the guarded epoch)
#   gpu1    waits for the old 25 ms queue (V5@25 -> RCL@25) while it runs, else selects + evaluates V5@25 itself; then the
#           remaining HEADS25 (default "rcl b2"; a finished head is skipped) on the same backbone; reports whether the
#           50 epochs reproduce the 10-25 run (EV-UAV-v63-w25) bit for bit
#   gpu0    seed 38: V5 (kept when already done) -> HEADS (default "rcl b2") -> EV-Flying RCL head in the final layout
#   gpu3    waits for B2 + occupancy (seed 37) -> seed 39: V5 -> HEADS
#   cpu     calibrated commit for the seed-37 B2 heads (the running commit_cal.sh covers the RCL heads)
# Epochs: the pre-registered guarded rule (SELECT_RULE=plain: best val only). Clean val / test evals only.
# Usage (code dir, after `conda activate evuav`):
#   bash tools/final/plan_a.sh stop          # look first
#   DRY=0 bash tools/final/plan_a.sh stop
#   nohup bash tools/final/plan_a.sh gpu1 > run/final/plan_a_gpu1.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
mkdir -p run/final log/final
CMD=${1:-}
ts() { date '+%m-%d %H:%M:%S'; }
busy() { pgrep -f "$1" > /dev/null; }      # some process line matches $1

descendants() {    # every process below pid $1
  local c
  for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done
}

ancestors() {      # pipeline scripts above pid $1 (tools/final/*, or the 11-04 morning waiter on queue_basic_e1)
  local p a
  p=$(ps -o ppid= -p "$1" | tr -d ' ')
  while [ -n "$p" ] && [ "$p" -gt 1 ]; do
    a=$(ps -o args= -p "$p")
    case "$a" in
      *tools/final/*|*queue_basic_e1*) echo "$p"; p=$(ps -o ppid= -p "$p" | tr -d ' ') ;;
      *) break ;;
    esac
  done
}

end_procs() {      # $1 = pids to end, $2 = what they must be (shown with the preview)
  local kill=$1 p a m
  echo "== 流水线的全部进程（* = 将结束）"
  pgrep -f "tools/final/|train_stream_v1.py|train_track_head.py|select_checkpoint.py|final_analysis.py|commit_calibration.py" \
    | while read -r p; do
        a=$(ps -o pid=,etime=,args= -p "$p" | cut -c1-165)
        case "$a" in *plan_a.sh*|"") continue ;; esac
        case " $kill " in *" $p "*) m="*" ;; *) m=" " ;; esac
        echo "$m $a"
      done
  [ -n "$kill" ] || { echo "没有要结束的进程"; return 0; }
  if [ "${DRY:-1}" = "1" ]; then
    echo "以上是预览。核对 * 行只有：$2"
    echo "确认后：DRY=0 bash tools/final/plan_a.sh $CMD"
    return 0
  fi
  kill $kill
  sleep 3
  echo "== 结束后还在的流水线进程"
  pgrep -af "tools/final/|train_stream_v1.py|train_track_head.py" | grep -v "plan_a.sh" | cut -c1-165
}

stop() {
  local c0 q38 b38="" c3 qb2 e3b p
  c0=$(pgrep -f "tools/final/chains.sh gpu0" | xargs)
  q38=$(pgrep -f "tools/final/queue_seed.sh" | xargs)
  for p in $q38; do b38="$b38 $(pgrep -P "$p" -f "tools/final/backbone.sh" | xargs)"; done
  c3=$(pgrep -f "tools/final/chains.sh gpu3" | xargs)
  qb2=$(pgrep -f "tools/final/queue_b2_evf.sh" | xargs)
  e3b=$(pgrep -f "train_stream_v1.py --config configs/final/bb_e3b_50.yaml --mode train" | xargs)
  end_procs "$(echo $c0 $q38 $b38 $c3 $qb2 $e3b | tr ' ' '\n' | sort -un | xargs)" \
    "chains.sh gpu0 / gpu3、queue_seed.sh、它下面的 backbone.sh、queue_b2_evf.sh、E3b 种子 38 的训练"
}

stop25() {
  local q25 up="" p
  q25=$(pgrep -f "tools/final/queue_25ms.sh" | xargs)
  for p in $q25; do up="$up $(ancestors "$p" | xargs) $(descendants "$p" | xargs)"; done
  end_procs "$(echo $q25 $up | tr ' ' '\n' | sort -un | xargs)" \
    "卡 1 的整个旧 25 ms 队列（等待任务、queue_25ms.sh 和它下面的脚本、python）"
}

calibrate() {      # $1 = head run, $2 = its config name, $3 = seed: calibrated commit (temperature on val), CPU only
  RUNS="$1:$2" SEED=$3 bash tools/final/commit_cal.sh > run/final/plan_a_cal_$1_seed$3.log 2>&1
  RUNS="$1:$2" SEED=$3 bash tools/final/cal_analysis.sh >> run/final/plan_a_cal_$1_seed$3.log 2>&1
  echo "  校准 $1 seed $3：$(grep -m1 '温度 T\|FAILED' run/final/plan_a_cal_$1_seed$3.log | cut -c1-100)"
}

head_on() {        # $1 = rcl | b2 | b2occ | causal, $2 = backbone checkpoint, $3 = seed, $4 = 50 | 25
  local h=$1 base=$2 s=$3 w=$4 name run dir
  case "$h@$w" in
    rcl@50)    name=rcl_feature_50;        run=rcl_feature_50_on_v5 ;;
    b2@50)     name=b2_lookahead_50;       run=b2_lookahead_50_on_v5 ;;
    b2occ@50)  name=b2_lookahead_occ_50;   run=b2_lookahead_occ_50_on_v5 ;;
    causal@50) name=rcl_feature_causal_50; run=rcl_feature_causal_50_on_v5 ;;
    rcl@25)    name=rcl_feature_25;        run=rcl_feature_25_on_v5w25 ;;
    b2@25)     name=b2_lookahead_25;       run=b2_lookahead_25_on_v5w25 ;;
    *) echo "[$(ts)] 没有这个判决头：$h @ $w ms"; return 1 ;;
  esac
  dir=log/final/${run}_seed$s
  if [ -f $dir/HEAD_DONE ]; then echo "[$(ts)] $run seed $s 已完成，跳过"; return 0; fi
  if [ -d $dir ]; then           # left by a stopped queue: the trainer refuses a directory with metrics.jsonl
    mv $dir ${dir}_stale_$(date +%m%d%H%M) && echo "  未完成的旧目录已移开：${dir}_stale_*"
  fi
  NAME=$name RUN=$run SEED=$s BASE=$base HZ="" B1=$([ "$h@$w" = "rcl@50" ] && echo 1 || echo 0) \
    PAST=$([ "$h" = "causal" ] && echo 0 || echo 1) bash tools/final/head.sh || return 1
  # RCL@25 of seed 37 is calibrated by the commit_cal.sh / cal_analysis.sh already waiting on the server
  [ "$h" = "causal" ] || [ "$h@$w@$s" = "rcl@25@37" ] || calibrate $run $name $s
}

seed_queue() {     # $1 = seed: V5 (clean evals; kept when already done) -> HEADS on it
  local s=$1 out=log/final/bb_v5_50_seed$1 train=1 h
  if [ -f $out/BACKBONE_DONE ]; then
    echo "[$(ts)] 种子 $s 的 V5 已完成：$(sed -n '3p' run/final/bb_v5_50_seed${s}_select.log 2>/dev/null | cut -c1-70)"
  else
    if busy "log/final/bb_v5_50_seed$s( |/|$)"; then
      echo "[$(ts)] 等种子 $s 正在跑的 V5 训练 / 评估结束"
      while busy "log/final/bb_v5_50_seed$s( |/|$)"; do sleep 120; done
    fi
    if [ -d $out ]; then
      tail -n 1 run/final/bb_v5_50_seed${s}_train.log 2>/dev/null | grep -q "TRAINING FINISHED" \
        || { echo "[$(ts)] PLAN A SEED $s FAILED：$out 已存在但训练没有正常结束（移开该目录后重跑）"; return 1; }
      train=0
    fi
    NAME=bb_v5_50 SEED=$s TRAIN=$train HZ="" bash tools/final/backbone.sh || { echo "[$(ts)] PLAN A SEED $s FAILED at V5"; return 1; }
  fi
  for h in ${HEADS:-rcl b2}; do
    head_on $h $out/selected_seed$s.pt $s 50 || echo "[$(ts)] 种子 $s 的 $h 失败，接着跑下一个"
  done
}

gpu1() {
  local out=log/final/bb_v5_25_seed37 ref=configs/final/reference/bb_v5_25_seed37_v63_record.jsonl h n waited=0
  if busy "tools/final/queue_25ms.sh"; then
    echo "[$(ts)] 旧的 25 ms 队列在跑（V5@25 → RCL@25）：等它结束，再在同一份骨干上接着跑"
    while busy "tools/final/queue_25ms.sh"; do sleep 300; done
    waited=1
  fi
  if busy "log/final/bb_v5_25_seed37( |/|$)"; then echo "卡 1 上还有 V5@25 的进程，先查清再跑"; return 1; fi
  tail -n 1 run/final/bb_v5_25_seed37_train.log 2>/dev/null | grep -q "TRAINING FINISHED" \
    || { echo "[$(ts)] PLAN A GPU1 FAILED：V5@25 没有训完（run/final/bb_v5_25_seed37_train.log）"; return 1; }
  if n=$(python -c "import sys; sys.path.insert(0, 'tools'); import select_checkpoint as S
print(S.check_replica(S.read_records(sys.argv[1]), S.read_records(sys.argv[2])))" $out/metrics.jsonl $ref 2>&1); then
    echo "  V5@25 的 $n 轮 val IoU 与 10-25 那次训练逐位相同"
  else
    echo "  V5@25 与 10-25 那次训练不逐位相同：$(echo "$n" | tail -n 1 | cut -c1-120)"
  fi
  if [ $waited = 0 ] || [ ! -f $out/BACKBONE_DONE ]; then       # select (current rule) + clean evals
    NAME=bb_v5_25 SEED=37 TRAIN=0 HZ="" bash tools/final/backbone.sh || { echo "[$(ts)] PLAN A GPU1 FAILED at V5@25"; return 1; }
  fi
  for h in ${HEADS25:-rcl b2}; do
    head_on $h $out/selected_seed37.pt 37 25 || echo "[$(ts)] 25 ms 的 $h 失败，接着跑下一个"
  done
}

gpu0() {
  if busy "tools/final/queue_seed.sh" || busy "tools/final/chains.sh gpu0"; then
    echo "旧的种子 38 队列还在：先 DRY=0 bash tools/final/plan_a.sh stop"; return 1
  fi
  seed_queue 38
  local V5E=log/stream_tif_commpeak_msg_evflying_x4_seed37/best_val_iou_seed37.pt SRC=log/changepoint_v41_evflying_feature_seed37
  if [ -f $V5E ] && [ -f $SRC/best_val_iou_seed37.pt ]; then      # EV-Flying: the RCL head trained on V5-EVF, eval only
    mkdir -p log/final/rcl_feature_evf_on_v5evf_seed37
    cp -n $SRC/best_val_iou_seed37.pt $SRC/metrics.jsonl $SRC/run_config.json log/final/rcl_feature_evf_on_v5evf_seed37/
    NAME=rcl_feature_evf RUN=rcl_feature_evf_on_v5evf BASE=$V5E TRAIN=0 HZ="" bash tools/final/head.sh \
      || echo "[$(ts)] EV-Flying 判决头评估失败"
  else
    echo "跳过 EV-Flying：缺 $V5E 或 $SRC/best_val_iou_seed37.pt"
  fi
}

gpu3() {
  if busy "tools/final/queue_b2_evf.sh" || busy "tools/final/chains.sh gpu3"; then
    echo "旧的卡 3 队列还在：先 DRY=0 bash tools/final/plan_a.sh stop"; return 1
  fi
  echo "[$(ts)] 等 B2 + 占据监督（种子 37）结束"
  until [ -f log/final/b2_lookahead_occ_50_on_v5_seed37/HEAD_DONE ] \
        || grep -q "HEAD b2_lookahead_occ_50_on_v5 seed 37 FAILED" run/final/queue_b2_evf.log 2>/dev/null; do
    sleep 300
  done
  seed_queue 39
}

cpu() {
  calibrate b2_lookahead_50_on_v5 b2_lookahead_50 37
  calibrate b2_lookahead_occ_50_on_v5 b2_lookahead_occ_50 37        # commit_cal.sh waits for its HEAD_DONE
}

case "$CMD" in
  stop)   stop; exit 0 ;;
  stop25) stop25; exit 0 ;;
  gpu0)   export GPU=${GPU:-0}; gpu0 ;;
  gpu1)   export GPU=${GPU:-1}; gpu1 ;;
  gpu3)   export GPU=${GPU:-3}; gpu3 ;;
  cpu)    cpu ;;
  *) echo "用法：bash tools/final/plan_a.sh stop | stop25 | gpu0 | gpu1 | gpu3 | cpu"; exit 1 ;;
esac || { echo "[$(ts)] PLAN A $CMD FAILED"; exit 1; }
echo "[$(ts)] PLAN A $CMD FINISHED"
