#!/usr/bin/env bash
# V9-final: what each GPU runs after its basic-batch queue, one queue after another.
# Usage (code dir, after `conda activate evuav`), one line per GPU:
#   nohup bash tools/final/chains.sh gpu1 > run/final/chain_gpu1.log 2>&1 &
#   gpu1: after queue_basic_e1  -> 25 ms core experiment -> EV-Flying E3b (runs only when E3b is the chosen backbone)
#   gpu0: after queue_basic_e3b -> seed 38 (V5 + chosen backbone, RCL on both, B2, B2 + occupancy, causal head)
#   gpu3: after the E3a backbone of queue_basic_v5 -> B2 two versions + EV-Flying head + V5 retrain -> seed 39
# A failed queue does not stop the chain; its own log says where it failed.
set -u
cd "$(dirname "$0")/../.."
ts() { date '+%m-%d %H:%M:%S'; }
wait_queue() {      # $1 = queue log, $2 = its name in the last line, e.g. "QUEUE BASIC E1"
  until grep -q "$2 \(FINISHED\|FAILED\)" "$1" 2>/dev/null; do sleep 300; done
  echo "[$(ts)] $(grep "$2 \(FINISHED\|FAILED\)" "$1" | tail -n 1)"
}
choice() {          # config of the chosen module-two backbone; empty when the basic V5 queue failed before choosing
  until [ -f run/final/backbone_choice_seed37.json ] || grep -q "QUEUE BASIC V5 FAILED" run/final/queue_basic_v5.log 2>/dev/null; do
    sleep 300
  done
  [ -f run/final/backbone_choice_seed37.json ] && python -c "
import json; print('bb_%s_50' % json.load(open('run/final/backbone_choice_seed37.json', encoding='utf-8'))['winner'])"
}
run() {             # $1 = log name under run/final, then VAR=value ... command
  local name=$1; shift
  echo "[$(ts)] 开始 $name：$*"
  env "$@" > run/final/$name.log 2>&1
  echo "[$(ts)] $name 结束：$(tail -n 1 run/final/$name.log)"
}
CHAIN=${1:?gpu0 / gpu1 / gpu3}
echo "[$(ts)] CHAIN $CHAIN 开始，python $(command -v python)"
case "$CHAIN" in
  gpu1)
    wait_queue run/final/queue_basic_e1.log "QUEUE BASIC E1"
    run queue_25ms GPU=1 bash tools/final/queue_25ms.sh
    M2=$(choice)
    if [ -n "$M2" ]; then run queue_evf GPU=1 M2=$M2 bash tools/final/queue_evf.sh; else echo "[$(ts)] 没有选骨干的结果：跳过 queue_evf"; fi
    ;;
  gpu0)
    wait_queue run/final/queue_basic_e3b.log "QUEUE BASIC E3B"
    M2=$(choice)
    if [ -n "$M2" ]; then run queue_seed38 GPU=0 SEED=38 M2=$M2 B2OCC=1 bash tools/final/queue_seed.sh
    else echo "[$(ts)] 没有选骨干的结果：跳过种子 38"; fi
    ;;
  gpu3)
    until [ -f log/final/bb_e3a_50_seed37/BACKBONE_DONE ] || grep -q "QUEUE BASIC V5 FAILED" run/final/queue_basic_v5.log 2>/dev/null; do
      sleep 300
    done
    echo "[$(ts)] E3a 骨干已完成（或基础 V5 队列失败），卡 3 空出"
    run queue_b2_evf GPU=3 bash tools/final/queue_b2_evf.sh
    M2=$(choice)
    if [ -n "$M2" ]; then run queue_seed39 GPU=3 SEED=39 M2=$M2 B2OCC=1 bash tools/final/queue_seed.sh
    else echo "[$(ts)] 没有选骨干的结果：跳过种子 39"; fi
    ;;
  *) echo "未知的 CHAIN：$CHAIN（gpu0 / gpu1 / gpu3）"; exit 1 ;;
esac
echo "[$(ts)] CHAIN $CHAIN FINISHED"
