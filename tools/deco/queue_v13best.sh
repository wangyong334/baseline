#!/usr/bin/env bash
# V13 best version (11-10): the VoW unit with a risk per side (VOW_ARGS, default "--alpha-bg 0.001 --alpha-tg 0.003":
# stopping as background, which can miss a target (Pd), needs P(the SDU would still flip it) <= 0.001; stopping as
# target, which can be a false alarm (Fa), <= 0.003; gate and commit alike).
#   (1) SDU seed 0 of every backbone: copied from OLD (run_v13.sh REUSE_FROM) when the job starts, then the VoW stage
#       -> dnm_m0.001_s0_nr_v13/vow_best (without OLD, or if OLD has no finished SDU, it is trained here)
#   (2) SDU seeds 1 and 2 of every backbone: training + VoW -> dnm_m0.001_s<seed>_nr_v13/vow_best (shared evidence)
# All jobs go into the pool of this code dir; its dispatcher must run on GPU 1 only (about 7-8 h in total).
# Gates: the 11-10 versions of fit_vow.py / run_v13.sh, a dispatcher of THIS dir running, z0 of every tag present
# (tools/deco/setup_v13_dir.sh links it), no job of this round queued yet.
# Usage (code dir, after `conda activate evuav`):
#   OLD=<old code dir> [VOW_ARGS="--alpha-bg 0.001 --alpha-tg 0.003"] [TAGS="v5s37 v5s38 v5s39"] bash tools/deco/queue_v13best.sh
# Results: run/deco/<TAG>/dnm_m0.001_s{0,1,2}_nr_v13/vow_best/vow.json; pack: bash tools/deco/pack_v13.sh
set -u
cd "$(dirname "$0")/../.."
TAGS=${TAGS:-"v5s37 v5s38 v5s39"}
VOW_ARGS=${VOW_ARGS:-"--alpha-bg 0.001 --alpha-tg 0.003"}
OLD=${OLD:-}
die() { echo "QUEUE V13 BEST NOT STARTED: $*"; exit 1; }
grep -q "alpha-bg" tools/deco/fit_vow.py || die "tools/deco/fit_vow.py has no --alpha-bg (upload the 11-10 code)"
grep -q "REUSE_FROM" tools/deco/run_v13.sh || die "tools/deco/run_v13.sh has no REUSE_FROM (upload the 11-10 code)"
here=$(pwd -P)
mine=""
for p in $(pgrep -f "tools/final/gpu_pool.sh"); do        # a dispatcher whose working dir is this code dir
  [ "$(readlink -f "/proc/$p/cwd" 2>/dev/null)" = "$here" ] && mine=$p
done
[ -n "$mine" ] || die "no dispatcher of this dir is running (GPUS=\"1\" HOLD_GPUS=\"1\" KEEP=1 nohup bash tools/final/gpu_pool.sh ...)"
ls run/queue/pool/pending 2>/dev/null | grep -q "v13best_" && die "v13best jobs are queued already"
[ -z "$OLD" ] || [ -d "$OLD/run/deco" ] || die "OLD=$OLD has no run/deco"
for t in $TAGS; do
  for s in train val test; do [ -f "run/deco/$t/z0_$s/DONE" ] || die "run/deco/$t/z0_$s/DONE missing (run setup_v13_dir.sh)"; done
done
add() {                                      # name tag seed [reuse dir]
  local reuse=""
  [ -n "${4:-}" ] && reuse="REUSE_FROM='$4' "
  bash tools/final/pool_add.sh "$1" "${reuse}TAG=$2 SEED=$3 VOW_NAME=vow_best VOW_ARGS='$VOW_ARGS' WORKERS=8 bash tools/deco/run_v13.sh" \
    > /dev/null || die "pool_add $1"
}
for t in $TAGS; do add "v13best_${t}_s0" "$t" 0 "$OLD"; done
for seed in 1 2; do
  for t in $TAGS; do add "v13best_${t}_s$seed" "$t" "$seed"; done
done
echo "pool: $(ls run/queue/pool/pending | tr '\n' ' ')"
echo "QUEUE V13 BEST DONE ($VOW_ARGS${OLD:+, seed-0 SDUs from $OLD})"
