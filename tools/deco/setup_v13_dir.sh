#!/usr/bin/env bash
# Prepare a fresh V13 code dir (11-10): link the inputs of every backbone tag from an older code dir instead of copying
# them: run/deco/<TAG>/z0_{train,val,test} (the BIR priors) and, if present, the evidence of the wide training band
# ev_m0.001_nr_{train,val,test}.npz (its log is copied, run_v13.sh checks it). Trained SDUs are not linked: the jobs copy
# them when they start (run_v13.sh REUSE_FROM), so an SDU still training in the old dir is picked up once it is done.
# Usage (new code dir, after `conda activate evuav`): OLD=<old code dir> [TAGS="v5s37 v5s38 v5s39"] bash tools/deco/setup_v13_dir.sh
# Last line: "SETUP V13 DIR DONE" or "SETUP V13 DIR FAILED: ...".
set -u
cd "$(dirname "$0")/../.."
OLD=${OLD:?OLD=<old code dir>}
TAGS=${TAGS:-"v5s37 v5s38 v5s39"}
die() { echo "SETUP V13 DIR FAILED: $*"; exit 1; }
[ "$(cd "$OLD" && pwd)" != "$(pwd)" ] || die "OLD is this dir"
for t in $TAGS; do
  mkdir -p "run/deco/$t"
  for s in train val test; do
    [ -f "$OLD/run/deco/$t/z0_$s/DONE" ] || die "$OLD/run/deco/$t/z0_$s/DONE missing"
    [ -e "run/deco/$t/z0_$s" ] || ln -s "$OLD/run/deco/$t/z0_$s" "run/deco/$t/z0_$s"
    ev=ev_m0.001_nr_$s
    if [ ! -e "run/deco/$t/$ev.npz" ] && [ -f "$OLD/run/deco/$t/$ev.npz" ] \
       && tail -n 1 "$OLD/run/deco/$t/$ev.log" 2>/dev/null | grep -q "EVIDENCE FINISHED"; then
      ln -s "$OLD/run/deco/$t/$ev.npz" "run/deco/$t/$ev.npz"
      cp "$OLD/run/deco/$t/$ev.log" "run/deco/$t/$ev.log"
    fi
  done
  echo "  $t: z0 linked, evidence $(ls run/deco/$t/ev_m0.001_nr_*.npz 2>/dev/null | wc -l) / 3 linked," \
       "SDU seed 0 in the old dir: $([ -f "$OLD/run/deco/$t/dnm_m0.001_s0_nr_v13/summary.json" ] && echo ready || echo 'not yet')"
done
echo "SETUP V13 DIR DONE"
