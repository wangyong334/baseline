#!/usr/bin/env bash
# Prepare the V17 code dir (11-13): link the priors of every backbone tag from an older code dir instead of copying
# them: run/deco/<TAG>/z0_{train,val,test} (the frozen V5 backbones' calibrated priors, tools/deco/dump_z0.py).
# Usage (V17 code dir, after `conda activate evuav`): OLD=<old code dir> [TAGS="v5s37 v5s38 v5s39"] bash tools/v17/setup_v17_dir.sh
# Last line: "SETUP V17 DIR DONE" or "SETUP V17 DIR FAILED: ...".
set -u
cd "$(dirname "$0")/../.."
OLD=${OLD:?OLD=<old code dir with run/deco/<TAG>/z0_*>}
TAGS=${TAGS:-"v5s37 v5s38 v5s39"}
die() { echo "SETUP V17 DIR FAILED: $*"; exit 1; }
[ "$(cd "$OLD" && pwd -P)" != "$(pwd -P)" ] || die "OLD is this dir"
for t in $TAGS; do
  mkdir -p "run/deco/$t"
  for s in train val test; do
    src=$(readlink -f "$OLD/run/deco/$t/z0_$s")
    [ -d "$src" ] || die "$OLD/run/deco/$t/z0_$s missing"
    n=$(ls "$src"/*.npz 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || die "$src has no priors"
    [ -e "run/deco/$t/z0_$s" ] || ln -s "$src" "run/deco/$t/z0_$s"
    echo "  $t $s: $n recordings ($src)"
  done
done
echo "SETUP V17 DIR DONE"
