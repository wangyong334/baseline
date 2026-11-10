#!/usr/bin/env bash
# V13 dependency check (11-10): V13 needs the V11 versions of the decision-layer files (the V12 package changed some of
# them). Compares the server copies with deps_v11/MANIFEST.md5 of the V13 package; APPLY=1 replaces the differing files
# by the packaged V11 copies (copy to a temp name, then mv; the old copy stays as <file>.pre_v13).
# Usage (code dir, after `conda activate evuav` and `tar xzf v13_pkg_*.tgz`):
#   bash tools/deco/check_v13_deps.sh            report only
#   APPLY=1 bash tools/deco/check_v13_deps.sh    replace the differing files (no V12 job may still be running)
set -u
cd "$(dirname "$0")/../.."
M=deps_v11/MANIFEST.md5
[ -f "$M" ] || { echo "CHECK V13 DEPS: $M missing (extract the V13 package in the code dir)"; exit 1; }
bad=0
while read -r sum f; do
  if [ -f "$f" ] && [ "$(md5sum "$f" | cut -d' ' -f1)" = "$sum" ]; then
    echo "  same     $f"
  else
    echo "  DIFFERS  $f"
    bad=$((bad + 1))
    if [ "${APPLY:-0}" = 1 ]; then
      [ -f "$f" ] && cp -p "$f" "$f.pre_v13"
      cp "deps_v11/$f" "$f.v13tmp" && mv "$f.v13tmp" "$f" && echo "           replaced (old copy: $f.pre_v13)"
    fi
  fi
done < "$M"
if [ "$bad" = 0 ]; then
  echo "CHECK V13 DEPS OK"
elif [ "${APPLY:-0}" = 1 ]; then
  echo "CHECK V13 DEPS APPLIED ($bad files replaced)"
else
  echo "CHECK V13 DEPS: $bad files differ (APPLY=1 to replace them by the V11 copies)"
fi
