#!/usr/bin/env bash
# Wait (up to 1 h) for the marker of tools/deco/smoke_dnm_w400.sh; exit 0 on OK, exit 1 on FAILED or timeout.
# Usage in a queue job: bash tools/deco/wait_smoke.sh && <job>
cd "$(dirname "$0")/../.."
for i in $(seq 120); do
  [ -f run/deco/SMOKE_DNM_W400_OK ] && exit 0
  [ -f run/deco/SMOKE_DNM_W400_FAILED ] && { echo "smoke failed: see run/deco/smoke_*.log"; exit 1; }
  sleep 30
done
echo "smoke marker missing after 1 h: see run/queue/gpu1/logs/005_smoke_dnm_w400.log"
exit 1
