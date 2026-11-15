#!/usr/bin/env bash
# Pack the V20 results for download (V20 code dir): per backbone the eval / long-stream json, metrics.jsonl and logs
# (no weights), the decision-layer runs without trajectories, and the card logs.
cd "$(dirname "$0")/../.."
stamp=$(date +%Y%m%d_%H%M%S)
archive=v20_results_$stamp.tar.gz
items=$(ls -d log/final/{v20,bb_v5}_*_seed*/{eval_*.json,long_stream_*.json,metrics.jsonl,run_config*.json,selection*.json} \
  run/final/{v20,bb_v5}_* run/v20 run/v18/p20_* run/pool_gpu3/pool.log 2>/dev/null)
tar --exclude='traj_*.npz' --exclude='*.pt' --exclude='messages_*.npz' -czf $archive $items
ls -lh $archive
echo "PACK V20 DONE $archive"
