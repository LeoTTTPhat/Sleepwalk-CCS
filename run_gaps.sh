#!/bin/sh
# Stage-4 supervisor: re-invokes the guarded, idempotent gaps_chain.sh until GAPS_OK.
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MG=${MEMGUARD:-$P/tools/memguard.py}
cd $P
i=0
while [ $i -lt 20 ]; do
  i=$((i+1))
  echo "[run_gaps $(date '+%Y-%m-%d %H:%M:%S')] pass $i"
  $MG --name p3-geofenced-sleepers-memory-gaps --ollama-models qwen2.5:7b -- sh $P/gaps_chain.sh && { echo "[run_gaps] done"; exit 0; }
  echo "[run_gaps $(date '+%Y-%m-%d %H:%M:%S')] chain exited non-zero; retry in 120s"
  sleep 120
done
echo "[run_gaps] giving up after $i passes"; exit 1
