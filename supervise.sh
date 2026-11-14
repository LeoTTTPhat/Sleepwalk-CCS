#!/bin/sh
# Outer supervisor: re-invokes the idempotent run_full.sh until it reports FULL_OK.
# Every step inside run_full.sh resumes from JSONL checkpoints + the sqlite call cache.
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $P
i=0
while [ $i -lt 30 ]; do
  i=$((i+1))
  echo "[supervise $(date '+%Y-%m-%d %H:%M:%S')] pass $i"
  SKIP_DATAPREP=1 sh $P/run_full.sh && { echo "[supervise] done"; exit 0; }
  echo "[supervise $(date '+%Y-%m-%d %H:%M:%S')] run_full.sh exited non-zero; retry in 120s"
  sleep 120
done
echo "[supervise] giving up after $i passes"; exit 1
