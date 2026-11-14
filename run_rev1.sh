#!/bin/sh
# Stage-7 (round-1 revision) supervisor: idempotent, resumable, memory-guarded.
# Every step re-runs safely (JSONL checkpoints skip finished keys), so an interrupted
# chain is restarted by re-invoking this script.  Steps are in reviewer-priority order:
# an interrupted chain still leaves every earlier answer complete.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="${MEMGUARD:-$P/tools/memguard.py}"
PY="${PY:-python}"
CFG=$P/src/configs/rev1.yaml
cd $P || exit 1
mkdir -p logs results/rev1

run() { # run <name> <exp>
  name=$1; exp=$2
  if [ -f .memguard/STOP ] || [ -f .memguard/STOP-p3-rev1 ]; then
    echo "[rev1] STOP file present, exiting"; exit 0
  fi
  i=0
  while [ $i -lt 4 ]; do
    i=$((i+1))
    echo "[rev1] === $name (attempt $i) $(date +%H:%M) ==="
    PYTHONPATH=$P/src "$GUARD" --name p3-geofenced-sleepers-memory-rev-$name \
      --ollama-models qwen2.5:7b -- \
      "$PY" -m geofence.revision --config $CFG --exp $exp && return 0
    rc=$?
    echo "[rev1] $name failed rc=$rc; retrying after cooldown"
    sleep 90
  done
  echo "[rev1] $name gave up after $i attempts"
  return 1
}

# 1. dormancy / ASR per request (the first major issue of all four reviewers)
run asrpool asr_pool
# 2. text-only baselines (static rules + LLM judge) on both corpora
run judge judge
# 3. CCS on corpus v2 (diverse, user-targeted Q-mass bins, adaptive variants, CRN)
run ccs2 ccs2
# 4. payload-specific ASR on corpus v2, by request type
run asr2 asr2
# 5. provenance-gate ablations + sleepers in the channels the gates admit
run provgate2 provgate2
# 6. routing loop with no-entry and phi-fail controls
run osrm2 osrm2
# 7. uniform probing on corpus v2 (Q vs uniform where triggers ARE user-targeted)
run ccs2uni ccs2_uniform
# 8. split pairs with oracle grouping (contrast with the oracle-free default)
run ccs2orc ccs2_oracle
# 9. optional: Hammer2.1-7B with its native function-calling schema
if [ "${REV_SKIP_HAMMER:-0}" != "1" ]; then run hammer2 hammer2; fi
echo "[rev1] REV1_OK $(date +%F' '%H:%M)"
