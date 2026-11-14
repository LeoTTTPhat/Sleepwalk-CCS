#!/bin/sh
# Smoke: full pipeline on tiny subsets with qwen2.5:7b (Ollama). Every step guarded.
set -e
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
MG=${MEMGUARD:-$P/tools/memguard.py}
CFG=$P/src/configs/smoke.yaml
export PYTHONPATH=$P/src
export HF_HUB_OFFLINE=1
# Smoke builds its own tiny derived sets; never touch data/derived (full-run inputs).
export GEOFENCE_DERIVED=$P/data/derived_smoke
cd $P
N=p3-geofenced-sleepers-memory

# Retry wrapper: the shared memguard occasionally dies on a transient psutil
# process-enumeration error. Every step is checkpointed/resumable, so retry safely.
retry() { n=0; while [ $n -lt 4 ]; do "$@" && return 0; n=$((n+1)); echo "[retry $n] $*"; sleep 8; done; echo "[FAIL] $*"; return 1; }

echo "[smoke] data prep"
retry $MG --name $N-dataprep -- $PY -m geofence.data_prep --city austin --n-users 2 --max-ctx 150 --n-legit 6 --seed 20261015

echo "[smoke] attack construction"
retry $MG --name $N-attack -- $PY -m geofence.attack --city austin --n-sleepers 6 --n-benign 6 --seed 20261015

for E in asr provgate ccs randomq shift loo osrm; do
  echo "[smoke] evaluate $E"
  retry $MG --name $N-eval-$E --ollama-models qwen2.5:7b -- $PY -m geofence.evaluate --config $CFG --exp $E
done

echo "[smoke] filters"
retry $MG --name $N-filters --ollama-models llama-guard3:8b -- $PY -m geofence.evaluate --config $CFG --exp filters

echo "[smoke] cost"
retry $MG --name $N-cost --ollama-models qwen2.5:7b -- $PY -m geofence.evaluate --config $CFG --exp cost

echo "[smoke] aggregate + plots"
retry $MG --name $N-aggregate -- $PY -m geofence.aggregate --run smoke
retry $MG --name $N-plots -- $PY -m geofence.plots --run smoke

echo "[smoke] outputs: results/smoke/seed_20261015/*.jsonl, results/smoke/summary.json, figures/smoke_*.pdf"
echo "SMOKE_OK"
