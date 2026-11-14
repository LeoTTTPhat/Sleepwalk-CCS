#!/bin/sh
# Full runs (Stage 3). Primary: qwen2.5:7b (Ollama) on Austin. Secondary: GeoLife Beijing
# contexts (qwen) and the Hammer2.1-7b tool-calling table (HF).
# Every step is guarded by memguard and resumes from JSONL checkpoints; agent calls are
# memoised in data/cache/agent_calls.sqlite (deterministic decoding), so re-running a
# finished step is ~free. Order follows PLAN.md priorities:
#   P1 (seed 1): asr, ccs(+calibration), provgate, filters
#   P2 (seed 1): randomq, shift          P3 (seed 1): loo, osrm, cost
#   seeds 2,3 for the headline asr + ccs (>=3 seeds for the main claims)
#   secondary: Beijing asr+ccs, Hammer asr
# Set SKIP_DATAPREP=1 to reuse data/derived (already built in Stage 3).
set -e
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
MG=${MEMGUARD:-$P/tools/memguard.py}
export PYTHONPATH=$P/src
export HF_HUB_OFFLINE=1
cd $P
N=p3-geofenced-sleepers-memory
FULL=$P/src/configs/full.yaml
BJ=$P/src/configs/full_beijing.yaml
HM=$P/src/configs/full_hammer.yaml
S1=20261015; S2=20261016; S3=20261017

# Retry wrapper (transient memguard/psutil crashes; all steps are resumable).
retry() { n=0; while [ $n -lt 4 ]; do "$@" && return 0; n=$((n+1)); echo "[retry $n] $*"; sleep 10; done; echo "[FAIL] $*"; return 1; }
stamp() { echo "[full $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# OSRM routing server for the non-mocked loop (lightweight, <4GB; not a model job).
if ! curl -s -m 3 "http://127.0.0.1:5000/route/v1/driving/-97.74,30.27;-97.75,30.28?overview=false" 2>/dev/null | grep -q '"code":"Ok"'; then
  stamp "starting OSRM"
  docker run -d --rm -m 4g -p 5000:5000 -v "$P/data/osrm:/data" \
    ghcr.io/project-osrm/osrm-backend:v5.27.1 osrm-routed --algorithm mld /data/austin.osrm || true
  for i in $(seq 1 20); do
    curl -s -m 3 "http://127.0.0.1:5000/route/v1/driving/-97.74,30.27;-97.75,30.28?overview=false" 2>/dev/null | grep -q '"code":"Ok"' && break
    sleep 2
  done
fi

if [ "${SKIP_DATAPREP:-0}" != "1" ]; then
  stamp "data prep (austin + beijing)"
  retry $MG --name $N-dataprep -- $PY -m geofence.data_prep --city both --n-users 20 --max-ctx 220 --n-legit 100 --seed $S1
  retry $MG --name $N-attack -- $PY -m geofence.attack --city austin --n-sleepers 100 --n-benign 100 --seed $S1
  retry $MG --name $N-attack-bj -- $PY -m geofence.attack --city beijing --n-sleepers 60 --n-benign 60 --seed $S1
fi

q() {  # q <config> <exp> <seed> <tag>
  stamp "qwen $4 $2 seed=$3"
  retry $MG --name $N-$4-$2-s$3 --ollama-models qwen2.5:7b -- $PY -m geofence.evaluate --config $1 --exp $2 --seed $3
}

# ---- P1 headline, seed 1
for E in asr ccs provgate; do q $FULL $E $S1 eval; done
stamp "content filters"
retry $MG --name $N-filters --ollama-models llama-guard3:8b -- $PY -m geofence.evaluate --config $FULL --exp filters
retry $MG --name $N-aggregate -- $PY -m geofence.aggregate --run full
# ---- P2 + P3, seed 1
for E in randomq shift loo osrm; do q $FULL $E $S1 eval; done
retry $MG --name $N-cost --ollama-models qwen2.5:7b -- $PY -m geofence.evaluate --config $FULL --exp cost --seed $S1
retry $MG --name $N-aggregate -- $PY -m geofence.aggregate --run full
# ---- seeds 2,3 for the headline claims
for S in $S2 $S3; do
  for E in asr ccs; do q $FULL $E $S eval; done
  retry $MG --name $N-aggregate -- $PY -m geofence.aggregate --run full
done
# ---- secondary: GeoLife Beijing (qwen)
for E in asr ccs; do q $BJ $E $S1 bj; done
retry $MG --name $N-bj-agg -- $PY -m geofence.aggregate --run full_beijing
# ---- secondary: Hammer2.1-7b tool-calling table (HF, one model at a time)
stamp "hammer asr"
retry $MG --name $N-hammer-asr-s$S1 -- $PY -m geofence.evaluate --config $HM --exp asr --seed $S1
retry $MG --name $N-aggregate-all -- $PY -m geofence.aggregate --run full full_beijing full_hammer
retry $MG --name $N-plots -- $PY -m geofence.plots --run full
stamp "FULL_OK"
