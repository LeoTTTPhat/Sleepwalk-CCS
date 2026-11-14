#!/bin/sh
# Stage-4 gap runs (see results/GAPS.md). Runs INSIDE one memguard invocation (run_gaps.sh)
# so the chain keeps its slot; every step is idempotent (JSONL resume keys + sqlite call
# cache), so a guard restart simply re-runs the chain and finished steps are skipped.
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
export PYTHONPATH=$P/src HF_HUB_OFFLINE=1
cd $P
FULL=$P/src/configs/full.yaml
S1=20261015
stamp() { echo "[gaps $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
retry() { n=0; while [ $n -lt 3 ]; do "$@" && return 0; n=$((n+1)); stamp "retry $n: $*"; sleep 15; done; stamp "FAIL: $*"; return 1; }
rc=0
stamp "1 pre-triage (embedding + classifier sweep inputs)"
retry $PY -m geofence.triage --run full --city austin || rc=1
# OSRM for the non-mocked loop (lightweight container, not a model job)
if ! curl -s -m 3 "http://127.0.0.1:5000/route/v1/driving/-97.74,30.27;-97.75,30.28?overview=false" 2>/dev/null | grep -q '"code":"Ok"'; then
  stamp "starting OSRM"
  docker run -d --rm -m 4g -p 5000:5000 -v "$P/data/osrm:/data" \
    ghcr.io/project-osrm/osrm-backend:v5.27.1 osrm-routed --algorithm mld /data/austin.osrm || true
  for i in $(seq 1 20); do
    curl -s -m 3 "http://127.0.0.1:5000/route/v1/driving/-97.74,30.27;-97.75,30.28?overview=false" 2>/dev/null | grep -q '"code":"Ok"' && break
    sleep 2
  done
fi
stamp "2 loo + osrm extended to all positives (seed 1)"
retry $PY -m geofence.evaluate --config $FULL --exp loo --seed $S1 || rc=1
retry $PY -m geofence.evaluate --config $FULL --exp osrm --seed $S1 || rc=1
stamp "3 Beijing CCS with disjoint, larger calibration"
retry $PY -m geofence.evaluate --config $P/src/configs/full_beijing_recal.yaml --exp asr --seed $S1 || rc=1
retry $PY -m geofence.evaluate --config $P/src/configs/full_beijing_recal.yaml --exp ccs --seed $S1 || rc=1
stamp "4 scope-granularity ablation (seed 1)"
for E in scope_none scope_tool; do retry $PY -m geofence.evaluate --config $FULL --exp $E --seed $S1 || rc=1; done
stamp "5 temperature 0.7 ablation"
retry $PY -m geofence.evaluate --config $P/src/configs/full_temp07.yaml --exp asr --seed $S1 || rc=1
retry $PY -m geofence.evaluate --config $P/src/configs/full_temp07.yaml --exp ccs --seed $S1 || rc=1
stamp "6 exploration weight 0 (pure Q) ablation"
retry $PY -m geofence.evaluate --config $FULL --exp explore0 --seed $S1 || rc=1
stamp "7 aggregate + plots"
retry $PY -m geofence.aggregate --run full full_beijing full_hammer full_beijing_recal full_temp07 || rc=1
retry $PY -m geofence.plots --run full || rc=1
if [ $rc -eq 0 ]; then stamp "GAPS_OK"; exit 0; fi
stamp "chain finished with failures (rc=$rc)"; exit 1
