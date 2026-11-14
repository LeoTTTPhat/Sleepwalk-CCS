#!/bin/sh
# Stage-7 addendum (2026-10-17), <= 2 h guarded compute, idempotent and resumable.
#  (a) osrm2: the routing decisions of the 00:03 run are valid, but the OSRM server was not
#      running (run_rev1.sh never started it), so no route was resolved. The old file is
#      kept as results/rev1/osrm2_osrmdown.jsonl; the rerun hits the agent-call cache and
#      only adds the OSRM resolution (and checks that the decisions reproduce).
#  (b) victims: transfer to two further Ollama victims (and Qwen2.5-7B under the same
#      design) on the hammer2 design, with a trigger-failing control (reviewer R4.9).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="${MEMGUARD:-$P/tools/memguard.py}"
PY="${PY:-python}"
CFG=$P/src/configs/rev1.yaml
cd $P || exit 1
mkdir -p logs results/rev1
osrm_ok() { curl -s -m 3 "http://127.0.0.1:5000/route/v1/driving/-97.74,30.27;-97.75,30.28?overview=false" 2>/dev/null | grep -q '"code":"Ok"'; }

run() { # run <name> <exp> [model]
  name=$1; exp=$2; model=${3:-}
  if [ -f .memguard/STOP ] || [ -f .memguard/STOP-p3-rev1b ]; then
    echo "[rev1b] STOP file present, exiting"; exit 0
  fi
  om=${model:-qwen2.5:7b}
  i=0
  while [ $i -lt 4 ]; do
    i=$((i+1))
    echo "[rev1b] === $name (attempt $i) $(date +%H:%M) ==="
    if [ -n "$model" ]; then
      PYTHONPATH=$P/src "$GUARD" --name p3-geofenced-sleepers-memory-rev-$name \
        --ollama-models "$om" -- "$PY" -m geofence.revision --config $CFG --exp $exp --model "$model" && return 0
    else
      PYTHONPATH=$P/src "$GUARD" --name p3-geofenced-sleepers-memory-rev-$name \
        --ollama-models "$om" -- "$PY" -m geofence.revision --config $CFG --exp $exp && return 0
    fi
    echo "[rev1b] $name failed; retrying after cooldown"; sleep 90
  done
  echo "[rev1b] $name gave up"; return 1
}

# (a) OSRM resolution for osrm2
if [ ! -f results/rev1/osrm2_osrmdown.jsonl ] && [ -f results/rev1/osrm2.jsonl ]; then
  mv results/rev1/osrm2.jsonl results/rev1/osrm2_osrmdown.jsonl
fi
STARTED=0
if ! osrm_ok; then
  echo "[rev1b] starting OSRM"
  docker run -d --rm --name p3-osrm-rev1b -m 4g -p 5000:5000 -v "$P/data/osrm:/data" \
    ghcr.io/project-osrm/osrm-backend:v5.27.1 osrm-routed --algorithm mld /data/austin.osrm && STARTED=1
  for i in $(seq 1 30); do osrm_ok && break; sleep 2; done
fi
if osrm_ok; then
  run osrm2b osrm2
else
  echo "[rev1b] OSRM not reachable; osrm2 rerun skipped"
fi
[ $STARTED = 1 ] && docker stop p3-osrm-rev1b >/dev/null 2>&1

# (b) further victims, one model per guarded invocation
run victims-qwen7b victims qwen2.5:7b
run victims-llama8b victims llama3.1:8b-instruct-q4_K_M
run victims-qwen14b victims qwen2.5:14b-instruct-q4_K_M
echo "[rev1b] REV1B_OK $(date +%F' '%H:%M)"
