#!/bin/sh
# Stage-8 chain, part 2 (reordered after measuring the real call rate under contention).
#
# run_rev2.sh assumed ~1.3 s/call (the rev1 rate with a free second slot); with the other
# slot busy the measured rate is ~2.6 s/call, which would have let the wall-clock cap cut
# off the two most important steps.  This part runs the remaining work in reviewer-weight
# order and sizes the n_max=40 corpus-B run as a PRE-SPECIFIED stratified subsample
# (n40_sleepers in rev2.yaml), so a run stopped by the cap is still a complete experiment
# on a subset fixed in advance.
#
#   1 retrieval        end-to-end arm with a real retriever (rigor major 7)
#   2 ccs3             corpus C at n_max=40: closure mimics + the corrected out-of-pool
#                      request-conditional class (rigor majors 6 and 8)
#   3 ccs-n40          corpus B at n_max=40, stratified subsample + the full held-out
#                      null set (validity major 4)
#   4 judge-r1c-*      the round-1 judge prompt on corpus C (76 short calls)
#   5 ccs-uniform-n40  the Q-vs-uniform arm at n_max=40 (lowest priority: the n_max=16
#                      paired comparison and the exact OC already cover the claim)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="${MEMGUARD:-$P/tools/memguard.py}"
PY="${PY:-python}"
CFG=$P/src/configs/rev2.yaml
CAP_S=${CAP_S:-10800}          # 3 h for this part (Stage-8 total stays under 4 h)
T0=$(date +%s)
cd $P || exit 1
mkdir -p logs results/rev1 .memguard

left() { echo $(( CAP_S - ($(date +%s) - T0) )); }

run() { # run <name> <exp> [model]
  name=$1; exp=$2; model=${3:-}
  if [ -f $ROOT/.memguard/STOP ] || [ -f .memguard/STOP-p3-rev2 ]; then
    echo "[rev2b] STOP file present, exiting"; exit 0
  fi
  if [ "$(left)" -le 0 ]; then
    echo "[rev2b] wall-clock cap reached; skipping $name and the rest"; return 2
  fi
  om=${model:-qwen2.5:7b}
  i=0
  while [ $i -lt 4 ]; do
    i=$((i+1))
    echo "[rev2b] === $name (attempt $i) $(date +%H:%M), $(left)s of budget left ==="
    if [ -n "$model" ]; then
      PYTHONPATH=$P/src "$GUARD" --name p3-geofenced-sleepers-memory-rev2-$name \
        --ollama-models "$om" -- "$PY" -m geofence.revision2 \
        --config $CFG --exp $exp --model "$model" && return 0
    else
      PYTHONPATH=$P/src "$GUARD" --name p3-geofenced-sleepers-memory-rev2-$name \
        --ollama-models "$om" -- "$PY" -m geofence.revision2 \
        --config $CFG --exp $exp && return 0
    fi
    echo "[rev2b] $name failed; retrying after cooldown"; sleep 90
  done
  echo "[rev2b] $name gave up"; return 1
}

run retrieval        retrieval
run ccs3             ccs3
run ccs-n40          ccs_n40
run judge-r1c-qwen   judge_round1_c qwen2.5:7b
run judge-r1c-llama  judge_round1_c llama3.1:8b-instruct-q4_K_M
run ccs-uniform-n40  ccs_uniform_n40

echo "[rev2b] REV2B_DONE $(date +%F' '%H:%M) after $(( ($(date +%s) - T0) / 60 )) min"
for f in judge2 asr3 retrieval ccs3 ccs2_n40 ccs2_uniform_n40; do
  n=$( [ -f results/rev1/$f.jsonl ] && wc -l < results/rev1/$f.jsonl || echo 0 )
  echo "[rev2b]   $f.jsonl rows=$n"
done
