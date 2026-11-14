#!/bin/sh
# Stage-8 (round-2 revision) chain.  Budget <= 4 h guarded; idempotent and resumable
# (every step appends JSONL with resume keys and skips finished ones).
#
# Priority order, so an interrupted chain still leaves the most important answers done:
#   1 corpusC     build corpus C (no model calls): closure mimics + out-of-pool
#                 request-conditional sleepers, with the disjointness check asserted.
#   2 judge2      neutral judge prompt on corpora A/B/C, two judge models
#                 (validity major 6, rigor major 8: the round-1 judge prompt exempted
#                 exactly corpus B's null classes).
#   3 asr3        corpus C attack arm, full-predicate verification.
#   4 retrieval   end-to-end arm with a real retriever over a 50-entry store
#                 (rigor major 7).
#   5 ccs3        CCS on corpus C at n_max=40 (rigor majors 6 and 8).
#   6 ccs_n40     corpus B CCS at n_max=40 (validity major 4: the deployed n_max=16 has
#                 exact power 0.012 at its own alternative).
#   7 ccs_uniform_n40  the Q-vs-uniform arm at the same budget.
#
# A wall-clock cap stops the chain before it can exceed the stage's compute allowance;
# steps left undone are reported and the offline analysis covers them as bounds.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="${MEMGUARD:-$P/tools/memguard.py}"
PY="${PY:-python}"
CFG=$P/src/configs/rev2.yaml
CAP_S=${CAP_S:-14400}          # 4 h hard cap on this chain
T0=$(date +%s)
cd $P || exit 1
mkdir -p logs results/rev1

left() { echo $(( CAP_S - ($(date +%s) - T0) )); }

run() { # run <name> <exp> [model]
  name=$1; exp=$2; model=${3:-}
  if [ -f .memguard/STOP ] || [ -f .memguard/STOP-p3-rev2 ]; then
    echo "[rev2] STOP file present, exiting"; exit 0
  fi
  if [ "$(left)" -le 0 ]; then
    echo "[rev2] wall-clock cap reached; skipping $name and the rest"; return 2
  fi
  om=${model:-qwen2.5:7b}
  i=0
  while [ $i -lt 4 ]; do
    i=$((i+1))
    echo "[rev2] === $name (attempt $i) $(date +%H:%M), $(left)s of budget left ==="
    if [ -n "$model" ]; then
      PYTHONPATH=$P/src "$GUARD" --name p3-geofenced-sleepers-memory-rev2-$name \
        --ollama-models "$om" -- "$PY" -m geofence.revision2 \
        --config $CFG --exp $exp --model "$model" && return 0
    else
      PYTHONPATH=$P/src "$GUARD" --name p3-geofenced-sleepers-memory-rev2-$name \
        --ollama-models "$om" -- "$PY" -m geofence.revision2 \
        --config $CFG --exp $exp && return 0
    fi
    echo "[rev2] $name failed; retrying after cooldown"; sleep 90
  done
  echo "[rev2] $name gave up"; return 1
}

# 1 corpus C (deterministic, no model calls; refuses to write if the request conjunct is
#   not disjoint from every probe request)
if [ ! -f data/derived_c/sleepers_austin.jsonl ]; then
  PYTHONPATH=$P/src "$GUARD" --name p3-geofenced-sleepers-memory-rev2-corpusC \
    -- "$PY" -m geofence.attack3 || exit 1
fi

# 2 judge: neutral prompt, two models (cheap: one short call per entry)
run judge2-qwen   judge2 qwen2.5:7b
run judge2-llama  judge2 llama3.1:8b-instruct-q4_K_M

# 2b the round-1 judge prompt on corpus C (76 short calls)
run judge-r1c-qwen  judge_round1_c qwen2.5:7b
run judge-r1c-llama judge_round1_c llama3.1:8b-instruct-q4_K_M

# 3-4 corpus C attack arm and the end-to-end retrieval arm
run asr3      asr3
run retrieval retrieval

# 5-7 the n_max=40 budget (the first 16 contexts of each stream are cache hits)
run ccs3             ccs3
run ccs-n40          ccs_n40
run ccs-uniform-n40  ccs_uniform_n40

echo "[rev2] REV2_DONE $(date +%F' '%H:%M) after $(( ($(date +%s) - T0) / 60 )) min"
for f in judge2 asr3 retrieval ccs3 ccs2_n40 ccs2_uniform_n40; do
  n=$( [ -f results/rev1/$f.jsonl ] && wc -l < results/rev1/$f.jsonl || echo 0 )
  echo "[rev2]   $f.jsonl rows=$n"
done
