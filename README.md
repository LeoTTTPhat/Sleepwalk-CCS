# Sleepers and Counterfactual Context Sampling: reproduction files

A *sleeper* is a memory entry for a location-aware LLM agent whose instruction is conditioned on the user's place and
time. It reads like an ordinary local tip, and it is meant to act only inside its stated geofence. Counterfactual Context
Sampling (CCS) tests an entry before the agent stores it. CCS replays the agent on sampled user contexts and flags
induced actions outside the entry's stated scope, with a calibrated sequential test.

This folder holds the code and drivers, the configurations, the entry corpora in anonymised form, the public POI tables,
all result files and the figures.

## Layout

| Path | Contents |
|---|---|
| `src/geofence/` | Core package: agent and tools (`agent.py`, `poi.py`), context pools (`data_prep.py`), corpus generators (`attack.py`: corpus A, `attack2.py`: corpus B, `attack3.py`: corpus C), predicates and scopes (`predicate.py`, `scope.py`), CCS (`ccs.py`), baseline defenses (`defenses.py`), experiment drivers (`evaluate.py`, `revision.py`, `revision2.py`), statistics (`stats.py`), aggregation and plots |
| `src/*.py` | Analyses (`analyze.py`, `analyze_rev.py`, `analyze_rev2.py`), cache-exact replay of the CCS runs (`replay.py`), static text rules (`text_rules.py`), figures (`make_figures.py`, `make_figures_rev.py`), corpus anonymisation (`anonymize_release.py`), data checks |
| `run_*.sh`, `supervise.sh`, `gaps_chain.sh` | Experiment drivers; every step appends JSONL records with resume keys and skips finished work |
| `src/configs/` | One YAML configuration per run (Austin main matrix, Beijing and its recalibration, Hammer tool-calling, temperature ablation, follow-ups, smoke test) |
| `src/tests/` | Unit tests (65, plus 3 skipped) |
| `src/tools/osrm_prepare.sh` | Builds the OSRM routing graph for the Austin extract |
| `data/release/` | Entry corpora A, B and C (Austin) and A (Beijing): sleepers, benign entries and legitimate conditional facts, anonymised (see below) |
| `data/derived/pois_*.jsonl` | OpenStreetMap POI tables of Austin and Beijing |
| `data/SHA256SUMS.txt` | Checksums of the raw downloads we used |
| `results/` | `summary.json`, per-run metrics (`*_metrics.csv`), statistics (`analysis_stats.json`, `revision_stats.json`, `revision2_stats.json`, `replay_stats.json`, `text_rules.json`), per-run JSONL records, and the gap register `GAPS.md` |
| `figures/` | All figures |
| `figkit/` | Script for the icon figures |
| `tools/memguard.py` | Pass-through stand-in for the memory guard that wrapped each step on our shared machine |
| `requirements.txt` | Exact Python package versions used (Python 3.11) |

## Anonymised corpora and withheld files

The attack entries send users to, or share their data with, real places taken from OpenStreetMap. To avoid implicating
real businesses, this folder does not pair any real place with an attack payload:
- **Placeholders.** In `data/release/`, every attacker place name, map id and phone number is replaced by a placeholder
  (for example `Venue a-012`, `anon_poi_a_012`, `+1-555-0112`). Where the attacker place is the entry's own scope place,
  the scope fields get the placeholder too.
- **Scope places.** The place an entry legitimately describes keeps its real name.

For the same reason, four kinds of file are not included:
- **Original-name corpora** (`data/derived*/`). The drivers rebuild them from the public OpenStreetMap data with the
  fixed seeds in the configurations.
- **Six per-context CCS logs** (`results/rev1/ccs2*.jsonl`, `results/rev1/ccs3.jsonl`).
- **The agent-call cache** (`data/cache/agent_calls.sqlite`).
- **User context pools** (`data/derived/users_*.jsonl`). These are reshaped check-in histories of Gowalla and GeoLife
  users; the data-preparation step rebuilds them from the public downloads.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from this folder. The drivers read two optional environment variables:
- `PY`: the Python interpreter (default `python`).
- `MEMGUARD`: a wrapper called as `$MEMGUARD --name NAME [options] -- command` (default `tools/memguard.py`, which runs
  the command directly).

## 1. Check the results from the shipped files (CPU, under 1 minute)

```bash
PYTHONPATH=src python -m pytest -q src/tests
python src/analyze.py
python src/make_figures.py
PYTHONPATH=src python src/make_figures_rev.py
python figkit/p3_figures.py
```

`src/analyze.py` reproduces `results/analysis_stats.json` from the shipped results.

Four scripts need the withheld files: `src/text_rules.py`, `src/analyze_rev.py`, `src/analyze_rev2.py` and
`src/replay.py`. Run them only after a full re-run (section 2). On this folder alone, the first three overwrite
`results/text_rules.json`, `results/revision_stats.json` and `results/revision2_stats.json` with incomplete values.

## 2. Re-run the experiments

**Data.** Put the raw downloads under `data/raw/` and compare them with `data/SHA256SUMS.txt`:
- `data/raw/geolife/`: GeoLife 1.3,
  https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip,
  unpacked to `Geolife Trajectories 1.3/`.
- `data/raw/gowalla/`: `loc-gowalla_totalCheckins.txt.gz` and `loc-gowalla_edges.txt.gz` from SNAP,
  https://snap.stanford.edu/data/loc-gowalla.html.
- `data/raw/osm/`: `Austin.osm.pbf` and `Beijing.osm.pbf` from BBBike, https://download.bbbike.org. BBBike refreshes its
  extracts, so a new download can differ from the checksums listed.

**Models.**
- Ollama: `qwen2.5:7b` (main agent), `llama3.1:8b-instruct-q4_K_M` and `qwen2.5:14b` (judge and transfer runs),
  `llama-guard3:8b` (content filter).
- Hugging Face: `MadeAgents/Hammer2.1-7b` (tool-calling agent), `deepset/deberta-v3-base-injection` and
  `protectai/deberta-v3-base-prompt-injection-v2` (content filters).
- `meta-llama/Llama-Prompt-Guard-2-86M` is gated and was not run.

**Routing.** The routing experiments need an OSRM server on `127.0.0.1:5000`. To set it up:
1. Copy `data/raw/osm/Austin.osm.pbf` to `data/osrm/austin.osm.pbf`.
2. Run `sh src/tools/osrm_prepare.sh`. It builds the graph with Docker, using the image
   `ghcr.io/project-osrm/osrm-backend:v5.27.1`.
3. Serve the graph:

```bash
docker run --rm -p 5000:5000 -v "$PWD/data/osrm:/data" ghcr.io/project-osrm/osrm-backend:v5.27.1 osrm-routed --algorithm mld /data/austin.osrm
```

**Runs.**

```bash
bash run_smoke.sh      # tiny end-to-end run (builds its own small corpora)
bash supervise.sh      # main runs (run_full.sh: data preparation, corpus A, Austin, Beijing, Hammer, filters, cost),
                       # restarted until complete
bash run_gaps.sh       # remaining seeds and descoped items (gaps_chain.sh)
PYTHONPATH=src python -m geofence.attack2   # corpus B (data/derived_rev/, seed 20261017)
bash run_rev1.sh       # follow-up experiments on corpus B
bash run_rev1b.sh      # controlled routing loop with OSRM
bash run_rev2.sh; bash run_rev2b.sh   # corpus C (built inside run_rev2.sh), judges, retrieval, CCS variants, transfer
```

Then run, in this order:

```bash
PYTHONPATH=src python src/replay.py
PYTHONPATH=src python src/text_rules.py
PYTHONPATH=src python src/analyze_rev.py
PYTHONPATH=src python src/analyze_rev2.py
python src/analyze.py
python src/make_figures.py
PYTHONPATH=src python src/make_figures_rev.py
```

Agent calls are memoised in `data/cache/agent_calls.sqlite`, so an interrupted run resumes without repeating them.
`results/GAPS.md` lists, for every planned experiment, whether it ran and on how many seeds. We ran everything on one
Apple M3 Ultra (512 GB unified memory).

## Notes

- The POI tables and map extracts are © OpenStreetMap contributors, available under the Open Database License (ODbL).
- The icon set in `figkit/icons/` is Phosphor Icons v2.1.1 (MIT licence, `figkit/icons/LICENSE-phosphor-MIT.txt`).
