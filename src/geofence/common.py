"""Shared helpers: paths, deterministic seeding, JSONL checkpoint/resume, memguard scale."""
import json
import os
import random
import hashlib
import threading

_APPEND_LOCK = threading.Lock()

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
DERIVED_BASE = os.path.join(DATA, "derived")  # shared inputs (POIs) built by src/data_check.py
# Per-run derived sets (users/sleepers/benign/legit). The smoke test sets
# GEOFENCE_DERIVED=data/derived_smoke so it never overwrites the full-run inputs.
DERIVED = os.environ.get("GEOFENCE_DERIVED", DERIVED_BASE)
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
LOGS = os.path.join(ROOT, "logs")

POIS = {"austin": os.path.join(DERIVED_BASE, "pois_austin.jsonl"),
        "beijing": os.path.join(DERIVED_BASE, "pois_beijing.jsonl")}
GOWALLA = os.path.join(DATA, "raw", "gowalla", "loc-gowalla_totalCheckins.txt.gz")
GEOLIFE = os.path.join(DATA, "raw", "geolife", "Geolife Trajectories 1.3", "Data")
OSRM_HOST = os.environ.get("OSRM_HOST", "http://127.0.0.1:5000")

# City bounding boxes (match src/data_check.py).
BBOX = {"beijing": (39.75, 40.10, 116.15, 116.65),
        "austin": (30.10, 30.55, -97.95, -97.55)}


def memguard_scale():
    """Convenience multiplier in (0,1]; shrink batch/workers/n when < 1."""
    try:
        return float(os.environ.get("MEMGUARD_SCALE", "1.0"))
    except ValueError:
        return 1.0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def stable_int(*parts):
    """Deterministic 32-bit int from arbitrary string parts (seed derivation)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj) + "\n"
    with _APPEND_LOCK:
        with open(path, "a") as f:
            f.write(line)
            f.flush()


def write_jsonl(path, objs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")


def done_keys(path, key="id"):
    """Set of already-completed keys in a checkpoint JSONL (for resume)."""
    return {r.get(key) for r in read_jsonl(path) if key in r}


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)
