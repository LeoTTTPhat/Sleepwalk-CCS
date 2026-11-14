"""Build user mobility distributions Q and context pools from real trajectories.

Primary:  Gowalla Austin  (users with >=100 check-ins in the Austin bbox).
Secondary: GeoLife Beijing (users with enough Beijing-bbox points).

A *context* c rendered for the agent/predicates is:
  {user, city, poi_id, place, area, dow, hour, tod, lat, lon, recent[list of names]}
The user's empirical set of training contexts IS the proposal Q over (place x time);
CCS samples from it (plus an exploration mixture built in ccs.py). We keep a 70/30
temporal split per user for the distribution-shift experiment (triggers placed in the
held-out last-30% region are 'low-Q' to the detector that only sees the first 70%).

Checkpointed per user to data/derived/users_<city>.jsonl (resume skips done users).
"""
import argparse
import datetime
import gzip
import os
import collections

import numpy as np

from . import common
from .poi import PoiIndex

TOD = [(0, 6, "early morning"), (6, 11, "morning"), (11, 14, "midday"),
       (14, 18, "afternoon"), (18, 22, "evening"), (22, 24, "night")]


def tod_label(hour):
    for a, b, lab in TOD:
        if a <= hour < b:
            return lab
    return "night"


def _context(idx, user, city, lat, lon, dt, recent):
    near = idx.nearest(lat, lon, k=1)[0]
    return {"user": user, "city": city, "poi_id": near["id"], "place": near["name"],
            "category": near.get("category"), "area": idx.area_label(lat, lon),
            "dow": dt.weekday(), "hour": dt.hour, "tod": tod_label(dt.hour),
            "lat": round(lat, 6), "lon": round(lon, 6), "recent": recent}


def gowalla_austin(idx, n_users, max_ctx, seed):
    b = common.BBOX["austin"]
    per_user = collections.defaultdict(list)
    with gzip.open(common.GOWALLA, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            u, ts, la, lo = p[0], p[1], float(p[2]), float(p[3])
            if b[0] < la < b[1] and b[2] < lo < b[3]:
                per_user[u].append((ts, la, lo))
    users = [(u, len(v)) for u, v in per_user.items() if len(v) >= 100]
    users.sort(key=lambda t: (-t[1], t[0]))
    users = users[:n_users]
    out = []
    for u, _ in users:
        chk = sorted(per_user[u], key=lambda r: r[0])  # ascending time
        ctxs = []
        recent = []
        for ts, la, lo in chk:
            try:
                dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ") - datetime.timedelta(hours=6)
            except ValueError:
                continue
            c = _context(idx, u, "austin", la, lo, dt, list(recent[-3:]))
            ctxs.append(c)
            recent.append({"place": c["place"], "category": c["category"]})
        if max_ctx and len(ctxs) > max_ctx:
            # keep temporal order; subsample evenly
            sel = np.linspace(0, len(ctxs) - 1, max_ctx).astype(int)
            ctxs = [ctxs[i] for i in sel]
        split = int(0.7 * len(ctxs))
        out.append({"id": u, "city": "austin", "n_checkins": len(chk),
                    "train": ctxs[:split], "test": ctxs[split:]})
    return out


def geolife_beijing(idx, n_users, max_ctx, seed):
    b = common.BBOX["beijing"]
    root = common.GEOLIFE
    out = []
    if not os.path.isdir(root):
        return out
    udirs = sorted(os.listdir(root))
    picked = 0
    for ud in udirs:
        if picked >= n_users:
            break
        tdir = os.path.join(root, ud, "Trajectory")
        if not os.path.isdir(tdir):
            continue
        pts = []
        for plt in sorted(os.listdir(tdir))[:50]:
            with open(os.path.join(tdir, plt)) as fh:
                for i, line in enumerate(fh):
                    if i < 6:
                        continue
                    f = line.strip().split(",")
                    if len(f) < 7:
                        continue
                    la, lo = float(f[0]), float(f[1])
                    if not (b[0] < la < b[1] and b[2] < lo < b[3]):
                        continue
                    try:
                        dt = datetime.datetime.strptime(f[5] + " " + f[6], "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=8)
                    except ValueError:
                        continue
                    pts.append((dt, la, lo))
        if len(pts) < 100:
            continue
        pts.sort(key=lambda r: r[0])
        # thin to roughly hourly to approximate visited places, not GPS stream
        thinned, last = [], None
        for dt, la, lo in pts:
            if last is None or (dt - last).total_seconds() >= 1800:
                thinned.append((dt, la, lo))
                last = dt
        if len(thinned) < 60:
            continue
        ctxs, recent = [], []
        for dt, la, lo in thinned:
            c = _context(idx, ud, "beijing", la, lo, dt, list(recent[-3:]))
            ctxs.append(c)
            recent.append({"place": c["place"], "category": c["category"]})
        if max_ctx and len(ctxs) > max_ctx:
            sel = np.linspace(0, len(ctxs) - 1, max_ctx).astype(int)
            ctxs = [ctxs[i] for i in sel]
        split = int(0.7 * len(ctxs))
        out.append({"id": ud, "city": "beijing", "n_checkins": len(thinned),
                    "train": ctxs[:split], "test": ctxs[split:]})
        picked += 1
    return out


def build_legit_conditionals(idx, n, seed):
    """Legitimate conditional facts (untrusted but benign) from real opening_hours."""
    rng = np.random.default_rng(seed)
    pool = idx.conditional_pois()
    rng.shuffle(pool)
    out = []
    for r in pool[:n]:
        out.append({"id": f"legit_{r['id']}", "kind": "legit_conditional",
                    "poi_id": r["id"], "place": r["name"], "area": idx.area_label(r["lat"], r["lon"]),
                    "category": r.get("category"), "opening_hours": r["opening_hours"],
                    "provenance": "tool:poi_hours",
                    "text": f"{r['name']} opening hours: {r['opening_hours']}."})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="austin", choices=["austin", "beijing", "both"])
    ap.add_argument("--n-users", type=int, default=20)
    ap.add_argument("--max-ctx", type=int, default=220)
    ap.add_argument("--n-legit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()
    scale = common.memguard_scale()
    n_users = max(1, int(args.n_users * scale))
    common.seed_everything(args.seed)

    cities = ["austin", "beijing"] if args.city == "both" else [args.city]
    for city in cities:
        idx = PoiIndex(city)
        out_path = os.path.join(common.DERIVED, f"users_{city}.jsonl")
        done = common.done_keys(out_path, "id")
        if city == "austin":
            users = gowalla_austin(idx, n_users, args.max_ctx, args.seed)
        else:
            users = geolife_beijing(idx, n_users, args.max_ctx, args.seed)
        for u in users:
            if u["id"] in done:
                continue
            common.append_jsonl(out_path, u)
        print(f"[{city}] users written: {len([u for u in users if u['id'] not in done])} "
              f"(total requested {n_users})")
        # legitimate conditionals from this city's POIs
        legit = build_legit_conditionals(idx, args.n_legit, args.seed + 7)
        common.write_jsonl(os.path.join(common.DERIVED, f"legit_{city}.jsonl"), legit)
        print(f"[{city}] legit conditionals: {len(legit)}")


if __name__ == "__main__":
    main()
