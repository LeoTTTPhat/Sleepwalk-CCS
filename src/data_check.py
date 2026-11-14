"""Basic statistics for the raw datasets of p3-geofenced-sleepers-memory, plus a light POI extract.

Run (guarded):
  shared/memguard.py --name p3-geofenced-sleepers-memory-datacheck -- \
      shared/.venv/bin/python src/data_check.py
Outputs:
  results/data_stats.json            dataset statistics (printed as well)
  data/derived/pois_<city>.jsonl     named POIs (id, name, name_en, category, lat, lon, opening_hours, tags subset)
Memory: streams every file; peak < 2 GB. Idempotent: POI extracts are skipped if present (use --force to redo).
"""
import collections, gzip, json, os, sys, time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DER = os.path.join(ROOT, "data", "derived")
os.makedirs(DER, exist_ok=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
FORCE = "--force" in sys.argv

# City bounding boxes (lat_min, lat_max, lon_min, lon_max)
BBOX = {"beijing": (39.75, 40.10, 116.15, 116.65),   # within 5th ring road approx.
        "austin": (30.10, 30.55, -97.95, -97.55)}
# Safety-critical POI classes used by the CCS exploration mixture
SAFETY = {("amenity", "hospital"), ("amenity", "clinic"), ("amenity", "doctors"), ("amenity", "pharmacy"),
          ("amenity", "police"), ("amenity", "fire_station"), ("amenity", "shelter"), ("healthcare", "hospital"),
          ("emergency", "yes"), ("amenity", "social_facility")}
POI_KEYS = ("amenity", "shop", "tourism", "leisure", "healthcare", "office", "public_transport", "railway", "emergency")


def inbox(lat, lon, b):
    return b[0] <= lat <= b[1] and b[2] <= lon <= b[3]


def geolife_stats():
    base = os.path.join(RAW, "geolife", "Geolife Trajectories 1.3", "Data")
    users = sorted(os.listdir(base))
    n_traj = n_pts = n_bj = 0
    tmin, tmax = "9999", "0000"
    pts_per_user = {}
    labeled = 0
    for u in users:
        tdir = os.path.join(base, u, "Trajectory")
        if os.path.exists(os.path.join(base, u, "labels.txt")):
            labeled += 1
        cnt = 0
        for f in os.listdir(tdir):
            if not f.endswith(".plt"):
                continue
            n_traj += 1
            with open(os.path.join(tdir, f)) as fh:
                for i, line in enumerate(fh):
                    if i < 6:
                        continue
                    p = line.rstrip().split(",")
                    if len(p) < 7:
                        continue
                    cnt += 1
                    if cnt % 50 == 1:  # sample 2% for bbox share and time range
                        lat, lon = float(p[0]), float(p[1])
                        n_bj += inbox(lat, lon, BBOX["beijing"])
                        d = p[5]
                        tmin, tmax = min(tmin, d), max(tmax, d)
        pts_per_user[u] = cnt
        n_pts += cnt
    sampled = sum((c + 49) // 50 for c in pts_per_user.values())
    heavy = sum(1 for c in pts_per_user.values() if c >= 100_000)
    return {"users": len(users), "trajectories": n_traj, "points": n_pts, "users_with_mode_labels": labeled,
            "share_points_in_beijing_bbox(sampled)": round(n_bj / max(sampled, 1), 4),
            "date_range": [tmin, tmax], "users_with_>=100k_points": heavy,
            "median_points_per_user": sorted(pts_per_user.values())[len(users) // 2]}


def gowalla_stats():
    fn = os.path.join(RAW, "gowalla", "loc-gowalla_totalCheckins.txt.gz")
    users, locs = set(), set()
    n = 0
    tmin, tmax = "9999", "0000"
    city = {c: {"checkins": 0, "users": collections.Counter(), "locs": set()} for c in BBOX}
    with gzip.open(fn, "rt") as fh:
        for line in fh:
            p = line.rstrip().split("\t")
            if len(p) != 5:
                continue
            u, t, lat, lon, l = p
            lat, lon = float(lat), float(lon)
            n += 1
            users.add(u); locs.add(l)
            d = t[:10]
            tmin, tmax = min(tmin, d), max(tmax, d)
            for c, b in BBOX.items():
                if inbox(lat, lon, b):
                    city[c]["checkins"] += 1; city[c]["users"][u] += 1; city[c]["locs"].add(l)
    edges = sum(1 for _ in gzip.open(os.path.join(RAW, "gowalla", "loc-gowalla_edges.txt.gz"), "rt"))
    out = {"checkins": n, "users": len(users), "locations": len(locs), "date_range": [tmin, tmax],
           "friendship_edges_directed": edges}
    for c, s in city.items():
        out[f"{c}_bbox"] = {"checkins": s["checkins"], "users": len(s["users"]), "locations": len(s["locs"]),
                            "users_with_>=50_checkins": sum(1 for v in s["users"].values() if v >= 50),
                            "users_with_>=100_checkins": sum(1 for v in s["users"].values() if v >= 100)}
    return out


def osm_stats(city):
    import osmium
    pbf = os.path.join(RAW, "osm", f"{city.capitalize()}.osm.pbf")
    outp = os.path.join(DER, f"pois_{city}.jsonl")
    write = FORCE or not os.path.exists(outp)
    st = {"pois_named": 0, "pois_with_opening_hours": 0, "safety_critical": collections.Counter(),
          "top_categories": collections.Counter(), "road_ways": 0, "pois_with_name_en": 0}
    fo = open(outp + ".tmp", "w") if write else None
    fp = osmium.FileProcessor(pbf).with_locations().with_areas()
    for obj in fp:
        tags = obj.tags
        if obj.is_way() and "highway" in tags:
            st["road_ways"] += 1
        if obj.is_way():
            continue  # closed ways are handled as areas below
        cat = next(((k, tags[k]) for k in POI_KEYS if k in tags), None)
        if cat is None or "name" not in tags:
            continue
        try:
            if obj.is_node():
                lat, lon = obj.location.lat, obj.location.lon
                oid = f"n{obj.id}"
            elif obj.is_area():
                ring = next(iter(obj.outer_rings()), None)
                if ring is None:
                    continue
                xs = [(n.lat, n.lon) for n in ring]
                lat, lon = sum(x[0] for x in xs) / len(xs), sum(x[1] for x in xs) / len(xs)
                oid = f"a{obj.orig_id()}"
            else:
                continue
        except Exception:
            continue
        st["pois_named"] += 1
        st["top_categories"][f"{cat[0]}={cat[1]}"] += 1
        oh = tags.get("opening_hours")
        st["pois_with_opening_hours"] += oh is not None
        st["pois_with_name_en"] += "name:en" in tags
        safe = any((k, tags.get(k)) in SAFETY for k in ("amenity", "healthcare", "emergency"))
        if safe:
            st["safety_critical"][f"{cat[0]}={cat[1]}"] += 1
        if fo:
            fo.write(json.dumps({"id": oid, "name": tags["name"], "name_en": tags.get("name:en"),
                                 "category": f"{cat[0]}={cat[1]}", "lat": round(lat, 6), "lon": round(lon, 6),
                                 "opening_hours": oh, "safety_critical": safe,
                                 "addr_street": tags.get("addr:street"), "cuisine": tags.get("cuisine"),
                                 "wheelchair": tags.get("wheelchair")}, ensure_ascii=False) + "\n")
    if fo:
        fo.close(); os.replace(outp + ".tmp", outp)
    st["safety_critical_total"] = sum(st["safety_critical"].values())
    st["safety_critical"] = dict(st["safety_critical"].most_common())
    st["top_categories"] = dict(st["top_categories"].most_common(15))
    st["poi_extract"] = os.path.relpath(outp, ROOT)
    return st


def main():
    t0 = time.time()
    stats = {"generated": datetime.now().isoformat(timespec="seconds")}
    print("== GeoLife =="); stats["geolife"] = geolife_stats(); print(json.dumps(stats["geolife"], indent=1))
    print("== Gowalla =="); stats["gowalla"] = gowalla_stats(); print(json.dumps(stats["gowalla"], indent=1))
    for c in ("beijing", "austin"):
        print(f"== OSM {c} =="); stats[f"osm_{c}"] = osm_stats(c); print(json.dumps(stats[f"osm_{c}"], indent=1, ensure_ascii=False))
    stats["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(ROOT, "results", "data_stats.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote results/data_stats.json in", stats["elapsed_s"], "s")


if __name__ == "__main__":
    main()
