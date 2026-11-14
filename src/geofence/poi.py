"""POI index: load OSM POI extracts, nearest-POI lookup, coarse area labels."""
import math

import numpy as np

from . import common


def _deg_scale(lat0):
    # metres per degree at lat0 (approx) -> project lon/lat to a local plane
    return (111320.0 * math.cos(math.radians(lat0)), 110540.0)


class PoiIndex:
    def __init__(self, city="austin", max_pois=None):
        self.city = city
        rows = common.read_jsonl(common.POIS[city])
        rows = [r for r in rows if r.get("lat") is not None and r.get("name")]
        if max_pois:
            rows = rows[:max_pois]
        self.pois = rows
        self.lat = np.array([r["lat"] for r in rows], dtype=np.float64)
        self.lon = np.array([r["lon"] for r in rows], dtype=np.float64)
        self.lat0 = float(np.median(self.lat))
        self.sx, self.sy = _deg_scale(self.lat0)
        self.x = (self.lon - np.median(self.lon)) * self.sx
        self.y = (self.lat - self.lat0) * self.sy
        try:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(np.column_stack([self.x, self.y]))
        except Exception:
            self.tree = None
        self.by_id = {r["id"]: r for r in rows}
        # coarse ~2km grid for "area" labels
        self.cell = 2000.0
        self._area_cache = {}

    def _proj(self, lat, lon):
        return ((lon - np.median(self.lon)) * self.sx, (lat - self.lat0) * self.sy)

    def nearest(self, lat, lon, k=1):
        x, y = self._proj(lat, lon)
        if self.tree is not None:
            d, idx = self.tree.query([x, y], k=k)
            idx = np.atleast_1d(idx)
            return [self.pois[i] for i in idx]
        d2 = (self.x - x) ** 2 + (self.y - y) ** 2
        idx = np.argsort(d2)[:k]
        return [self.pois[i] for i in idx]

    def area_label(self, lat, lon):
        """Human-readable area: grid cell named after a prominent nearby POI."""
        x, y = self._proj(lat, lon)
        key = (int(x // self.cell), int(y // self.cell))
        if key in self._area_cache:
            return self._area_cache[key]
        # prominent POI = nearest named POI to the cell centre
        cx, cy = (key[0] + 0.5) * self.cell, (key[1] + 0.5) * self.cell
        if self.tree is not None:
            _, i = self.tree.query([cx, cy], k=1)
        else:
            i = int(np.argmin((self.x - cx) ** 2 + (self.y - cy) ** 2))
        name = self.pois[i]["name"]
        label = f"the {name} area"
        self._area_cache[key] = label
        return label

    def pois_in_area(self, area):
        """POIs whose area label equals `area` (built once, cached).

        matching_context_v2 asks for this repeatedly; scanning all POIs per call made
        context synthesis quadratic in the corpus size."""
        tbl = getattr(self, "_area_members", None)
        if tbl is None:
            tbl = {}
            for r in self.pois:
                tbl.setdefault(self.area_label(r["lat"], r["lon"]), []).append(r)
            self._area_members = tbl
        return tbl.get(area, [])

    def pois_in_category(self, cat):
        """POIs whose OSM category contains `cat` (built once, cached)."""
        tbl = getattr(self, "_cat_members", None)
        if tbl is None:
            tbl = {}
            for r in self.pois:
                c = str(r.get("category") or "")
                key = c.split("=")[-1] if "=" in c else c
                tbl.setdefault(key, []).append(r)
            self._cat_members = tbl
        if cat in tbl:
            return tbl[cat]
        out = [r for k, v in tbl.items() if cat and cat in k for r in v]
        tbl[cat] = out
        return out

    def conditional_pois(self):
        """POIs carrying opening_hours -> source of legitimate conditional facts."""
        return [r for r in self.pois if r.get("opening_hours")]

    def safety_critical(self):
        return [r for r in self.pois if r.get("safety_critical")]

    def sample_targets(self, rng, n):
        idx = rng.choice(len(self.pois), size=min(n, len(self.pois)), replace=False)
        return [self.pois[i] for i in idx]
