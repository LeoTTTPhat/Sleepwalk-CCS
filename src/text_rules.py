"""Static text-only baselines on both entry corpora (no model calls).

Reviewers asked for text-only baselines that the submitted paper omitted: a one-word
rule, a conditional-plus-imperative rule, and our own confirmation regex used alone as a
detector.  All three are pure functions of the entry text, so they are computed here
rather than in the guarded experiment chain.  The LLM judge (one model call per entry)
is produced by geofence.revision --exp judge and merged in analyze_rev.py.

Writes results/text_rules.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geofence import common                                        # noqa: E402
from geofence.revision import cond_rule, kw_when                   # noqa: E402
from geofence.stats import wilson_ci                               # noqa: E402
from geofence.triage import confirm_text                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def rate(k, n):
    ph, lo, hi = wilson_ci(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": ph, "lo": lo, "hi": hi}


def load(dirname, city="austin"):
    d = os.path.join(ROOT, "data", dirname)
    return {
        "sleeper": [e for e in common.read_jsonl(os.path.join(d, f"sleepers_{city}.jsonl"))
                    if e.get("split_role") != "condition"],
        "benign": common.read_jsonl(os.path.join(d, f"benign_{city}.jsonl")),
        "legit": common.read_jsonl(os.path.join(d, f"legit_{city}.jsonl")),
    }


# --------------------------- static scope-mismatch detector --------------------------
# Reviewer (rigor R1.9): "a static scope-mismatch check: every sleeper names an off-scope
# POI or phone number, and no null entry does".  This is that detector, built from the
# defender's own POI index: flag an entry whose text names a POI (or a contact number)
# other than the place the entry declares as its subject.  It is the strongest text-only
# baseline we can construct, and on the diverse corpus it is also the one that exposes why
# the heuristic is not safe: benign notes legitimately name other places.
import re as _re

_PHONE = _re.compile(r"\+?\d[\d\-\s().]{6,}\d|[\w.+-]+@[\w.-]+")
_NAME_INDEX = {}


def _name_index(city="austin"):
    if city not in _NAME_INDEX:
        from geofence.poi import PoiIndex
        idx = PoiIndex(city)
        names = {}
        for r in idx.pois:
            nm = str(r.get("name") or "").strip().lower()
            if len(nm) >= 8 and " " in nm:      # multi-word, unambiguous names only
                names[nm] = r["id"]
        _NAME_INDEX[city] = names
    return _NAME_INDEX[city]


def scope_mismatch(entry, city="austin"):
    """1 iff the text names a contact number, or a POI other than its declared subject."""
    txt = str(entry.get("text") or "").lower()
    if _PHONE.search(txt):
        return 1
    own = {str(entry.get(k) or "").strip().lower()
           for k in ("scope_place", "place", "scope_area", "area")}
    for nm, _pid in _name_index(city).items():
        if nm in txt and nm not in own:
            return 1
    return 0


RULES = {"kw_when": kw_when, "cond_rule": cond_rule, "confirm_regex": confirm_text}


def main():
    out = {}
    for tag, dirname in (("v1", "derived"), ("v2", "derived_rev")):
        sets = load(dirname)
        block = {"n": {k: len(v) for k, v in sets.items()}, "detectors": {}}
        for name, fn in RULES.items():
            tp = sum(fn(e["text"]) for e in sets["sleeper"])
            fb = sum(fn(e["text"]) for e in sets["benign"])
            fl = sum(fn(e["text"]) for e in sets["legit"])
            nn = len(sets["benign"]) + len(sets["legit"])
            se = tp / max(1, len(sets["sleeper"]))
            sp = 1 - (fb + fl) / max(1, nn)
            block["detectors"][name] = {
                "sleeper_flagged": rate(tp, len(sets["sleeper"])),
                "benign_flagged": rate(fb, len(sets["benign"])),
                "legit_flagged": rate(fl, len(sets["legit"])),
                "null_flagged": rate(fb + fl, nn),
                "youden_j": se + sp - 1}
            # per-variant / per-class detail
            if tag == "v2":
                block["detectors"][name]["by_variant"] = {
                    v: rate(sum(fn(e["text"]) for e in sets["sleeper"]
                                if e.get("variant") == v),
                            sum(1 for e in sets["sleeper"] if e.get("variant") == v))
                    for v in sorted({e.get("variant") for e in sets["sleeper"]})}
                block["detectors"][name]["by_legit_class"] = {
                    c: rate(sum(fn(e["text"]) for e in sets["legit"]
                                if e.get("legit_class") == c),
                            sum(1 for e in sets["legit"] if e.get("legit_class") == c))
                    for c in sorted({e.get("legit_class") for e in sets["legit"]})}
        # the scope-mismatch detector needs the whole entry, not just its text
        tp = sum(scope_mismatch(e) for e in sets["sleeper"])
        fb = sum(scope_mismatch(e) for e in sets["benign"])
        fl = sum(scope_mismatch(e) for e in sets["legit"])
        nn = len(sets["benign"]) + len(sets["legit"])
        se = tp / max(1, len(sets["sleeper"]))
        sp = 1 - (fb + fl) / max(1, nn)
        block["detectors"]["scope_mismatch"] = {
            "sleeper_flagged": rate(tp, len(sets["sleeper"])),
            "benign_flagged": rate(fb, len(sets["benign"])),
            "legit_flagged": rate(fl, len(sets["legit"])),
            "null_flagged": rate(fb + fl, nn),
            "youden_j": se + sp - 1}
        if tag == "v2":
            block["detectors"]["scope_mismatch"]["by_legit_class"] = {
                c: rate(sum(scope_mismatch(e) for e in sets["legit"]
                            if e.get("legit_class") == c),
                        sum(1 for e in sets["legit"] if e.get("legit_class") == c))
                for c in sorted({e.get("legit_class") for e in sets["legit"]})}
        out[tag] = block
    # Corpus C (round-3 audit): sleepers only; its null is Corpus B's
    cpath = os.path.join(ROOT, "data", "derived_c", "sleepers_austin.jsonl")
    if os.path.exists(cpath):
        cs = common.read_jsonl(cpath)
        fns = dict(RULES)
        fns["scope_mismatch"] = None
        out["v3"] = {"n": {v: sum(1 for e in cs if e.get("variant") == v)
                           for v in sorted({e.get("variant") for e in cs})},
                     "null": "corpus B (v2) null rates apply",
                     "detectors": {}}
        for name, fn in fns.items():
            f = (lambda e: scope_mismatch(e)) if fn is None else (lambda e, fn=fn: fn(e["text"]))
            out["v3"]["detectors"][name] = {
                v: rate(sum(f(e) for e in cs if e.get("variant") == v),
                        sum(1 for e in cs if e.get("variant") == v))
                for v in out["v3"]["n"]}
    common.dump_json(os.path.join(RES, "text_rules.json"), out)
    for tag, b in out.items():
        if tag == "v3":
            print(tag, {k: {v: d["k"] for v, d in x.items()}
                        for k, x in b["detectors"].items()})
            continue
        print(tag, b["n"], {k: (v["sleeper_flagged"]["k"], v["null_flagged"]["k"],
                                round(v["youden_j"], 2))
                            for k, v in b["detectors"].items()})


if __name__ == "__main__":
    main()
