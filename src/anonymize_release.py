"""Build the release copies of the entry corpora with attacker identities removed.

Round-1 ethics point: the submitted release named real Austin businesses as the
destinations of reroute / location-disclosure / deposit payloads, and carried
512-area-code phone numbers.  The attacker POI is drawn uniformly from the OSM extract
and carries no meaning, so naming it implicates a real business for nothing.  This script
replaces every attacker place name and identifier with a synthetic placeholder and every
recipient with a reserved 555 test number, inside the entry text as well as the metadata.
The scope POI -- the place the entry legitimately describes -- keeps its real name,
because the attack is about writing a review of a real place.  The exception is an entry
whose attacker place IS its scope place (the self-describing variant of corpus B): there
the scope fields would name the real place next to the payload, so they get the entry's
placeholder too.

Writes data/release/{sleepers,benign,legit}_<city>_corpus<tag>.jsonl for corpora A
(data/derived), B (data/derived_rev) and C (data/derived_c), for Austin and Beijing where
the corpus exists, and fails if any released entry still carries its own original
attacker name, map id or a non-555 recipient anywhere in the record.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geofence import common                                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA = (("a", "derived"), ("b", "derived_rev"), ("c", "derived_c"))
PHONE = re.compile(r"\+?1?[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}")


def anonymise(rows, tag):
    out = []
    for i, e in enumerate(rows):
        e = dict(e)
        fake_place = f"Venue {tag}-{i:03d}"
        fake_id = f"anon_poi_{tag}_{i:03d}"
        fake_rec = f"+1-555-01{i % 100:02d}"
        self_describing = bool(e.get("attacker_poi")) and e.get("attacker_poi") == e.get("scope_poi")
        old_scope_poi = e.get("scope_poi")
        txt = e.get("text", "")
        for old in (e.get("attacker_place"), e.get("attacker_poi")):
            if old:
                txt = txt.replace(str(old), fake_place)
        if e.get("recipient"):
            txt = txt.replace(str(e["recipient"]), fake_rec)
        txt = PHONE.sub(fake_rec, txt)
        if e.get("attacker_place"):
            e["attacker_place"] = fake_place
        if e.get("attacker_poi"):
            e["attacker_poi"] = fake_id
        if e.get("recipient"):
            e["recipient"] = fake_rec
        if self_describing:
            e["scope_poi"], e["scope_place"] = fake_id, fake_place
            scope = dict(e.get("scope") or {})
            scope["targets"] = [fake_id if t == old_scope_poi else t for t in scope.get("targets", [])]
            e["scope"] = scope
        e["text"] = txt
        e["anonymised"] = True
        out.append(e)
    return out


def leaks(original_rows, released_rows):
    """(id, field, value) for every released entry that still contains its own original attacker place name, map id
    or a recipient outside the reserved 555 range, anywhere in the record."""
    orig = {o["id"]: o for o in original_rows}
    found = []
    for r in released_rows:
        o = orig[r["id"]]
        blob = json.dumps(r, ensure_ascii=False)
        for field in ("attacker_place", "attacker_poi", "recipient"):
            v = str(o.get(field) or "")
            if len(v) < 4 or (field == "recipient" and "555" in v):
                continue
            if v in blob:
                found.append((r["id"], field, v))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="both", choices=["austin", "beijing", "both"])
    args = ap.parse_args()
    cities = ("austin", "beijing") if args.city == "both" else (args.city,)
    outdir = os.path.join(ROOT, "data", "release")
    os.makedirs(outdir, exist_ok=True)
    total, bad, leaked = 0, 0, []
    for city in cities:
        for tag, src in CORPORA:
            for kind in ("sleepers", "benign", "legit"):
                p = os.path.join(ROOT, "data", src, f"{kind}_{city}.jsonl")
                if not os.path.exists(p):
                    continue
                original = common.read_jsonl(p)
                rows = anonymise(original, tag)
                common.write_jsonl(os.path.join(outdir, f"{kind}_{city}_corpus{tag}.jsonl"), rows)
                total += len(rows)
                bad += sum(1 for r in rows if re.search(r"512[-\s]?555", r.get("text", "")))
                leaked += [(f"{kind}_{city}_corpus{tag}",) + x for x in leaks(original, rows)]
    for x in leaked:
        print("LEAK", *x)
    print(f"release: {total} entries -> {outdir}; residual 512 numbers: {bad}; leaked identities: {len(leaked)}")
    if bad or leaked:
        sys.exit(1)


if __name__ == "__main__":
    main()
