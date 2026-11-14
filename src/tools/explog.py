"""Markdown table of this project's Stage-3 memguard jobs (from .memguard state + logs).

Usage: python src/tools/explog.py [--since 'YYYY-MM-DD HH:MM']
Reads only the guard's own records; prints name, status, attempts, peak GiB, wall, end.
"""
import argparse, glob, json, os, re

MG = "../.memguard"
ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-10-15 11:20")
a = ap.parse_args()
rows = []
for st in glob.glob(os.path.join(MG, "state", "p3-geofenced-sleepers-memory-*.json")):
    try:
        j = json.load(open(st))
    except Exception:
        continue
    if j.get("updated", "") < a.since:
        continue
    log = st.replace("/state/", "/logs/").replace(".json", ".log")
    wall, start, breaches = None, None, 0
    if os.path.exists(log):
        txt = open(log, errors="ignore").read()
        m = re.findall(r"wall (\d+)s", txt)
        wall = int(m[-1]) if m else None
        s = re.search(r"\[memguard (\S+ \S+) [^\]]*\] started attempt 0", txt)
        start = s.group(1) if s else None
        breaches = len(re.findall(r"(?i)breach|killing", txt))
    rows.append((start or j.get("updated"), j["name"].replace("p3-geofenced-sleepers-memory-", ""),
                 j.get("status"), j.get("attempt"), j.get("peak_gb"), wall, j.get("updated"), breaches))
rows.sort()
print("| start | job | status | attempt | peak GiB | wall | last update | breach lines |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    w = f"{r[5]/3600:.2f} h" if r[5] else "-"
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {w} | {r[6]} | {r[7]} |")
