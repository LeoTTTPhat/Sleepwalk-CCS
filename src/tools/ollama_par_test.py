"""Throughput probe: sequential vs concurrent Ollama calls (decides worker count)."""
import time, sys
from concurrent.futures import ThreadPoolExecutor
from geofence.agent import OllamaAgent
a = OllamaAgent()
ctx = {"user": "t", "city": "austin", "poi_id": "x", "place": "Test Cafe",
       "category": "amenity=cafe", "area": "the Test area", "dow": 1, "hour": 12,
       "tod": "midday", "lat": 30.27, "lon": -97.74, "recent": []}
qs = [f"What should I do near here? ({i})" for i in range(16)]
a.act(ctx, [], qs[0])  # warm
t = time.time()
for q in qs[:6]: a.act(ctx, [], q)
seq = (time.time() - t) / 6
print(f"sequential {seq:.2f}s/call", flush=True)
for w in (2, 4, 8):
    t = time.time()
    with ThreadPoolExecutor(w) as ex:
        list(ex.map(lambda q: a.act(ctx, [], q), qs))
    print(f"workers={w}: {(time.time()-t)/len(qs):.2f}s/call effective", flush=True)
