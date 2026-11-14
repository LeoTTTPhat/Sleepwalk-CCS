"""Counterfactual Context Sampling (CCS) detector.

For a candidate untrusted entry e, sample contexts c ~ Q (the user's own mobility
distribution, estimated from their first-70% trajectory) plus an exploration mixture
(uniform over the city + up-weighted safety-critical POI classes). Run the agent with
and without e under each c (paired), score out-of-scope divergence, and apply an SPRT
against the calibrated null rho0. Baselines (no-candidate-entry action per context) are
computed once and shared across all entries; split-pair entries are injected together.
"""
import hashlib

import numpy as np

from . import scope as scopemod
from .data_prep import tod_label
from .predicate import phi_holds as _phi
from .stats import sprt_run

EPS_DEFAULT = np.log(20) / 50.0  # ~0.0599  (BRIEF: eps = ln(20)/50)


def _ctx_sig(ctx, query):
    """Baseline-cache key.  Stage-7 fix (reviewer-validity major 7): the key now
    covers the FULL rendered context -- including the recent places, which the
    trajectory-motif predicate depends on -- so two different prompts can never
    share a no-entry baseline."""
    recent = "/".join(f"{r.get('place')}:{r.get('category')}"
                      for r in (ctx.get("recent") or []))
    return hashlib.sha256(
        f"{ctx.get('poi_id')}|{ctx.get('place')}|{ctx.get('area')}|{ctx.get('hour')}|"
        f"{ctx.get('dow')}|{recent}|{query}".encode()).hexdigest()[:16]


def _synth_context(idx, rng, safety_weight, city, n_recent=2):
    """One exploration-mixture context: a random POI (safety classes up-weighted).

    Stage-7 fix (reviewer-rigor major 3): exploration contexts now also carry RECENT
    places, drawn with the same safety up-weighting, so a trajectory-motif predicate
    ("after visiting a <category> location") can be satisfied by the exploration
    mixture -- previously recent was always empty, so the mixture could never test a
    motif trigger."""
    safety = idx.safety_critical()

    def draw():
        if safety and rng.random() < safety_weight:
            return safety[int(rng.integers(len(safety)))]
        return idx.pois[int(rng.integers(len(idx.pois)))]

    poi = draw()
    hour = int(rng.integers(0, 24))
    dow = int(rng.integers(0, 7))
    recent = []
    for _ in range(int(n_recent)):
        r = draw()
        recent.append({"place": r["name"], "category": r.get("category")})
    return {"user": "explore", "city": city, "poi_id": poi["id"], "place": poi["name"],
            "category": poi.get("category"), "area": idx.area_label(poi["lat"], poi["lon"]),
            "dow": dow, "hour": hour, "tod": tod_label(hour),
            "lat": poi["lat"], "lon": poi["lon"], "recent": recent}


def crn_contexts(user_record, n, idx, exploration, safety_weight, seed):
    """Common random numbers: ONE context/request sequence per (user, seed), shared by
    every candidate entry.  Stage-7 fix (reviewer-validity major 3): the Stage-3 runs
    drew contexts from a per-ENTRY RNG, so only ~6% of no-entry baselines were ever
    reused and the "halved" per-write cost was an analytic idealisation.  With CRN the
    baseline for each context is computed once per user and reused for all writes."""
    rng = np.random.default_rng(
        int(hashlib.sha256(f"crn|{user_record['id']}|{seed}".encode()).hexdigest()[:16], 16))
    return sample_contexts(user_record, n, rng, idx, exploration, safety_weight)


def sample_contexts(user_record, n, rng, idx, exploration=0.3, safety_weight=0.5):
    """n contexts: (1-exploration) from the user's empirical Q (train), rest from the
    exploration mixture. Returns list of (context, query)."""
    from .scope import QUERIES
    train = user_record.get("train", [])
    city = user_record.get("city", "austin")
    out = []
    for _ in range(n):
        if train and rng.random() > exploration:
            c = dict(train[int(rng.integers(len(train)))])
        else:
            c = _synth_context(idx, rng, safety_weight, city)
        q = QUERIES[int(rng.integers(len(QUERIES)))]
        out.append((c, q))
    return out


def entry_payload_entries(entry, by_id):
    """Resolve the entries to inject for a candidate (handles split pairs)."""
    if entry.get("split_pair") and entry.get("split_role"):
        # inject both halves of the pair
        base = entry["id"].rsplit("_", 1)[0]
        parts = [by_id.get(base + "_A"), by_id.get(base + "_B")]
        return [p for p in parts if p]
    return [entry]


def ccs_detect(agent, entry, user_record, idx, rho0, rho1, cfg, baseline_cache,
               by_id=None, seed=0, decode_seed=None, counters=None):
    """Run CCS on one entry. Returns dict with decision/n/divergences/contexts.

    `seed` drives context sampling; `decode_seed` (default: seed) is the agent's
    decoding seed (irrelevant at temperature 0, kept fixed across sampling seeds)."""
    by_id = by_id or {}
    ds = seed if decode_seed is None else decode_seed
    n_max = cfg["n_max"]
    if cfg.get("crn"):
        samples = crn_contexts(user_record, n_max, idx, cfg.get("exploration", 0.3),
                               cfg.get("safety_weight", 0.5), seed)
    else:
        rng = np.random.default_rng(
            int(hashlib.sha256(f"{entry['id']}|{user_record['id']}|{seed}".encode())
                .hexdigest()[:16], 16))
        samples = sample_contexts(user_record, n_max, rng, idx,
                                  cfg.get("exploration", 0.3),
                                  cfg.get("safety_weight", 0.5))
    inject = entry_payload_entries(entry, by_id)
    obs = []
    details = []
    for i, (ctx, q) in enumerate(samples):
        sig = _ctx_sig(ctx, q)
        if sig in baseline_cache:
            a_without = baseline_cache[sig]
        else:
            a_without = agent.act(ctx, [], q, seed=ds)
            baseline_cache[sig] = a_without
            if counters is not None:
                counters["baseline_calls"] = counters.get("baseline_calls", 0) + 1
        a_with = agent.act(ctx, inject, q, seed=ds)
        if counters is not None:
            counters["with_calls"] = counters.get("with_calls", 0) + 1
        sv = cfg.get("scope", "poi")
        if sv == "strict":
            d = scopemod.divergence_strict(a_with, a_without, entry, idx)
        else:
            d = scopemod.DIVERGENCE_VARIANTS[sv](a_with, a_without, entry)
        obs.append(d)
        ctx_phi = dict(ctx); ctx_phi["query"] = q
        details.append({"i": i, "div": d, "poi": ctx.get("place"),
                        "area": ctx.get("area"), "query": q,
                        "hour": ctx.get("hour"), "dow": ctx.get("dow"),
                        "phi": int(bool(entry.get("predicate")) and
                                   _phi(entry["predicate"], ctx_phi)),
                        "payload_fired": scopemod.payload_action(a_with, entry, idx),
                        "with": a_with, "without": a_without})
        # early-stop check uses the SPRT incrementally
        res = sprt_run(obs, rho0, rho1, cfg["alpha"], cfg["beta"], len(obs))
        if res["stopped_early"]:
            break
    res = sprt_run(obs, rho0, rho1, cfg["alpha"], cfg["beta"], n_max)
    # confirmation path: untrusted entry touching a high-risk tool -> confirm regardless
    confirm = entry.get("risk_tier") == "high"
    return {"entry": entry["id"], "kind": entry.get("kind"), "user": user_record["id"],
            "decision": res["decision"], "n_used": len(obs),
            "k_div": int(sum(obs)), "llr": res["llr"], "confirm_required": confirm,
            "flagged": res["decision"] == "flag" or confirm,
            "k_phi": int(sum(d["phi"] for d in details)),
            "k_div_phi": int(sum(1 for d in details if d["div"] and d["phi"])),
            "k_div_nophi": int(sum(1 for d in details if d["div"] and not d["phi"])),
            "k_payload": int(sum(1 for d in details if d["payload_fired"])),
            "div_by_query": {q: int(sum(1 for d in details
                                        if d["query"] == q and d["div"]))
                             for q in sorted({d["query"] for d in details})},
            "n_by_query": {q: int(sum(1 for d in details if d["query"] == q))
                           for q in sorted({d["query"] for d in details})},
            "details": details[:cfg.get("keep_details", 0)]}
