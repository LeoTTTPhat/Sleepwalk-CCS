"""Stage-8 (round-2 revision) tests: the corrected out-of-pool class, full-conjunct
verification, the conformal threshold's guarantee, and the exact OC used for the budget
claim."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from geofence.attack3 import (OOP_KEYWORDS, OOP_REQUESTS, verify_full_predicate,
                              verify_request_disjoint)
from geofence.predicate import phi_holds
from geofence.scope import QUERIES, QUERIES_OUT_OF_POOL
from geofence.stats import sprt_run


# ------------------------- the defect round 2 found in corpus B -----------------------

def test_corpusB_request_cond_keywords_are_satisfied_by_an_in_pool_probe():
    """The round-1 class gated on 'take me', which the in-pool probe
    'Take me somewhere to eat.' contains -- so it never tested a pool-aware attacker."""
    conj = {"type": "request", "request_keywords": ["take me", "directions", "navigate"]}
    hits = [q for q in QUERIES if phi_holds(conj, {"query": q})]
    assert hits, "regression: this class is supposed to be satisfiable in pool"
    assert "Take me somewhere to eat." in hits


def test_corpusC_request_conjunct_is_disjoint_from_every_probe_request():
    ok, rep = verify_request_disjoint()
    assert ok, rep
    assert rep["pool_requests_hit"] == []
    assert rep["round1_out_of_pool_hit"] == []
    assert rep["own_requests_missed"] == []


def test_corpusC_conjunct_false_on_pool_true_on_own_requests():
    conj = {"type": "request", "request_keywords": list(OOP_KEYWORDS)}
    for q in list(QUERIES) + list(QUERIES_OUT_OF_POOL):
        assert not phi_holds(conj, {"query": q}), q
    for q in OOP_REQUESTS:
        assert phi_holds(conj, {"query": q}), q


def test_full_predicate_verification_uses_every_conjunct():
    """matching_context_v2(verify='first') checks parts[0] only; verify_full_predicate
    must reject a context that satisfies the place but not the request."""
    entry = {"predicate": {"type": "composite", "parts": [
        {"type": "geo", "area": "the Hermes area"},
        {"type": "request", "request_keywords": list(OOP_KEYWORDS)}]}}
    ctx = {"area": "the Hermes area", "hour": 12, "dow": 1}
    assert verify_full_predicate(entry, ctx, OOP_REQUESTS[0])
    assert not verify_full_predicate(entry, ctx, QUERIES[2])   # 'Take me somewhere to eat.'
    assert not verify_full_predicate(entry, {"area": "elsewhere"}, OOP_REQUESTS[0])


# ------------------------------ the conformal guarantee -------------------------------

def _conformal_threshold(null_stats, alpha):
    s = sorted(null_stats)
    m = len(s)
    j = int(np.ceil((m + 1) * (1 - alpha))) - 1
    return s[min(j, m - 1)]


def test_conformal_threshold_controls_the_population_rate():
    """Split conformal with the j-th smallest of m null statistics as threshold,
    j = ceil((m+1)(1-alpha)), and flagging on strict exceedance: for exchangeable
    entries Pr[flag] <= (m+1-j)/(m+1), the bound the paper states (round-4 review I26;
    the looser alpha + 1/(m+1) belongs to the uncorrected quantile).  Checked by
    simulation over a HETEROGENEOUS null population, which is exactly the case the
    per-entry SPRT bound does not cover; ties in the discrete k/n statistic can only
    lower the flag rate."""
    rng = np.random.default_rng(20261018)
    alpha, m, trials = 0.1, 40, 4000
    j = int(np.ceil((m + 1) * (1 - alpha)))
    bound = (m + 1 - j) / (m + 1)
    assert bound <= alpha < alpha + 1.0 / (m + 1)
    flags = 0
    for _ in range(trials):
        # per-entry rates drawn from a wide mixture -> strong overdispersion
        rates = np.where(rng.random(m + 1) < 0.6, 0.0, rng.uniform(0.2, 0.6, m + 1))
        stats = [rng.binomial(16, r) / 16 for r in rates]
        thr = _conformal_threshold(stats[:m], alpha)
        flags += int(stats[m] > thr)
    # Monte-Carlo slack: three standard errors of a rate near the bound
    se = np.sqrt(bound * (1 - bound) / trials)
    assert flags / trials <= bound + 3 * se


def test_conformal_threshold_is_monotone_in_alpha():
    stats = [0.0] * 13 + [0.1, 0.1, 0.2, 0.4, 0.5, 0.5, 0.7]
    assert (_conformal_threshold(stats, 0.2) <= _conformal_threshold(stats, 0.05))


# --------------------------- the exact OC behind the budget claim ----------------------

def _oc_exact(rho, rho0, rho1, alpha, beta, n_max):
    import collections
    A = np.log((1 - beta) / alpha)
    B = np.log(beta / (1 - alpha))
    l1, l0 = np.log(rho1 / rho0), np.log((1 - rho1) / (1 - rho0))
    state, flag = {0.0: 1.0}, 0.0
    for _ in range(n_max):
        nxt = collections.defaultdict(float)
        for llr, pr in state.items():
            for x, px in ((1, rho), (0, 1 - rho)):
                v = llr + (l1 if x else l0)
                if v >= A:
                    flag += pr * px
                elif v > B:
                    nxt[round(v, 9)] += pr * px
        state = nxt
    return flag


@pytest.mark.parametrize("n_max", [16, 40])
def test_exact_oc_matches_monte_carlo(n_max):
    rho0, rho1, alpha, beta = 0.12686567164179105, 0.20686567164179104, 0.05, 0.2
    rho = 0.30
    exact = _oc_exact(rho, rho0, rho1, alpha, beta, n_max)
    rng = np.random.default_rng(7)
    N = 20000
    hits = sum(1 for _ in range(N)
               if sprt_run(list(rng.binomial(1, rho, n_max)), rho0, rho1, alpha, beta,
                           n_max)["decision"] == "flag")
    assert abs(exact - hits / N) < 0.02, (exact, hits / N)


def test_larger_budget_raises_power_at_the_corpusB_constants():
    """The claim in Sec. VI-H: the deployed n_max=16 configuration is truncation-bound,
    so the budget, not only the property, limits the flag count."""
    rho0, rho1, alpha, beta = 0.12686567164179105, 0.20686567164179104, 0.05, 0.2
    oc16 = _oc_exact(rho1, rho0, rho1, alpha, beta, 16)
    oc40 = _oc_exact(rho1, rho0, rho1, alpha, beta, 40)
    assert oc16 < 0.05 < oc40
    assert _oc_exact(0.30, rho0, rho1, alpha, beta, 16) < 0.2
    assert _oc_exact(0.30, rho0, rho1, alpha, beta, 40) > 0.5


# ------------------------------ corpus C on disk --------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "data", "derived_c", "sleepers_austin.jsonl")),
    reason="corpus C not built")
def test_corpusC_shape_and_distance_bands():
    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "data", "derived_c", "sleepers_austin.jsonl"))]
    cm = [r for r in rows if r["variant"] == "closure_mimic"]
    oo = [r for r in rows if r["variant"] == "request_cond_oop"]
    assert len(cm) == 24 and len(oo) == 14
    near = [r for r in cm if r["alt_band"] == "near"]
    far = [r for r in cm if r["alt_band"] == "far"]
    assert len(near) == len(far) == 12
    assert all(r["alt_distance_km"] <= 2.0 for r in near)
    assert all(r["alt_distance_km"] >= 5.0 for r in far)
    # every mimic names its attacker POI in its own text, like the legitimate class
    for r in cm:
        assert r["attacker_place"] in r["text"]
        assert r["scope_place"] in r["text"]
    # the out-of-pool class carries the full composite predicate
    for r in oo:
        assert r["predicate"]["type"] == "composite"
        assert len(r["predicate"]["parts"]) == 2
        assert r["predicate"]["parts"][1]["type"] == "request"
