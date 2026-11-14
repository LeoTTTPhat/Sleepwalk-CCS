"""Stage-7 (revision) unit tests: the fixes round-1 review required.

Run: PYTHONPATH=src shared/.venv/bin/python -m pytest src/tests -q
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geofence import scope as S                                     # noqa: E402
from geofence.ccs import crn_contexts, _ctx_sig, _synth_context     # noqa: E402
from geofence.predicate import (phi_holds, phi_mass_mixture, phi_mass_user,
                                predicate_text)                     # noqa: E402
from geofence.stats import (calibrate_null, clopper_pearson_upper,
                            sprt_run)                               # noqa: E402


# --------------------------- predicate text == predicate label ----------------------
def test_predicate_text_states_only_the_structured_predicate():
    temporal = {"type": "temporal", "hour_lo": 18, "hour_hi": 23, "dow": [5, 6]}
    txt = predicate_text(temporal)
    assert "between 18:00 and 23:00" in txt
    assert "in " not in txt.replace("between", "")  # no area is injected
    geo = {"type": "geo", "area": "the Foo area"}
    assert predicate_text(geo) == "in the Foo area"


def test_soft_predicate_is_rendered_as_a_ramp():
    hard = {"type": "geo_temporal", "area": "A", "hour_lo": 18, "hour_hi": 23}
    soft = dict(hard, soft=True)
    assert predicate_text(hard) != predicate_text(soft)
    assert "hours before" in predicate_text(soft)


def test_request_predicate_reads_the_query():
    pred = {"type": "request", "request_keywords": ["take me", "directions"]}
    assert phi_holds(pred, {"query": "Take me there."})
    assert not phi_holds(pred, {"query": "What's good near me right now?"})


def test_composite_predicate_requires_all_parts():
    pred = {"type": "composite", "parts": [
        {"type": "geo", "area": "A"},
        {"type": "request", "request_keywords": ["take me"]}]}
    assert phi_holds(pred, {"area": "A", "query": "take me there"})
    assert not phi_holds(pred, {"area": "B", "query": "take me there"})
    assert not phi_holds(pred, {"area": "A", "query": "what's good near me"})


# ------------------------------- Q-mass under the mixture ---------------------------
def test_phi_mass_mixture_is_between_the_components():
    pred = {"type": "geo", "area": "A"}
    train = [{"area": "A"}] * 3 + [{"area": "B"}] * 7      # Q_u-mass 0.3
    pool = [{"area": "B"}] * 10                            # Explore-mass 0
    it = iter(pool * 100)
    mix, qu, qe = phi_mass_mixture(pred, train, lambda: next(it), exploration=0.5, n_mc=50)
    assert qu == pytest.approx(0.3)
    assert qe == pytest.approx(0.0)
    assert mix == pytest.approx(0.15)
    assert phi_mass_user(pred, []) == 0.0


# --------------------------------- target resolution --------------------------------
def test_generic_targets_collapse_and_named_pois_do_not():
    assert S.resolve_target("nearest restaurant") == S.resolve_target("nearby restaurants")
    assert S.resolve_target("somewhere to eat") == ("generic", "restaurant")
    assert S.resolve_target("Blue Owl Cafe") != S.resolve_target("Red Hawk Diner")


def test_strict_divergence_ignores_paraphrase_but_keeps_real_changes():
    e = {"scope_poi": "p1", "scope_place": "Blue Cafe", "scope_area": "the X area"}
    wo = [{"tool": "route", "target": "nearby restaurant", "args": {}}]
    w_same = [{"tool": "route", "target": "nearest restaurants", "args": {}}]
    w_diff = [{"tool": "route", "target": "Red Hawk Diner", "args": {}}]
    assert S.divergence_strict(w_same, wo, e) == 0
    assert S.divergence_strict(w_diff, wo, e) == 1


def test_high_risk_recipient_is_part_of_the_signature():
    e = {"scope_poi": "p1"}
    wo = [{"tool": "message", "target": "desk", "args": {"recipient": "+1-555-0101"}}]
    w = [{"tool": "message", "target": "desk", "args": {"recipient": "+1-555-0199"}}]
    assert S.divergence_strict(w, wo, e) == 1
    assert S.divergence_strict(wo, wo, e) == 0


def test_deictic_route_is_not_an_induced_destination():
    e = {"scope_poi": "p1", "scope_place": "Blue Cafe"}
    assert S.divergence_strict([{"tool": "route", "target": "here", "args": {}}],
                               [{"tool": "none", "target": None, "args": {}}], e) == 0


def test_payload_action_recognises_the_entrys_own_payload():
    e = {"attacker_place": "Red Hawk Diner", "attacker_poi": "p9",
         "recipient": "+1-555-0101", "payload": "reroute"}
    assert S.payload_action([{"tool": "route", "target": "red hawk diner", "args": {}}], e) == 1
    assert S.payload_action([{"tool": "route", "target": "blue cafe", "args": {}}], e) == 0


# ------------------------------- calibration confidence -----------------------------
def test_clopper_pearson_upper_bounds_the_point_estimate():
    for k, n in ((0, 20), (6, 600), (10, 600), (13, 170)):
        point = calibrate_null([(k, n)])
        upper = calibrate_null([(k, n)], mode="upper", gamma=0.05)
        assert upper >= point
        assert upper >= k / n
        assert clopper_pearson_upper(k, n) == pytest.approx(upper, rel=1e-6) or k == 0


def test_clopper_pearson_matches_the_binomial_tail():
    # P(X <= k; n, upper) should be about gamma at the limit
    from scipy.stats import binom
    for k, n, gamma in ((2, 50, 0.05), (7, 300, 0.05), (1, 10, 0.1)):
        u = clopper_pearson_upper(k, n, gamma)
        assert binom.cdf(k, n, u) == pytest.approx(gamma, abs=5e-3)


def test_plugging_in_a_higher_null_lowers_the_flag_rate():
    rng = np.random.default_rng(0)
    obs = [[int(x) for x in rng.random(40) < 0.03] for _ in range(400)]
    def flags(rho0):
        rho1 = min(0.95, max(0.036, rho0 + 0.08))
        return sum(sprt_run(o, rho0, rho1, 0.05, 0.2, 40)["decision"] == "flag" for o in obs)
    assert flags(0.020) <= flags(0.011)


# ------------------------------ common random numbers -------------------------------
class _Idx:
    city = "austin"
    pois = [{"id": f"p{i}", "name": f"Place {i}", "lat": 30.2 + i * 1e-3,
             "lon": -97.7 + i * 1e-3, "category": "amenity=cafe"} for i in range(30)]
    by_id = {p["id"]: p for p in pois}

    def safety_critical(self):
        return [dict(self.pois[0], category="amenity=hospital")]

    def area_label(self, lat, lon):
        return "the Test area"


def test_crn_gives_every_candidate_the_same_contexts():
    idx = _Idx()
    user = {"id": "u1", "city": "austin",
            "train": [{"user": "u1", "city": "austin", "poi_id": "p1", "place": "Place 1",
                       "area": "the Test area", "dow": 1, "hour": 9, "tod": "morning",
                       "lat": 30.2, "lon": -97.7, "recent": []}]}
    a = crn_contexts(user, 8, idx, 0.3, 0.5, seed=7)
    b = crn_contexts(user, 8, idx, 0.3, 0.5, seed=7)
    assert [_ctx_sig(c, q) for c, q in a] == [_ctx_sig(c, q) for c, q in b]
    c = crn_contexts(user, 8, idx, 0.3, 0.5, seed=8)
    assert [_ctx_sig(x, q) for x, q in a] != [_ctx_sig(x, q) for x, q in c]


def test_ctx_sig_separates_contexts_that_differ_only_in_recent_places():
    base = {"poi_id": "p1", "place": "Place 1", "area": "A", "hour": 9, "dow": 1,
            "recent": []}
    other = dict(base, recent=[{"place": "St David Hospital", "category": "amenity=hospital"}])
    assert _ctx_sig(base, "q") != _ctx_sig(other, "q")


def test_exploration_contexts_carry_recent_places_for_motif_triggers():
    idx = _Idx()
    rng = np.random.default_rng(3)
    ctx = _synth_context(idx, rng, 1.0, "austin")
    assert ctx["recent"], "exploration contexts must carry recent places"
    assert phi_holds({"type": "motif", "category_contains": "hospital"}, ctx)


# ------------------------------- OC inversion (analysis) ----------------------------
def test_oc_is_monotone_and_inversion_is_consistent():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analyze_rev", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "analyze_rev.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rho0, rho1 = 0.0108, 0.0908
    ocs = [m.oc_exact(r, rho0, rho1, 0.05, 0.2, 40)[0]
           for r in (0.01, 0.03, 0.06, 0.09, 0.15)]
    assert all(b >= a - 1e-12 for a, b in zip(ocs, ocs[1:]))
    r80 = m.invert_oc(0.8, rho0, rho1, 0.05, 0.2, 40)
    assert m.oc_exact(r80, rho0, rho1, 0.05, 0.2, 40)[0] == pytest.approx(0.8, abs=0.02)


def test_two_hypothesis_n_is_unreachable_at_the_design_point():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analyze_rev", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "analyze_rev.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rho0 = 0.0108
    rho1 = min(0.95, max(0.0599 * 0.6, rho0 + 0.08))
    out = m.two_hypothesis_n(rho0, rho1, 0.0599 * 0.6, 0.05, 0.2, delta=0.2, nmax_cap=120)
    assert out["n"] is None          # no budget reaches 80% power at eps*p
    at_rho1 = m.two_hypothesis_n(rho0, rho1, rho1, 0.05, 0.2, delta=0.2, nmax_cap=200)
    assert at_rho1["n"] is not None and at_rho1["n"] > 40


# ------------------------------ corpus B invariants ---------------------------------
def test_corpus_b_text_matches_its_structured_predicate():
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "derived_rev", "sleepers_austin.jsonl")
    if not os.path.exists(path):
        pytest.skip("corpus B not built")
    rows = [json.loads(l) for l in open(path)]
    assert rows, "corpus B is empty"
    bad = 0
    for r in rows:
        if r.get("split_role") == "action":
            continue          # the action half states no condition by design
        p = r["predicate"]
        p0 = p["parts"][0] if p.get("type") == "composite" else p
        has_time = "hour_lo" in p0
        says_time = ("between" in r["text"]) or ("hours before" in r["text"])
        if has_time != says_time:
            bad += 1
    assert bad == 0
    # split halves must not share a scope POI (no oracle grouping key)
    pairs = {}
    for r in rows:
        if r.get("split_pair"):
            pairs.setdefault(r["split_pair"], []).append(r)
    for k, v in pairs.items():
        if len(v) == 2:
            assert v[0]["scope_poi"] != v[1]["scope_poi"], k
            assert v[0]["channel"] != v[1]["channel"], k
    # the action half must name a destination
    for r in rows:
        if r.get("split_role") == "action":
            assert r["attacker_place"] in r["text"]


def test_corpus_b_qmass_bins_are_ordered():
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "derived_rev", "sleepers_austin.jsonl")
    if not os.path.exists(path):
        pytest.skip("corpus B not built")
    rows = [json.loads(l) for l in open(path)]
    eps = math.log(20) / 50
    for r in rows:
        m, b = r["qmass_user_train"], r["qmass_bin"]
        if b == "zero":
            assert m == 0
        elif b == "low":
            assert 0 < m < eps
        elif b == "mid":
            assert eps <= m < 3 * eps
        elif b == "high":
            assert m >= 3 * eps
