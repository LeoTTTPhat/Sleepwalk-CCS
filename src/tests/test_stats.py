"""Unit tests for the CCS statistical core (no models, no I/O).

Covers: exact-ish Type-I (FPR) control of the SPRT under a synthetic Bernoulli
null, power under the alternative, Ville's inequality for the e-process
(supermartingale validity), the Theorem-1 sample-complexity / miss-probability
relation, Wilson-CI sanity, calibration flooring, and the scope invariance lemma.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geofence import stats  # noqa: E402

RNG = np.random.default_rng(12345)
ALPHA, BETA = 0.05, 0.2
RHO0, RHO1 = 0.05, 0.20
NMAX = 600
TRIALS = 4000


def _stream(rate, n):
    return (RNG.random(n) < rate).astype(int).tolist()


def test_sprt_type_one_control():
    flags = 0
    for _ in range(TRIALS):
        obs = _stream(RHO0, NMAX)
        res = stats.sprt_run(obs, RHO0, RHO1, ALPHA, BETA, NMAX)
        flags += int(res["decision"] == "flag")
    fpr = flags / TRIALS
    bound = ALPHA / (1 - BETA)  # Wald upper bound on Type-I ~ 0.0625
    assert fpr <= bound + 0.03, f"FPR {fpr:.3f} exceeds Wald bound {bound:.3f}"


def test_sprt_power():
    hits = 0
    for _ in range(TRIALS):
        obs = _stream(RHO1, NMAX)
        res = stats.sprt_run(obs, RHO0, RHO1, ALPHA, BETA, NMAX)
        hits += int(res["decision"] == "flag")
    power = hits / TRIALS
    assert power >= (1 - BETA) - 0.05, f"power {power:.3f} below target"


def test_sprt_terminates_early_on_strong_signal():
    # a trigger that always diverges should stop in a handful of samples
    obs = [1] * NMAX
    res = stats.sprt_run(obs, RHO0, RHO1, ALPHA, BETA, NMAX)
    assert res["decision"] == "flag" and res["n"] <= 10 and res["stopped_early"]


def test_e_process_ville():
    # Under H0, P(sup_n E_n >= 1/alpha) <= alpha  (Ville's inequality)
    thresh = 1.0 / ALPHA
    exceed = 0
    trials = 4000
    for _ in range(trials):
        obs = _stream(RHO0, 300)
        traj = stats.e_process_trajectory(obs, RHO0, RHO1)
        if max(traj) >= thresh:
            exceed += 1
    rate = exceed / trials
    assert rate <= ALPHA + 0.02, f"Ville violated: {rate:.3f} > {ALPHA}"


def test_e_value_mean_one_under_null():
    # E[E_1] = 1 exactly for a single Bernoulli step under H0
    inc1 = np.exp(stats.llr_increment(1, RHO0, RHO1))
    inc0 = np.exp(stats.llr_increment(0, RHO0, RHO1))
    mean = RHO0 * inc1 + (1 - RHO0) * inc0
    assert abs(mean - 1.0) < 1e-9


def test_sample_complexity_miss_prob():
    for eps, p, delta in [(0.06, 0.6, 0.05), (0.1, 0.8, 0.1), (0.2, 0.5, 0.01)]:
        n = stats.sample_complexity(eps, p, delta)
        assert stats.miss_probability(eps, p, n) <= delta + 1e-9


def test_residual_impact_bounded():
    assert stats.residual_impact(0.06, 0.0) == 0.06
    assert stats.residual_impact(0.9, 0.5) == 1.0


def test_wilson_ci_contains_and_orders():
    phat, lo, hi = stats.wilson_ci(5, 20)
    assert lo < phat < hi and 0 <= lo and hi <= 1
    # zero events -> lower bound 0
    _, lo0, hi0 = stats.wilson_ci(0, 30)
    assert lo0 == 0.0 and hi0 > 0


def test_calibrate_null_floor_and_value():
    assert stats.calibrate_null([(0, 50), (0, 50)]) >= 1e-3  # floored, never 0
    r = stats.calibrate_null([(5, 100), (3, 100)])
    assert 0.03 < r < 0.07


def test_invariance_lemma_no_effect():
    scope = {"tools": {"poi_info", "route"}, "targets": {"poi_x"}}
    a = [{"tool": "poi_info", "target": "poi_x"}]
    # identical paired runs -> no divergence, regardless of scope
    assert stats.out_of_scope_divergence(a, a, scope) == 0


def test_invariance_in_scope_change_not_flagged():
    scope = {"tools": {"poi_info", "route"}, "targets": {"poi_x"}}
    # in-scope answer changes (poi_info vs route to same in-scope target) -> no OOS div
    a_with = [{"tool": "route", "target": "poi_x"}]
    a_without = [{"tool": "poi_info", "target": "poi_x"}]
    assert stats.out_of_scope_divergence(a_with, a_without, scope) == 0


def test_invariance_out_of_scope_change_flagged():
    scope = {"tools": {"poi_info", "route"}, "targets": {"poi_x"}}
    a_with = [{"tool": "share_location", "target": "attacker"}]
    a_without = [{"tool": "poi_info", "target": "poi_x"}]
    assert stats.out_of_scope_divergence(a_with, a_without, scope) == 1
