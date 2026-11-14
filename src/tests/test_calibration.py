"""Calibration checks at the deployed operating point (Stage-2 re-verification).

* SPRT FPR with a *plug-in* null rate rho0_hat (calibrated from a finite set of
  benign paired runs, as evaluate.py does: 30 entries x 20 contexts, Jeffreys
  floor) and the deployed truncation n_max=40, rho1 = max(eps*p, rho0_hat+0.08).
* Wilson-interval coverage by simulation (the CI used for every rate we report).
* The GEOFENCE_DERIVED override isolates smoke inputs from full-run inputs.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geofence import stats  # noqa: E402

ALPHA, BETA = 0.05, 0.2
EPS, P = 0.0599, 0.6


def test_sprt_fpr_with_plugin_rho0_at_deployed_nmax():
    rng = np.random.default_rng(2026)
    true_rho0 = 0.04
    trials, flags = 3000, 0
    for _ in range(trials):
        counts = [(int((rng.random(20) < true_rho0).sum()), 20) for _ in range(30)]
        r0 = stats.calibrate_null(counts)
        r1 = min(0.95, max(EPS * P, r0 + 0.08))
        obs = (rng.random(40) < true_rho0).astype(int).tolist()
        flags += stats.sprt_run(obs, r0, r1, ALPHA, BETA, 40)["decision"] == "flag"
    fpr = flags / trials
    # Wald bound alpha/(1-beta)=0.0625 plus Monte-Carlo slack (3 s.e. ~ 0.013)
    assert fpr <= ALPHA / (1 - BETA) + 0.013, fpr


def test_sprt_detects_strong_trigger_at_deployed_nmax():
    rng = np.random.default_rng(7)
    r0 = 0.04
    r1 = max(EPS * P, r0 + 0.08)
    trials, flags = 2000, 0
    for _ in range(trials):
        obs = (rng.random(40) < 0.5).astype(int).tolist()  # Q-mass ~0.8, p ~0.6
        flags += stats.sprt_run(obs, r0, r1, ALPHA, BETA, 40)["decision"] == "flag"
    assert flags / trials >= 0.95


def test_wilson_coverage():
    rng = np.random.default_rng(11)
    for p, n in [(0.05, 60), (0.3, 85), (0.9, 100)]:
        ks = rng.binomial(n, p, size=4000)
        cover = np.mean([stats.wilson_ci(int(k), n)[1] <= p <= stats.wilson_ci(int(k), n)[2]
                         for k in ks])
        assert cover >= 0.92, (p, n, cover)


def test_derived_override_isolates_smoke():
    src = os.path.join(os.path.dirname(__file__), "..")
    code = "from geofence import common; print(common.DERIVED); print(common.POIS['austin'])"
    env = dict(os.environ, GEOFENCE_DERIVED="/tmp/x_derived_smoke", PYTHONPATH=src)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout.split()
    assert out[0] == "/tmp/x_derived_smoke"
    assert out[1].endswith(os.path.join("data", "derived", "pois_austin.jsonl"))
