"""Stage-5 analysis layer: the exact SPRT operating characteristic used in
results/analysis_stats.json and the paper must agree with the implemented sprt_run."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyze import oc_exact, sprt_theory  # noqa: E402
from geofence.stats import sprt_run  # noqa: E402

R0, R1 = 0.010815307820299502, 0.0908153078202995


def test_oc_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for rho in (0.0108, 0.036, 0.0908):
        sims = 6000
        fl = sum(sprt_run(list((rng.uniform(size=40) < rho).astype(int)), R0, R1, 0.05, 0.2,
                          40)["decision"] == "flag" for _ in range(sims))
        p, _ = oc_exact(rho, R0, R1, 0.05, 0.2, 40)
        se = (p * (1 - p) / sims) ** 0.5
        assert abs(fl / sims - p) < 4 * se + 1e-3


def test_oc_type_one_below_wald_bound():
    p, _ = oc_exact(R0, R0, R1, 0.05, 0.2, 40)
    assert p <= 0.05 / (1 - 0.2)


def test_theory_constants():
    t = sprt_theory(R0, R1, 0.05, 0.2, 40)
    assert t["min_divergences_to_flag"] == 2
    assert t["clean_contexts_to_clear"] == 19
    # 19 clean contexts from LLR 0 reach the lower boundary
    assert sprt_run([0] * 40, R0, R1, 0.05, 0.2, 40)["n"] == 19
