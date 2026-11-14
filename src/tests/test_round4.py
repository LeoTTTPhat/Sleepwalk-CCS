"""Round-4 review tests (I26, I12): the conformal order-statistic bound at the paper's
constants, the bracket for the resolved-target re-test (replay.sprt_rescore_bracket),
class separation in the n_max=40 vs n_max=16 pairing (analyze_rev2.block_budget), and
the stale-string guard.  Toy streams only; no model calls and no result files."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import analyze_rev2 as ar2                                           # noqa: E402
from replay import sprt_rescore_bracket                              # noqa: E402


# ------------------------- the conformal bound at the paper's constants ----------------

def test_conformal_bound_is_exact_for_continuous_statistics_at_m40():
    """m = 40, alpha = 0.05 -> j = 39 and Pr[flag] = (m+1-j)/(m+1) = 2/41 exactly for
    continuous exchangeable statistics; the loose alpha + 1/(m+1) = 0.074 is not the
    rate of this rule."""
    m, alpha = 40, 0.05
    j = int(np.ceil((m + 1) * (1 - alpha)))
    assert j == 39 and abs((m + 1 - j) / (m + 1) - 2 / 41) < 1e-12
    rng = np.random.default_rng(4)
    trials = 60000
    x = rng.random((trials, m + 1))
    thr = np.sort(x[:, :m], axis=1)[:, j - 1]          # j-th smallest of the m nulls
    rate = float(np.mean(x[:, m] > thr))
    se = np.sqrt((2 / 41) * (1 - 2 / 41) / trials)
    assert abs(rate - 2 / 41) < 4 * se
    assert rate < alpha + 1 / (m + 1) - 0.02


# --------------------------- the resolved-target re-test bracket -----------------------

def _ctx(cls, entry, seq, key="div_strict"):
    return [{"class": cls, "entry": entry, key: int(x)} for x in seq]


def _toy_streams(key="div_strict"):
    rows = []
    rows += _ctx("sleeper", "sA", [1, 1, 0, 0], key)   # flags at n=2 (k_min = 2)
    rows += _ctx("sleeper", "sB", [0] * 5, key)        # runs out of logged contexts
    rows += _ctx("sleeper", "sC", [0] * 25, key)       # clears at n=19
    rows += _ctx("benign", "nCal", [1, 1], key)        # calibration entry, flags
    rows += _ctx("benign", "nB", [0] * 3, key)         # held out, runs out
    rows += _ctx("legit", "nF", [0, 1, 1], key)        # held out, flags at n=3
    return rows


def test_rescore_bracket_lower_and_upper_on_toy_streams():
    out = sprt_rescore_bracket(_toy_streams(), calib_entries={"nCal"}, rho0=0.017471,
                               nmax=40, use_strict=True)
    s, na, nh = out["sleeper"], out["null_all"], out["null_heldout"]
    # lower = exhausted streams scored as clears (what sprt_run does)
    assert (s["lower"]["k"], s["lower"]["n"]) == (1, 3)
    assert (na["lower"]["k"], na["lower"]["n"]) == (2, 3)
    assert (nh["lower"]["k"], nh["lower"]["n"]) == (1, 2)
    # upper = exhausted streams scored as flags
    assert s["upper"]["k"] == 2 and na["upper"]["k"] == 3 and nh["upper"]["k"] == 2
    assert s["exhausted_streams"] == 1 and na["exhausted_streams"] == 1
    assert nh["exhausted_streams"] == 1
    # the calibration entry is in null_all but never in the held-out count
    assert na["lower"]["n"] - nh["lower"]["n"] == 1
    assert abs(out["rho1"] - (0.017471 + 0.08)) < 1e-12


def test_rescore_bracket_full_length_stream_is_not_exhausted():
    """A stream that reaches n_max without crossing is a genuine truncation-to-clear,
    not an exhausted log, so it must not widen the bracket."""
    seq = [1] + [0] * 19 + [1] + [0] * 19               # 40 contexts, never crosses
    assert len(seq) == 40
    rows = _ctx("sleeper", "s40", seq)
    out = sprt_rescore_bracket(rows, calib_entries=set(), nmax=40)
    assert out["sleeper"]["exhausted_streams"] == 0
    assert out["sleeper"]["lower"]["k"] == out["sleeper"]["upper"]["k"] == 0


def test_rescore_bracket_uses_the_requested_scorer():
    rows = _toy_streams(key="div")
    out = sprt_rescore_bracket(rows, calib_entries={"nCal"}, use_strict=False)
    assert out["scoring"] == "containment"
    assert (out["sleeper"]["lower"]["k"], out["sleeper"]["upper"]["k"]) == (1, 2)


# -------------------- class separation in the n_max=40 vs n_max=16 pairing ---------------

def _row(entry, cls, decision, n_used, k_div, variant="standard"):
    return {"key": f"{entry}:k", "entry": entry, "class": cls, "decision": decision,
            "flagged": decision == "flag", "n_used": n_used, "k_div": k_div,
            "rho0": 0.12686567164179105, "rho1": 0.20686567164179104,
            "variant": variant if cls == "sleeper" else None}


@pytest.fixture
def toy_budget(tmp_path, monkeypatch):
    n16 = [_row("s1", "sleeper", "clear", 16, 2), _row("s2", "sleeper", "flag", 10, 6),
           _row("s3", "sleeper", "clear", 16, 1), _row("s4", "sleeper", "flag", 9, 6),
           _row("b1", "benign", "clear", 16, 0), _row("b2", "benign", "clear", 16, 1),
           _row("l1", "legit", "flag", 12, 6), _row("l2", "legit", "clear", 16, 3)]
    # n_max=40 on a sleeper subsample (s1-s3) and every null
    n40 = [_row("s1", "sleeper", "flag", 30, 10), _row("s2", "sleeper", "flag", 12, 7),
           _row("s3", "sleeper", "clear", 40, 3),
           _row("b1", "benign", "flag", 25, 9), _row("b2", "benign", "clear", 40, 2),
           _row("l1", "legit", "clear", 40, 6), _row("l2", "legit", "flag", 22, 9)]
    for name, rs in (("ccs2.jsonl", n16), ("ccs2_n40.jsonl", n40)):
        with open(tmp_path / name, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rs) + "\n")
    monkeypatch.setattr(ar2, "REV", str(tmp_path))
    monkeypatch.setattr(ar2, "RES", str(tmp_path))
    monkeypatch.setattr(ar2.rows, "__defaults__", ("key", str(tmp_path)))
    return ar2.block_budget()


def test_nmax40_pairing_separates_sleepers_from_nulls(toy_budget):
    pr = toy_budget["n_max_40"]["vs_nmax16_paired"]
    sl, nu = pr["sleepers"], pr["nulls"]
    # sleepers: only the 3 subsampled units, s1 gained, nothing lost
    assert sl["n"] == 3 and sl["gained"] == 1 and sl["lost"] == 0
    assert (sl["flagged_at_nmax16"]["k"], sl["flagged_at_nmax40"]["k"]) == (1, 2)
    # nulls: b1 and l2 gained, l1 lost -- none of it counted as detection
    assert nu["n"] == 4 and nu["gained"] == 2 and nu["lost"] == 1
    assert (nu["flagged_at_nmax16"]["k"], nu["flagged_at_nmax40"]["k"]) == (1, 2)
    # pooling the two (the round-3 defect) would have reported 3 gained, 1 lost
    assert sl["gained"] + nu["gained"] == 3
    nb = pr["nulls_by_class"]
    assert nb["benign"] == {"n": 2, "flagged_at_nmax16": 0, "flagged_at_nmax40": 1}
    assert nb["legit"] == {"n": 2, "flagged_at_nmax16": 1, "flagged_at_nmax40": 1}
    # matched Youden J on the same units at both budgets
    assert abs(pr["youden_J_nmax16_same_units"] - (1 / 3 - 1 / 4)) < 1e-12
    assert abs(pr["youden_J_nmax40_same_units"] - (2 / 3 - 2 / 4)) < 1e-12
    assert toy_budget["n_max_40"]["write_mix"] == {"sleeper_writes": 3, "null_writes": 4}


# ------------------------------------ stale-string guard --------------------------------
