"""Stage-5 derived statistics -> results/analysis_stats.json.

Pure post-processing of results/ (raw JSONL + results/summary.json). No model calls,
no network, deterministic (fixed RNG seeds). Every number that the paper quotes but that
is not already a leaf of results/summary.json is written here, so that make_tables.py,
make_figures.py and results/analysis.md can trace it to a file.

Blocks written:
  sprt_theory        Wald boundaries, per-divergence / per-clean LLR steps, number of
                     divergences needed to flag, clean contexts needed to clear, Theorem-1
                     n for (eps, p, delta), miss probability at the configured n_max.
  oc                 exact operating characteristic of the truncated SPRT (P(flag), E[n])
                     on a grid of true per-context divergence rates, per run.
  fpr_model          predicted null FPR (i) at the calibrated rho0, (ii) at the null
                     divergence actually realised in the CCS runs, (iii) under a
                     beta-binomial (entry-heterogeneous) null fitted to the calibration
                     counts; versus the observed held-out FPR.
  overdispersion     Monte-Carlo test that calibration-entry divergence counts are
                     homogeneous binomial (variance-of-counts statistic) + Tarone Z.
  pit                randomized PIT values of calibration counts under the pooled binomial
                     (for the QQ plot).
  qmass_dependence   does SPRT detection increase with trigger Q-mass? (Mann-Whitney U,
                     logistic slope, Spearman) per seed.
  subgroup_tests     Fisher / chi-square tests: effective vs inert, payload, split, soft.
  paired_diffs       entry-bootstrap 95% CIs of paired detection / FPR differences for
                     every ablation (complements the exact McNemar p in summary.json).
  provgate_extra     spotlight vs TMA-NM sign test; per-entry ASR difference CIs.
  cost               per-write cost shared / unshared / shared + pre-triage (seed 1).
  power              sample sizes the open claims would need (GAPS.md D1, D3).
"""
import collections
import glob
import json
import math
import os

import numpy as np
import yaml
from scipy import stats as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CFG = os.path.join(ROOT, "src", "configs")
SEEDS = ("20261015", "20261016", "20261017")
EPS = 0.0599
RUNS_CCS = [("full", s) for s in SEEDS] + [("full_beijing", SEEDS[0]),
                                          ("full_beijing_recal", SEEDS[0]),
                                          ("full_temp07", SEEDS[0])]


# ----------------------------------------------------------------------------- io
def read_rows(path, keyf=None):
    """JSONL rows de-duplicated on the resume key (first occurrence kept), as aggregate.py."""
    if not os.path.exists(path):
        return []
    out, seen = [], set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = keyf(r) if keyf else r.get("key", (r.get("entry"), r.get("user"), r.get("mode")))
            k = json.dumps(k, sort_keys=True, default=str)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out


def ccs_rows(run, seed, exp="ccs"):
    return read_rows(os.path.join(RES, run, f"seed_{seed}", f"{exp}.jsonl"))


def calib_rows(run, seed, exp="calib"):
    return read_rows(os.path.join(RES, run, f"seed_{seed}", f"{exp}.jsonl"),
                     keyf=lambda r: (r["entry"], r.get("user")))


def cfg(run):
    with open(os.path.join(CFG, f"{run}.yaml")) as fh:
        return yaml.safe_load(fh)


def wilson(k, n, z=1.96):
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": int(k), "n": int(n), "rate": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h)}


# ------------------------------------------------------------------ SPRT theory
def _steps(rho0, rho1):
    rho0 = min(max(rho0, 1e-9), 1 - 1e-9)
    rho1 = min(max(rho1, 1e-9), 1 - 1e-9)
    return math.log(rho1 / rho0), math.log((1 - rho1) / (1 - rho0))


def oc_exact(rho, rho0, rho1, alpha, beta, nmax):
    """Exact P(flag) and E[n] of geofence.stats.sprt_run under X_i ~iid Bern(rho)."""
    logA = math.log((1 - beta) / alpha)
    logB = math.log(beta / (1 - alpha))
    a, b = _steps(rho0, rho1)
    dist = {0: 1.0}
    pflag = en = 0.0
    for n in range(1, nmax + 1):
        new = collections.defaultdict(float)
        for k, p in dist.items():
            for x, px in ((1, rho), (0, 1 - rho)):
                if px == 0:
                    continue
                k2 = k + x
                llr = k2 * a + (n - k2) * b
                q = p * px
                if llr >= logA:
                    pflag += q
                    en += n * q
                elif llr <= logB:
                    en += n * q
                else:
                    new[k2] += q
        dist = new
    rest = sum(dist.values())
    en += nmax * rest
    return pflag, en


def sprt_theory(rho0, rho1, alpha, beta, nmax):
    logA = math.log((1 - beta) / alpha)
    logB = math.log(beta / (1 - alpha))
    a, b = _steps(rho0, rho1)
    clear_n = math.ceil(logB / b)  # consecutive clean contexts to accept H0 from 0
    # minimum divergences needed to flag at some n <= nmax (best case: all up front)
    k_min = math.ceil(logA / a)
    # earliest n at which k divergences flag: smallest n>=k with k*a+(n-k)*b>=logA -> n=k
    # latest n at which exactly 2 divergences still flag (2 early, then clean):
    #   2a + (n-2) b >= logA  ->  n <= 2 + (logA - 2a)/b
    out = {"rho0": rho0, "rho1": rho1, "alpha": alpha, "beta": beta, "n_max": nmax,
           "logA": logA, "logB": logB, "llr_per_divergence": a, "llr_per_clean": b,
           "clean_contexts_to_clear": clear_n, "min_divergences_to_flag": k_min,
           "wald_fpr_bound": alpha / (1 - beta)}
    return out


def theorem1(eps, p, delta, nmax):
    n = math.ceil(math.log(1 / delta) / (eps * p))
    return {"eps": eps, "p": p, "delta": delta, "n_required": n,
            "miss_prob_at_nmax": (1 - eps * p) ** nmax,
            "eps_p_guaranteed_at_nmax": math.log(1 / delta) / nmax}


# --------------------------------------------------------------- null modelling
def betabinom_fit(ks, ns):
    """MLE of a beta-binomial (mean m, ICC rho) for counts ks out of ns."""
    from scipy.optimize import minimize
    ks, ns = np.asarray(ks), np.asarray(ns)

    def nll(th):
        m = 1 / (1 + math.exp(-th[0]))
        rho = 1 / (1 + math.exp(-th[1]))
        a = m * (1 - rho) / rho
        b = (1 - m) * (1 - rho) / rho
        return -float(np.sum(st.betabinom.logpmf(ks, ns, a, b)))

    m0 = max(ks.sum() / ns.sum(), 1e-4)
    best = None
    for r0 in (1e-3, 1e-2, 0.05, 0.2):
        res = minimize(nll, [math.log(m0 / (1 - m0)), math.log(r0 / (1 - r0))],
                       method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8,
                                                      "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    m = 1 / (1 + math.exp(-best.x[0]))
    rho = 1 / (1 + math.exp(-best.x[1]))
    nll_binom = -float(np.sum(st.binom.logpmf(ks, ns, ks.sum() / ns.sum())))
    lr = max(0.0, 2 * (nll_binom - best.fun))
    # boundary LRT: 0.5 chi2_0 + 0.5 chi2_1
    p_lrt = 0.5 * float(st.chi2.sf(lr, 1)) if lr > 0 else 1.0
    return {"mean": m, "icc": rho, "a": m * (1 - rho) / rho, "b": (1 - m) * (1 - rho) / rho,
            "lrt_stat": lr, "lrt_p_boundary": p_lrt}


def overdispersion_mc(ks, ns, sims=20000, seed=0):
    ks, ns = np.asarray(ks), np.asarray(ns)
    p = ks.sum() / ns.sum()
    rng = np.random.default_rng(seed)
    obs = float(np.var(ks / ns))
    sim = rng.binomial(ns[None, :], p, size=(sims, len(ns))) / ns[None, :]
    v = sim.var(axis=1)
    pmc = float((1 + np.sum(v >= obs - 1e-15)) / (sims + 1))
    # Tarone (1979) score statistic
    S = float(np.sum((ks - ns * p) ** 2) / (p * (1 - p))) if 0 < p < 1 else float("nan")
    Z = (S - ns.sum()) / math.sqrt(2 * np.sum(ns * (ns - 1))) if 0 < p < 1 else float("nan")
    return {"pooled_rate": float(p), "var_obs": obs, "var_null_mean": float(v.mean()),
            "dispersion_ratio": obs / float(v.mean()) if v.mean() > 0 else None,
            "p_mc": pmc, "tarone_z": Z, "tarone_p": float(st.norm.sf(Z)) if Z == Z else None,
            "n_entries": int(len(ks)), "n_entries_k_ge_2": int(np.sum(ks >= 2))}


def pit_values(ks, ns, seed=0):
    ks, ns = np.asarray(ks), np.asarray(ns)
    p = ks.sum() / ns.sum()
    rng = np.random.default_rng(seed)
    v = rng.uniform(size=len(ks))
    lo = st.binom.cdf(ks - 1, ns, p)
    return list(map(float, lo + v * st.binom.pmf(ks, ns, p)))


def predicted_fpr_mixture(fit, rho0, rho1, alpha, beta, nmax, grid=400):
    """E_{rho~Beta(a,b)}[OC(rho)] by midpoint quadrature on the Beta quantiles."""
    qs = (np.arange(grid) + 0.5) / grid
    rhos = st.beta.ppf(qs, fit["a"], fit["b"])
    return float(np.mean([oc_exact(float(r), rho0, rho1, alpha, beta, nmax)[0] for r in rhos]))


# ------------------------------------------------------------------- helpers
def boot_diff(a, b, B=10000, seed=0):
    """Entry bootstrap of mean(a) - mean(b) for paired per-unit outcomes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {"diff": float(a.mean() - b.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "n": int(len(a))}


def flag(r):
    return int(r["decision"] == "flag")


def pair(ra, rb, key=lambda r: (r["class"], r["entry"], r.get("user"))):
    A = {key(r): r for r in ra}
    Bm = {key(r): r for r in rb}
    ks = sorted(set(A) & set(Bm), key=str)
    return [A[k] for k in ks], [Bm[k] for k in ks]


# ---------------------------------------------------------------------- main
def main():
    S = json.load(open(os.path.join(RES, "summary.json")))
    out = {"_note": "generated by src/analyze.py from results/ (deterministic); "
                    "see module docstring for definitions"}

    # ---------------- SPRT theory / OC / FPR model, per CCS run
    theory, ocs, fprm, od, pits = {}, {}, {}, {}, {}
    grid = [0.0, 0.002, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03,
            0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3,
            0.4, 0.5]
    for run, seed in RUNS_CCS:
        c = cfg(run)
        rows = ccs_rows(run, seed)
        if not rows:
            continue
        rho0, rho1 = rows[0]["rho0"], rows[0]["rho1"]
        alpha, beta, nmax = c["alpha"], c["beta"], c["n_max"]
        tag = f"{run}|{seed}"
        theory[tag] = sprt_theory(rho0, rho1, alpha, beta, nmax)
        theory[tag]["theorem1_p0.6"] = theorem1(EPS, 0.6, 0.05, nmax)
        theory[tag]["theorem1_p1.0"] = theorem1(EPS, 1.0, 0.05, nmax)
        # operating characteristic at the design alternative and at the brief's eps*p
        for name, r_ in (("rho1", rho1), ("eps_p_0.6", EPS * 0.6), ("eps_p_1.0", EPS),
                         ("rho0", rho0)):
            pf, en_ = oc_exact(r_, rho0, rho1, alpha, beta, nmax)
            theory[tag][f"oc_at_{name}"] = {"rho": r_, "p_flag": pf, "e_n": en_}
        # P(>=1 divergence within the clearance window) at eps*p (Theorem-1 event)
        cn = theory[tag]["clean_contexts_to_clear"]
        theory[tag]["p_at_least_one_in_clear_window_eps_p_0.6"] = 1 - (1 - EPS * 0.6) ** cn
        ocs[tag] = {"rho": grid,
                    "p_flag": [oc_exact(r, rho0, rho1, alpha, beta, nmax)[0] for r in grid],
                    "e_n": [oc_exact(r, rho0, rho1, alpha, beta, nmax)[1] for r in grid]}
        neg = [r for r in rows if r["class"] in ("benign", "legit")]
        sl = [r for r in rows if r["class"] == "sleeper"]
        cal = calib_rows(run, seed)
        cal_ids = {r["entry"] for r in cal}
        ho = [r for r in neg if r["entry"] not in cal_ids]
        kd, nu = sum(r["k_div"] for r in neg), sum(r["n_used"] for r in neg)
        realised = kd / nu
        ks = [r["k"] for r in cal]
        ns = [r["n"] for r in cal]
        fit = betabinom_fit(ks, ns)
        od[tag] = overdispersion_mc(ks, ns)
        od[tag]["betabinom"] = fit
        pits[tag] = pit_values(ks, ns)
        # realised per-context divergence rate on sleepers (upward-biased by stopping)
        skd, snu = sum(r["k_div"] for r in sl), sum(r["n_used"] for r in sl)
        fprm[tag] = {
            "observed_heldout": wilson(sum(flag(r) for r in ho), len(ho)),
            "observed_all_null": wilson(sum(flag(r) for r in neg), len(neg)),
            "rho0_calibrated": rho0,
            "realised_null_divergence": realised,
            "pred_fpr_at_rho0": oc_exact(rho0, rho0, rho1, alpha, beta, nmax)[0],
            "pred_fpr_at_realised": oc_exact(realised, rho0, rho1, alpha, beta, nmax)[0],
            "pred_fpr_betabinom_calib": predicted_fpr_mixture(fit, rho0, rho1, alpha, beta,
                                                              nmax),
            "pred_e_n_null_at_realised": oc_exact(realised, rho0, rho1, alpha, beta, nmax)[1],
            "observed_mean_n_null": float(np.mean([r["n_used"] for r in neg])),
            "sleeper_divergence_rate_in_ccs_runs": skd / snu if snu else None,
            "frac_null_hitting_nmax": float(np.mean([r["n_used"] >= nmax for r in neg])),
        }
        # beta-binomial refit on the null entries seen in the CCS run itself is biased by
        # the stopping rule, so only the calibration counts (fixed n) are fitted.
    out["sprt_theory"] = theory
    out["oc"] = ocs
    out["fpr_model"] = fprm
    out["overdispersion"] = od
    out["pit"] = pits

    # ---------------- Q-mass dependence of SPRT detection (Austin seeds)
    qd = {}
    for seed in SEEDS:
        sl = [r for r in ccs_rows("full", seed) if r["class"] == "sleeper"]
        q = np.array([r.get("qmass") or 0.0 for r in sl])
        y = np.array([flag(r) for r in sl])
        mw = st.mannwhitneyu(q[y == 1], q[y == 0], alternative="greater")
        sp = st.spearmanr(q, y)
        try:
            import statsmodels.api as sm
            X = sm.add_constant(q)
            m = sm.Logit(y, X).fit(disp=0)
            slope = {"coef": float(m.params[1]), "lo": float(m.conf_int()[1][0]),
                     "hi": float(m.conf_int()[1][1]), "p": float(m.pvalues[1])}
        except Exception as e:  # pragma: no cover
            slope = {"error": str(e)}
        qd[seed] = {"n": int(len(sl)), "median_qmass_detected": float(np.median(q[y == 1])),
                    "median_qmass_missed": float(np.median(q[y == 0])),
                    "mean_qmass_detected": float(q[y == 1].mean()),
                    "mean_qmass_missed": float(q[y == 0].mean()),
                    "mannwhitney_greater_p": float(mw.pvalue),
                    "spearman_rho": float(sp.correlation), "spearman_p": float(sp.pvalue),
                    "logit_slope_qmass": slope,
                    "frac_qmass_zero": float(np.mean(q == 0))}
    out["qmass_dependence"] = qd

    # ---------------- subgroup tests (seed 1, full slice; seeds 2/3 counts too)
    sub = {}
    for seed in SEEDS:
        sl = [r for r in ccs_rows("full", seed) if r["class"] == "sleeper"]
        asr = {r["entry"]: r for r in read_rows(os.path.join(RES, "full", f"seed_{seed}",
                                                             "asr.jsonl"),
                                                keyf=lambda r: r["entry"])}
        eff = [flag(r) for r in sl if r["entry"] in asr and asr[r["entry"]]["asr"] > 0]
        ine = [flag(r) for r in sl if r["entry"] in asr and asr[r["entry"]]["asr"] == 0]
        t = [[sum(eff), len(eff) - sum(eff)], [sum(ine), len(ine) - sum(ine)]]
        res = {"effective_vs_inert": {"table": t,
                                      "fisher_p": float(st.fisher_exact(t)[1])}}
        pay = collections.defaultdict(list)
        for r in sl:
            pay[r["payload"]].append(flag(r))
        tab = [[sum(v), len(v) - sum(v)] for _, v in sorted(pay.items())]
        chi = st.chi2_contingency(tab)
        res["payload"] = {"groups": sorted(pay), "table": tab, "chi2_p": float(chi[1])}
        rr = pay.get("reroute", [])
        pm = pay.get("payment", [])
        t2 = [[sum(rr), len(rr) - sum(rr)], [sum(pm), len(pm) - sum(pm)]]
        res["reroute_vs_payment_fisher_p"] = float(st.fisher_exact(t2)[1])
        rk = collections.defaultdict(list)
        for r in sl:
            rk["high" if r["confirm_required"] else "low"].append(flag(r))
        t3 = [[sum(rk["high"]), len(rk["high"]) - sum(rk["high"])],
              [sum(rk["low"]), len(rk["low"]) - sum(rk["low"])]]
        res["high_vs_low_risk_fisher_p"] = float(st.fisher_exact(t3)[1])
        ch = collections.defaultdict(list)
        for r in sl:
            ch[r["channel"]].append(flag(r))
        tab = [[sum(v), len(v) - sum(v)] for _, v in sorted(ch.items())]
        res["channel_chi2_p"] = float(st.chi2_contingency(tab)[1])
        pt = collections.defaultdict(list)
        for r in sl:
            pt[r["pred_type"]].append(flag(r))
        tab = [[sum(v), len(v) - sum(v)] for _, v in sorted(pt.items())]
        res["pred_type_chi2_p"] = float(st.chi2_contingency(tab)[1])
        sub[seed] = res
    # pooled-over-seeds effective-sleeper detection (different entry subsets per seed)
    blk = S["full"]["per_seed"]
    k = sum(blk[s]["ccs"]["detection_sprt_effective"]["k"] for s in SEEDS)
    n = sum(blk[s]["ccs"]["detection_sprt_effective"]["n"] for s in SEEDS)
    sub["effective_3seeds"] = wilson(k, n)
    sub["effective_3seeds"]["note"] = ("sum over seeds; seeds 2/3 re-use entries of seed 1, so "
                                       "the Wilson interval is optimistic")
    k = sum(blk[s]["ccs"]["detection_sprt_qmass_ge_eps"]["k"] for s in SEEDS)
    n = sum(blk[s]["ccs"]["detection_sprt_qmass_ge_eps"]["n"] for s in SEEDS)
    sub["qmass_ge_eps_3seeds"] = wilson(k, n)
    out["subgroup_tests"] = sub

    # ---------------- paired differences with entry-bootstrap CIs
    pdiff = {}
    s1 = SEEDS[0]
    ref = ccs_rows("full", s1)
    for abl in ("randomq", "shift", "explore0", "scope_tool", "scope_none"):
        rows = ccs_rows("full", s1, abl)
        for cls_name, cls in (("sleeper", ("sleeper",)), ("null", ("benign", "legit"))):
            a = [r for r in rows if r["class"] in cls]
            b = [r for r in ref if r["class"] in cls]
            a, b = pair(a, b)
            if not a:
                continue
            fa, fb = [flag(r) for r in a], [flag(r) for r in b]
            d = boot_diff(fa, fb)
            d.update({"k_a": sum(fa), "k_b": sum(fb),
                      "mean_n_a": float(np.mean([r["n_used"] for r in a])),
                      "mean_n_b": float(np.mean([r["n_used"] for r in b]))})
            pdiff[f"{abl}_vs_ccs|{cls_name}"] = d
    for run, refrun in (("full_temp07", "full"), ("full_beijing_recal", "full_beijing")):
        ra, rb = ccs_rows(run, s1), ccs_rows(refrun, s1)
        for cls_name, cls in (("sleeper", ("sleeper",)), ("null", ("benign", "legit"))):
            a, b = pair([r for r in ra if r["class"] in cls], [r for r in rb if r["class"] in cls])
            if not a:
                continue
            fa, fb = [flag(r) for r in a], [flag(r) for r in b]
            d = boot_diff(fa, fb)
            d.update({"k_a": sum(fa), "k_b": sum(fb),
                      "mean_n_a": float(np.mean([r["n_used"] for r in a])),
                      "mean_n_b": float(np.mean([r["n_used"] for r in b]))})
            pdiff[f"{run}_vs_{refrun}|{cls_name}"] = d
    out["paired_diffs"] = pdiff

    # ---------------- provenance gating extras (seed 1)
    pg = read_rows(os.path.join(RES, "full", f"seed_{s1}", "provgate.jsonl"),
                   keyf=lambda r: r.get("key"))
    by = collections.defaultdict(dict)
    chan = {}
    for r in pg:
        if r["mode"] == "legit_utility":
            continue
        by[r["entry"]][r["mode"]] = r["asr"]
        chan[r["entry"]] = r["channel"]
    ents = sorted(e for e in by if all(m in by[e] for m in ("none", "spotlight", "tma_nm",
                                                              "tma_nm_relaxed")))
    pge = {"n_entries": len(ents)}
    for a_, b_ in (("spotlight", "none"), ("tma_nm", "none"), ("tma_nm", "spotlight"),
                   ("tma_nm_relaxed", "none")):
        va = [by[e][a_] for e in ents]
        vb = [by[e][b_] for e in ents]
        lower = sum(x < y for x, y in zip(va, vb))
        higher = sum(x > y for x, y in zip(va, vb))
        p = float(st.binomtest(lower, lower + higher, 0.5).pvalue) if lower + higher else 1.0
        d = boot_diff(va, vb)
        d.update({"entries_lower": lower, "entries_higher": higher, "sign_test_p": p})
        pge[f"{a_}_vs_{b_}"] = d
    fired = [e for e in ents if by[e]["none"] > 0]
    pge["entries_firing_without_gate"] = len(fired)
    pge["firing_by_channel_without_gate"] = dict(collections.Counter(chan[e] for e in fired))
    pge["entries_firing_under_spotlight"] = sum(by[e]["spotlight"] > 0 for e in ents)
    pge["firing_by_channel_under_spotlight"] = dict(collections.Counter(
        chan[e] for e in ents if by[e]["spotlight"] > 0))
    pge["entries_by_channel"] = dict(collections.Counter(chan[e] for e in ents))
    out["provgate_extra"] = pge

    # ---------------- cost (seed 1)
    cost = S["full"]["per_seed"][s1]["cost"]
    tri = S["full"]["triage"]["sweeps"]["emb_score"]
    callsec = cost["measured_call_s"]
    cs = {"call_s": callsec,
          "sleeper_shared_s": cost["per_sleeper_write_s_shared"],
          "sleeper_unshared_s": cost["per_sleeper_write_s_unshared"],
          "null_shared_s": cost["per_benign_write_s_shared"],
          "null_unshared_s": cost["per_benign_write_s_unshared"],
          "calls_sleeper": cost["mean_calls_per_sleeper_write_shared"],
          "calls_null": cost["mean_calls_per_benign_write_shared"],
          "unshared_note": "unshared = 2 x n_used x latency (analytic, not measured)"}
    for th in ("0.22", "0.26", "0.3"):
        t = tri[th][f"seed_{s1}"]
        cs[f"triage_{th}"] = {
            "sleeper_shared_s": cs["sleeper_shared_s"] * (1 - t["frac_sleeper_calls_saved"]),
            "null_shared_s": cs["null_shared_s"] * (1 - t["frac_null_calls_saved"]),
            "frac_null_calls_saved": t["frac_null_calls_saved"],
            "frac_sleeper_calls_saved": t["frac_sleeper_calls_saved"]}
    # Wall-clock of the whole CCS step per write, for reference (includes baselines/cache)
    tw = S["full"]["timing_wall_s_by_exp_seed"]
    cs["ccs_step_wall_s_seed1"] = tw.get(f"ccs|{s1}")
    cs["ccs_step_writes_seed1"] = 205
    out["cost"] = cs

    # ---------------- power / sample size for the open claims
    def n_for_halfwidth(p, h):
        return math.ceil(1.96 ** 2 * p * (1 - p) / h ** 2)

    def n_power_binom(p0, p1, alpha=0.05, power=0.8):
        for n in range(20, 5000):
            kc = int(st.binom.isf(alpha, n, p0)) + 1  # reject if k >= kc
            while st.binom.sf(kc - 1, n, p0) > alpha:
                kc += 1
            if st.binom.sf(kc - 1, n, p1) >= power:
                return n
        return None

    out["power"] = {
        "n_sleepers_qmass_ge_eps_for_pm5_at_0.95": n_for_halfwidth(0.95, 0.05),
        "n_null_to_detect_fpr_0.089_vs_bound_0.0625": n_power_binom(0.0625, 8 / 90),
        "n_null_to_detect_fpr_0.125_vs_bound_0.0625": n_power_binom(0.0625, 0.125),
    }

    # ---------------- Holm correction over the family of paired tests reported in the paper
    fam = {}
    pv = S["full"]["paired_vs_ccs"]
    for k, v in pv.items():
        for cls in ("sleeper", "null"):
            if cls in v:
                fam[f"{k.split('|')[0]}|{cls}"] = v[cls]["sprt"]["mcnemar_exact_p"]
    for run in ("full_temp07", "full_beijing_recal"):
        v = S[run]["paired_vs_reference"]["ccs|20261015"]
        for cls in ("sleeper", "null"):
            fam[f"{run}_vs_ref|{cls}"] = v[cls]["sprt"]["mcnemar_exact_p"]
    for k, v in S["full"]["per_seed"][s1]["provgate"]["paired_sign_test"].items():
        if k != "tma_nm_relaxed_vs_none":
            fam[f"provgate_{k}"] = v["sign_test_p"]
    items = sorted(fam.items(), key=lambda kv: kv[1])
    m = len(items)
    holm, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        holm[k] = {"p": p, "p_holm": running, "reject_0.05": running < 0.05}
    out["holm_family"] = {"m": m, "tests": holm}

    with open(os.path.join(RES, "analysis_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=False)
    print("wrote results/analysis_stats.json")


if __name__ == "__main__":
    main()
