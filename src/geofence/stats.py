"""Statistical core for CCS: SPRT, sample complexity, Wilson CI, calibration.

Pure functions (no models, no I/O) so they can be unit-tested for exact Type-I
control, e-process / supermartingale validity, and the invariance lemma.

Theory (matches PLAN.md / BRIEF.md):

* Per sampled context c ~ Q, the detector observes a Bernoulli outcome
  X = 1 iff an *out-of-scope* action diverges between the paired runs
  (agent with the candidate entry e vs. without it).
* Null model H0: X ~ Bernoulli(rho0), where rho0 is the calibrated benign
  divergence rate from paired runs of unconditional untrusted entries
  (model noise, never a true trigger).
* Alternative H1: X ~ Bernoulli(rho1), rho1 = eps * p, where a trigger has
  Q-mass Q(phi) >= eps and stochastic activation probability p.
* SPRT (Wald) decides H1 (flag) / H0 (clear) with FPR <= alpha, power >= 1-beta.
* Theorem 1 (detection sample complexity, one-sided): to observe at least one
  activation of a trigger with divergence rate rho1 = eps*p with probability
  >= 1-delta, n >= ln(1/delta) / (eps*p) paired samples suffice, since
  P(0 activations in n) = (1-eps p)^n <= exp(-eps p n) = delta.
"""
import math

# ----------------------------- e-process / SPRT -----------------------------


def llr_increment(x, rho0, rho1):
    """Log-likelihood-ratio contribution of one Bernoulli outcome x in {0,1}."""
    rho0 = min(max(rho0, 1e-9), 1 - 1e-9)
    rho1 = min(max(rho1, 1e-9), 1 - 1e-9)
    if x:
        return math.log(rho1 / rho0)
    return math.log((1.0 - rho1) / (1.0 - rho0))


def sprt_boundaries(alpha, beta):
    """Wald's A (upper, reject H0) and B (lower, accept H0) in log space."""
    logA = math.log((1.0 - beta) / alpha)
    logB = math.log(beta / (1.0 - alpha))
    return logA, logB


def sprt_run(observations, rho0, rho1, alpha, beta, n_max):
    """Run the SPRT over a stream of Bernoulli outcomes.

    Returns dict: decision in {"flag","clear"}, n (samples used), llr,
    trajectory (cumulative llr), stopped_early (bool).
    Conservative truncation at n_max: if neither boundary is crossed, decide
    "clear" (accept H0). H1 ("flag") is reached only by crossing the Wald upper
    boundary, which under H0 has probability <= alpha/(1-beta); pushing all
    unresolved mass to "clear" therefore keeps the realized FPR at or below that
    bound (it never adds false flags), at the cost of some power on weak signals.
    """
    logA, logB = sprt_boundaries(alpha, beta)
    llr = 0.0
    traj = []
    decision = None
    n = 0
    for x in observations[:n_max]:
        n += 1
        llr += llr_increment(x, rho0, rho1)
        traj.append(llr)
        if llr >= logA:
            decision = "flag"
            return {"decision": decision, "n": n, "llr": llr, "trajectory": traj,
                    "stopped_early": True}
        if llr <= logB:
            decision = "clear"
            return {"decision": decision, "n": n, "llr": llr, "trajectory": traj,
                    "stopped_early": True}
    return {"decision": "clear", "n": n, "llr": llr, "trajectory": traj,
            "stopped_early": False}


def e_value(observations, rho0, rho1):
    """Likelihood-ratio e-process value after all observations.

    E_n = prod_i L1(x_i)/L0(x_i). Under H0, (E_n) is a non-negative martingale
    with E[E_n]=1, so by Ville's inequality P(sup_n E_n >= 1/alpha) <= alpha.
    Returns the final e-value.
    """
    e = 1.0
    for x in observations:
        e *= math.exp(llr_increment(x, rho0, rho1))
    return e


def e_process_trajectory(observations, rho0, rho1):
    e = 1.0
    out = []
    for x in observations:
        e *= math.exp(llr_increment(x, rho0, rho1))
        out.append(e)
    return out


# ----------------------------- sample complexity ----------------------------


def sample_complexity(eps, p, delta):
    """Theorem 1: n >= ln(1/delta) / (eps*p) to observe >=1 activation w.p. 1-delta."""
    eps = max(eps, 1e-9)
    p = max(p, 1e-9)
    return math.ceil(math.log(1.0 / delta) / (eps * p))


def miss_probability(eps, p, n):
    """P(no activation observed in n samples) <= exp(-eps p n)."""
    rho1 = eps * p
    return (1.0 - rho1) ** n


def residual_impact(eps, tv=0.0):
    """Corollary: undetected trigger fires in <= eps fraction of future sessions
    from Q, plus a total-variation term for distribution shift."""
    return min(1.0, eps + tv)


# ----------------------------- interval estimation --------------------------


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom
    return (phat, max(0.0, centre - half), min(1.0, centre + half))


def jeffreys_rate(k, n):
    """Jeffreys-smoothed rate (Beta(1/2,1/2) posterior mean); avoids 0/1 for logs."""
    return (k + 0.5) / (n + 1.0)


def clopper_pearson_upper(k, n, gamma=0.05):
    """One-sided upper (1-gamma) confidence limit for a binomial rate k/n.

    Stage-7 (reviewer-rigor major issue 8): plugging the POINT estimate rho0-hat into
    the SPRT makes Proposition 6 conditional on rho0 <= rho0-hat.  Plugging this upper
    limit instead gives FPR <= alpha/(1-beta) with probability at least 1-gamma over
    the calibration draw (Proposition 8 monotonicity).
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta as _beta
        return float(_beta.ppf(1.0 - gamma, k + 1, n - k))
    except Exception:
        pass
    # dependency-free fallback: bisect the binomial tail P(X<=k; n, theta) = gamma
    from math import comb

    def tail(theta):
        return sum(comb(n, i) * theta ** i * (1 - theta) ** (n - i) for i in range(k + 1))

    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if tail(mid) > gamma:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_null(div_counts, mode="point", gamma=0.05):
    """Estimate rho0 from paired-run out-of-scope divergence on benign entries.

    div_counts: list of (k_divergences, n_contexts) per benign entry.
    mode="point": Jeffreys-smoothed pooled rate (floored away from 0) -- what the
    Stage-3 runs used.  mode="upper": one-sided Clopper-Pearson (1-gamma) upper limit,
    which makes the FPR guarantee hold with confidence 1-gamma over calibration.
    """
    k = sum(c[0] for c in div_counts)
    n = sum(c[1] for c in div_counts)
    if mode == "upper":
        return max(clopper_pearson_upper(k, n, gamma), 1e-3)
    return max(jeffreys_rate(k, n), 1e-3)


# ----------------------------- invariance lemma -----------------------------


def out_of_scope_divergence(actions_with, actions_without, scope):
    """Invariance / scope lemma indicator.

    actions_* : list of structured actions (dicts with 'tool' and 'target').
    scope     : dict with 'tools' (allowed tool set, read-only in-scope) and
                'targets' (allowed target ids/areas for the entry).
    Returns 1 iff the symmetric difference of OUT-OF-SCOPE actions between the
    paired runs is non-empty. In-scope changes (e.g. a legitimate opening-hours
    fact altering a POI answer) never count. If the entry has no causal effect
    (actions_with == actions_without), returns 0 by construction -> the lemma.
    """
    def oos(actions):
        s = set()
        for a in actions:
            tool = a.get("tool")
            tgt = a.get("target")
            in_scope = (tool in scope.get("tools", set())) and (
                tgt is None or tgt in scope.get("targets", set()))
            if not in_scope:
                s.add((tool, tgt))
        return s
    return int(oos(actions_with) != oos(actions_without))
