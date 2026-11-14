"""Spatiotemporal trigger predicates phi(context) and stochastic activation.

A predicate is a small dict. phi_holds(pred, ctx) is a deterministic boolean;
activation_prob(pred, ctx, base_p) returns the probability the payload fires given
the context, modelling soft-edged (ramped) predicates.
"""


def _geo(pred, ctx):
    if "area" in pred and ctx.get("area") != pred["area"]:
        return False
    if "poi_id" in pred and ctx.get("poi_id") != pred["poi_id"]:
        return False
    return "area" in pred or "poi_id" in pred


def _temporal(pred, ctx):
    ok = True
    if "dow" in pred:
        ok = ok and ctx.get("dow") in pred["dow"]
    if "hour_lo" in pred:
        h = ctx.get("hour", 12)
        lo, hi = pred["hour_lo"], pred["hour_hi"]
        if lo <= hi:
            ok = ok and (lo <= h < hi)
        else:  # wraps midnight
            ok = ok and (h >= lo or h < hi)
    return ok


def _motif(pred, ctx):
    cat = pred.get("category_contains", "")
    for r in ctx.get("recent", []):
        if cat in (r.get("category") or ""):
            return True
    return False


def phi_holds(pred, ctx):
    t = pred["type"]
    if t == "geo":
        return _geo(pred, ctx)
    if t == "temporal":
        return _temporal(pred, ctx)
    if t == "geo_temporal":
        return _geo(pred, ctx) and _temporal(pred, ctx)
    if t == "motif":
        return _motif(pred, ctx)
    if t == "request":
        q = str(ctx.get("query", "")).lower()
        return any(k in q for k in pred.get("request_keywords", ["take me", "directions"]))
    if t == "composite":
        return all(phi_holds(sub, ctx) for sub in pred.get("parts", []))
    if t == "always":
        return True
    raise ValueError(f"unknown predicate type {t}")


def activation_prob(pred, ctx, base_p):
    """Payload fires w.p. base_p when phi holds (hard predicate), or a ramped
    probability near the temporal boundary for soft-edged predicates."""
    if not pred.get("soft"):
        return base_p if phi_holds(pred, ctx) else 0.0
    # soft temporal: full strength inside the window; linear ramp over a 2-hour
    # margin as the hour approaches the window start; 0 further out.
    if "hour_lo" in pred:
        h = ctx.get("hour", 12)
        lo, hi = pred["hour_lo"], pred["hour_hi"]
        inside = (lo <= h < hi) if lo <= hi else (h >= lo or h < hi)
        if inside:
            if "dow" in pred and ctx.get("dow") not in pred["dow"]:
                return 0.0
            return base_p
        hours_to_start = (lo - h) % 24  # forward distance to window opening
        margin = 2.0
        frac = max(0.0, 1.0 - hours_to_start / margin)
        return base_p * frac
    return base_p if phi_holds(pred, ctx) else 0.0


def predicate_text(pred, soft_margin=2.0):
    """Natural-language rendering of the trigger condition.

    Contract (Stage-7 fix): the rendered text states EXACTLY the structured
    predicate and nothing else, so phi_holds(pred, c) is true iff the condition
    stated in the entry text holds in c.  Earlier revisions injected the entry's
    scope area/place into every rendering, which made the text strictly stronger
    than the label used for phi and Q-mass (reviewer-rigor major issue 2).
    Soft-edged predicates render the ramp (reviewer: soft edges were text-identical
    to hard ones).
    """
    t = pred.get("type", "always")
    bits = []
    if t in ("geo", "geo_temporal"):
        if pred.get("area"):
            bits.append(f"in {pred['area']}")
        if pred.get("poi_id") and pred.get("place"):
            bits.append(f"near {pred['place']}")
    if t in ("temporal", "geo_temporal"):
        if "dow" in pred:
            names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"]
            bits.append("on " + "/".join(names[d] for d in pred["dow"]))
        if "hour_lo" in pred:
            if pred.get("soft"):
                bits.append(f"from about {int(soft_margin)} hours before "
                            f"{pred['hour_lo']:02d}:00 (more strongly the closer it is) "
                            f"until {pred['hour_hi']:02d}:00")
            else:
                bits.append(f"between {pred['hour_lo']:02d}:00 and "
                            f"{pred['hour_hi']:02d}:00")
    if t == "motif":
        bits.append("shortly after visiting a "
                    f"{pred.get('category_contains','')} location")
    if t == "request":
        bits.append(f"asking {pred.get('request_text','for directions')}")
    return " ".join(bits) if bits else "any time"


# --------------------------- Q-mass under the sampler ------------------------
# Theorem-side Q is the SAMPLING MIXTURE Qt = (1-lambda) Q_u + lambda Explore,
# not the user's empirical Q_u alone (reviewer-rigor: "_qmass uses the empirical
# Q_u while the theorem is stated for the mixture").  phi-mass is the probability
# that the predicate holds under a draw; activation p is reported separately.


def phi_mass_user(pred, contexts):
    if not contexts:
        return 0.0
    return sum(1 for c in contexts if phi_holds(pred, c)) / len(contexts)


def phi_mass_mixture(pred, contexts, explore_sampler, exploration=0.3, n_mc=400):
    """(1-lambda)*Q_u(phi) + lambda*Explore(phi); Explore(phi) by Monte Carlo."""
    qu = phi_mass_user(pred, contexts)
    if exploration <= 0:
        return qu, qu, 0.0
    hits = 0
    for _ in range(n_mc):
        if phi_holds(pred, explore_sampler()):
            hits += 1
    qe = hits / n_mc
    return (1 - exploration) * qu + exploration * qe, qu, qe
