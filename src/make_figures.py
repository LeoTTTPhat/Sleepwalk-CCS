"""Paper figures (vector PDF) from results/summary.json + results/analysis_stats.json + raw JSONL.

Run:  shared/.venv/bin/python src/make_figures.py      (after src/analyze.py)
Writes figures/fig_*.pdf. No model calls; deterministic.

Style: IEEE two-column (single column 3.5 in, double column 7.16 in), 8 pt serif text,
Okabe-Ito colour-blind-safe palette (validated; CVD separation of the pink/green pair is
in the 6-8 dE band, so every series also carries a distinct marker / line style).
Uncertainty: Wilson 95% intervals for proportions, entry-bootstrap 95% intervals for
means of per-entry rates and paired differences (as stored in the results files).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SEEDS = ("20261015", "20261016", "20261017")

# Okabe-Ito, fixed order (never cycled)
C = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "pink": "#CC79A7",
     "orange": "#E69F00", "sky": "#56B4E9", "grey": "#7F7F7F", "black": "#000000"}
SERIES = [C["blue"], C["verm"], C["green"], C["pink"], C["orange"]]
MARK = ["o", "s", "^", "D", "v"]
LS = ["-", "--", "-.", ":", (0, (5, 1))]
W1, W2 = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 1.2, "lines.markersize": 4,
    "errorbar.capsize": 1.8, "axes.grid": True, "grid.color": "#DDDDDD",
    "grid.linewidth": 0.4, "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})

S = json.load(open(os.path.join(RES, "summary.json")))
A = json.load(open(os.path.join(RES, "analysis_stats.json")))
F = S["full"]
P1 = F["per_seed"][SEEDS[0]]


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def err(d, rate_key="rate"):
    m = d.get(rate_key, d.get("mean"))
    return m, [[m - d["lo"]], [d["hi"] - m]]


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right")


# ----------------------------------------------------------------- Fig 1: attack
def fig_attack():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
    rows = []  # (label, dict with mean/lo/hi, group, n, seed2/3 values)
    boot = F["across_seeds_entry_bootstrap"]["asr.asr"]
    rows.append(("Qwen2.5-7B, Austin (3 seeds)", boot, "setting", boot["n"], None))
    pb = S["full_beijing"]["per_seed"][SEEDS[0]]["asr"]["asr"]
    rows.append(("Qwen2.5-7B, Beijing", pb, "setting", pb["n"], None))
    pt = S["full_temp07"]["per_seed"][SEEDS[0]]["asr"]["asr"]
    rows.append(("Qwen2.5-7B, Austin, T=0.7", pt, "setting", pt["n"], None))
    ph = S["full_hammer"]["per_seed"][SEEDS[0]]["asr"]["asr"]
    rows.append(("Hammer2.1-7B, Austin", ph, "setting", ph["n"], None))
    names = {"exfil": "exfiltration", "payment": "payment", "reroute": "reroute",
             "forwarded_doc": "forwarded doc", "poi_review": "POI review",
             "tool_output": "tool output", "web_snippet": "web snippet",
             "geo": "geographic", "geo_temporal": "geo+temporal", "motif": "trajectory motif",
             "temporal": "temporal"}
    for grp, key in (("payload", "asr_by_payload"), ("channel", "asr_by_channel"),
                     ("predicate", "asr_by_pred_type")):
        for k, v in sorted(P1["asr"][key].items()):
            others = [F["per_seed"][s]["asr"][key].get(k, {}).get("mean") for s in SEEDS[1:]]
            rows.append((names.get(k, k), v, grp, v["n"], others))
    ypos, y = [], 0
    last = None
    ticks = []
    for lab, d, grp, n, others in rows:
        if last is not None and grp != last:
            y += 0.6
        last = grp
        ypos.append(y)
        col = C["black"] if grp == "setting" else SERIES[["payload", "channel", "predicate"].index(grp)]
        mk = "o" if grp == "setting" else MARK[["payload", "channel", "predicate"].index(grp) + 1]
        m, e = err(d)
        ax.errorbar([m], [y], xerr=e, fmt=mk, color=col, ms=3.5, lw=0.9)
        if others:
            for o in others:
                if o is not None:
                    ax.plot([o], [y], marker="|", color=C["grey"], ms=5, lw=0)
        ticks.append(f"{lab}  (n={n})")
        y += 1
    ax.set_yticks(ypos)
    ax.set_yticklabels(ticks)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 0.82)   # headroom so the widest CI (web snippet) is not clipped
    ax.set_xlabel(r"ASR when $\varphi$ holds (mean of per-entry rates)")
    for gname, gy in (("setting", ypos[0]), ("payload", ypos[4]), ("channel", ypos[7]),
                      ("predicate", ypos[11])):
        ax.text(0.74, gy, gname, ha="right", va="center", fontsize=6.5, color=C["grey"],
                style="italic")
    ax.plot([], [], marker="|", color=C["grey"], lw=0, ms=5, label="seeds 2, 3 (50-entry slice)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    panel_label(ax, "(a)", x=-0.62)

    # (b) stealth: content filters vs CCS
    tri = F["triage"]["filters_on_positive_sleepers"]
    fl = F["filters"]
    labs = [("llama_guard", "Llama-Guard-3-8B"), ("protectai", "ProtectAI DeBERTa"),
            ("deepset", "deepset DeBERTa")]
    cats = [("sleeper (85 pos.)", None), ("benign uncond.", "benign"), ("legit cond.", "legit")]
    yb = np.arange(len(labs) + 1)
    h = 0.25
    for j, (cname, ck) in enumerate(cats):
        for i, (fk, fn) in enumerate(labs):
            d = tri[fk] if ck is None else fl[fk][ck]
            m, e = err(d)
            bx.barh(yb[i] + (j - 1) * h, m, height=h * 0.85, color=SERIES[j], edgecolor="white",
                    linewidth=0.4, label=cname if i == 0 else None)
            bx.errorbar([m], [yb[i] + (j - 1) * h], xerr=e, fmt="none", ecolor="#333333", lw=0.6)
    # CCS row: SPRT detection on 85 positives; held-out null FPR (benign / legit)
    c = P1["ccs"]
    for j, d in enumerate([c["detection_sprt_only"], c["fpr_benign_sprt_heldout"],
                           c["fpr_legit_sprt_heldout"]]):
        m, e = err(d)
        bx.barh(yb[-1] + (j - 1) * h, m, height=h * 0.85, color=SERIES[j], edgecolor="white",
                linewidth=0.4, hatch="////" if j == 0 else None)
        bx.errorbar([m], [yb[-1] + (j - 1) * h], xerr=e, fmt="none", ecolor="#333333", lw=0.6)
    bx.set_yticks(yb)
    bx.set_yticklabels([fn for _, fn in labs] + ["CCS (SPRT, ours)"])
    bx.invert_yaxis()
    bx.set_xlim(0, 1.0)
    bx.set_xlabel("flag rate (Wilson 95% CI)")
    bx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2,
              columnspacing=0.8, handlelength=1.2)
    panel_label(bx, "(b)", x=-0.42)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_attack.pdf")


# --------------------------------------------------------------- Fig 2: provgate
def fig_provgate():
    pg = P1["provgate"]
    modes = [("none", "no gate"), ("spotlight", "spotlighting"), ("tma_nm", "origin-tag rule"),
             ("tma_nm_relaxed", "origin-tag (relaxed)")]
    chans = [("all", "all"), ("web_snippet", "web snippet"), ("poi_review", "POI review"),
             ("forwarded_doc", "fwd. doc"), ("tool_output", "tool output")]
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    x = np.arange(len(chans))
    w = 0.19
    for j, (mk, mn) in enumerate(modes):
        for i, (ck, cn) in enumerate(chans):
            d = pg["asr_by_mode"][mk] if ck == "all" else pg["asr_by_mode_channel"][f"{mk}|{ck}"]
            m, e = err(d)
            ax.errorbar([x[i] + (j - 1.5) * w], [m], yerr=e, fmt=MARK[j], color=SERIES[j],
                        ms=3.2, lw=0.8, label=mn if i == 0 else None,
                        mfc=SERIES[j] if j < 2 else "white")
    n = {ck: pg["asr_by_mode_channel"][f"none|{ck}"]["n"] for ck, _ in chans[1:]}
    n["all"] = pg["asr_by_mode"]["none"]["n"]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cn}\n(n={n[ck]})" for ck, cn in chans], fontsize=6.5)
    ax.set_ylabel("ASR when $\\varphi$ holds")
    ax.set_ylim(-0.03, 0.9)
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_provgate.pdf")


# ------------------------------------------------------------ Fig 3: calibration
RUNLAB = {"full|20261015": "Austin s1", "full|20261016": "Austin s2",
          "full|20261017": "Austin s3", "full_beijing|20261015": "Beijing (orig.)",
          "full_beijing_recal|20261015": "Beijing (recal.)",
          "full_temp07|20261015": "Austin T=0.7"}
RUNCOL = {"full|20261015": C["blue"], "full|20261016": C["blue"], "full|20261017": C["blue"],
          "full_beijing|20261015": C["verm"], "full_beijing_recal|20261015": C["green"],
          "full_temp07|20261015": C["pink"]}
RUNMK = {"full|20261015": "o", "full|20261016": "s", "full|20261017": "^",
         "full_beijing|20261015": "D", "full_beijing_recal|20261015": "v",
         "full_temp07|20261015": "P"}


def _summary_ccs(tag):
    run, seed = tag.split("|")
    return S[run]["per_seed"][seed]


def fig_calibration():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3))
    ax, bx, cx = axs
    tags = list(RUNLAB)
    for t in tags:
        p = _summary_ccs(t)
        cal = p["calibration"]["pooled_divergence"]
        real = p["ccs"]["null_divergence_in_ccs_runs"]
        ax.errorbar([cal["rate"]], [real["rate"]],
                    xerr=[[cal["rate"] - cal["lo"]], [cal["hi"] - cal["rate"]]],
                    yerr=[[real["rate"] - real["lo"]], [real["hi"] - real["rate"]]],
                    fmt=RUNMK[t], color=RUNCOL[t], ms=3.5, lw=0.7, label=RUNLAB[t])
    lim = [0, 0.085]
    ax.plot(lim, lim, color=C["grey"], ls=":", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"calibrated null divergence $\hat\rho_0$")
    ax.set_ylabel("null divergence realised in CCS runs")
    ax.legend(loc="upper left", fontsize=5.4, handletextpad=0.2, borderaxespad=0.15,
              labelspacing=0.25)
    panel_label(ax, "(a)", x=-0.18)

    # (b) FPR: observed held-out vs model predictions vs bound
    fm = A["fpr_model"]
    y = np.arange(len(tags))
    for i, t in enumerate(tags):
        o = fm[t]["observed_heldout"]
        m, e = err(o)
        bx.errorbar([m], [i], xerr=e, fmt=RUNMK[t], color=RUNCOL[t], ms=3.8, lw=0.9)
        bx.plot([fm[t]["pred_fpr_at_rho0"]], [i - 0.25], marker="|", color=C["grey"], ms=6, lw=0)
        bx.plot([fm[t]["pred_fpr_at_realised"]], [i - 0.25], marker="x", color="#333333", ms=3.5,
                lw=0)
        bx.text(min(o["hi"] + 0.012, 0.44), i - 0.3, f"{o['k']}/{o['n']}", va="center",
                fontsize=5.8, color=RUNCOL[t])
    bx.axvline(0.0625, color=C["verm"], ls="--", lw=0.8)
    bx.text(0.068, -0.55, r"$\alpha/(1-\beta)$", color=C["verm"], fontsize=6.5)
    bx.set_yticks(y)
    bx.set_yticklabels([RUNLAB[t] for t in tags])
    bx.invert_yaxis()
    bx.set_xlim(0, 0.5)
    bx.set_xlabel("held-out null FPR (SPRT flags)")
    bx.plot([], [], marker="|", color=C["grey"], lw=0, ms=6, label=r"OC at $\hat\rho_0$")
    bx.plot([], [], marker="x", color="#333333", lw=0, ms=3.5, label="OC at realised rate")
    bx.legend(loc="lower right", fontsize=5.8, borderaxespad=0.1)
    panel_label(bx, "(b)", x=-0.33)

    # (c) QQ of randomized PIT of calibration counts under the pooled binomial
    groups = [("Austin s1-s3 (70 entries)", [f"full|{s}" for s in SEEDS], C["blue"], "o"),
              ("Beijing recal. (40)", ["full_beijing_recal|20261015"], C["green"], "v"),
              ("Austin T=0.7 (20)", ["full_temp07|20261015"], C["pink"], "P")]
    rng = np.random.default_rng(1)
    for lab, tl, col, mk in groups:
        u = np.sort(np.concatenate([A["pit"][t] for t in tl]))
        n = len(u)
        q = (np.arange(1, n + 1) - 0.5) / n
        cx.plot(q, u, marker=mk, ms=2.2, lw=0, color=col, label=lab, alpha=0.9)
        if tl[0].startswith("full|"):
            sims = np.sort(rng.uniform(size=(4000, n)), axis=1)
            lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
            cx.fill_between(q, lo, hi, color=C["blue"], alpha=0.12, lw=0,
                            label="95% band (n=70)")
    cx.plot([0, 1], [0, 1], color=C["grey"], ls=":", lw=0.8)
    cx.set_xlabel("uniform quantile")
    cx.set_ylabel("randomised PIT, calib. counts")
    cx.set_xlim(0, 1)
    cx.set_ylim(0, 1)
    cx.legend(loc="lower right", fontsize=5.6, borderaxespad=0.1)
    panel_label(cx, "(c)", x=-0.18)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_calibration.pdf")


# ------------------------------------------------------------------ Fig 4: power
def fig_power():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    ax, bx, cx = axs
    oc = A["oc"]
    th = A["sprt_theory"]
    for i, (t, lab) in enumerate((("full|20261015", r"Austin s1 ($\hat\rho_0$=%.3f)"),
                                  ("full_temp07|20261015", r"Austin T=0.7 ($\hat\rho_0$=%.3f)"),
                                  ("full_beijing_recal|20261015", r"Beijing recal. ($\hat\rho_0$=%.3f)"))):
        o = oc[t]
        ax.plot(o["rho"], o["p_flag"], ls=LS[i], color=SERIES[[0, 3, 2][i]],
                label=lab % th[t]["rho0"])
    t1 = th["full|20261015"]
    for key, txt, yy in (("oc_at_eps_p_0.6", r"$\epsilon p$=0.036", 0.45),
                         ("oc_at_rho1", r"$\rho_1$", 0.86)):
        r = t1[key]["rho"]
        ax.axvline(r, color=C["grey"], ls=":", lw=0.7)
        ax.text(r + 0.004, yy, txt, fontsize=6, color="#333333")
    ax.axhline(0.0625, color=C["verm"], ls="--", lw=0.7)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"true per-context divergence rate $\rho$")
    ax.set_ylabel("P(flag) of truncated SPRT (exact)")
    ax.legend(loc="lower right", fontsize=5.8)
    panel_label(ax, "(a)", x=-0.18)

    B = [5, 10, 20, 30, 40]
    curves = [("Austin s1 (85)", F["per_seed"][SEEDS[0]]["ccs"]),
              ("Austin s2 (50)", F["per_seed"][SEEDS[1]]["ccs"]),
              ("Austin s3 (50)", F["per_seed"][SEEDS[2]]["ccs"]),
              ("Austin T=0.7 (30)", S["full_temp07"]["per_seed"][SEEDS[0]]["ccs"]),
              ("Beijing recal. (36)", S["full_beijing_recal"]["per_seed"][SEEDS[0]]["ccs"])]
    cols = [C["blue"], C["blue"], C["blue"], C["pink"], C["green"]]
    for i, (lab, c) in enumerate(curves):
        for axx, key in ((bx, "detection_sprt_by_budget"), (cx, "fpr_null_sprt_by_budget")):
            bb = [b for b in B if str(b) in c[key]]
            ys = [c[key][str(b)]["rate"] for b in bb]
            axx.plot(bb, ys, marker=MARK[i], ls=LS[i], color=cols[i], ms=3, lw=1.0, label=lab)
            if i == 0:
                lo = [c[key][str(b)]["lo"] for b in bb]
                hi = [c[key][str(b)]["hi"] for b in bb]
                axx.fill_between(bb, lo, hi, color=cols[i], alpha=0.13, lw=0)
    bx.set_xlabel("per-entry budget B (contexts)")
    bx.set_ylabel("sleepers flagged by n <= B")
    bx.set_ylim(0, 1)
    bx.axvline(19, color=C["grey"], ls=":", lw=0.7)
    bx.text(19.5, 0.88, "19 clean contexts\nclear an entry", fontsize=5.8, color="#333333")
    bx.legend(loc="lower right", fontsize=5.8)
    panel_label(bx, "(b)", x=-0.18)
    cx.axhline(0.0625, color=C["verm"], ls="--", lw=0.7)
    cx.text(5, 0.068, r"$\alpha/(1-\beta)$", color=C["verm"], fontsize=6)
    cx.set_xlabel("per-entry budget B (contexts)")
    cx.set_ylabel("null entries flagged by n <= B")
    cx.set_ylim(0, 0.2)
    panel_label(cx, "(c)", x=-0.18)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_power.pdf")


# ---------------------------------------------------- Fig 5: detection breakdown
def fig_breakdown():
    c1 = P1["ccs"]
    rows = [("all sleepers", c1["detection_sprt_only"], "all", "detection_sprt_only", None)]
    rows += [("fire under $\\varphi$ (ASR>0)", c1["detection_sprt_effective"], "activation",
              "detection_sprt_effective", None),
             ("never fire (ASR=0)", c1["detection_sprt_ineffective"], "activation",
              "detection_sprt_ineffective", None)]
    nm = {"exfil": "exfiltration", "payment": "payment", "reroute": "reroute"}
    for k in ("reroute", "exfil", "payment"):
        rows.append((nm[k], c1["detection_sprt_by_payload"][k], "payload",
                     "detection_sprt_by_payload", k))
    rows.append(("low-risk tier", c1["detection_low_risk_sprt"], "tier", "detection_low_risk_sprt", None))
    rows.append(("high-risk tier", c1["detection_high_risk_sprt"], "tier", "detection_high_risk_sprt", None))
    for k in ("0.00-0.06", "0.06-0.15", "0.15-0.40", "0.40-1.01"):
        if k in c1["detection_sprt_by_qmass"]:
            rows.append((f"Q-mass {k.replace('1.01', '1.0')}", c1["detection_sprt_by_qmass"][k],
                         "qmass", "detection_sprt_by_qmass", k))
    rows.append(("Q-mass >= eps", c1["detection_sprt_qmass_ge_eps"], "qmass",
                 "detection_sprt_qmass_ge_eps", None))
    rows.append(("split pair", c1["detection_sprt_by_split"]["True"], "robust",
                 "detection_sprt_by_split", "True"))
    rows.append(("soft-edged", c1["detection_sprt_by_soft"]["True"], "robust",
                 "detection_sprt_by_soft", "True"))
    gcol = {"all": C["black"], "activation": C["blue"], "payload": C["verm"],
            "tier": C["green"], "qmass": C["pink"], "robust": C["orange"]}
    gmk = {"all": "o", "activation": "s", "payload": "^", "tier": "D", "qmass": "v",
           "robust": "P"}
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    y, last, ys, labs = 0, None, [], []
    for lab, d, g, key, sub in rows:
        if last is not None and g != last:
            y += 0.5
        last = g
        m, e = err(d)
        ax.errorbar([m], [y], xerr=e, fmt=gmk[g], color=gcol[g], ms=3.3, lw=0.9)
        for s in SEEDS[1:]:
            cc = F["per_seed"][s]["ccs"].get(key)
            if cc is not None and sub is not None:
                cc = cc.get(sub)
            if cc:
                ax.plot([cc["rate"]], [y], marker="|", color=C["grey"], ms=5, lw=0)
        ax.text(1.03, y, f"{d['k']}/{d['n']}", va="center", fontsize=6, transform=ax.get_yaxis_transform())
        ys.append(y)
        labs.append(lab)
        y += 1
    ax.set_yticks(ys)
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("SPRT detection rate, seed 1 (Wilson 95% CI)")
    ax.plot([], [], marker="|", color=C["grey"], lw=0, ms=5, label="seeds 2, 3")
    ax.legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, "fig_breakdown.pdf")


# -------------------------------------------------------------- Fig 6: ablations
def fig_ablations():
    pdif = A["paired_diffs"]
    pv = F["paired_vs_ccs"]
    rows = [
        ("scope: tool-only", "scope_tool_vs_ccs", pv["scope_tool_vs_ccs|20261015"]),
        ("scope: none (any change)", "scope_none_vs_ccs", pv["scope_none_vs_ccs|20261015"]),
        ("contexts: pure Q (expl. 0)", "explore0_vs_ccs", pv["explore0_vs_ccs|20261015"]),
        ("contexts: uniform, no Q", "randomq_vs_ccs", pv["randomq_vs_ccs|20261015"]),
        ("contexts: shifted Q", "shift_vs_ccs", pv["shift_vs_ccs|20261015"]),
        ("decoding: T=0.7", "full_temp07_vs_full",
         S["full_temp07"]["paired_vs_reference"]["ccs|20261015"]),
        ("Beijing: recalibrated", "full_beijing_recal_vs_full_beijing",
         S["full_beijing_recal"]["paired_vs_reference"]["ccs|20261015"]),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.1), sharey=True)
    for j, (cls, ax, title) in enumerate(((("sleeper"), axs[0], "change in SPRT detection"),
                                          (("null"), axs[1], "change in null FPR (all null)"))):
        for i, (lab, key, mc) in enumerate(rows):
            d = pdif.get(f"{key}|{cls}")
            if d is None:
                ax.text(0, i, "n/a (sleepers only)", ha="center", va="center", fontsize=6,
                        color=C["grey"])
                continue
            m = d["diff"]
            col = SERIES[0] if i < 2 else (SERIES[1] if i < 5 else SERIES[2])
            mk = MARK[0] if i < 2 else (MARK[1] if i < 5 else MARK[2])
            ax.errorbar([m], [i], xerr=[[m - d["lo"]], [d["hi"] - m]], fmt=mk, color=col,
                        ms=3.5, lw=0.9)
            p = mc.get(cls, {}).get("sprt", {}).get("mcnemar_exact_p")
            ps = "" if p is None else (f"p={p:.0e}" if p < 1e-3 else f"p={p:.2f}" if p >= 0.01 else f"p={p:.3f}")
            ax.text(1.02, i, f"{d['k_a']}/{d['n']} vs {d['k_b']}/{d['n']}  {ps}", va="center",
                    fontsize=5.8, transform=ax.get_yaxis_transform())
        ax.axvline(0, color=C["grey"], lw=0.8)
        ax.set_xlabel(title + " (paired, 95% entry bootstrap)")
        ax.set_xlim(-0.8, 0.8)
    axs[0].set_yticks(range(len(rows)))
    axs[0].set_yticklabels([r[0] for r in rows])
    axs[0].invert_yaxis()
    fig.tight_layout(w_pad=7.0)
    save(fig, "fig_ablations.pdf")


# ------------------------------------------------------ Fig 7: triage and cost
def fig_triage_cost():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.45), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax, bx, cx = axs
    sw = F["triage"]["sweeps"]["emb_score"]
    th = sorted(sw, key=float)
    x = [float(t) for t in th]
    series = [("sleepers passing", [sw[t]["pass_sleeper_positive"]["rate"] for t in th]),
              ("benign passing", [sw[t]["pass_benign"]["rate"] for t in th]),
              ("legit passing", [sw[t]["pass_legit"]["rate"] for t in th]),
              ("SPRT detection (s1)",
               [sw[t]["seed_20261015"]["detection_sprt_after_triage"]["rate"] for t in th]),
              ("null calls saved (s1)",
               [sw[t]["seed_20261015"]["frac_null_calls_saved"] for t in th])]
    for i, (lab, ys) in enumerate(series):
        ax.plot(x, ys, marker=MARK[i], ls=LS[i], color=SERIES[i], ms=2.8, lw=1.0, label=lab)
    ax.axvline(0.22, color=C["grey"], ls=":", lw=0.7)
    ax.set_xlabel("embedding pre-triage threshold")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.6,
              borderaxespad=0.2, columnspacing=0.6, handlelength=1.6)
    panel_label(ax, "(a)", x=-0.15)

    rows = [json.loads(line) for line in open(os.path.join(RES, "full", f"seed_{SEEDS[0]}",
                                                           "ccs.jsonl"))]
    seen, R = set(), []
    for r in rows:
        if r["key"] not in seen:
            seen.add(r["key"])
            R.append(r)
    for i, (cls, lab) in enumerate((("sleeper", "sleepers (85)"), ("benign", "benign (60)"),
                                    ("legit", "legit (60)"))):
        n = np.sort([r["n_used"] for r in R if r["class"] == cls])
        bx.step(n, np.arange(1, len(n) + 1) / len(n), where="post", color=SERIES[i], ls=LS[i],
                lw=1.1, label=lab)
    bx.axvline(19, color=C["grey"], ls=":", lw=0.7)
    bx.set_xlabel("contexts used per entry (n)")
    bx.set_ylabel("ECDF")
    bx.set_xlim(0, 41)
    bx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.8,
              borderaxespad=0.2, handlelength=1.6)
    panel_label(bx, "(b)", x=-0.15)

    cs = A["cost"]
    groups = ["sleeper write", "null write"]
    vals = {"unshared (analytic)": [cs["sleeper_unshared_s"], cs["null_unshared_s"]],
            "shared baselines": [cs["sleeper_shared_s"], cs["null_shared_s"]],
            "shared + pre-triage 0.22": [cs["triage_0.22"]["sleeper_shared_s"],
                                         cs["triage_0.22"]["null_shared_s"]]}
    xg = np.arange(2)
    w = 0.26
    for j, (lab, v) in enumerate(vals.items()):
        bars = cx.bar(xg + (j - 1) * w, v, width=w * 0.9, color=SERIES[j], edgecolor="white",
                      lw=0.4, label=lab, hatch="////" if j == 0 else None)
        for b_, vv in zip(bars, v):
            cx.text(b_.get_x() + b_.get_width() / 2, vv + 3, f"{vv:.0f}", ha="center",
                    fontsize=5.8)
    cx.set_xticks(xg)
    cx.set_xticklabels(groups)
    cx.set_ylabel("wall-clock per write (s)")
    cx.set_ylim(0, 205)
    cx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.6,
              borderaxespad=0.2, handlelength=1.4, columnspacing=0.6)
    panel_label(cx, "(c)", x=-0.15)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_triage_cost.pdf")


def main():
    fig_attack()
    fig_provgate()
    fig_calibration()
    fig_power()
    fig_breakdown()
    fig_ablations()
    fig_triage_cost()


if __name__ == "__main__":
    main()
