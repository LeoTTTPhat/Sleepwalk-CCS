"""Stage-7 figures: figures/fig_request.pdf, figures/fig_qmass.pdf.

fig_request  where the detector's signal comes from: induced out-of-scope divergence by
             probe request, split by whether the entry's stated trigger holds (Corpus A,
             replayed from the call cache) plus the payload-specific rate.
fig_qmass    detection against the trigger's realised Q_u-mass bin on Corpus B, with
             uniform probing on the same entries -- the test of the paper's hypothesis.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
W1, W2 = 3.5, 7.16
OK = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 1.8,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.4,
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})
SHORT = {"What's good near me right now?": "good near me?",
         "Where should I go next?": "where next?",
         "Take me somewhere to eat.": "somewhere to eat",
         "I need a pharmacy, what do you suggest?": "need a pharmacy",
         "Plan my next stop.": "plan next stop",
         "Any recommendations around here?": "recommendations?",
         "Take me there.": "take me there*"}
R = json.load(open(os.environ.get("REV_STATS", os.path.join(RES, "revision_stats.json"))))
RP = json.load(open(os.path.join(RES, "replay_stats.json")))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def errbars(ax, ys, ds, color, label, marker="o"):
    m = [d["rate"] for d in ds]
    lo = [d["rate"] - d["lo"] for d in ds]
    hi = [d["hi"] - d["rate"] for d in ds]
    ax.errorbar([100 * v for v in m], ys, xerr=[[100 * v for v in lo], [100 * v for v in hi]],
                fmt=marker, color=color, label=label, linestyle="none")


def fig_request():
    """(a) divergence by probe request, split by whether the stated condition holds;
    (b) the deployed test re-run on subsets of its own probe stream.  Both come from the
    cache-exact replay, so the figure needs no new model calls."""
    rp = RP["sleeper_divergence_by_query"]
    qs = sorted(rp, key=lambda q: -rp[q]["all"]["rate"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.25),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    ys = np.arange(len(qs))
    errbars(ax, ys + 0.17, [rp[q]["phi_text_fails"] for q in qs], OK[1],
            "stated trigger fails", "o")
    # trigger-holding contexts are rare under the user's own mobility; requests with
    # none are not plotted (an empty cell has no rate)
    hy = [y for y, q in zip(ys, qs) if rp[q]["phi_text_holds"]["n"] > 0]
    hd = [rp[q]["phi_text_holds"] for q in qs if rp[q]["phi_text_holds"]["n"] > 0]
    errbars(ax, np.array(hy) - 0.17, hd, OK[0], "stated trigger holds", "s")
    for y, q in zip(ys, qs):
        d = rp[q]["phi_text_fails"]
        ax.annotate(f"{d['k']}/{d['n']}", (100 * d["hi"] + 1.0, y + 0.17), fontsize=5.6,
                    va="center", color="#333333")
        h = rp[q]["phi_text_holds"]
        txt = f"{h['k']}/{h['n']}" if h["n"] else "no context"
        ax.annotate(txt, (100 * h["rate"] + 1.0 if h["n"] else 0.5, y - 0.17),
                    fontsize=5.6, va="center", color="#333333",
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT.get(q, q) for q in qs])
    ax.set_xlabel("induced out-of-scope divergence (%)")
    ax.set_xlim(-1, 45)
    ax.set_title("(a) by probe request", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), ncol=1, borderaxespad=0.2,
              handletextpad=0.3, framealpha=1.0, frameon=True, edgecolor="none")

    sub = RP.get("sprt_subsets", {})
    order = [("all_contexts", "full stream"),
             ("routing_requests_only", "routing requests"),
             ("info_requests_only", "information requests"),
             ("trigger_failing_only", "trigger fails only"),
             ("one_context_per_request", "one ctx per request")]
    order = [(k, lab) for k, lab in order if k in sub]
    ys2 = np.arange(len(order))
    det = [sub[k]["detection"] for k, _ in order]
    bx.barh(ys2, [100 * d["rate"] for d in det], color=OK[2], height=0.55)
    bx.errorbar([100 * d["rate"] for d in det], ys2,
                xerr=[[100 * (d["rate"] - d["lo"]) for d in det],
                      [100 * (d["hi"] - d["rate"]) for d in det]],
                fmt="none", ecolor="black", elinewidth=0.7)
    for y, (k, _lab) in zip(ys2, order):
        d = sub[k]["detection"]
        bx.annotate(f"{d['k']}/{d['n']}, $\\bar n$={sub[k]['mean_contexts_used']:.0f}",
                    (2, y), fontsize=5.6, va="center", color="white")
    bx.set_yticks(ys2)
    bx.set_yticklabels([lab for _, lab in order])
    bx.invert_yaxis()
    bx.set_xlabel("sleepers flagged by the deployed test (%)")
    bx.set_title("(b) re-run on subsets of the probe stream", loc="left")
    fig.tight_layout(w_pad=0.9)
    save(fig, "fig_request.pdf")


def fig_qmass():
    cc = R.get("ccs2") or {}
    c = cc.get("ccs2")
    qvu = cc.get("q_vs_uniform") or {}
    if not c:
        print("fig_qmass: no ccs2 data yet")
        return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2, 2.3))
    order = ("zero", "low", "mid", "high", "shift")
    if qvu.get("by_bin"):
        # (a) paired: the same standard-variant entries probed by the mixture and uniformly
        bins = [b for b in order if b in qvu["by_bin"]]
        xs = np.arange(len(bins))
        w = 0.38
        for j, (arm, col, lab) in enumerate((("Q", OK[0], "mixture $\\widetilde{Q}$"),
                                            ("uniform", OK[1], "uniform"))):
            ds = [qvu["by_bin"][b][arm] for b in bins]
            xx = xs + (j - 0.5) * w
            ax.bar(xx, [100 * d["rate"] for d in ds], color=col, width=w - 0.04, label=lab)
            ax.errorbar(xx, [100 * d["rate"] for d in ds],
                        yerr=[[100 * (d["rate"] - d["lo"]) for d in ds],
                              [100 * (d["hi"] - d["rate"]) for d in ds]],
                        fmt="none", ecolor="black", elinewidth=0.6)
            for x, d in zip(xx, ds):
                ax.annotate(f"{d['k']}", (x, 100 * d["hi"] + 1.5), ha="center", va="bottom",
                        fontsize=6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b}\n(n={qvu['by_bin'][b]['Q']['n']})" for b in bins])
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        tq, tu = qvu.get("trigger_contexts_Q"), qvu.get("trigger_contexts_uniform")
        if tq and tu:
            ax.text(0.02, 0.97, f"probes inside the trigger: "
                    f"{100 * tq['rate']:.1f}% vs {100 * tu['rate']:.1f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=6, frameon=False)
        ax.set_title("(a) same 60 entries, two proposals", loc="left")
    ax.set_ylabel("flagged by SPRT (%)")
    ax.set_xlabel("realised $Q_u$-mass bin of the trigger")
    var = c.get("by_variant", {})
    vorder = [v for v in ("standard", "soft", "split", "request_cond",
                          "leak_suppressed", "self_describing") if v in var]
    lab = {"standard": "standard", "soft": "soft", "split": "split (singly)",
           "request_cond": "request-cond.", "leak_suppressed": "leak-suppr.",
           "self_describing": "self-descr."}
    ys = np.arange(len(vorder))
    errbars(bx, ys, [var[v] for v in vorder], OK[2], "mixture probing, all 120", "o")
    for y, v in zip(ys, vorder):
        bx.annotate(f"{var[v]['k']}/{var[v]['n']}", (100 * var[v]["hi"] + 3, y),
                    va="center", fontsize=6)
    bx.set_xlim(0, 100)
    bx.set_yticks(ys)
    bx.set_yticklabels([lab[v] for v in vorder])
    bx.set_xlabel("flagged by SPRT (%)")
    bx.set_title("(b) Corpus B by attack variant", loc="left")
    save(fig, "fig_qmass.pdf")


if __name__ == "__main__":
    fig_request()
    fig_qmass()
