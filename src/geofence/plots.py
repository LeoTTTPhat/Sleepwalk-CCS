"""Generate figures from results/<run>/summary.json (pooled over seeds) and raw JSONL."""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import common  # noqa: E402


def _summary(d):
    return common.load_json(os.path.join(d, "summary.json"))


def fig_detection_vs_qmass(run, d):
    s = _summary(d).get("pooled", {})
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    plotted = False
    for mode, key, style, lab in [("ccs", "detection_sprt_by_qmass", "o-", "CCS (Q + exploration)"),
                                  ("randomq", "detection_sprt_by_qmass", "s--", "random probing (no Q)"),
                                  ("shift", "detection_sprt_by_qmass", "^:", "CCS, shifted Q")]:
        qb = s.get(mode, {}).get(key)
        if not qb:
            continue
        xs, ys, yerr = [], [], []
        for k in sorted(qb, key=lambda z: float(z.split("-")[0])):
            a, b = k.split("-")
            xs.append((float(a) + min(float(b), 1.0)) / 2)
            r = qb[k]
            ys.append(r["rate"])
            yerr.append([r["rate"] - r["lo"], r["hi"] - r["rate"]])
        ax.errorbar(xs, ys, yerr=np.array(yerr).T, fmt=style, capsize=2, label=lab)
        plotted = True
    ax.set_xlabel("trigger Q-mass (user's train region)")
    ax.set_ylabel("SPRT detection rate")
    ax.set_ylim(-0.02, 1.02); ax.axvline(0.06, color="grey", ls=":", lw=0.8)
    ax.legend(fontsize=7); fig.tight_layout()
    if plotted:
        fig.savefig(os.path.join(common.FIGURES, f"{run}_detection_vs_qmass.pdf"))
    plt.close(fig)


def fig_provgate(run, d):
    pg = _summary(d).get("pooled", {}).get("provgate", {}).get("asr_by_mode_channel")
    if not pg:
        return
    modes = ["none", "spotlight", "tma_nm", "tma_nm_relaxed"]
    chans = sorted({k.split("|")[1] for k in pg})
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    w = 0.8 / len(modes)
    for i, m in enumerate(modes):
        vals = [pg.get(f"{m}|{c}", {}).get("mean", np.nan) for c in chans]
        lo = [pg.get(f"{m}|{c}", {}).get("lo", np.nan) for c in chans]
        hi = [pg.get(f"{m}|{c}", {}).get("hi", np.nan) for c in chans]
        err = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
        ax.bar(np.arange(len(chans)) + i * w, vals, w, yerr=err, capsize=2, label=m)
    ax.set_xticks(np.arange(len(chans)) + 0.4 - w / 2)
    ax.set_xticklabels(chans, fontsize=7)
    ax.set_ylabel("sleeper ASR (phi holds)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(common.FIGURES, f"{run}_provgate.pdf"))
    plt.close(fig)


def fig_n_distribution(run, d):
    sl = []
    for p in glob.glob(os.path.join(d, "seed_*", "ccs.jsonl")):
        sl += [r["n_used"] for r in common.read_jsonl(p) if r.get("class") == "sleeper"]
    if not sl:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.hist(sl, bins=range(1, max(sl) + 2), color="#3a7")
    ax.set_xlabel("contexts until SPRT stop (n)"); ax.set_ylabel("sleepers (all seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(common.FIGURES, f"{run}_n_dist.pdf"))
    plt.close(fig)


def fig_filters(run, d):
    flt = _summary(d).get("filters")
    if not flt:
        return
    names = list(flt)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    w = 0.25
    for i, ds in enumerate(["sleeper", "benign", "legit"]):
        vals = [flt[n].get(ds, {}).get("rate", np.nan) for n in names]
        ax.bar(np.arange(len(names)) + i * w, vals, w, label=f"flag rate: {ds}")
    ax.set_xticks(np.arange(len(names)) + w); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("fraction flagged"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(common.FIGURES, f"{run}_filters.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    os.makedirs(common.FIGURES, exist_ok=True)
    d = os.path.join(common.RESULTS, args.run)
    for fn in (fig_detection_vs_qmass, fig_provgate, fig_n_distribution, fig_filters):
        try:
            fn(args.run, d)
        except Exception as e:
            print(f"skip {fn.__name__}: {e}")
    print("figures written to", common.FIGURES)


if __name__ == "__main__":
    main()
