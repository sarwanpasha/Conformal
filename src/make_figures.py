"""Figures of the paper from the summary written by summarize.py.

    python src/make_figures.py --summary results/summary.json --out Figures

The summary must contain the groups named in GROUPS (left-deep plan space,
MLP estimator); groups that are missing are skipped.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 13, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.8,
})
AL = ["0.20", "0.10", "0.05", "0.01"]  # left to right: increasing target coverage
XT = ["0.80", "0.90", "0.95", "0.99"]
METHODS = ["point", "T1p", "T2_clean", "T2_reuse", "T3_CQR"]
COLORS = {"point": "#1b1b1b", "T1p": "#009E73", "T2_clean": "#D55E00",
          "T2_reuse": "#E69F00", "T3_CQR": "#0072B2"}
LABELS = {"point": "Point / SCP", "T1p": "SCP-$k$", "T2_clean": "LW-CP",
          "T2_reuse": "LW-CP (shared)", "T3_CQR": "CQR"}
STYLES = {"point": "-", "T1p": ":", "T2_clean": "-", "T2_reuse": "--", "T3_CQR": "-."}
MARKS = {"point": "o", "T1p": "v", "T2_clean": "s", "T2_reuse": "D", "T3_CQR": "^"}
GROUPS = [("tpch_sf1_leftdeep_mlp", "TPC-H"), ("tpcds_sf1_leftdeep_mlp", "TPC-DS"),
          ("job_sf1_leftdeep_mlp", "JOB"), ("tpch_sf10_leftdeep_mlp", "TPC-H SF 10")]


def find(S, name):
    for k, g in S["groups"].items():
        if k.split(":")[-1] == name and "a" in g:
            return g
    return None


def first(x):
    if isinstance(x, list):
        x = x[0] if x else None
    return np.nan if x is None else float(x)


def fig_ccdf(S, out):
    gs = [(find(S, g), n) for g, n in GROUPS]
    gs = [(g, n) for g, n in gs if g is not None and "ccdf" in g["a"]["0.01"]]
    xs = np.array(S["xs"], dtype=float)
    fig, axes = plt.subplots(1, len(gs), figsize=(2.75 * len(gs), 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (g, n) in zip(axes, gs):
        C = g["a"]["0.01"]["ccdf"]
        for m in METHODS:
            if m not in C:
                continue
            y = np.array([np.nan if v is None else v for v in C[m]], dtype=float)
            y[y <= 0] = np.nan
            ax.step(xs, y, where="post", color=COLORS[m], ls=STYLES[m], lw=1.8,
                    label=LABELS[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 1e5)
        ax.set_ylim(5e-4, 1)
        ax.set_xticks([1, 1e2, 1e4])
        ax.set_xlabel(r"Plan regret $r$")
        ax.text(0.96, 0.96, n, transform=ax.transAxes, ha="right", va="top", fontsize=14)
    axes[0].set_ylabel(r"Fraction with $\rho>r$")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), frameon=False,
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_p99(S, out):
    gs = [(find(S, g), n) for g, n in GROUPS[:3]]
    gs = [(g, n) for g, n in gs if g is not None]
    x = list(range(len(AL)))
    fig, axes = plt.subplots(1, len(gs), figsize=(3.4 * len(gs), 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (g, n) in zip(axes, gs):
        for m in METHODS:
            y = [first(g["a"][a]["reg"][m]["p99"]) for a in AL]
            ax.plot(x, y, color=COLORS[m], ls=STYLES[m], marker=MARKS[m], ms=6, lw=1.8,
                    label=LABELS[m])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(XT)
        ax.set_xlabel(r"Target $1-\alpha$")
        ax.text(0.04, 0.96, n, transform=ax.transAxes, ha="left", va="top", fontsize=14)
    axes[0].set_ylabel("P99 plan regret")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), frameon=False,
               bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_coverage(S, out):
    gs = [(find(S, g), n) for g, n in GROUPS]
    gs = [(g, n) for g, n in gs if g is not None]
    x = list(range(len(AL)))
    colors = ["#0072B2", "#D55E00", "#009E73", "#1b1b1b"]
    marks = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7))
    for (g, n), c, mk in zip(gs, colors, marks):
        A = g["a"]
        axes[0].plot(x, [first(A[a]["prcp"]["cov"]) for a in AL], color=c, marker=mk,
                     ms=6, lw=1.8, label=n)
        axes[1].plot(x, [first(A[a]["prcp"]["cov_hard"]) for a in AL], color=c,
                     marker=mk, ms=6, lw=1.8)
        axes[1].plot(x, [first(A[a]["mond"]["cov_hard"]) for a in AL], color=c,
                     marker=mk, ms=6, lw=1.5, ls="--", mfc="white")
    axes[0].plot(x, [1 - float(a) for a in AL], color="0.55", lw=1.2, ls=":")
    axes[0].set_ylim(0.78, 1.005)
    axes[1].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Coverage (all)")
    axes[1].set_ylabel("Coverage (hard)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(XT)
        ax.set_xlabel(r"Target $1-\alpha$")
    h, lab = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="0.3", lw=1.8),
          plt.Line2D([], [], color="0.3", lw=1.5, ls="--")]
    lab += ["PR-CP", "Mondrian PR-CP"]
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.2), handlelength=1.8, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="Figures")
    a = ap.parse_args()
    S = json.load(open(a.summary))
    os.makedirs(a.out, exist_ok=True)
    fig_ccdf(S, os.path.join(a.out, "regret_ccdf.pdf"))
    fig_p99(S, os.path.join(a.out, "p99_alpha.pdf"))
    fig_coverage(S, os.path.join(a.out, "prcp_coverage.pdf"))
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
