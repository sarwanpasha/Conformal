"""Print a readable summary of one or more result files written by pipeline.py.

Example:
    python src/report.py results/example/*.json
"""
import json
import sys

import numpy as np

PICKERS = [("point", "point estimate"), ("T1", "split CP"), ("T1p", "size-stratified CP"),
           ("T2_clean", "LW-S-CP"), ("T2_reuse", "LW-S-CP (reused split)"),
           ("T3_CQR", "CQR")]


def load(path):
    return json.load(open(path), parse_constant=lambda s: None)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main(paths):
    runs = [load(p) for p in paths]
    r0 = runs[0]
    print("benchmark=%s  sf=%g  plan space=%s  estimator=%s  seeds=%d" % (
        r0["bench"], r0["sf"], r0.get("plan_space"), r0.get("estimator"), len(runs)))
    print("queries: train=%d  calibration=%d  test=%d   candidate plans used per query=%.2f"
          % (r0["n_train"], r0["n_cal"], r0["n_test"], mean([r["cand_used_mean"] for r in runs])))
    for a in ("0.01", "0.10"):
        A = [r["alphas"][a] for r in runs]
        print("\nalpha = %s   plan regret of the chosen plan (mean over seeds)" % a)
        print("  %-24s %8s %8s %10s %12s" % ("picker", "P50", "P90", "P99", "max"))
        for key, name in PICKERS:
            st = [x["regret"][key] for x in A]
            print("  %-24s %8.2f %8.2f %10.2f %12.1f" % (
                name, mean([s["p50"] for s in st]), mean([s["p90"] for s in st]),
                mean([s["p99"] for s in st]), mean([s["max"] for s in st])))
        same = all(x["picks"]["T1"] == x["picks"]["point"] for x in A)
        print("  split CP picks the same plan as the point estimate on every query: %s" % same)
        pr = [x["prcp"] for x in A]
        print("  PR-CP certificate exp(tau) = %.3f   coverage: all=%.3f  hard=%.3f" % (
            mean([p["bound_unshifted"] for p in pr]),
            mean([p["coverage_unshifted"]["all"] for p in pr]),
            mean([p["coverage_unshifted"]["hard"] for p in pr])))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
