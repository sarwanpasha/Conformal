"""Compact summary of every result file (numbers for the paper and figures)."""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np

AL = ["0.01", "0.05", "0.10", "0.20"]
PK = ["point", "T1", "T1p", "T2_clean", "T2_reuse", "T3_CQR"]
XS = [1.0, 1.01, 1.1, 1.5, 2, 3, 5, 10, 20, 50, 100, 300, 1e3, 3e3, 1e4, 1e5]


def L(f):
    return json.load(open(f), parse_constant=lambda s: None)


def r4(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float("%.4g" % x)


def ms(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None
    return [r4(np.mean(v)), r4(np.std(v)), len(v)]


def regrets(r, a, m):
    out = []
    for q, oi in zip(r["test_queries"], r["alphas"][a]["picks"][m]):
        o = min(q["true_costs"])
        if o > 0:
            out.append(q["true_costs"][oi] / o)
    return out


def group(files):
    R = [L(f) for f in files]
    g = {"n_seeds": len(R), "n_queries": R[0]["n_queries"], "n_train": R[0]["n_train"],
         "n_cal": R[0]["n_cal"], "n_test": R[0]["n_test"],
         "cand_used": ms([r["cand_used_mean"] for r in R]),
         "cand_total": ms([r["cand_total_mean"] for r in R]),
         "cand_max": max(r["cand_total_max"] for r in R),
         "zero_excl": sum(r.get("n_zero_oracle_excluded", 0) for r in R), "a": {}}
    k_hist = defaultdict(int)
    for r in R:
        for q in r["test_queries"]:
            k_hist[len(q["tables"])] += 1
    g["test_k_hist"] = dict(sorted(k_hist.items()))
    for a in AL:
        A = [r["alphas"][a] for r in R]
        d = {"reg": {}, "chg": {}, "card": {}}
        for m in PK:
            d["reg"][m] = {s: ms([x["regret"][m][s] for x in A]) for s in
                           ("p50", "p90", "p99", "mean", "max")}
            d["chg"][m] = [r4(np.mean([x["plan_changes"][m]["changed"] for x in A])),
                           r4(np.mean([x["plan_changes"][m]["worse"] for x in A]))]
        for m, c in A[0]["cardinality"].items():
            d["card"][m] = {s: ms([x["cardinality"][m][s] for x in A])
                            for s in ("pooled", "per_query", "log_width")}
        d["prop1"] = all(x["picks"]["T1"] == x["picks"]["point"] for x in A)
        d["opt_frac"] = ms([np.mean([q["true_costs"][i] <= min(q["true_costs"])
                                     for q, i in zip(r["test_queries"], x["picks"]["point"])])
                            for r, x in zip(R, A)])
        P = [x["prcp"] for x in A]
        d["prcp"] = {"bound": ms([p["bound_unshifted"] for p in P]),
                     "bound_sh": ms([p["bound_shifted"] for p in P]),
                     "cov": ms([p["coverage_unshifted"]["all"] for p in P]),
                     "cov_easy": ms([p["coverage_unshifted"]["easy"] for p in P]),
                     "cov_hard": ms([p["coverage_unshifted"]["hard"] for p in P]),
                     "n_hard": ms([p["coverage_unshifted"]["n_hard"] for p in P])}
        M = [x["prcp_mondrian"] for x in A]
        d["mond"] = {"med_bound": ms([m["median_bound"] for m in M]),
                     "max_bound": ms([m["max_bound"] for m in M]),
                     "cov": ms([m["cov_all"] for m in M]),
                     "cov_hard": ms([m["cov_hard"] for m in M])}
        if a in ("0.01", "0.10"):
            d["ccdf"] = {}
            for m in ("point", "T1p", "T2_clean", "T2_reuse", "T3_CQR"):
                v = np.array(sum((regrets(r, a, m) for r in R), []))
                d["ccdf"][m] = [r4(np.mean(v > x)) for x in XS]
        g["a"][a] = d
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--pg", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G = defaultdict(list)
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            tag = os.path.basename(d.rstrip("/")) + ":" + re.sub(r"_seed\d+", "", os.path.basename(f)[:-5])
            G[tag].append(f)
    out = {"xs": XS, "groups": {}}
    for t, fs in sorted(G.items()):
        try:
            out["groups"][t] = group(fs)
        except Exception as e:
            out["groups"][t] = {"error": repr(e)}
    if a.pg:
        P = {}
        for f in sorted(glob.glob(os.path.join(a.pg, "pgest_*.json"))):
            P.setdefault("pgest", {})[os.path.basename(f)[:-5]] = {
                k: {s: r4(v) for s, v in L(f)[k].items()} for k in ("postgres_native", "learned_point")}
        for f in sorted(glob.glob(os.path.join(a.pg, "wall_*.json"))):
            w = L(f)
            S = w["summary"]
            P.setdefault("wall", {})[os.path.basename(f)[:-5]] = {
                "n": w["n"], "timeout": w["timeout_s"], "n_disagree": w.get("n_disagree"),
                "oracle_med_s": r4(float(np.median([r["oracle_s"] for r in w["rows"]]))),
                "pooled": S.get("pooled_spearman"),
                "m": {m: {k: (r4(v) if not isinstance(v, dict) else v) for k, v in s.items()}
                      for m, s in S.items() if m != "pooled_spearman"}}
        for f in sorted(glob.glob(os.path.join(a.pg, "inject_*.json"))):
            w = L(f)
            P.setdefault("inject", {})[os.path.basename(f)[:-5]] = {
                "n": w["n"], "timeout": w["timeout_s"],
                "m": {m: {k: r4(v) for k, v in s.items()} for m, s in w["summary"].items()}}
        for f in sorted(glob.glob(os.path.join(a.pg, "latcp_*.json"))):
            w = L(f)
            lat = w["latency"]
            best = [min(x[0] for x in Lq) for Lq in lat]
            P.setdefault("latcp", {})[os.path.basename(f)[:-5]] = {
                "n": w["n"], "timeout": w["timeout_s"], "workers": w["workers"],
                "plans_timed": int(sum(len(x) for x in lat)),
                "timeouts": int(sum(t for Lq in lat for _, t in Lq)),
                "best_med_s": r4(float(np.median(best))),
                "a": {ak: {"reg": {m: {k: r4(v) for k, v in s.items()}
                                   for m, s in an["regret"].items()},
                           "prcp": {k: r4(v) if not isinstance(v, (bool, type(None))) else v
                                    for k, v in an["latency_prcp"].items()}}
                      for ak, an in w["analysis"].items()}}
        out["pg"] = P
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("SUMMARY_WRITTEN", a.out, len(out["groups"]), "groups")


if __name__ == "__main__":
    main()
