"""PostgreSQL experiments (PostgreSQL 16 with the pg_hint_plan extension).

  load      copy a DuckDB benchmark into PostgreSQL, index join keys, ANALYZE
  wallclock force each picker's left-deep join order with pg_hint_plan Leading()
            and time it, alongside the oracle plan and PostgreSQL's own plan
  pgest     use PostgreSQL's native cardinality estimates as the base
            estimator on the stored test queries and measure plan regret
  inject    inject point estimates or conformal upper bounds as Rows() hints
            and time the plan PostgreSQL chooses
  latcp     time every candidate join order and calibrate PR-CP on latency

The server must already be running; this module only connects. Connection
settings come from PRCP_PG_HOST (socket directory or host name, default /tmp),
PRCP_PG_PORT (default 5432) and PRCP_PG_USER (default: $USER).
"""
import argparse, json, math, os, re, time
import numpy as np
import psycopg2
from pipeline import BENCH, join_conditions

SOCK = os.environ.get("PRCP_PG_HOST", "/tmp")
PORT = int(os.environ.get("PRCP_PG_PORT", "5432"))
USER = os.environ.get("PRCP_PG_USER", os.environ.get("USER", "postgres"))


def dbname(bench, sf):
    return ("%s_sf%g" % (bench, sf)).replace(".", "_")


def connect(db):
    c = psycopg2.connect(host=SOCK, port=PORT, dbname=db, user=USER)
    c.autocommit = True
    return c


def join_columns(spec):
    al2t = {a: t for t, a in spec["tables"].items()}
    cols = set()
    for _, _, cond in spec["edges"]:
        for m in re.finditer(r"\b(\w+)\.(\w+)\b", cond):
            if m.group(1) in al2t:
                cols.add((al2t[m.group(1)], m.group(2)))
    return sorted(cols)


def cmd_load(a):
    import duckdb
    spec = BENCH[a.bench]
    db = dbname(a.bench, a.sf)
    c = connect("postgres")
    cur = c.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (db,))
    if not cur.fetchone():
        cur.execute("CREATE DATABASE %s" % db)
    c.close()
    dk = duckdb.connect()  # in-memory; source attached read-only, pg read-write
    dk.execute("ATTACH '%s' AS src (READ_ONLY)"
               % os.path.join(a.data_dir, "%s_sf%g.duckdb" % (a.bench, a.sf)))
    dk.execute("SET enable_progress_bar = false")
    dk.execute("INSTALL postgres")
    dk.execute("LOAD postgres")
    dk.execute("ATTACH 'host=%s port=%d dbname=%s user=%s' AS pg (TYPE postgres)"
               % (SOCK, PORT, db, USER))
    for t in spec["tables"]:
        t0 = time.time()
        dk.execute("DROP TABLE IF EXISTS pg.public.%s" % t)
        dk.execute("CREATE TABLE pg.public.%s AS SELECT * FROM src.%s" % (t, t))
        print("  loaded %-24s %.1fs" % (t, time.time() - t0), flush=True)
    dk.close()
    c = connect(db)
    cur = c.cursor()
    for t, col in join_columns(spec):
        t0 = time.time()
        cur.execute("CREATE INDEX IF NOT EXISTS ix_%s_%s ON %s (%s)" % (t, col, t, col))
        print("  index %s(%s) %.1fs" % (t, col, time.time() - t0), flush=True)
    cur.execute("ANALYZE")
    print("LOAD_DONE", db, flush=True)


def base_sql(spec, tables, preds, select="COUNT(*)"):
    al = spec["tables"]
    frm = ", ".join("%s %s" % (t, al[t]) for t in tables)
    aliases = set(al[t] for t in tables)
    conds = join_conditions(spec, tables) + [p for p in preds
                                             if p.split(".")[0] in aliases]
    sql = "SELECT %s FROM %s" % (select, frm)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    return sql


def leading_hint(spec, order):
    al = [spec["tables"][t] for t in order]
    expr = al[0]
    for x in al[1:]:
        expr = "(%s %s)" % (expr, x)
    return "/*+ Leading(%s) */ " % expr


def _safe(cur, sql):
    """A statement_timeout that fires just as the previous query finishes can
    cancel the NEXT statement instead; retry it."""
    for _ in range(3):
        try:
            return cur.execute(sql)
        except psycopg2.errors.QueryCanceled:
            time.sleep(0.2)
    return cur.execute(sql)


def timed(cur, sql, reps, timeout_s):
    _safe(cur, "SET statement_timeout = %d" % int(timeout_s * 1000))
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        try:
            cur.execute(sql)
            cur.fetchall()
        except psycopg2.errors.QueryCanceled:
            return timeout_s, True
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, False


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 6:
        return None
    r, p = spearmanr(x, y)
    return {"rho": float(r), "p": float(p), "n": len(x)}


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def cmd_wallclock(a):
    R = json.load(open(a.result), parse_constant=lambda s: None)
    spec = BENCH[R["bench"]]
    tq = R["test_queries"]
    picks = R["alphas"][a.alpha]["picks"]
    methods = [m for m in ("point", "T1p", "T2_clean", "T3_CQR") if m in picks]
    if not tq or not tq[0].get("orders"):
        raise SystemExit("wall-clock needs left-deep orders in the result file")
    dis = [i for i in range(len(tq)) if len({picks[m][i] for m in methods}) > 1]
    rest = [i for i in range(len(tq)) if i not in set(dis)]
    sel = (dis + rest)[:a.n]
    c = connect(dbname(R["bench"], R["sf"]))
    cur = c.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET jit = off")
    rows = []
    for k, i in enumerate(sel):
        q = tq[i]
        oi_star = int(np.argmin(q["true_costs"]))
        rec = {"qi": i, "k": len(q["tables"]),
               "oracle_cout": q["true_costs"][oi_star]}
        plain = base_sql(spec, q["tables"], q["preds"])
        timed(cur, plain, 1, a.timeout)
        rec["oracle_s"], rec["oracle_to"] = timed(
            cur, leading_hint(spec, q["orders"][oi_star]) + plain, a.reps, a.timeout)
        rec["pgnative_s"], rec["pgnative_to"] = timed(cur, plain, a.reps, a.timeout)
        for m in methods:
            oi = picks[m][i]
            rec[m + "_cout_regret"] = q["true_costs"][oi] / max(1.0, q["true_costs"][oi_star])
            rec[m + "_s"], rec[m + "_to"] = timed(
                cur, leading_hint(spec, q["orders"][oi]) + plain, a.reps, a.timeout)
        rows.append(rec)
        if (k + 1) % 10 == 0:
            print("  timed %d/%d" % (k + 1, len(sel)), flush=True)
    out = {"result": os.path.basename(a.result), "bench": R["bench"], "sf": R["sf"],
           "alpha": a.alpha, "n": len(rows), "reps": a.reps, "timeout_s": a.timeout,
           "n_disagree": len(dis), "rows": rows, "summary": {}}
    for m in methods + ["pgnative"]:
        lat = [r[m + "_s"] / max(r["oracle_s"], 1e-4) for r in rows]
        s = {"lat_regret_p50": pct(lat, 50), "lat_regret_p90": pct(lat, 90),
             "lat_regret_p99": pct(lat, 99), "lat_regret_max": max(lat) if lat else None,
             "timeouts": sum(1 for r in rows if r[m + "_to"])}
        if m != "pgnative":
            s["spearman_cout_vs_latency"] = spearman(
                [r[m + "_cout_regret"] for r in rows], lat)
        out["summary"][m] = s
    pc, pl = [], []
    for r in rows:
        for m in methods:
            pc.append(r[m + "_cout_regret"])
            pl.append(r[m + "_s"] / max(r["oracle_s"], 1e-4))
    out["summary"]["pooled_spearman"] = spearman(pc, pl)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    print("WALLCLOCK_DONE", a.out, flush=True)


def cmd_pgest(a):
    R = json.load(open(a.result), parse_constant=lambda s: None)
    spec = BENCH[R["bench"]]
    tq = R["test_queries"]
    point = R["alphas"]["0.10"]["picks"]["point"]
    c = connect(dbname(R["bench"], R["sf"]))
    cur = c.cursor()
    cache = {}

    def est(tables, preds):
        key = (tuple(sorted(tables)), tuple(sorted(preds)))
        if key not in cache:
            cur.execute("EXPLAIN (FORMAT JSON) " + base_sql(spec, sorted(tables), preds,
                                                            select="1"))
            cache[key] = float(cur.fetchone()[0][0]["Plan"]["Plan Rows"])
        return cache[key]

    reg_pg, reg_ml = [], []
    for i, q in enumerate(tq):
        oracle = max(1.0, min(q["true_costs"]))
        best, bc = None, None
        for oi, pl in enumerate(q["plans"]):
            tot = sum(est(k.split("|"), q["preds"]) for k in pl)
            if bc is None or tot < bc:
                best, bc = oi, tot
        reg_pg.append(q["true_costs"][best] / oracle)
        reg_ml.append(q["true_costs"][point[i]] / oracle)

    def stats(v):
        return {"p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99),
                "mean": float(np.mean(v)), "max": float(np.max(v)),
                "frac_optimal": float(np.mean([x <= 1.0 + 1e-9 for x in v]))}

    out = {"result": os.path.basename(a.result), "bench": R["bench"],
           "n": len(tq), "explains": len(cache),
           "postgres_native": stats(reg_pg), "learned_point": stats(reg_ml)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("PGEST_DONE", a.out, flush=True)


# -------------------------------------------------------------- inject ----

def plan_signature(node):
    """Join tree shape + operators of an EXPLAIN plan, to tell whether an
    injection changed the plan PostgreSQL chose."""
    kids = node.get("Plans", [])
    me = node["Node Type"] + (":" + node["Alias"] if "Alias" in node else "")
    if not kids:
        return me
    return "%s(%s)" % (me, ",".join(plan_signature(k) for k in kids))


def rows_hint(spec, keys, vals):
    parts = []
    for k, v in zip(keys, vals):
        if v is None:
            return None
        rows = max(1, int(round(min(math.exp(min(v, 34.0)) - 1.0, 1e15))))
        parts.append("Rows(%s #%d)" % (" ".join(spec["tables"][t] for t in k.split("|")), rows))
    return "/*+ " + " ".join(parts) + " */ "


def cmd_inject(a):
    """Inject per-subset cardinalities (point estimate or a CP upper bound)
    into PostgreSQL via pg_hint_plan Rows() and let PostgreSQL plan freely
    (any join shape, any operator). Time the plan it picks."""
    R = json.load(open(a.result), parse_constant=lambda s: None)
    spec = BENCH[R["bench"]]
    tq = R["test_queries"]
    inj = R["alphas"][a.alpha].get("inject")
    if not inj:
        raise SystemExit("result file has no injection estimates (run pipeline.py with --n-inject > 0)")
    methods = list(inj[0]["est"].keys())
    c = connect(dbname(R["bench"], R["sf"]))
    cur = c.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET jit = off")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(min(a.n, len(inj))):
        q, e = tq[i], inj[i]
        plain = base_sql(spec, q["tables"], q["preds"])
        timed(cur, plain, 1, a.timeout)  # warm
        variants = [("pgnative", "")]
        for m in methods:
            h = rows_hint(spec, e["keys"], e["est"][m])
            if h is not None:
                variants.append((m, h))
        rec = {"qi": i, "k": len(q["tables"])}
        order = rng.permutation(len(variants))
        for j in order:
            name, h = variants[j]
            _safe(cur, "SET statement_timeout = 0")
            _safe(cur, "EXPLAIN (FORMAT JSON) " + h + plain)
            rec[name + "_plan"] = plan_signature(cur.fetchone()[0][0]["Plan"])
            rec[name + "_s"], rec[name + "_to"] = timed(cur, h + plain, a.reps, a.timeout)
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print("  injected %d/%d" % (i + 1, min(a.n, len(inj))), flush=True)
    names = ["pgnative"] + methods
    out = {"result": os.path.basename(a.result), "bench": R["bench"],
           "estimator": R.get("estimator"), "alpha": a.alpha, "n": len(rows),
           "reps": a.reps, "timeout_s": a.timeout, "rows": rows, "summary": {}}
    for m in names:
        have = [r for r in rows if (m + "_s") in r]
        if not have:
            continue
        best = [min(r[x + "_s"] for x in names if (x + "_s") in r) for r in have]
        rel_best = [r[m + "_s"] / max(b, 1e-4) for r, b in zip(have, best)]
        rel_pg = [r[m + "_s"] / max(r["pgnative_s"], 1e-4) for r in have]
        out["summary"][m] = {
            "n": len(have),
            "total_s": float(sum(r[m + "_s"] for r in have)),
            "timeouts": sum(1 for r in have if r[m + "_to"]),
            "vs_best_p50": pct(rel_best, 50), "vs_best_p90": pct(rel_best, 90),
            "vs_best_p99": pct(rel_best, 99), "vs_best_max": float(max(rel_best)),
            "vs_pg_geomean": float(np.exp(np.mean(np.log(np.maximum(rel_pg, 1e-6))))),
            "vs_pg_p99": pct(rel_pg, 99),
            "plan_differs_from_pg": float(np.mean(
                [r[m + "_plan"] != r["pgnative_plan"] for r in have])),
        }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    print("INJECT_DONE", a.out, flush=True)


# --------------------------------------------------------------- latcp ----

_LW = {}


def _lat_init(db, timeout, reps):
    c = connect(db)
    cur = c.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET jit = off")
    _LW.update(cur=cur, timeout=timeout, reps=reps)


def _lat_job(item):
    qi, bench, tables, preds, orders = item
    spec = BENCH[bench]
    plain = base_sql(spec, tables, preds)
    cur = _LW["cur"]
    timed(cur, plain, 1, _LW["timeout"])  # warm
    res = []
    for o in orders:
        s, to = timed(cur, leading_hint(spec, o) + plain, _LW["reps"], _LW["timeout"])
        res.append([s, bool(to)])
    return qi, res


def cmd_latcp(a):
    """Time EVERY candidate join order of each test query, so latency regret
    (and the latency-optimal plan) is defined; then calibrate PR-CP directly
    on latency regret. Raw timings are saved for offline re-analysis."""
    from multiprocessing import get_context
    from pipeline import conformal_quantile
    R = json.load(open(a.result), parse_constant=lambda s: None)
    tq = R["test_queries"][:a.n]
    items = [(i, R["bench"], q["tables"], q["preds"], q["orders"])
             for i, q in enumerate(tq)]
    lat = [None] * len(items)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers, initializer=_lat_init,
                                   initargs=(dbname(R["bench"], R["sf"]),
                                             a.timeout, a.reps)) as pool:
        for n_done, (qi, res) in enumerate(pool.imap_unordered(_lat_job, items), 1):
            lat[qi] = res
            if n_done % 20 == 0:
                print("  timed %d/%d queries, %.0fs" % (n_done, len(items),
                                                        time.time() - t0), flush=True)
    out = {"result": os.path.basename(a.result), "bench": R["bench"],
           "n": len(tq), "reps": a.reps, "timeout_s": a.timeout,
           "workers": a.workers, "latency": lat, "analysis": {}}
    # latency regret of every picker stored in the result file
    for akey, ares in R["alphas"].items():
        picks = ares["picks"]
        an = {"regret": {}}
        lr = {}
        for m, pk in picks.items():
            v = []
            for i, L in enumerate(lat):
                best = min(x[0] for x in L)
                v.append(L[pk[i]][0] / max(best, 1e-4))
            lr[m] = v
            an["regret"][m] = {"p50": pct(v, 50), "p90": pct(v, 90),
                               "p99": pct(v, 99), "max": float(max(v)),
                               "mean": float(np.mean(v))}
        # PR-CP calibrated on latency regret of the point picker:
        # 50 random cal/test halves, mean +- sd
        sc = np.log(np.maximum(np.array(lr["point"]), 1.0))
        hard = np.array(lr["point"]) > 1.10
        alpha = float(akey)
        covs, covh, bnds, cout_cov = [], [], [], []
        b_cout = ares.get("prcp", {}).get("bound_unshifted")
        rg = np.random.default_rng(0)
        for _ in range(50):
            p = rg.permutation(len(sc))
            cal, te = p[:len(p) // 2], p[len(p) // 2:]
            tau = conformal_quantile(list(sc[cal]), alpha)
            if not np.isfinite(tau):
                continue
            cv = sc[te] <= tau + 1e-12
            covs.append(float(np.mean(cv)))
            if hard[te].any():
                covh.append(float(np.mean(cv[hard[te]])))
            bnds.append(float(np.exp(tau)))
            if b_cout:
                cout_cov.append(float(np.mean(np.exp(sc[te]) <= b_cout + 1e-12)))
        an["latency_prcp"] = {
            "bound_mean": float(np.mean(bnds)) if bnds else None,
            "bound_sd": float(np.std(bnds)) if bnds else None,
            "cov_all_mean": float(np.mean(covs)) if covs else None,
            "cov_all_sd": float(np.std(covs)) if covs else None,
            "cov_hard_mean": float(np.mean(covh)) if covh else None,
            "frac_hard": float(np.mean(hard)),
            "cout_bound": b_cout,
            "cout_bound_cov_on_latency": float(np.mean(cout_cov)) if cout_cov else None}
        out["analysis"][akey] = an
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["analysis"], indent=1)[:6000])
    print("LATCP_DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("load")
    s.add_argument("--bench", required=True)
    s.add_argument("--sf", type=float, default=1.0)
    s.add_argument("--data-dir", default="data")
    s = sub.add_parser("wallclock")
    s.add_argument("--result", required=True)
    s.add_argument("--alpha", default="0.01")
    s.add_argument("--n", type=int, default=80)
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--timeout", type=float, default=60.0)
    s.add_argument("--out", required=True)
    s = sub.add_parser("pgest")
    s.add_argument("--result", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("inject")
    s.add_argument("--result", required=True)
    s.add_argument("--alpha", default="0.01")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--timeout", type=float, default=300.0)
    s.add_argument("--out", required=True)
    s = sub.add_parser("latcp")
    s.add_argument("--result", required=True)
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--timeout", type=float, default=120.0)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"load": cmd_load, "wallclock": cmd_wallclock, "pgest": cmd_pgest, "inject": cmd_inject, "latcp": cmd_latcp}[a.cmd](a)


if __name__ == "__main__":
    main()
