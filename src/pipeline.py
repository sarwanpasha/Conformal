"""Plan-regret conformal prediction (PR-CP) experiment pipeline.

Generates a join workload over a benchmark schema, computes exact intermediate
cardinalities with DuckDB, trains a cardinality estimator, and compares plan
choices made from point estimates and from conformal upper bounds. It also
calibrates PR-CP, a conformal certificate on the plan regret of the chosen plan.

Run `python src/pipeline.py --help` for options; see README.md for examples.
"""
import argparse, itertools, json, math, os, random, sys, time, zlib
import numpy as np
import duckdb

ALPHAS = [0.01, 0.05, 0.10, 0.20]

# Every intermediate costs at least this much: a plan that produces an empty
# intermediate still performs the join. With COST_FLOOR = 1 the oracle cost is
# >= k-1 >= 1 for every query, so the multiplicative plan-regret ratio is always
# well defined and no +1 smoothing is needed anywhere.
COST_FLOOR = 1

# "leftdeep" or "bushy"
PLAN_SPACE = "leftdeep"

TPCH = {
    "tables": {
        "region": "r", "nation": "n", "customer": "c", "supplier": "s",
        "part": "p", "partsupp": "ps", "orders": "o", "lineitem": "l",
    },
    "edges": [
        ("region", "nation", "r.r_regionkey = n.n_regionkey"),
        ("nation", "customer", "n.n_nationkey = c.c_nationkey"),
        ("nation", "supplier", "n.n_nationkey = s.s_nationkey"),
        ("customer", "orders", "c.c_custkey = o.o_custkey"),
        ("orders", "lineitem", "o.o_orderkey = l.l_orderkey"),
        ("part", "partsupp", "p.p_partkey = ps.ps_partkey"),
        ("supplier", "partsupp", "s.s_suppkey = ps.ps_suppkey"),
        ("lineitem", "part", "l.l_partkey = p.p_partkey"),
        ("lineitem", "supplier", "l.l_suppkey = s.s_suppkey"),
    ],
    "preds": {
        "lineitem": [("l.l_quantity", "n", 1, 50), ("l.l_discount", "n", 0, 1),
                     ("l.l_shipmode", "c", None, None)],
        "orders": [("o.o_totalprice", "n", 800, 560000),
                   ("o.o_orderpriority", "c", None, None)],
        "customer": [("c.c_acctbal", "n", -1000, 10000),
                     ("c.c_mktsegment", "c", None, None)],
        "part": [("p.p_size", "n", 1, 50), ("p.p_brand", "c", None, None)],
        "supplier": [("s.s_acctbal", "n", -1000, 10000)],
        "partsupp": [("ps.ps_supplycost", "n", 1, 1000)],
        "nation": [("n.n_name", "c", None, None)],
        "region": [("r.r_name", "c", None, None)],
    },
}

TPCDS = {
    "tables": {
        "store_sales": "ss", "store_returns": "sr", "date_dim": "dd",
        "item": "i", "customer": "cu", "store": "st",
        "customer_address": "ca", "customer_demographics": "cd",
        "household_demographics": "hd",
    },
    "edges": [
        ("store_sales", "date_dim", "ss.ss_sold_date_sk = dd.d_date_sk"),
        ("store_sales", "item", "ss.ss_item_sk = i.i_item_sk"),
        ("store_sales", "customer", "ss.ss_customer_sk = cu.c_customer_sk"),
        ("store_sales", "store", "ss.ss_store_sk = st.s_store_sk"),
        ("store_sales", "customer_demographics", "ss.ss_cdemo_sk = cd.cd_demo_sk"),
        ("store_sales", "household_demographics", "ss.ss_hdemo_sk = hd.hd_demo_sk"),
        ("store_sales", "store_returns",
         "ss.ss_item_sk = sr.sr_item_sk AND ss.ss_ticket_number = sr.sr_ticket_number"),
        ("customer", "customer_address", "cu.c_current_addr_sk = ca.ca_address_sk"),
        ("store_returns", "item", "sr.sr_item_sk = i.i_item_sk"),
    ],
    "preds": {
        "store_sales": [("ss.ss_quantity", "n", 1, 100),
                        ("ss.ss_sales_price", "n", 0, 200)],
        "store_returns": [("sr.sr_return_quantity", "n", 1, 100)],
        "date_dim": [("dd.d_year", "n", 1998, 2003), ("dd.d_moy", "n", 1, 12)],
        "item": [("i.i_current_price", "n", 0, 100), ("i.i_category", "c", None, None)],
        "customer": [("cu.c_birth_year", "n", 1920, 1990)],
        "store": [("st.s_number_employees", "n", 200, 320)],
        "customer_address": [("ca.ca_state", "c", None, None)],
        "customer_demographics": [("cd.cd_gender", "c", None, None)],
        "household_demographics": [("hd.hd_dep_count", "n", 0, 9)],
    },
}

JOB = {
    "tables": {
        "title": "t", "movie_info": "mi", "movie_info_idx": "mii",
        "movie_companies": "mc", "movie_keyword": "mk", "cast_info": "ci",
    },
    "edges": [
        ("title", "movie_info", "t.id = mi.movie_id"),
        ("title", "movie_info_idx", "t.id = mii.movie_id"),
        ("title", "movie_companies", "t.id = mc.movie_id"),
        ("title", "movie_keyword", "t.id = mk.movie_id"),
        ("title", "cast_info", "t.id = ci.movie_id"),
    ],
    "preds": {
        "title": [("t.production_year", "n", 1900, 2015),
                  ("t.kind_id", "c", None, None)],
        "movie_info": [("mi.info_type_id", "c", None, None)],
        "movie_info_idx": [("mii.info_type_id", "c", None, None)],
        "movie_companies": [("mc.company_type_id", "c", None, None)],
        "movie_keyword": [("mk.keyword_id", "n", 1, 134170)],
        "cast_info": [("ci.role_id", "c", None, None)],
    },
}

BENCH = {"tpch": TPCH, "tpcds": TPCDS, "job": JOB}


def open_db(bench, sf, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "%s_sf%g.duckdb" % (bench, sf))
    fresh = not os.path.exists(path)
    if bench == "job":
        if fresh:
            raise SystemExit("JOB database missing: run scripts/prepare_imdb.py first")
        return duckdb.connect(path, read_only=True)
    con = duckdb.connect(path, read_only=not fresh)
    ext = "tpch" if bench == "tpch" else "tpcds"
    con.execute("INSTALL %s;" % ext)
    con.execute("LOAD %s;" % ext)
    if fresh:
        gen = "dbgen" if bench == "tpch" else "dsdgen"
        con.execute("CALL %s(sf=%g);" % (gen, sf))
        # reopen read-only so that worker processes can open the file too
        con.close()
        con = duckdb.connect(path, read_only=True)
    return con


def categorical_values(con, spec, rng):
    vals = {}
    for tbl, plist in spec["preds"].items():
        for col, kind, _lo, _hi in plist:
            if kind != "c":
                continue
            raw = col.split(".")[1]
            rows = con.execute(
                "SELECT DISTINCT %s FROM %s WHERE %s IS NOT NULL LIMIT 40"
                % (raw, tbl, raw)).fetchall()
            vals[col] = [r[0] for r in rows]
    return vals


def neighbours(spec, tbl):
    out = set()
    for a, b, _ in spec["edges"]:
        if a == tbl:
            out.add(b)
        if b == tbl:
            out.add(a)
    return out


def connected_subset(spec, k, rng):
    tables = list(spec["tables"])
    cur = {rng.choice(tables)}
    while len(cur) < k:
        cand = set()
        for t in cur:
            cand |= neighbours(spec, t)
        cand -= cur
        if not cand:
            return None
        cur.add(rng.choice(sorted(cand)))
    return sorted(cur)


def gen_query(spec, catvals, rng, kmin=2, kmax=5):
    for _ in range(50):
        k = rng.randint(kmin, kmax)
        tabs = connected_subset(spec, k, rng)
        if tabs:
            break
    else:
        return None
    if not tabs:
        return None
    preds = []
    n_pred = rng.randint(0, 2)
    for _ in range(n_pred):
        t = rng.choice(tabs)
        plist = spec["preds"].get(t, [])
        if not plist:
            continue
        col, kind, lo, hi = plist[rng.randrange(len(plist))]
        if kind == "n":
            a = rng.uniform(lo, hi)
            op = rng.choice(["<", ">", "<=", ">="])
            preds.append("%s %s %s" % (col, op, round(a, 3)))
        else:
            vals = catvals.get(col) or []
            if not vals:
                continue
            v = str(rng.choice(vals)).replace("'", "''")
            preds.append("%s = '%s'" % (col, v))
    return {"tables": tabs, "preds": preds}


def join_conditions(spec, tabs):
    conds = []
    ts = set(tabs)
    for a, b, cond in spec["edges"]:
        if a in ts and b in ts:
            conds.append(cond)
    return conds


def is_connected(spec, tabs):
    if len(tabs) <= 1:
        return True
    seen = {tabs[0]}
    frontier = [tabs[0]]
    ts = set(tabs)
    while frontier:
        t = frontier.pop()
        for nb in neighbours(spec, t) & ts:
            if nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    return len(seen) == len(ts)


def count_subset(con, spec, tabs, preds, cache):
    key = (tuple(sorted(tabs)), tuple(sorted(preds)))
    if key in cache:
        return cache[key]
    frm = ", ".join("%s %s" % (t, spec["tables"][t]) for t in sorted(tabs))
    conds = join_conditions(spec, tabs)
    aliases = [spec["tables"][t] for t in tabs]
    used = [p for p in preds if p.split(".")[0] in aliases]
    where = " AND ".join(conds + used)
    sql = "SELECT COUNT(*) FROM %s%s" % (frm, (" WHERE " + where) if where else "")
    val = con.execute(sql).fetchone()[0]
    cache[key] = int(val)
    return int(val)


def enumerate_left_deep(spec, tabs, max_plans, struct_rng):
    all_orders = []
    for perm in itertools.permutations(sorted(tabs)):
        ok = True
        for i in range(2, len(perm) + 1):
            if not is_connected(spec, list(perm[:i])):
                ok = False
                break
        if ok:
            all_orders.append(perm)
    total = len(all_orders)
    if total == 0:
        return [], 0
    if total > max_plans:
        idx = struct_rng.sample(range(total), max_plans)
        all_orders = [all_orders[i] for i in sorted(idx)]
    return all_orders, total


def join_edge_between(spec, A, B):
    sa, sb = set(A), set(B)
    for x, y, _ in spec["edges"]:
        if (x in sa and y in sb) or (y in sa and x in sb):
            return True
    return False


def enumerate_bushy(spec, tabs, max_plans, struct_rng):
    """All bushy plans, each represented by the SET of intermediate table-sets
    it materialises (distinct trees with the same intermediates have the same
    C_out cost, so we deduplicate on that set)."""
    memo = {}

    def rec(key):
        if key in memo:
            return memo[key]
        if len(key) == 1:
            memo[key] = [()]
            return memo[key]
        out = set()
        first, rest = key[0], key[1:]
        n = len(rest)
        for mask in range(0, 1 << n):
            A = tuple(sorted([first] + [rest[i] for i in range(n) if mask >> i & 1]))
            B = tuple(sorted(t for t in key if t not in set(A)))
            if not B:
                continue
            if not is_connected(spec, list(A)) or not is_connected(spec, list(B)):
                continue
            if not join_edge_between(spec, A, B):
                continue
            for pa in rec(A):
                for pb in rec(B):
                    out.add(tuple(sorted(set(pa) | set(pb) | {"|".join(key)})))
        memo[key] = sorted(out)
        return memo[key]

    allp = rec(tuple(sorted(tabs)))
    total = len(allp)
    if total > max_plans:
        idx = struct_rng.sample(range(total), max_plans)
        allp = [allp[i] for i in sorted(idx)]
    return allp, total


def plan_prefixes(order):
    return [list(order[:i]) for i in range(2, len(order) + 1)]


class Featurizer:
    def __init__(self, spec, base_rows):
        self.tabs = sorted(spec["tables"])
        self.edges = [(a, b) for a, b, _ in spec["edges"]]
        self.base_rows = base_rows
        self.spec = spec

    @property
    def dim(self):
        return 2 * len(self.tabs) + len(self.edges) + 3

    def __call__(self, tabs, preds):
        ts = set(tabs)
        v = np.zeros(self.dim, dtype=np.float32)
        o = 0
        for i, t in enumerate(self.tabs):
            v[o + i] = 1.0 if t in ts else 0.0
        o += len(self.tabs)
        alias2tab = {self.spec["tables"][t]: t for t in self.tabs}
        pc = {}
        for p in preds:
            al = p.split(".")[0]
            t = alias2tab.get(al)
            if t in ts:
                pc[t] = pc.get(t, 0) + 1
        for i, t in enumerate(self.tabs):
            v[o + i] = pc.get(t, 0)
        o += len(self.tabs)
        for i, (a, b) in enumerate(self.edges):
            v[o + i] = 1.0 if (a in ts and b in ts) else 0.0
        o += len(self.edges)
        v[o] = sum(math.log(max(1, self.base_rows[t])) for t in tabs)
        v[o + 1] = len(tabs)
        v[o + 2] = sum(pc.values())
        return v


def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def regret_stats(r):
    r = np.asarray(r, dtype=float)
    return {"p50": float(np.percentile(r, 50)), "p90": float(np.percentile(r, 90)),
            "p99": float(np.percentile(r, 99)), "mean": float(np.mean(r)),
            "max": float(np.max(r))}


def bootstrap_ci(vals, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    n = len(v)
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = fn(v[rng.integers(0, n, n)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


_W = {}


def _winit(bench, sf, data_dir, threads, plan_space):
    """Process-pool initialiser: one read-only DuckDB connection per worker."""
    global PLAN_SPACE
    PLAN_SPACE = plan_space
    _W["con"] = open_db(bench, sf, data_dir)
    _W["con"].execute("PRAGMA threads=%d" % threads)
    # progress bars would flood batch logs
    _W["con"].execute("SET enable_progress_bar = false")
    # cap each worker's DuckDB memory: the default is ~80% of RAM per connection,
    # so N workers on a large database would exceed the machine's memory
    _W["con"].execute("PRAGMA memory_limit='%dMB'"
                      % int(os.environ.get("DUCKDB_MEM_MB", "8192")))
    # each worker gets its own spill directory so concurrent read-only
    # connections never share (and delete) each other's spill files
    td = os.path.join(os.environ.get("DUCKDB_TMP_ROOT", "/tmp"),
                      "duckdb_%s_%d" % (os.environ.get("SLURM_JOB_ID", "local"), os.getpid()))
    os.makedirs(td, exist_ok=True)
    _W["con"].execute("PRAGMA temp_directory='%s'" % td)
    _W["spec"] = BENCH[bench]
    _W["cache"] = {}


def _wjob(item):
    """_wjob_inner with an optional wall-clock cap per query (QUERY_TIMEOUT_S).
    A query whose true cardinalities cannot be computed within the cap is
    dropped, exactly like an empty-result query."""
    qt = float(os.environ.get("QUERY_TIMEOUT_S", "0") or 0)
    if qt <= 0:
        return _wjob_inner(item)
    import threading
    hit = []
    def _stop():
        hit.append(1)
        _W["con"].interrupt()
    t = threading.Timer(qt, _stop)
    t.daemon = True
    t.start()
    try:
        r = _wjob_inner(item)
    finally:
        t.cancel()
    if hit:
        print("QUERY_TIMEOUT k=%d tables=%s" % (len(item[0]["tables"]), "|".join(item[0]["tables"])),
              flush=True)
        return None
    return r


def _wjob_inner(item):
    q, max_plans = item
    spec, con = _W["spec"], _W["con"]
    full = "|".join(sorted(q["tables"]))
    cache = _W["cache"]
    try:
        if count_subset(_W["con"], _W["spec"], q["tables"], q["preds"], cache) <= 0:
            return None
    except Exception:
        return None
    sig = repr((tuple(q["tables"]), tuple(sorted(q["preds"]))))
    # stable across processes: hash() is per-process randomised
    struct_rng = random.Random(zlib.crc32(sig.encode()))
    if PLAN_SPACE == "bushy":
        plans, total = enumerate_bushy(spec, q["tables"], max_plans, struct_rng)
        orders = []
    else:
        orders, total = enumerate_left_deep(spec, q["tables"], max_plans, struct_rng)
        plans = [tuple("|".join(sorted(pp)) for pp in plan_prefixes(o)) for o in orders]
    if len(plans) < 2:
        return None
    subsets = {}
    for pl in plans:
        for key in pl:
            if key not in subsets:
                try:
                    subsets[key] = count_subset(con, spec, key.split("|"),
                                                q["preds"], cache)
                except Exception:
                    return None
    q["plans"] = [list(pl) for pl in plans]
    q["orders"] = [list(o) for o in orders]
    q["n_candidates_total"] = total
    q["subsets"] = dict(subsets)
    q["true_costs"] = [sum(max(subsets[k], COST_FLOOR) for k in pl) for pl in plans]
    return q


def _card_map(items, workers, bench, sf, data_dir, threads, plan_space, t0):
    """Ordered parallel map of _wjob; keeps None results so indices line up."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    res = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_winit,
                             initargs=(bench, sf, data_dir, threads, plan_space)) as ex:
        for i, r in enumerate(ex.map(_wjob, items, chunksize=1)):
            res.append(r)
            if (i + 1) % 20 == 0:
                print("  %d/%d, %.1fs" % (i + 1, len(items), time.time() - t0), flush=True)
    return res


def build_workload(con, spec, n_queries, rng, max_plans, catvals, log_every=200,
                   workers=1, bench=None, sf=None, data_dir="data", threads=4,
                   plan_space="leftdeep", fixed_specs=None):
    """Generate query specs serially (cheap, no DB), then compute true
    cardinalities in parallel across worker processes."""
    t0 = time.time()
    # over-generate: some specs yield fewer than 2 distinct plans (e.g. 2-table
    # queries in the bushy space) and are dropped
    target = int(n_queries * float(os.environ.get("OVERGEN", "3.0"))) + 50
    specs, attempts = [], 0
    if fixed_specs is not None:
        specs, target = list(fixed_specs), 0
    while len(specs) < target and attempts < target * 20:
        attempts += 1
        q = gen_query(spec, catvals, rng)
        if q is not None:
            specs.append(q)
    print("  %d query specs generated, computing cardinalities with %d workers"
          % (len(specs), workers), flush=True)
    # ---- multi-node sharding of the true-cardinality computation ----------
    # CARD_DIR=<dir> CARD_MODE=prep      : save the query specs + RNG state, exit
    # CARD_DIR=<dir> CARD_MODE=shardK/N  : compute specs i with i % N == K, exit
    # CARD_DIR=<dir> CARD_MODE=assemble  : merge all shards in spec order and
    #                                      continue the normal pipeline
    # The merged workload is identical to what a single-node run would build.
    cdir = os.environ.get("CARD_DIR") if fixed_specs is None else None
    if cdir:
        os.makedirs(cdir, exist_ok=True)
        sp = os.path.join(cdir, "specs.json")
        mode = os.environ.get("CARD_MODE", "")
        if mode == "prep":
            json.dump({"specs": specs, "rng": rng.getstate()}, open(sp, "w"))
            print("CARD_PREP_DONE %d specs" % len(specs), flush=True)
            sys.exit(0)
        d = json.load(open(sp))
        specs = d["specs"]
        st = d["rng"]
        rng.setstate((st[0], tuple(st[1]), st[2]))
        if mode.startswith("shard"):
            sel = mode[5:]
            part = None
            if ":" in sel:
                sel, part = sel.split(":")
            k, N = [int(x) for x in sel.split("/")]
            idxs = [i for i in range(len(specs)) if i % N == k]
            tag = "shard_%d_of_%d" % (k, N)
            if part:
                m, M = [int(x) for x in part.split("/")]
                idxs = [i for j, i in enumerate(idxs) if j % M == m]
                tag += "_part_%d_of_%d" % (m, M)
            res = _card_map([(specs[i], max_plans) for i in idxs], workers, bench,
                            sf, data_dir, threads, plan_space, t0)
            json.dump({str(i): r for i, r in zip(idxs, res)},
                      open(os.path.join(cdir, tag + ".json"), "w"))
            print("CARD_SHARD_DONE %d/%d n=%d usable=%d %.0fs" % (
                k, N, len(idxs), sum(r is not None for r in res), time.time() - t0),
                flush=True)
            sys.exit(0)
        if mode == "assemble":
            import glob as _glob
            got = {}
            for f in _glob.glob(os.path.join(cdir, "shard_*_of_*.json")):
                got.update(json.load(open(f)))
            missing = [i for i in range(len(specs)) if str(i) not in got]
            if missing:
                raise SystemExit("missing %d shard results" % len(missing))
            out = [got[str(i)] for i in range(len(specs)) if got[str(i)] is not None]
            print("  assembled %d usable queries from %d specs" % (len(out), len(specs)),
                  flush=True)
            return out[:n_queries]
    items = [(q, max_plans) for q in specs]
    out = []
    if workers <= 1:
        _winit(bench, sf, data_dir, threads, plan_space)
        for i, it in enumerate(items):
            r = _wjob(it)
            if r is not None:
                out.append(r)
            if (i + 1) % log_every == 0:
                print("  %d/%d, %.1fs" % (i + 1, len(items), time.time() - t0), flush=True)
            if len(out) >= n_queries:
                break
        return out[:n_queries]
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    # "spawn", not fork: the parent already holds an open DuckDB connection
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_winit,
                             initargs=(bench, sf, data_dir, threads, plan_space)) as ex:
        for i, r in enumerate(ex.map(_wjob, items, chunksize=2)):
            if r is not None:
                out.append(r)
            if (i + 1) % log_every == 0:
                print("  %d/%d, %.1fs" % (i + 1, len(items), time.time() - t0), flush=True)
    return out[:n_queries]


def collect_intermediates(queries):
    rows = []
    for qi, q in enumerate(queries):
        for key, card in q["subsets"].items():
            rows.append({"qi": qi, "tables": key.split("|"), "preds": q["preds"],
                         "card": card, "k": len(key.split("|"))})
    return rows


def featurize_rows(fz, rows):
    return np.stack([fz(r["tables"], r["preds"]) for r in rows])


def fit_estimator(X, y, seed, kind="mlp"):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    if kind == "pg":
        return None, "pg"
    if kind == "gbm":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9,
                         random_state=seed, verbosity=0, n_jobs=int(
                             os.environ.get("NTHREADS", "8")))
        m.fit(sc.transform(X), y)
        return sc, m
    from sklearn.neural_network import MLPRegressor
    m = MLPRegressor(hidden_layer_sizes=(256, 256, 256, 256), activation="relu",
                     solver="adam", learning_rate_init=1e-3, max_iter=200,
                     early_stopping=True, n_iter_no_change=15, random_state=seed)
    m.fit(sc.transform(X), y)
    return sc, m


def predict(sc, m, X):
    if isinstance(m, str) and m == "pg":
        return X[:, -1].astype(float)
    return m.predict(sc.transform(X))


def augment_featurizer(fz, estimator, con, spec, bench, sf):
    """mlp_sel: append exact per-table filtered log-cardinalities (a base-table
    selectivity signal similar to what sample-based estimators obtain).
    pg: append PostgreSQL's own log-estimate for the subset as the LAST column;
    it is the point estimate, and the difficulty/CQR models see it as input."""
    if estimator not in ("mlp_sel", "pg"):
        return fz
    al2t = {a: t for t, a in spec["tables"].items()}
    tabs = sorted(spec["tables"])
    sel_cache, pg_cache, st = {}, {}, {}
    if estimator == "pg":
        import pg_experiments
        st["pg"] = pg_experiments
        st["cur"] = pg_experiments.connect(pg_experiments.dbname(bench, sf)).cursor()

    def tsel(t, preds):
        mine = tuple(sorted(p for p in preds if al2t.get(p.split(".")[0]) == t))
        key = (t, mine)
        if key not in sel_cache:
            sql = "SELECT COUNT(*) FROM %s %s" % (t, spec["tables"][t])
            if mine:
                sql += " WHERE " + " AND ".join(mine)
            sel_cache[key] = math.log(con.execute(sql).fetchone()[0] + 1.0)
        return sel_cache[key]

    def pgest(ts, preds):
        key = ("|".join(sorted(ts)), tuple(sorted(preds)))
        if key not in pg_cache:
            st["cur"].execute("EXPLAIN (FORMAT JSON) " + st["pg"].base_sql(
                spec, sorted(ts), preds, select="1"))
            rows = float(st["cur"].fetchone()[0][0]["Plan"]["Plan Rows"])
            pg_cache[key] = math.log(rows + 1.0)
        return pg_cache[key]

    def f(ts, preds):
        v = fz(ts, preds)
        if estimator == "mlp_sel":
            ex = [tsel(t, preds) if t in ts else 0.0 for t in tabs]
        else:
            ex = [pgest(ts, preds)]
        return np.concatenate([v, np.array(ex, dtype=np.float32)])
    return f


def parse_workload_file(path, spec):
    """Fixed test workload. Accepts the MSCN/JOB-light CSV format
    (tables#joins#col,op,val,...#card) or one 'SELECT COUNT(*) FROM .. WHERE ..'
    per line. Join conditions are dropped and regenerated from the schema."""
    import re
    rename = {"mi_idx": "mii"}
    out = []
    for line in open(path):
        line = line.strip().rstrip(";")
        if not line:
            continue
        if line.upper().startswith("SELECT"):
            m = re.match(r"(?is)select\s+count\(\*\)\s+from\s+(.*?)\s+where\s+(.*)$", line)
            if not m:
                continue
            frm, where = m.group(1), m.group(2)
            items = [x.split() for x in frm.split(",")]
            conds = [c.strip() for c in re.split(r"(?i)\s+and\s+", where)]
            preds = [c for c in conds
                     if not re.match(r"^\w+\.\w+\s*=\s*\w+\.\w+$", c)]
        else:
            parts = line.split("#")
            items = [x.split() for x in parts[0].split(",")]
            flat = [x for x in parts[2].split(",")] if len(parts) > 2 and parts[2] else []
            preds = ["%s %s %s" % (flat[i], flat[i + 1], flat[i + 2])
                     for i in range(0, len(flat) - 2, 3)]
        fixed = []
        for p in preds:
            for a, b in rename.items():
                p = re.sub(r"\b%s\." % a, b + ".", p)
            fixed.append(p)
        tabs = sorted(t for t, _a in items)
        if all(t in spec["tables"] for t in tabs):
            out.append({"tables": tabs, "preds": fixed})
    return out


def run(bench, sf, seed, n_queries, splits, max_plans, data_dir, out_dir,
        workers=1, plan_space="leftdeep", estimator="mlp", wthreads=4,
        shift="none", test_file=None, n_inject=200, tag_suffix=""):
    spec = BENCH[bench]
    rng = random.Random(seed)
    con = open_db(bench, sf, data_dir)
    con.execute("PRAGMA threads=%d" % int(os.environ.get("NTHREADS", "8")))
    con.execute("SET enable_progress_bar = false")
    catvals = categorical_values(con, spec, rng)
    base_rows = {t: con.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
                 for t in spec["tables"]}
    print("[%s sf=%g seed=%d] generating %d queries" % (bench, sf, seed, n_queries), flush=True)
    queries = build_workload(con, spec, n_queries, rng, max_plans, catvals,
                             workers=workers, bench=bench, sf=sf,
                             data_dir=data_dir, threads=wthreads,
                             plan_space=plan_space)
    print("  got %d queries" % len(queries), flush=True)
    test_fixed = None
    if test_file:
        fx = parse_workload_file(test_file, spec)
        test_fixed = build_workload(con, spec, len(fx), rng, max_plans, catvals,
                                    workers=workers, bench=bench, sf=sf,
                                    data_dir=data_dir, threads=wthreads,
                                    plan_space=plan_space, fixed_specs=fx)
        print("  fixed test workload: %d of %d queries usable"
              % (len(test_fixed), len(fx)), flush=True)

    n_tr, n_cal, n_te = splits
    want = n_tr + n_cal + n_te
    if len(queries) < want:
        f = len(queries) / float(want)
        n_tr, n_cal, n_te = int(n_tr * f), int(n_cal * f), int(n_te * f)
        print("  WARNING: only %d queries survived; splits rescaled to %d/%d/%d"
              % (len(queries), n_tr, n_cal, n_te), flush=True)
    idx = list(range(len(queries)))
    rng.shuffle(idx)
    if shift == "none":
        tr_i = idx[:n_tr]
        cal_i = idx[n_tr:n_tr + n_cal]
        te_i = idx[n_tr + n_cal:n_tr + n_cal + n_te]
    else:
        # train AND calibrate on one join-size regime, test on the other
        small = [i for i in idx if len(queries[i]["tables"]) <= 3]
        large = [i for i in idx if len(queries[i]["tables"]) >= 4]
        src, tgt = (small, large) if shift == "small2large" else (large, small)
        f = min(1.0, len(src) / float(n_tr + n_cal))
        a1, a2 = int(n_tr * f), int(n_cal * f)
        tr_i, cal_i, te_i = src[:a1], src[a1:a1 + a2], tgt[:n_te]
        print("  shift=%s: train %d, cal %d, test %d"
              % (shift, len(tr_i), len(cal_i), len(te_i)), flush=True)

    fz = Featurizer(spec, base_rows)
    fz = augment_featurizer(fz, estimator, con, spec, bench, sf)
    tr_rows = collect_intermediates([queries[i] for i in tr_i])
    Xtr = featurize_rows(fz, tr_rows)
    ytr = np.array([math.log(r["card"] + 1) for r in tr_rows], dtype=np.float32)
    print("  training estimator on %d intermediates" % len(tr_rows), flush=True)
    sc, model = fit_estimator(Xtr, ytr, seed, kind=estimator)

    def est_map(qlist):
        out = []
        for q in qlist:
            keys = list(q["subsets"])
            X = np.stack([fz(k.split("|"), q["preds"]) for k in keys])
            yh = predict(sc, model, X)
            out.append({"keys": keys, "yhat": yh, "X": X,
                        "true": np.array([q["subsets"][k] for k in keys], dtype=float)})
        return out

    cal_q = [queries[i] for i in cal_i]
    te_q = test_fixed if test_fixed is not None else [queries[i] for i in te_i]
    cal_e = est_map(cal_q)
    te_e = est_map(te_q)

    cal_abs, cal_k, cal_yt, cal_X = [], [], [], []
    for q, e in zip(cal_q, cal_e):
        yt = np.log(e["true"] + 1.0)
        cal_abs.append(np.abs(e["yhat"] - yt))
        cal_k.append(np.array([len(k.split("|")) for k in e["keys"]]))
        cal_yt.append(yt)
        cal_X.append(e["X"])
    A = np.concatenate(cal_abs)
    K = np.concatenate(cal_k)
    YT = np.concatenate(cal_yt)
    XC = np.concatenate(cal_X)
    half = len(cal_q) // 2
    q_half = np.concatenate([np.full(len(e["keys"]), 0 if i < half else 1)
                             for i, e in enumerate(cal_e)])

    from xgboost import XGBRegressor
    g_reuse = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                           random_state=seed, verbosity=0).fit(XC, A)
    g_clean = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                           random_state=seed, verbosity=0).fit(XC[q_half == 0], A[q_half == 0])
    from sklearn.ensemble import GradientBoostingRegressor
    cqr_models = {}
    for a in ALPHAS:
        lo = GradientBoostingRegressor(loss="quantile", alpha=a / 2.0,
                                       n_estimators=150, random_state=seed).fit(Xtr, ytr)
        hi = GradientBoostingRegressor(loss="quantile", alpha=1.0 - a / 2.0,
                                       n_estimators=150, random_state=seed).fit(Xtr, ytr)
        cqr_models[a] = (lo, hi)
    print("  fitted difficulty + quantile models", flush=True)

    results = {"bench": bench, "sf": sf, "seed": seed, "n_queries": len(queries),
               "n_train": len(tr_i), "n_cal": len(cal_i), "n_test": len(te_q),
               "base_rows": base_rows,
               "cand_total_mean": float(np.mean([q["n_candidates_total"] for q in queries])),
               "cand_total_max": int(np.max([q["n_candidates_total"] for q in queries])),
               "cand_used_mean": float(np.mean([len(q["plans"]) for q in queries])),
               "alphas": {}}

    def pick(q, logub_by_key):
        """argmin of sum_j exp(logub_j), computed in log space (monotone,
        overflow-free; the constant -1 per intermediate is identical across
        left-deep plans of a query and cannot change the argmin)."""
        best, bc = None, None
        for oi, pl in enumerate(q["plans"]):
            vals = np.array([logub_by_key[k] for k in pl], dtype=float)
            m = float(np.max(vals))
            tot = m + math.log(float(np.sum(np.exp(vals - m))))
            if bc is None or tot < bc:
                best, bc = oi, tot
        return best

    def point_ub(e):
        return {k: float(v) for k, v in zip(e["keys"], e["yhat"])}

    def regret_of(q, oi):
        oracle = min(q["true_costs"])
        if oracle <= 0:
            return None
        return q["true_costs"][oi] / oracle

    pt_pick_te = [pick(q, point_ub(e)) for q, e in zip(te_q, te_e)]
    pt_pick_cal = [pick(q, point_ub(e)) for q, e in zip(cal_q, cal_e)]
    pt_reg_te = [regret_of(q, oi) for q, oi in zip(te_q, pt_pick_te)]
    pt_reg_cal = [regret_of(q, oi) for q, oi in zip(cal_q, pt_pick_cal)]
    results["n_zero_oracle_excluded"] = int(sum(
        1 for r in pt_reg_cal + pt_reg_te if r is None))

    te_X = np.concatenate([e["X"] for e in te_e])
    te_A = np.concatenate([np.abs(e["yhat"] - np.log(e["true"] + 1.0)) for e in te_e])
    te_K = np.concatenate([np.array([len(k.split("|")) for k in e["keys"]]) for e in te_e])
    te_YT = np.concatenate([np.log(e["true"] + 1.0) for e in te_e])
    te_QID = np.concatenate([np.full(len(e["keys"]), i) for i, e in enumerate(te_e)])
    one_rng = np.random.default_rng(seed)
    one_idx = np.array([one_rng.choice(np.where(te_QID == u)[0]) for u in np.unique(te_QID)])

    for a in ALPHAS:
        res = {}
        q_glob = conformal_quantile(A, a)
        q_by_k = {int(k): conformal_quantile(A[K == k], a) for k in np.unique(K)}
        q_by_k = {k: (v if np.isfinite(v) else q_glob) for k, v in q_by_k.items()}
        sel = q_half == 1
        q_lw_clean = conformal_quantile(
            A[sel] / np.maximum(1e-6, g_clean.predict(XC[sel])), a)
        q_lw_reuse = conformal_quantile(A / np.maximum(1e-6, g_reuse.predict(XC)), a)
        lo_m, hi_m = cqr_models[a]
        q_cqr = conformal_quantile(
            np.maximum(lo_m.predict(XC) - YT, YT - hi_m.predict(XC)), a)

        picks = {"point": pt_pick_te}
        degenerate = {}
        qcheck = {"T1": q_glob, "T1p": q_glob, "T2_clean": q_lw_clean,
                  "T2_reuse": q_lw_reuse, "T3_CQR": q_cqr}
        for name in ("T1", "T1p", "T2_clean", "T2_reuse", "T3_CQR"):
            if not np.isfinite(qcheck[name]):
                picks[name] = pt_pick_te
                degenerate[name] = True
                continue
            pl = []
            for q, e in zip(te_q, te_e):
                if name == "T1":
                    ub = e["yhat"] + q_glob
                elif name == "T1p":
                    qk = np.array([q_by_k.get(len(k.split("|")), q_glob) for k in e["keys"]])
                    ub = e["yhat"] + qk
                elif name == "T2_clean":
                    ub = e["yhat"] + q_lw_clean * np.maximum(
                        1e-6, g_clean.predict(e["X"]))
                elif name == "T2_reuse":
                    ub = e["yhat"] + q_lw_reuse * np.maximum(
                        1e-6, g_reuse.predict(e["X"]))
                else:
                    ub = hi_m.predict(e["X"]) + q_cqr
                pl.append(pick(q, {k: float(v) for k, v in zip(e["keys"], ub)}))
            picks[name] = pl

        res["picks"] = {k: [int(x) for x in v] for k, v in picks.items()}
        res["degenerate_quantile"] = degenerate
        res["regret"] = {}
        res["plan_changes"] = {}
        for name, pk in picks.items():
            rs = [r for r in (regret_of(q, oi) for q, oi in zip(te_q, pk)) if r is not None]
            st = regret_stats(rs)
            st["p99_ci"] = bootstrap_ci(rs, lambda v: np.percentile(v, 99), seed=seed)
            st["max_ci"] = bootstrap_ci(rs, lambda v: np.max(v), seed=seed)
            st["mean_ci"] = bootstrap_ci(rs, np.mean, seed=seed)
            res["regret"][name] = st
            changed = sum(1 for x, y in zip(pk, pt_pick_te) if x != y)
            worse = sum(1 for q, x, y in zip(te_q, pk, pt_pick_te)
                        if x != y and q["true_costs"][x] > q["true_costs"][y])
            res["plan_changes"][name] = {"changed": changed, "worse": worse,
                                         "better": changed - worse}

        cov = {}
        cov["T1"] = {"pooled": float(np.mean(te_A <= q_glob)),
                     "per_query": float(np.mean(te_A[one_idx] <= q_glob)),
                     "log_width": float(2 * q_glob)}
        bk = np.array([q_by_k.get(int(k), q_glob) for k in te_K])
        cov["T1p"] = {"pooled": float(np.mean(te_A <= bk)),
                      "per_query": float(np.mean(te_A[one_idx] <= bk[one_idx])),
                      "log_width": float(np.median(2 * bk))}
        gw = np.maximum(1e-6, g_clean.predict(te_X))
        cov["T2_clean"] = {"pooled": float(np.mean(te_A <= q_lw_clean * gw)),
                           "per_query": float(np.mean(
                               te_A[one_idx] <= (q_lw_clean * gw)[one_idx])),
                           "log_width": float(np.median(2 * q_lw_clean * gw))}
        lo_p = lo_m.predict(te_X)
        hi_p = hi_m.predict(te_X)
        inside = (te_YT >= lo_p - q_cqr) & (te_YT <= hi_p + q_cqr)
        cov["T3_CQR"] = {"pooled": float(np.mean(inside)),
                         "per_query": float(np.mean(inside[one_idx])),
                         "log_width": float(np.median(hi_p - lo_p + 2 * q_cqr))}
        res["cardinality"] = cov

        s_un, s_sh = [], []
        for q, oi in zip(cal_q, pt_pick_cal):
            oracle = min(q["true_costs"])
            chosen = q["true_costs"][oi]
            s_sh.append(math.log((chosen + 1.0) / (oracle + 1.0)))
            if oracle > 0:
                s_un.append(math.log(chosen / oracle))
        tau_un = conformal_quantile(s_un, a)
        tau_sh = conformal_quantile(s_sh, a)
        te_r = [r for r in pt_reg_te if r is not None]
        te_hard = [r > 1.0 for r in te_r]

        def cover(tau):
            c = [r <= math.exp(tau) for r in te_r]
            easy = [x for x, h in zip(c, te_hard) if not h]
            hard = [x for x, h in zip(c, te_hard) if h]
            return {"all": float(np.mean(c)),
                    "easy": float(np.mean(easy)) if easy else None,
                    "hard": float(np.mean(hard)) if hard else None,
                    "n_easy": len(easy), "n_hard": len(hard)}

        miss = [r for r in te_r if r > math.exp(tau_un)]
        res["prcp"] = {"tau_unshifted": tau_un, "bound_unshifted": math.exp(tau_un),
                       "tau_shifted": tau_sh, "bound_shifted": math.exp(tau_sh),
                       "coverage_unshifted": cover(tau_un),
                       "coverage_shifted": cover(tau_sh),
                       "n_cal_scored": len(s_un),
                       "miss_n": len(miss),
                       "miss_min_regret": float(min(miss)) if miss else None,
                       "miss_median_regret": float(np.median(miss)) if miss else None}
        # ---- Mondrian PR-CP: one conformal quantile per difficulty stratum ----
        strata = {}
        for q, oi in zip(cal_q, pt_pick_cal):
            kk = len(q["tables"])
            oracle = min(q["true_costs"])
            if oracle > 0:
                strata.setdefault(kk, []).append(
                    math.log(q["true_costs"][oi] / oracle))
        tau_by_k = {kk: conformal_quantile(v, a) for kk, v in strata.items()}
        mcov, mbounds = [], []
        for q, r in zip(te_q, pt_reg_te):
            if r is None:
                continue
            t = tau_by_k.get(len(q["tables"]), tau_un)
            if not np.isfinite(t):
                t = tau_un
            mcov.append(bool(r <= math.exp(t)))
            mbounds.append(math.exp(min(t, 700)))
        m_easy = [c for c, h in zip(mcov, te_hard) if not h]
        m_hard = [c for c, h in zip(mcov, te_hard) if h]
        res["prcp_mondrian"] = {
            "tau_by_k": {str(kk): (v if np.isfinite(v) else None)
                         for kk, v in tau_by_k.items()},
            "n_cal_by_k": {str(kk): len(v) for kk, v in strata.items()},
            "median_bound": float(np.median(mbounds)) if mbounds else None,
            "max_bound": float(np.max(mbounds)) if mbounds else None,
            "cov_all": float(np.mean(mcov)) if mcov else None,
            "cov_easy": float(np.mean(m_easy)) if m_easy else None,
            "cov_hard": float(np.mean(m_hard)) if m_hard else None,
        }
        if n_inject > 0 and a in (0.01, 0.10):
            inj = []
            for q in te_q[:n_inject]:
                keys = []
                for r in range(2, len(q["tables"]) + 1):
                    for c in itertools.combinations(sorted(q["tables"]), r):
                        if is_connected(spec, list(c)):
                            keys.append("|".join(c))
                Xs = np.stack([fz(k.split("|"), q["preds"]) for k in keys])
                yh = predict(sc, model, Xs)
                kk = [len(k.split("|")) for k in keys]
                ubs = {"point": yh,
                       "T1": yh + q_glob,
                       "T1p": yh + np.array([q_by_k.get(x, q_glob) for x in kk]),
                       "T2_clean": yh + q_lw_clean * np.maximum(
                           1e-6, g_clean.predict(Xs)),
                       "T3_CQR": hi_m.predict(Xs) + q_cqr}
                inj.append({"keys": keys, "est": {
                    m: [float(x) if np.isfinite(x) else None for x in v]
                    for m, v in ubs.items()}})
            res["inject"] = inj
        results["alphas"]["%.2f" % a] = res
        print("  alpha=%.2f done" % a, flush=True)

    results["plan_space"] = plan_space
    results["estimator"] = estimator
    results["cost_floor"] = COST_FLOOR
    results["shift"] = shift
    results["test_file"] = os.path.basename(test_file) if test_file else None
    results["cal_queries"] = [
        {"k": len(q["tables"]), "true_costs": q["true_costs"], "point_pick": int(oi)}
        for q, oi in zip(cal_q, pt_pick_cal)]
    results["test_queries"] = [
        {"tables": q["tables"], "preds": q["preds"], "orders": q["orders"],
         "plans": q["plans"], "true_costs": q["true_costs"]} for q in te_q]
    os.makedirs(out_dir, exist_ok=True)
    tag = "%s_sf%g_%s_%s_seed%d" % (bench, sf, plan_space, estimator, seed)
    if tag_suffix:
        tag += "_" + tag_suffix
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(results, f, indent=1)
    print("WROTE", os.path.join(out_dir, tag + ".json"), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="tpch")
    ap.add_argument("--sf", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-queries", type=int, default=2500)
    ap.add_argument("--splits", default="1500,500,500")
    ap.add_argument("--max-plans", type=int, default=8)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--plan-space", default="leftdeep", choices=["leftdeep", "bushy"])
    ap.add_argument("--estimator", default="mlp", choices=["mlp", "gbm", "pg", "mlp_sel"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-threads", type=int, default=4)
    ap.add_argument("--shift", default="none",
                    choices=["none", "small2large", "large2small"])
    ap.add_argument("--test-file", default=None)
    ap.add_argument("--n-inject", type=int, default=200,
                    help="test queries for which per-subset estimates are stored "
                         "for cardinality injection into PostgreSQL")
    args = ap.parse_args()
    splits = tuple(int(x) for x in args.splits.replace(":", ",").split(","))
    if len(splits) != 3:
        raise SystemExit("--splits needs 3 values (train,cal,test); got %r" % (splits,))
    run(args.bench, args.sf, args.seed, args.n_queries, splits, args.max_plans,
        args.data_dir, args.out_dir, workers=args.workers,
        plan_space=args.plan_space, estimator=args.estimator,
        wthreads=args.worker_threads, shift=args.shift,
        test_file=args.test_file, n_inject=args.n_inject,
        tag_suffix="_".join(x for x in [
            args.shift if args.shift != "none" else "",
            os.path.splitext(os.path.basename(args.test_file))[0]
            if args.test_file else ""] if x))


if __name__ == "__main__":
    main()
