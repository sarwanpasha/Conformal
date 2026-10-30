# Plan-Regret Conformal Prediction (PR-CP)

Code for the paper *Plan-Regret Conformal Prediction for Join Ordering*.

A query optimizer chooses a join order by comparing estimated plan costs. A common
idea for making learned cardinality estimators safer is to plan with conformal
upper bounds instead of point estimates. This repository shows, and lets you
reproduce, two findings:

1. With a single global split-conformal quantile, the optimizer picks **exactly
   the same plan** as with the point estimates (the bounds rescale every plan's
   estimated cost by the same factor).
2. With adaptive intervals (locally weighted CP, conformalized quantile
   regression), the optimizer often picks **much worse** plans, even though every
   interval keeps its nominal coverage.

It also implements **PR-CP**, which calibrates a conformal quantile directly on the
*plan regret* of the selected plan,

    regret(Q) = C_out(selected plan) / C_out(best candidate plan),

and returns a distribution-free certificate: with probability at least 1 - alpha,
the selected plan is at most `exp(tau)` times more expensive than the best
candidate. A per-join-size (Mondrian) variant and a variant calibrated on measured
PostgreSQL latency are included.

## Contents

| Path | Purpose |
|---|---|
| `src/pipeline.py` | Workload generation, exact cardinalities (DuckDB), estimators, conformal upper bounds, plan selection, PR-CP. Writes one JSON file per run. |
| `src/pg_experiments.py` | PostgreSQL experiments: loading data, native estimates, forced join orders, cardinality injection, latency-calibrated PR-CP. |
| `src/report.py` | Prints a readable summary of result files. |
| `src/summarize.py` | Aggregates result files over seeds into one compact JSON file. |
| `src/make_figures.py` | Draws the figures of the paper from the output of `summarize.py`. |
| `scripts/run_example.sh` | Small end-to-end example (about 30 seconds on 4 cores). |
| `scripts/run_main.sh` | Main configurations of the paper on one machine. |
| `scripts/prepare_imdb.py`, `scripts/get_job_light.sh` | IMDB data for the JOB and JOB-light experiments. |
| `scripts/setup_postgres.sh`, `scripts/pg_ctl.sh`, `scripts/run_postgres.sh` | Build PostgreSQL 16.4 with `pg_hint_plan` 1.6.0 locally and run the PostgreSQL experiments. |

## Installation

Python 3.9 or newer.

```bash
git clone https://github.com/sarwanpasha/Conformal.git
cd Conformal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TPC-H and TPC-DS data are generated automatically by DuckDB's `tpch` and `tpcds`
extensions on first use (internet access is needed once to install them).

## Quick example

```bash
bash scripts/run_example.sh
```

This generates TPC-H at scale factor 0.01, builds a workload of 600 join queries
with 2 to 5 tables, computes all intermediate cardinalities exactly, trains the
MLP estimator, and evaluates every plan-selection rule for 2 seeds. It takes about
30 seconds on 4 cores and ends with a summary like the following (from our test
run, Python 3.12, DuckDB 2.0):

```
benchmark=tpch  sf=0.01  plan space=leftdeep  estimator=mlp  seeds=2
queries: train=300  calibration=150  test=150   candidate plans used per query=5.58

alpha = 0.01   plan regret of the chosen plan (mean over seeds)
  picker                        P50      P90        P99          max
  point estimate               1.00     1.00       1.55        216.3
  split CP                     1.00     1.00       1.55        216.3
  size-stratified CP           1.00     1.00       1.63        217.0
  LW-S-CP                      1.00     3.79     185.70        329.6
  LW-S-CP (reused split)       1.00     2.97      28.32        279.6
  CQR                          1.00     2.00      31.86         69.4
  split CP picks the same plan as the point estimate on every query: True
  PR-CP certificate exp(tau) = 4.325   coverage: all=0.997  hard=0.967
```

How to read it:

* *Plan regret* is the C_out cost of the selected plan divided by the cost of the
  best candidate (1.00 means the best candidate was selected).
* The `point estimate` and `split CP` rows are identical: global split-conformal
  upper bounds never change the selected plan.
* The adaptive methods (`LW-S-CP`, `CQR`) select much worse plans in the tail
  (P90 and P99), even though their cardinality intervals keep their nominal
  coverage.
* PR-CP certifies, for 99% of queries, that the selected plan costs at most
  `exp(tau)` times the best candidate. `coverage` is the observed fraction of test
  queries within the certificate, over all queries and over queries where the
  point estimate did not select the best plan.

Exact numbers vary slightly between runs and library versions. The paper uses
scale factors 1 and 10 and thousands of queries per run.

## Running a single configuration

```bash
python src/pipeline.py --bench tpch --sf 1 --seed 1 \
    --n-queries 2500 --splits 1500:500:500 \
    --estimator mlp --plan-space leftdeep --workers 16 \
    --data-dir data --out-dir results/main
python src/report.py results/main/tpch_sf1_leftdeep_mlp_seed1.json
```

Main options:

| Option | Values |
|---|---|
| `--bench` | `tpch`, `tpcds`, `job` (run `python scripts/prepare_imdb.py` first for `job`) |
| `--sf` | scale factor for `tpch` and `tpcds` |
| `--estimator` | `mlp`, `mlp_sel` (MLP with exact base-table selectivities), `gbm` (XGBoost), `pg` (PostgreSQL row estimates; needs a running server with the data loaded) |
| `--plan-space` | `leftdeep` or `bushy` |
| `--shift` | `none`, `small2large` (train/calibrate on 2-3 tables, test on 4-5), `large2small` |
| `--test-file` | fixed test workload, for example `data/job-light.csv` from `scripts/get_job_light.sh` |
| `--workers` | processes used to compute exact cardinalities |

Each run writes a JSON file with, for every miscoverage level alpha in
{0.01, 0.05, 0.10, 0.20}: the plan picked by each rule for every test query, plan
regret statistics, cardinality coverage of each conformal method, and the PR-CP
and Mondrian PR-CP certificates with their coverage.

### Plan-selection rules in the output

| Key | Rule |
|---|---|
| `point` | point estimates |
| `T1` | split conformal upper bounds (global quantile) |
| `T1p` | split conformal upper bounds with one quantile per intermediate size |
| `T2_clean` | locally weighted CP, difficulty model and quantile on disjoint calibration halves |
| `T2_reuse` | locally weighted CP, both fitted on the same calibration data |
| `T3_CQR` | conformalized quantile regression |

## Reproducing the paper

```bash
python scripts/prepare_imdb.py      # IMDB tables for JOB (about 1.2 GB download)
bash scripts/get_job_light.sh       # JOB-light queries
WORKERS=32 bash scripts/run_main.sh # DuckDB experiments, 3 seeds
python src/summarize.py --dirs results/main --out results/summary.json
python src/make_figures.py --summary results/summary.json --out Figures
```

PostgreSQL experiments:

```bash
bash scripts/setup_postgres.sh      # builds PostgreSQL 16.4 + pg_hint_plan 1.6.0 into ./pg
bash scripts/pg_ctl.sh start
for b in tpch tpcds job; do bash scripts/run_postgres.sh $b; done
bash scripts/pg_ctl.sh stop
```

Connection settings for `src/pg_experiments.py` and `--estimator pg` are read from
`PRCP_PG_HOST` (socket directory or host, default `/tmp`), `PRCP_PG_PORT` (default
5432) and `PRCP_PG_USER` (default `$USER`).

### Large scale factors on a cluster

Computing exact cardinalities at TPC-H scale factor 10 takes many CPU hours, because
a few generated queries have very large intermediate results. `pipeline.py` can
split this step over several machines. Every machine needs access to the same
directory.

```bash
A="--bench tpch --sf 10 --seed 1 --n-queries 1500 --splits 900:300:300 --data-dir data --out-dir results/sf10"
# 1. generate and save the query set once
CARD_DIR=cards/s1 CARD_MODE=prep python src/pipeline.py $A
# 2. compute shard K of N on any machine (repeat for K = 0..N-1, in parallel)
CARD_DIR=cards/s1 CARD_MODE=shard0/12 QUERY_TIMEOUT_S=3600 python src/pipeline.py $A --workers 32
# 3. merge the shards and run the rest of the pipeline
CARD_DIR=cards/s1 CARD_MODE=assemble python src/pipeline.py $A
```

A shard can be split further with `CARD_MODE=shardK/N:m/M`. `QUERY_TIMEOUT_S`
drops queries whose cardinalities cannot be computed within the given number of
seconds. The merged workload is the same as a single-machine run would produce.

## License

MIT License. See `LICENSE`.
