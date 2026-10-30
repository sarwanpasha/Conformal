#!/bin/bash
# Main configurations from the paper on a single machine (DuckDB only).
# WORKERS controls the number of processes that compute exact cardinalities.
set -e
cd "$(dirname "$0")/.."
W=${WORKERS:-16}
run() { python src/pipeline.py "$@" --workers $W --worker-threads 2 --data-dir data --out-dir results/main; }
for s in 1 2 3; do
  for est in mlp mlp_sel gbm; do
    run --bench tpch  --sf 1 --seed $s --n-queries 2500 --splits 1500:500:500 --estimator $est
    run --bench tpcds --sf 1 --seed $s --n-queries 1500 --splits 900:300:300  --estimator $est
    run --bench job   --sf 1 --seed $s --n-queries 800  --splits 400:200:200  --estimator $est
  done
  run --bench tpch  --sf 1 --seed $s --n-queries 2500 --splits 1500:500:500 --plan-space bushy
  run --bench tpcds --sf 1 --seed $s --n-queries 1500 --splits 900:300:300  --plan-space bushy
  run --bench job   --sf 1 --seed $s --n-queries 800  --splits 400:200:200  --plan-space bushy
  for sh in small2large large2small; do
    run --bench tpch  --sf 1 --seed $s --n-queries 2500 --splits 1500:500:500 --shift $sh
    run --bench tpcds --sf 1 --seed $s --n-queries 1500 --splits 900:300:300  --shift $sh
    run --bench job   --sf 1 --seed $s --n-queries 800  --splits 400:200:200  --shift $sh
  done
  run --bench job --sf 1 --seed $s --n-queries 800 --splits 400:200:200 --test-file data/job-light.csv
done
