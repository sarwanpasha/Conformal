#!/bin/bash
# PostgreSQL experiments for one benchmark. Requires a running server
# (scripts/pg_ctl.sh start) and result files from scripts/run_main.sh.
#   usage: scripts/run_postgres.sh tpch|tpcds|job
set -e
cd "$(dirname "$0")/.."
B=$1; R=results/main; O=results/pg
T=120; [ "$B" = job ] && T=300
python src/pg_experiments.py load --bench $B --sf 1 --data-dir data
for s in 1 2 3; do
  python src/pg_experiments.py pgest --result $R/${B}_sf1_leftdeep_mlp_seed$s.json --out $O/pgest_${B}_seed$s.json
done
F=$R/${B}_sf1_leftdeep_mlp_seed1.json
python src/pg_experiments.py wallclock --result $F --alpha 0.01 --n 80 --reps 3 --timeout $T --out $O/wall_${B}_seed1.json
python src/pg_experiments.py latcp --result $F --n 500 --reps 3 --timeout $T --workers 6 --out $O/latcp_${B}_seed1.json
for a in 0.01 0.10; do
  python src/pg_experiments.py inject --result $F --alpha $a --n 100 --reps 3 --timeout $T --out $O/inject_${B}_mlp_a$a.json
done
