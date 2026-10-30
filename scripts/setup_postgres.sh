#!/bin/bash
# Build PostgreSQL 16.4 with pg_hint_plan 1.6.0 into ./pg and create a cluster in
# ./pgdata (no root access needed). Start it with scripts/pg_ctl.sh start.
set -e
cd "$(dirname "$0")/.."
ROOT=$PWD
mkdir -p build && cd build
if [ ! -x $ROOT/pg/bin/postgres ]; then
  curl -sL -o pg.tar.bz2 https://ftp.postgresql.org/pub/source/v16.4/postgresql-16.4.tar.bz2
  tar xjf pg.tar.bz2 && cd postgresql-16.4
  ./configure --prefix=$ROOT/pg --without-icu --without-readline --without-zlib
  make -j"$(nproc)" && make install && cd ..
fi
if [ ! -f $ROOT/pg/lib/pg_hint_plan.so ]; then
  curl -sL -o hint.tar.gz https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL16_1_6_0.tar.gz
  tar xzf hint.tar.gz && cd pg_hint_plan-REL16_1_6_0
  make USE_PGXS=1 PG_CONFIG=$ROOT/pg/bin/pg_config
  make USE_PGXS=1 PG_CONFIG=$ROOT/pg/bin/pg_config install && cd ..
fi
if [ ! -f $ROOT/pgdata/PG_VERSION ]; then
  $ROOT/pg/bin/initdb -D $ROOT/pgdata -U "$USER"
  cat >> $ROOT/pgdata/postgresql.conf <<CONF
shared_preload_libraries = 'pg_hint_plan'
shared_buffers = 8GB
work_mem = 256MB
max_parallel_workers_per_gather = 0
listen_addresses = ''
CONF
fi
echo "PostgreSQL ready in $ROOT/pg, cluster in $ROOT/pgdata"
