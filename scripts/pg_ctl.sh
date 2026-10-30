#!/bin/bash
# Start or stop the local PostgreSQL cluster built by setup_postgres.sh.
# The Unix socket lives in /tmp, which is the default PRCP_PG_HOST.
cd "$(dirname "$0")/.."
PORT=${PRCP_PG_PORT:-5432}
case "$1" in
  start) pg/bin/pg_ctl -D pgdata -l pgdata/server.log -o "-p $PORT -k /tmp" -w start ;;
  stop)  pg/bin/pg_ctl -D pgdata -w stop ;;
  *) echo "usage: $0 start|stop"; exit 1 ;;
esac
