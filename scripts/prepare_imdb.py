"""Download the IMDB snapshot used by the Join Order Benchmark and build
data/job_sf1.duckdb with the six tables used in the experiments."""
import os
import subprocess
import sys

import duckdb

URL = "https://event.cwi.nl/da/job/imdb.tgz"
SCHEMA = {
    "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id",
              "phonetic_code", "episode_of_id", "season_nr", "episode_nr",
              "series_years", "md5sum"],
    "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
    "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
    "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
    "movie_keyword": ["id", "movie_id", "keyword_id"],
    "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order",
                  "role_id"],
}


def main(data_dir="data"):
    csv_dir = os.path.join(data_dir, "imdb")
    os.makedirs(csv_dir, exist_ok=True)
    if not os.path.exists(os.path.join(csv_dir, "title.csv")):
        tgz = os.path.join(data_dir, "imdb.tgz")
        subprocess.run(["curl", "-L", "--retry", "3", "-o", tgz, URL], check=True)
        subprocess.run(["tar", "xzf", tgz, "-C", csv_dir], check=True)
        os.remove(tgz)
    db = os.path.join(data_dir, "job_sf1.duckdb")
    if os.path.exists(db):
        print("exists:", db)
        return
    con = duckdb.connect(db)
    for t, cols in SCHEMA.items():
        names = ", ".join("'%s'" % c for c in cols)
        con.execute("CREATE TABLE %s AS SELECT * FROM read_csv('%s/%s.csv', header=false, "
                    "quote='\"', escape='\\', names=[%s], ignore_errors=true)"
                    % (t, csv_dir, t, names))
        print(t, con.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0])
    con.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
