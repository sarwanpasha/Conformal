#!/bin/bash
# JOB-light (70 queries over the same six IMDB tables), from the MSCN repository.
set -e
mkdir -p data
curl -sL -o data/job-light.csv \
  https://raw.githubusercontent.com/andreaskipf/learnedcardinalities/master/workloads/job-light.csv
wc -l data/job-light.csv
