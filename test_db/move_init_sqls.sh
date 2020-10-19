#!/usr/bin/env bash

# Moves all sql files from /initdb-sqls/** to /docker-entrypoint-initdb.d/
# and renames eg. a/b.sql into a_b.sql. This means subfolders appear in
# the filename. This is necessary because postgres only (alphabetically)
# executes files in /docker-entrypoint-initdb.d/ and not in subfolders.

# Important to also catch files directly in sql
shopt -s globstar

cd /initdb-sqls/
for f in **/*.sql; do
  mv -v "$f" "../docker-entrypoint-initdb.d/${f//\//_}"
done

