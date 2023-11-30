#!/usr/bin/env bash
set -x
set -e
PG_PATH=$(pwd)/pg_database
mkdir -p $PG_PATH
echo postgres > /tmp/pg
initdb --pwfile /tmp/pg -U postgres $PG_PATH
rm -f $PG_PATH/pg_hba.conf
cat <<EOM >$PG_PATH/pg_hba.conf
local   all             all                                     md5
host    all             all             0.0.0.0/0               md5
host    all             all             ::1/128                 md5
EOM
postgres -D $PG_PATH -k $PG_PATH -i
