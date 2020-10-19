FROM postgres:11

# For our schema and convenience
ENV POSTGRES_USER=data_engineer POSTGRES_DB=test_db

COPY sql /initdb-sqls/

# Move sql files to /container-entrypoint-initdb/ and rename them
# so that postgres takes foldernames into account when executing the files
COPY move_init_sqls.sh /move_init_sqls.sh
RUN chmod +x /move_init_sqls.sh
RUN ./move_init_sqls.sh

