# PostgreSQL deployment contract

SQLite remains the zero-setup backend for unit tests and manual fallback work.
Docker and deployment use PostgreSQL through `ONE_DATABASE_URL` and must
install the optional adapter dependency:

```bash
pip install -e '.[postgres]'
export ONE_DATABASE_URL='postgresql://one:<password>@postgres:5432/one'
```

The API does not serve requests until the schema migrations have completed.
Startup applies the idempotent, numbered SQL files in `migrations/` in lexical
order and records versions in `schema_migrations`. A fresh database receives
every table used by `app/main.py`, including the family-mode tables in
`002_family_mode.sql`. The Docker image copies this migration set; do not rely
on a manually prepared database.

## Verification

Run the normal deterministic suite first:

```bash
cd one
pytest
```

For an ephemeral PostgreSQL instance, set `ONE_TEST_POSTGRES_URL` and run the
opt-in contract checks:

```bash
ONE_TEST_POSTGRES_URL='postgresql://one:<password>@127.0.0.1:5432/one' \
  pytest -q tests/test_postgres_contract.py
```

Those checks verify the health response identifies `postgresql`, pairing and
writes work, nullable map lookup is portable, transaction rollback is real,
and household deletion cascades through foreign-key-backed rows. They do not
call LM Studio or any other external inference service.

`GET /api/v1/health` is a liveness/readiness signal for the API and database;
it should return HTTP 200 with `{"status":"ok","database":"postgresql"}`
only after the database connection and migrations are ready. Container
orchestration should wait for the PostgreSQL service health check rather than
assuming that process start means the database is accepting connections.
