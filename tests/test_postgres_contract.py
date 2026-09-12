"""Optional PostgreSQL contract checks.

The normal test suite remains zero-setup SQLite.  CI (and a developer with a
disposable database) opts into this module with ``ONE_TEST_POSTGRES_URL``.
These checks intentionally stop at database-backed API operations; no LM
Studio or other external service is contacted.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import _statements
from app.main import make_app


def _postgres_url() -> str | None:
    value = os.environ.get("ONE_TEST_POSTGRES_URL", "").strip()
    return value or None


@pytest.fixture
def postgres_client(tmp_path: Path):
    dsn = _postgres_url()
    if not dsn:
        pytest.skip("set ONE_TEST_POSTGRES_URL to run PostgreSQL contract checks")
    pytest.importorskip("psycopg")
    settings = Settings(
        database_url=dsn,
        object_store_path=tmp_path / "objects",
        bootstrap_secret="postgres-test",
        env="test",
        # A deliberately unreachable endpoint makes accidental LLM use fail
        # quickly.  This suite only exercises deterministic DB/API paths.
        lm_studio_url="http://127.0.0.1:9/v1",
    )
    with TestClient(make_app(settings)) as client:
        yield client


def _admin(client: TestClient) -> tuple[dict[str, str], str]:
    started = client.post(
        "/api/v1/pairing/start",
        json={"display_name": f"Postgres fixture {uuid4()}", "home_name": "Postgres Contract"},
    )
    assert started.status_code == 200, started.text
    completed = client.post(
        "/api/v1/pairing/complete", json={"code": started.json()["pairing_code"]}
    )
    assert completed.status_code == 200, completed.text
    payload = completed.json()
    return {"Authorization": f"Bearer {payload['access_token']}"}, payload["home_id"]


def test_migration_statement_splitter_ignores_comments_and_string_semicolons():
    script = """
    -- comment with a ; semicolon
    CREATE TABLE demo (value TEXT DEFAULT 'keep; this'); /* inline ; comment */
    INSERT INTO demo VALUES ('it''s still one statement');
    """
    assert _statements(script) == [
        "CREATE TABLE demo (value TEXT DEFAULT 'keep; this')",
        "INSERT INTO demo VALUES ('it''s still one statement')",
    ]


def test_postgres_health_pairing_transaction_and_cascade(postgres_client: TestClient):
    """Exercise health, writes, a nullable query, rollback, and FK cascade."""

    health = postgres_client.get("/api/v1/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok"
    assert health.json()["database"] == "postgresql"

    headers, home_id = _admin(postgres_client)
    room = postgres_client.post(
        f"/api/v1/homes/{home_id}/rooms", headers=headers, json={"name": "Kitchen"}
    )
    assert room.status_code == 200, room.text

    # A NULL room_id takes the same SQL path used by the production map route
    # and guards against a SQLite-only ``IS ?`` implementation.
    room_map = postgres_client.post(
        f"/api/v1/homes/{home_id}/maps",
        headers=headers,
        json={"room_id": None, "map_data": {"objects": []}},
    )
    assert room_map.status_code == 200, room_map.text
    assert room_map.json()["revision"] == 1

    db = postgres_client.app.state.db
    actual_tables = {
        row["table_name"]
        for row in db.many(
            "SELECT table_name FROM information_schema.tables WHERE table_schema=?",
            ("public",),
        )
    }
    assert {
        "homes",
        "users",
        "memberships",
        "home_runtime",
        "sessions",
        "pairing_codes",
        "consents",
        "cameras",
        "rooms",
        "room_maps",
        "calibrations",
        "objects",
        "observations",
        "events",
        "clips",
        "summaries",
        "audit_log",
        "deletion_requests",
        "family_invites",
        "medication_plans",
        "medication_check_ins",
    } <= actual_tables
    assert [row["version"] for row in db.many("SELECT version FROM schema_migrations ORDER BY version")] == [1, 2]
    assert db.one("SELECT '100%' AS label")["label"] == "100%"

    # A second application process must see the same migration set.  This
    # catches non-idempotent or multi-statement initialization before serving
    # requests, rather than only proving that the first connection booted.
    second_app = make_app(postgres_client.app.state.settings)
    assert second_app.state.db.health() == {"status": "ok", "backend": "postgresql"}
    assert [row["version"] for row in second_app.state.db.many("SELECT version FROM schema_migrations ORDER BY version")] == [1, 2]

    rollback_home = str(uuid4())
    with pytest.raises(RuntimeError, match="fixture rollback"):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO homes (id, name, created_at) VALUES (?,?,?)",
                (rollback_home, "must roll back", "2026-01-01T00:00:00+00:00"),
            )
            raise RuntimeError("fixture rollback")
    assert db.one("SELECT id FROM homes WHERE id=?", (rollback_home,)) is None

    # Use the API's audited deletion workflow as deterministic cleanup and as
    # proof that PostgreSQL foreign keys cascade across the household schema.
    deleted = postgres_client.post(
        f"/api/v1/homes/{home_id}/privacy/delete", headers=headers
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["status"] == "completed"
    assert db.one("SELECT id FROM homes WHERE id=?", (home_id,)) is None
    assert db.one("SELECT id FROM room_maps WHERE home_id=?", (home_id,)) is None


def test_migration_files_are_ordered_and_cover_schema_when_postgres_enabled():
    """Fail loudly in opt-in runs if the deployment migration set is partial."""

    if not _postgres_url():
        pytest.skip("set ONE_TEST_POSTGRES_URL to validate deployment migrations")
    migration_dir = Path(__file__).parents[1] / "migrations"
    migrations = sorted(migration_dir.glob("*.sql"))
    assert migrations, "PostgreSQL deployment requires at least one SQL migration"
    prefixes = [path.name.split("_", 1)[0] for path in migrations]
    assert all(prefix.isdigit() for prefix in prefixes)
    assert len(prefixes) == len(set(prefixes)), "migration numbers must be unique"

    sql = "\n".join(path.read_text(encoding="utf-8").lower() for path in migrations)
    required_tables = {
        "homes",
        "users",
        "memberships",
        "home_runtime",
        "sessions",
        "pairing_codes",
        "consents",
        "cameras",
        "rooms",
        "room_maps",
        "calibrations",
        "objects",
        "observations",
        "events",
        "clips",
        "summaries",
        "audit_log",
        "deletion_requests",
        "family_invites",
        "medication_plans",
        "medication_check_ins",
    }
    missing = [
        table
        for table in sorted(required_tables)
        if not re.search(rf"create\s+table(?:\s+if\s+not\s+exists)?\s+{table}\b", sql)
    ]
    assert not missing, f"migration set does not declare required tables: {missing}"
