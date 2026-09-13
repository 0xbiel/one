import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS homes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    care_setting TEXT NOT NULL DEFAULT 'home',
    support_focus TEXT NOT NULL DEFAULT 'general'
);
CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, display_name TEXT NOT NULL, email TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memberships (home_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('resident','caregiver','admin','publisher')), PRIMARY KEY(home_id,user_id), FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS home_runtime (home_id TEXT PRIMARY KEY, paused INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, home_id TEXT NOT NULL, expires_at TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS pairing_codes (code_hash TEXT PRIMARY KEY, home_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL, expires_at TEXT NOT NULL, used_at TEXT, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS email_verifications (id TEXT PRIMARY KEY, email TEXT NOT NULL, user_id TEXT NOT NULL, home_id TEXT NOT NULL, purpose TEXT NOT NULL CHECK(purpose IN ('create','login')), code_hash TEXT NOT NULL UNIQUE, expires_at TEXT NOT NULL, used_at TEXT, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS consents (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, subject_user_id TEXT NOT NULL, purpose TEXT NOT NULL, policy_version TEXT NOT NULL, granted_at TEXT NOT NULL, revoked_at TEXT, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS cameras (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, name TEXT NOT NULL, room_id TEXT, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, resolution_width INTEGER, resolution_height INTEGER, metadata_json TEXT NOT NULL DEFAULT '{}', FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS rooms (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS room_maps (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, room_id TEXT, revision INTEGER NOT NULL, coordinate_frame TEXT NOT NULL, artifact_key TEXT, map_json TEXT NOT NULL, created_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'manual', dimension TEXT NOT NULL DEFAULT '2d', approximate INTEGER NOT NULL DEFAULT 0, localization_status TEXT NOT NULL DEFAULT 'unlocalized', metadata_json TEXT NOT NULL DEFAULT '{}', usdz_artifact_key TEXT, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS calibrations (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, camera_id TEXT NOT NULL, map_id TEXT NOT NULL, intrinsics_json TEXT NOT NULL, extrinsics_json TEXT NOT NULL, accuracy_m REAL, created_at TEXT NOT NULL, resolution_width INTEGER, resolution_height INTEGER, camera_metadata_json TEXT NOT NULL DEFAULT '{}', metrics_json TEXT NOT NULL DEFAULT '{}', source TEXT NOT NULL DEFAULT 'manual', status TEXT NOT NULL DEFAULT 'active', invalidated_at TEXT, invalidation_reason TEXT, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS camera_map_generation_jobs (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, camera_id TEXT NOT NULL, room_id TEXT, room_label TEXT NOT NULL DEFAULT 'Room', orientation TEXT NOT NULL DEFAULT 'portrait', status TEXT NOT NULL CHECK(status IN ('collecting','processing','ready','needs_rescan','unavailable','failed')), frame_count INTEGER NOT NULL DEFAULT 0, resolution_width INTEGER NOT NULL, resolution_height INTEGER NOT NULL, map_id TEXT, error_code TEXT, error_message TEXT, metrics_json TEXT NOT NULL DEFAULT '{}', model_version TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE, FOREIGN KEY(map_id) REFERENCES room_maps(id) ON DELETE SET NULL);
CREATE INDEX IF NOT EXISTS camera_map_generation_jobs_camera_idx ON camera_map_generation_jobs(home_id, camera_id, updated_at);
CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, label TEXT NOT NULL, display_name TEXT, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, object_id TEXT, camera_id TEXT, map_id TEXT, x REAL, y REAL, z REAL, uncertainty_m REAL, confidence REAL, detector_version TEXT, observed_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, event_type TEXT NOT NULL, status TEXT NOT NULL, explanation TEXT, confidence REAL, evidence_json TEXT NOT NULL, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, expires_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS clips (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, event_id TEXT NOT NULL, object_key TEXT NOT NULL, starts_at TEXT NOT NULL, ends_at TEXT NOT NULL, expires_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS summaries (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, subject_user_id TEXT, status TEXT NOT NULL, trend TEXT NOT NULL, explanation TEXT NOT NULL, evidence_json TEXT NOT NULL, limitations TEXT NOT NULL, model_version TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS audit_log (id TEXT PRIMARY KEY, home_id TEXT, user_id TEXT, action TEXT NOT NULL, target_type TEXT, target_id TEXT, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS deletion_requests (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, requested_by TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT);
CREATE TABLE IF NOT EXISTS family_invites (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, invited_by TEXT NOT NULL, email TEXT, display_name TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('resident','caregiver')), code_hash TEXT NOT NULL UNIQUE, expires_at TEXT NOT NULL, accepted_at TEXT, created_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(invited_by) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS medication_plans (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, subject_user_id TEXT NOT NULL, name TEXT NOT NULL, dose TEXT NOT NULL, schedule TEXT NOT NULL, instructions TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, version INTEGER NOT NULL DEFAULT 1, created_by TEXT NOT NULL, assigned_caregiver_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(subject_user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(assigned_caregiver_id) REFERENCES users(id) ON DELETE SET NULL);
CREATE TABLE IF NOT EXISTS medication_check_ins (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, plan_id TEXT NOT NULL, subject_user_id TEXT NOT NULL, scheduled_for TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','taken','skipped','missed')), note TEXT NOT NULL DEFAULT '', marked_by TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(plan_id) REFERENCES medication_plans(id) ON DELETE CASCADE, FOREIGN KEY(subject_user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(marked_by) REFERENCES users(id) ON DELETE SET NULL, UNIQUE(plan_id, scheduled_for));
"""


SCHEMA_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""


def _statements(script: str) -> list[str]:
    """Split migration SQL while respecting strings and SQL comments.

    This is intentionally a small parser rather than a general SQL parser:
    migrations are trusted repository files, but comments and semicolons in a
    string literal must not alter statement boundaries during PostgreSQL
    startup. Both line and block comments are accepted.
    """
    statements: list[str] = []
    buffer: list[str] = []
    state = "normal"
    index = 0
    while index < len(script):
        char = script[index]
        next_char = script[index + 1] if index + 1 < len(script) else ""
        if state == "normal":
            if char == "'":
                state = "single"
                buffer.append(char)
            elif char == '"':
                state = "double"
                buffer.append(char)
            elif char == "-" and next_char == "-":
                state = "line_comment"
                buffer.append(" ")
                index += 1
            elif char == "/" and next_char == "*":
                state = "block_comment"
                buffer.append(" ")
                index += 1
            elif char == ";":
                statement = "".join(buffer).strip()
                if statement:
                    statements.append(statement)
                buffer = []
            else:
                buffer.append(char)
        elif state == "line_comment":
            if char in "\r\n":
                state = "normal"
                buffer.append(char)
        elif state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 1
        elif state == "single":
            buffer.append(char)
            if char == "'":
                if next_char == "'":
                    buffer.append(next_char)
                    index += 1
                else:
                    state = "normal"
            elif char == "\\" and next_char:
                buffer.append(next_char)
                index += 1
        else:  # double-quoted identifier
            buffer.append(char)
            if char == '"':
                if next_char == '"':
                    buffer.append(next_char)
                    index += 1
                else:
                    state = "normal"
        index += 1
    statement = "".join(buffer).strip()
    if statement:
        statements.append(statement)
    return statements


def _portable_schema() -> str:
    """Remove SQLite's foreign-key pragma before sending the schema to Postgres."""
    return "\n".join(
        line for line in SCHEMA.splitlines() if not line.strip().upper().startswith("PRAGMA ")
    )


def _migration_file(filename: str, fallback: str = "") -> str:
    migration = Path(__file__).resolve().parent.parent / "migrations" / filename
    try:
        return migration.read_text(encoding="utf-8")
    except OSError:
        return fallback


def _core_migration() -> str:
    # Keep the deployment schema source in the numbered migration set. The
    # fallback keeps installed wheels/self-contained images bootable when the
    # repository's migration directory is not packaged.
    return _migration_file("001_initial.sql", _portable_schema())


def _family_migration() -> str:
    # The family tables are also present in the current SQLite baseline, so an
    # absent migration directory remains safe for packaged/local test builds.
    return _migration_file("002_family_mode.sql")

def _mapping_migration() -> str:
    return _migration_file("003_camera_roomplan.sql")


def _identity_migration() -> str:
    return _migration_file("004_email_identity.sql")


def _camera_generation_migration() -> str:
    return _migration_file("005_camera_map_generation.sql")


def _camera_generation_metadata_migration() -> str:
    return _migration_file("006_camera_map_generation_metadata.sql")


def _roomplan_usdz_migration() -> str:
    return _migration_file("007_roomplan_usdz.sql")


def _postgres_migrations() -> list[tuple[int, str]]:
    migration_dir = Path(__file__).resolve().parent.parent / "migrations"
    paths = sorted(migration_dir.glob("*.sql"))
    if not paths:
        raise RuntimeError("PostgreSQL requires the numbered SQL files in migrations/")

    migrations: list[tuple[int, str]] = []
    seen: set[int] = set()
    for path in paths:
        prefix = path.name.split("_", 1)[0]
        if not prefix.isdigit():
            raise RuntimeError(f"Invalid PostgreSQL migration filename: {path.name}")
        version = int(prefix)
        if version in seen:
            raise RuntimeError(f"Duplicate PostgreSQL migration version: {version}")
        seen.add(version)
        migrations.append((version, path.read_text(encoding="utf-8")))
    return migrations


def _postgres_sql(sql: str) -> str:
    """Translate the intentionally SQLite-shaped query API to psycopg SQL."""
    # The application uses qmark placeholders everywhere so isolated SQLite
    # tests and Postgres share the same query strings.
    # psycopg uses `%` for its parameter grammar. Escape literal percent signs
    # first (for example a SQL LIKE pattern), then introduce the `%s` markers.
    translated = sql.replace("%", "%%").replace("?", "%s")
    # SQLite's `IS ?` is a null-safe equality comparison. PostgreSQL only
    # permits IS with NULL/TRUE/FALSE literals, so use its equivalent.
    return re.sub(r"\bIS\s+%s\b", "IS NOT DISTINCT FROM %s", translated, flags=re.IGNORECASE)


class _TransactionConnection:
    """Small connection facade that preserves qmark SQL inside transactions."""

    def __init__(self, database: "Database"):
        self._database = database

    def execute(self, sql: str, params: tuple = ()):
        return self._database.conn.execute(self._database._sql(sql), params)

    def __getattr__(self, name: str):
        return getattr(self._database.conn, name)


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self.backend = settings.database_backend
        self.path: str | None = None
        if self.backend == "sqlite":
            path = settings.sqlite_path
            if path is None:  # defensive; Settings validates the scheme below
                raise RuntimeError("Invalid SQLite database URL")
            self.path = path
            if path != ":memory:":
                import os
                os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._initialize_sqlite()
        elif self.backend == "postgresql":
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL is configured but psycopg is not installed; "
                    "install the optional dependency with `pip install -e '.[postgres]'`"
                ) from exc
            try:
                self.conn = psycopg.connect(settings.database_url, row_factory=dict_row)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to connect to PostgreSQL for ONE_DATABASE_URL "
                    f"({type(exc).__name__}); verify the database is reachable and credentials are valid"
                ) from exc
            self._initialize_postgresql()
            # Keep ordinary reads outside a transaction. Explicit multi-step
            # writes use ``transaction()`` below and still commit/rollback as
            # one unit.
            self.conn.autocommit = True
        else:
            raise RuntimeError(
                "Unsupported ONE_DATABASE_URL scheme; use sqlite:///... or postgresql://..."
            )

    def _initialize_sqlite(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.executescript(SCHEMA_MIGRATIONS)
            self._record_sqlite_migration(1)
            if not self._sqlite_migration_applied(2):
                self.conn.executescript(_family_migration())
                self._record_sqlite_migration(2)
            if not self._sqlite_migration_applied(3):
                tables = {
                    table: {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
                    for table in ("cameras", "room_maps", "calibrations")
                }
                additions = {
                    "cameras": [("resolution_width", "INTEGER"), ("resolution_height", "INTEGER"), ("metadata_json", "TEXT NOT NULL DEFAULT '{}'")],
                    "room_maps": [("source", "TEXT NOT NULL DEFAULT 'manual'"), ("approximate", "INTEGER NOT NULL DEFAULT 0"), ("localization_status", "TEXT NOT NULL DEFAULT 'unlocalized'"), ("metadata_json", "TEXT NOT NULL DEFAULT '{}'")],
                    "calibrations": [("resolution_width", "INTEGER"), ("resolution_height", "INTEGER"), ("camera_metadata_json", "TEXT NOT NULL DEFAULT '{}'") , ("metrics_json", "TEXT NOT NULL DEFAULT '{}'") , ("source", "TEXT NOT NULL DEFAULT 'manual'"), ("status", "TEXT NOT NULL DEFAULT 'active'"), ("invalidated_at", "TEXT"), ("invalidation_reason", "TEXT")],
                }
                for table, columns in additions.items():
                    for column, definition in columns:
                        if column not in tables[table]:
                            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                self._record_sqlite_migration(3)
            if not self._sqlite_migration_applied(4):
                self.conn.executescript(_identity_migration())
                self._record_sqlite_migration(4)
            if not self._sqlite_migration_applied(5):
                room_map_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(room_maps)").fetchall()}
                if "dimension" not in room_map_columns:
                    self.conn.execute("ALTER TABLE room_maps ADD COLUMN dimension TEXT NOT NULL DEFAULT '2d'")
                calibration_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(calibrations)").fetchall()}
                if "metrics_json" not in calibration_columns:
                    self.conn.execute("ALTER TABLE calibrations ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'")
                self.conn.execute(
                    """UPDATE room_maps
                       SET source='legacy-2d', approximate=1,
                           localization_status='rescan-required', dimension='2d'
                     WHERE source IN ('manual', 'camera-provisional', 'roomplan-normalized')"""
                )
                self._record_sqlite_migration(5)
            if not self._sqlite_migration_applied(6):
                job_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(camera_map_generation_jobs)").fetchall()}
                if "room_label" not in job_columns:
                    self.conn.execute("ALTER TABLE camera_map_generation_jobs ADD COLUMN room_label TEXT NOT NULL DEFAULT 'Room'")
                if "orientation" not in job_columns:
                    self.conn.execute("ALTER TABLE camera_map_generation_jobs ADD COLUMN orientation TEXT NOT NULL DEFAULT 'portrait'")
                self._record_sqlite_migration(6)
            if not self._sqlite_migration_applied(7):
                room_map_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(room_maps)").fetchall()}
                if "usdz_artifact_key" not in room_map_columns:
                    self.conn.execute("ALTER TABLE room_maps ADD COLUMN usdz_artifact_key TEXT")
                self._record_sqlite_migration(7)
            if not self._sqlite_migration_applied(8):
                home_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(homes)").fetchall()}
                if "care_setting" not in home_columns:
                    self.conn.execute("ALTER TABLE homes ADD COLUMN care_setting TEXT NOT NULL DEFAULT 'home'")
                if "support_focus" not in home_columns:
                    self.conn.execute("ALTER TABLE homes ADD COLUMN support_focus TEXT NOT NULL DEFAULT 'general'")
                self._record_sqlite_migration(8)
            # Keep the zero-setup SQLite adapter forward-compatible with a
            # database created before caregiver assignment was introduced.
            plan_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(medication_plans)").fetchall()}
            if "assigned_caregiver_id" not in plan_columns:
                self.conn.execute("ALTER TABLE medication_plans ADD COLUMN assigned_caregiver_id TEXT")
            self.conn.commit()

    def _sqlite_migration_applied(self, version: int) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version=?", (version,)
        ).fetchone() is not None

    def _record_sqlite_migration(self, version: int) -> None:
        if not self._sqlite_migration_applied(version):
            self.conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, now_iso()),
            )

    def _initialize_postgresql(self) -> None:
        with self._lock:
            try:
                # Serialize startup across API processes so two replicas do
                # not both observe a missing migration and race its INSERT.
                self.conn.execute("SELECT pg_advisory_xact_lock(hashtext('one.schema.migrations'))")
                self.conn.execute(SCHEMA_MIGRATIONS)
                applied = {
                    row["version"]
                    for row in self.conn.execute("SELECT version FROM schema_migrations").fetchall()
                }
                for version, script in _postgres_migrations():
                    if version in applied:
                        continue
                    for statement in _statements(script):
                        self.conn.execute(statement)
                    self.conn.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (%s, %s)",
                        (version, now_iso()),
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    @contextmanager
    def transaction(self):
        with self._lock:
            if self.backend == "postgresql":
                with self.conn.transaction():
                    yield _TransactionConnection(self)
                return
            try:
                yield _TransactionConnection(self)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _sql(self, sql: str) -> str:
        return _postgres_sql(sql) if self.backend == "postgresql" else sql

    def execute(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self.conn.execute(self._sql(sql), params)
            if self.backend == "sqlite":
                self.conn.commit()
            return cur

    def one(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            row = self.conn.execute(self._sql(sql), params).fetchone()
            return dict(row) if row else None

    def many(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.conn.execute(self._sql(sql), params).fetchall()]

    def health(self) -> dict:
        """Return a safe, non-secret connectivity report for the API health route."""
        with self._lock:
            try:
                self.conn.execute(self._sql("SELECT 1")).fetchone()
                return {"status": "ok", "backend": self.backend}
            except Exception as exc:
                return {"status": "error", "backend": self.backend, "error": type(exc).__name__}

    def export_home(self, home_id: str) -> dict:
        # Export user-visible records, including the minimal rights/audit
        # trail. Never export bearer-token hashes or one-time pairing hashes.
        tables = ["homes", "users", "memberships", "consents", "cameras", "rooms", "room_maps", "calibrations", "camera_map_generation_jobs", "objects", "observations", "events", "clips", "summaries", "family_invites", "medication_plans", "medication_check_ins", "audit_log", "deletion_requests"]
        result = {}
        for table in tables:
            if table == "homes": query, params = "SELECT * FROM homes WHERE id=?", (home_id,)
            elif table == "users": query, params = "SELECT u.* FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.home_id=?", (home_id,)
            elif table == "memberships": query, params = "SELECT * FROM memberships WHERE home_id=?", (home_id,)
            else: query, params = f"SELECT * FROM {table} WHERE home_id=?", (home_id,)
            result[table] = self.many(query, params)
        return result
