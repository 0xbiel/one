import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import Settings


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS homes (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, display_name TEXT NOT NULL, email TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memberships (home_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('resident','caregiver','admin','publisher')), PRIMARY KEY(home_id,user_id), FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS home_runtime (home_id TEXT PRIMARY KEY, paused INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, home_id TEXT NOT NULL, expires_at TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS pairing_codes (code_hash TEXT PRIMARY KEY, home_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL, expires_at TEXT NOT NULL, used_at TEXT, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS consents (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, subject_user_id TEXT NOT NULL, purpose TEXT NOT NULL, policy_version TEXT NOT NULL, granted_at TEXT NOT NULL, revoked_at TEXT, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS cameras (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, name TEXT NOT NULL, room_id TEXT, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS rooms (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS room_maps (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, room_id TEXT, revision INTEGER NOT NULL, coordinate_frame TEXT NOT NULL, artifact_key TEXT, map_json TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS calibrations (id TEXT PRIMARY KEY, home_id TEXT NOT NULL, camera_id TEXT NOT NULL, map_id TEXT NOT NULL, intrinsics_json TEXT NOT NULL, extrinsics_json TEXT NOT NULL, accuracy_m REAL, created_at TEXT NOT NULL, FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE);
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


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        path = settings.sqlite_path
        if path is None:
            raise RuntimeError("Production PostgreSQL is configured via ONE_DATABASE_URL; install psycopg and add the adapter before deployment")
        self.path = path
        if path != ":memory:":
            import os
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            # Keep the zero-setup SQLite adapter forward-compatible with a
            # database created before caregiver assignment was introduced.
            plan_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(medication_plans)").fetchall()}
            if "assigned_caregiver_id" not in plan_columns:
                self.conn.execute("ALTER TABLE medication_plans ADD COLUMN assigned_caregiver_id TEXT")
            self.conn.commit()

    @contextmanager
    def transaction(self):
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def one(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            row = self.conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def many(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def export_home(self, home_id: str) -> dict:
        # Export user-visible records, including the minimal rights/audit
        # trail. Never export bearer-token hashes or one-time pairing hashes.
        tables = ["homes", "users", "memberships", "consents", "cameras", "rooms", "room_maps", "calibrations", "objects", "observations", "events", "clips", "summaries", "family_invites", "medication_plans", "medication_check_ins", "audit_log", "deletion_requests"]
        result = {}
        for table in tables:
            if table == "homes": query, params = "SELECT * FROM homes WHERE id=?", (home_id,)
            elif table == "users": query, params = "SELECT u.* FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.home_id=?", (home_id,)
            elif table == "memberships": query, params = "SELECT * FROM memberships WHERE home_id=?", (home_id,)
            else: query, params = f"SELECT * FROM {table} WHERE home_id=?", (home_id,)
            result[table] = self.many(query, params)
        return result
