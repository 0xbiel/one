import json

import pytest

from app.config import Settings
from app.db import Database
from app.privacy_data import build_privacy_export, validate_deletion_confirmation


@pytest.fixture
def disposable_db():
    db = Database(Settings(database_url="sqlite:///:memory:", env="test"))
    for home in ("a", "b"):
        db.execute("INSERT INTO homes(id,name,created_at) VALUES (?,?,?)", (home, "Disposable " + home, "2026-10-01T00:00:00Z"))
        db.execute("INSERT INTO users VALUES (?,?,?,?)", ("u" + home, home, home + "@example.invalid", "now"))
        db.execute("INSERT INTO memberships VALUES (?,?,?)", (home, "u" + home, "admin"))
        db.execute("INSERT INTO care_recipients VALUES (?,?,?,?,?,?)", ("r" + home, home, "Recipient " + home, "relative", "Room", "now"))
    yield db
    db.conn.close()


def test_reviewed_export_excludes_credentials_and_biometrics(disposable_db):
    db = disposable_db
    db.execute("INSERT INTO family_invites VALUES (?,?,?,?,?,?,?,?,?,?)", ("invite", "a", "ua", None, "Caregiver", "caregiver", "invite-secret", "later", None, "now"))
    db.execute("INSERT INTO face_profiles VALUES (?,?,?,?,?,?,?,?,?)", ("face", "a", "ra", "biometric-private-key", "model", 3, "ready", "now", "now"))
    db.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", ("session-secret", "ua", "a", "later", "now"))
    db.execute("INSERT INTO cameras(id,home_id,name,created_at,metadata_json) VALUES (?,?,?,?,?)", ("camera", "a", "Hall", "now", json.dumps({"provenance": {"synthetic": True, "source": "test"}, "nested": [{"access_token": "nested-secret", "embedding": [1, 2], "transcript": "private speech", "safe": "retained"}], "visual_landmarks": {"descriptors": [1, 2]}})))
    db.execute("INSERT INTO home_runtime VALUES (?,?,?)", ("a", 1, "now"))
    exported = build_privacy_export(db, "a", exported_at="test-time")
    serialized = json.dumps(exported["data"])
    for secret in ("invite-secret", "biometric-private-key", "session-secret", "nested-secret", "private speech", "descriptors"):
        assert secret not in serialized
    assert exported["data"]["face_profiles"][0]["status"] == "ready"
    assert exported["data"]["care_recipients"][0]["id"] == "ra"
    assert exported["data"]["home_runtime"][0]["paused"] == 1
    metadata = json.loads(exported["data"]["cameras"][0]["metadata_json"])
    assert metadata == {"provenance": {"synthetic": True, "source": "test"}, "nested": [{"safe": "retained"}]}
    assert exported["scope"]["includes_media_bytes"] is False
    assert exported["exclusions"]["retention"]
    assert exported["exported_at"] == "test-time"


def test_every_export_table_is_home_scoped(disposable_db):
    data = disposable_db.export_home("a")
    assert data["homes"][0]["id"] == "a"
    assert data["users"][0]["id"] == "ua"
    assert data["memberships"][0]["user_id"] == "ua"
    for table, records in data.items():
        if table not in {"homes", "users"}:
            assert all(row["home_id"] == "a" for row in records)
    assert disposable_db.export_home("missing")["users"] == []


def test_new_columns_and_invalid_json_fail_closed(disposable_db):
    db = disposable_db
    db.execute("ALTER TABLE cameras ADD COLUMN unexpected_secret TEXT")
    db.execute("INSERT INTO cameras(id,home_id,name,created_at,metadata_json,unexpected_secret) VALUES (?,?,?,?,?,?)", ("c", "a", "Camera", "now", "malformed-secret", "future-secret"))
    data = db.export_home("a")
    assert data["cameras"][0]["metadata_json"] == "{}"
    assert "secret" not in json.dumps(data)


@pytest.mark.parametrize("confirmed,target", [(False, "a"), (None, "a"), (1, "a"), (True, None), (True, "b"), (True, " a ")])
def test_confirmation_rejects_absent_or_wrong_target(confirmed, target):
    with pytest.raises(ValueError):
        validate_deletion_confirmation("a", confirmed, target)


def test_disposable_cascade_preserves_other_home(disposable_db):
    db = disposable_db
    validate_deletion_confirmation("a", True, "a")
    with db.transaction() as conn:
        conn.execute("DELETE FROM homes WHERE id=?", ("a",))
    assert db.one("SELECT id FROM homes WHERE id=?", ("a",)) is None
    assert db.one("SELECT id FROM care_recipients WHERE id=?", ("ra",)) is None
    assert db.one("SELECT id FROM care_recipients WHERE id=?", ("rb",))["id"] == "rb"
    assert db.one("SELECT id FROM homes WHERE id=?", ("b",))["id"] == "b"
