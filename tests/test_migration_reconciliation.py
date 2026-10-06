from app.config import Settings
from app.db import Database, _postgres_migrations


def test_legacy_event_aggregation_version_16_upgrades_without_data_loss(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'legacy.db'}")
    db = Database(settings)
    db.conn.execute("INSERT INTO homes(id,name,created_at) VALUES ('test-home','Retained home','2026-10-01')")
    db.conn.execute("INSERT INTO events(id,home_id,event_type,status,evidence_json,first_seen_at,last_seen_at,expires_at,event_group_key,observation_count) VALUES ('test-event','test-home','activity','open','{}','2026-10-01','2026-10-01','2026-10-02','retained-group',5)")
    db.conn.commit()
    # Reconstruct the older local schema: 16 was event aggregation, and the
    # upstream tracking/care tables had not yet been added.
    db.conn.execute("PRAGMA foreign_keys=OFF")
    for table in ('location_clear_watermarks','location_points','safe_places','tracking_devices','care_entries'):
        db.conn.execute(f"DROP TABLE {table}")
    db.conn.execute("DELETE FROM schema_migrations WHERE version >= 16")
    db.conn.execute("INSERT INTO schema_migrations(version,applied_at) VALUES (16,'2026-10-01')")
    db.conn.commit()
    db.conn.close()
    upgraded = Database(settings)
    assert upgraded.one("SELECT name FROM homes WHERE id='test-home'")["name"] == "Retained home"
    event = upgraded.one("SELECT event_group_key,observation_count FROM events WHERE id='test-event'")
    assert event["event_group_key"] == "retained-group"
    assert event["observation_count"] == 5
    assert upgraded.one("SELECT name FROM sqlite_master WHERE type='table' AND name='tracking_devices'")
    assert 'street_name' in {row[1] for row in upgraded.conn.execute('PRAGMA table_info(location_points)')}
    assert upgraded.one("SELECT name FROM sqlite_master WHERE type='table' AND name='care_entries'")
    assert [row['version'] for row in upgraded.many('SELECT version FROM schema_migrations ORDER BY version')] == list(range(1,20))
    upgraded.conn.close()


def test_postgres_migration_numbers_remain_unique():
    versions = [version for version, _script in _postgres_migrations()]
    assert versions == list(range(1,20))
