CREATE TABLE IF NOT EXISTS event_snapshots (
    event_id TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    object_key TEXT NOT NULL,
    content_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS event_snapshots_home_idx
    ON event_snapshots(home_id, expires_at);
