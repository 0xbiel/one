CREATE TABLE IF NOT EXISTS care_entries (
    id TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    care_recipient_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('note','appointment')),
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    location TEXT NOT NULL DEFAULT '',
    starts_at TEXT,
    ends_at TEXT,
    timezone_name TEXT,
    reminder_minutes INTEGER,
    version INTEGER NOT NULL DEFAULT 1,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(care_recipient_id) REFERENCES care_recipients(id) ON DELETE CASCADE,
    FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS care_entries_recipient_idx ON care_entries(home_id,care_recipient_id,starts_at,created_at);
