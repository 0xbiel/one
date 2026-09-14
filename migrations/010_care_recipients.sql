CREATE TABLE IF NOT EXISTS care_recipients (
    id TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    relationship TEXT,
    room_label TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS care_recipients_home_idx
    ON care_recipients(home_id, created_at);
