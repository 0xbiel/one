CREATE TABLE IF NOT EXISTS camera_reconnect_tokens (
    token_hash TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at TEXT,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS camera_reconnect_tokens_camera_idx
    ON camera_reconnect_tokens(home_id, camera_id, revoked_at);
