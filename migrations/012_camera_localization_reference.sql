CREATE TABLE IF NOT EXISTS camera_localization_references (
    home_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    map_id TEXT NOT NULL,
    x REAL NOT NULL,
    z REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual-floor-reference',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(home_id, camera_id, map_id),
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE,
    FOREIGN KEY(map_id) REFERENCES room_maps(id) ON DELETE CASCADE
);
