CREATE TABLE IF NOT EXISTS tracking_devices (
    id TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    care_recipient_id TEXT NOT NULL,
    label TEXT NOT NULL,
    platform TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active','paused','revoked')),
    registered_by TEXT NOT NULL,
    last_seen_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(care_recipient_id) REFERENCES care_recipients(id) ON DELETE CASCADE,
    FOREIGN KEY(registered_by) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS tracking_devices_recipient_idx
    ON tracking_devices(home_id, care_recipient_id, status, updated_at);

CREATE TABLE IF NOT EXISTS location_points (
    id TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    care_recipient_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    client_sample_id TEXT NOT NULL,
    latitude REAL NOT NULL CHECK(latitude >= -90 AND latitude <= 90),
    longitude REAL NOT NULL CHECK(longitude >= -180 AND longitude <= 180),
    accuracy_m REAL,
    speed_mps REAL,
    bearing_deg REAL,
    battery_percent INTEGER,
    captured_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(care_recipient_id) REFERENCES care_recipients(id) ON DELETE CASCADE,
    FOREIGN KEY(device_id) REFERENCES tracking_devices(id) ON DELETE CASCADE,
    UNIQUE(device_id, client_sample_id)
);

CREATE INDEX IF NOT EXISTS location_points_recipient_time_idx
    ON location_points(home_id, care_recipient_id, captured_at);

CREATE TABLE IF NOT EXISTS safe_places (
    id TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    care_recipient_id TEXT NOT NULL,
    name TEXT NOT NULL,
    latitude REAL NOT NULL CHECK(latitude >= -90 AND latitude <= 90),
    longitude REAL NOT NULL CHECK(longitude >= -180 AND longitude <= 180),
    radius_m REAL NOT NULL CHECK(radius_m >= 25 AND radius_m <= 5000),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(care_recipient_id) REFERENCES care_recipients(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS safe_places_recipient_idx
    ON safe_places(home_id, care_recipient_id, created_at);
