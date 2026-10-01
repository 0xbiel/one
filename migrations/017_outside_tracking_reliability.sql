ALTER TABLE location_points ADD COLUMN street_name TEXT;
ALTER TABLE location_points ADD COLUMN dwell_duration_millis INTEGER NOT NULL DEFAULT 0;
ALTER TABLE safe_places ADD COLUMN kind TEXT NOT NULL DEFAULT 'safe';
ALTER TABLE safe_places ADD COLUMN revision INTEGER NOT NULL DEFAULT 1;
CREATE UNIQUE INDEX IF NOT EXISTS safe_places_one_home_idx ON safe_places(care_recipient_id) WHERE kind = 'home';
CREATE INDEX IF NOT EXISTS location_points_retention_idx ON location_points(captured_at);
CREATE TABLE IF NOT EXISTS location_clear_watermarks (
    home_id TEXT NOT NULL,
    care_recipient_id TEXT NOT NULL PRIMARY KEY,
    cleared_at TEXT NOT NULL,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(care_recipient_id) REFERENCES care_recipients(id) ON DELETE CASCADE
);
