ALTER TABLE events
    ADD COLUMN IF NOT EXISTS event_group_key TEXT;

ALTER TABLE events
    ADD COLUMN IF NOT EXISTS observation_count INTEGER NOT NULL DEFAULT 1;

CREATE INDEX IF NOT EXISTS events_group_idx
    ON events(home_id, event_type, event_group_key, last_seen_at);
