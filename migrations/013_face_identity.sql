ALTER TABLE objects
  ADD COLUMN IF NOT EXISTS care_recipient_id TEXT REFERENCES care_recipients(id) ON DELETE SET NULL;

ALTER TABLE observations
  ADD COLUMN IF NOT EXISTS care_recipient_id TEXT REFERENCES care_recipients(id) ON DELETE SET NULL;
ALTER TABLE observations
  ADD COLUMN IF NOT EXISTS identity_confidence REAL;

ALTER TABLE events
  ADD COLUMN IF NOT EXISTS care_recipient_id TEXT REFERENCES care_recipients(id) ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS face_profiles (
    id TEXT PRIMARY KEY,
    home_id TEXT NOT NULL,
    care_recipient_id TEXT NOT NULL,
    template_artifact_key TEXT NOT NULL,
    model_version TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('ready','unavailable','revoked')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(home_id) REFERENCES homes(id) ON DELETE CASCADE,
    FOREIGN KEY(care_recipient_id) REFERENCES care_recipients(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS face_profiles_recipient_idx
    ON face_profiles(home_id, care_recipient_id);
CREATE INDEX IF NOT EXISTS face_profiles_home_status_idx
    ON face_profiles(home_id, status);
CREATE INDEX IF NOT EXISTS objects_recipient_idx
    ON objects(home_id, care_recipient_id);
CREATE INDEX IF NOT EXISTS observations_recipient_idx
    ON observations(home_id, care_recipient_id, observed_at);
