-- ONE family mode slice. Keep identifiers/timestamps portable between the
-- SQLite demo adapter and the PostgreSQL deployment migration runner.
CREATE TABLE IF NOT EXISTS family_invites (
  id TEXT PRIMARY KEY,
  home_id TEXT NOT NULL REFERENCES homes(id) ON DELETE CASCADE,
  invited_by TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  email TEXT,
  display_name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('resident', 'caregiver')),
  code_hash TEXT NOT NULL UNIQUE,
  expires_at TEXT NOT NULL,
  accepted_at TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medication_plans (
  id TEXT PRIMARY KEY,
  home_id TEXT NOT NULL REFERENCES homes(id) ON DELETE CASCADE,
  subject_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  dose TEXT NOT NULL,
  schedule TEXT NOT NULL,
  instructions TEXT NOT NULL DEFAULT '',
  active INTEGER NOT NULL DEFAULT 1,
  version INTEGER NOT NULL DEFAULT 1,
  created_by TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  assigned_caregiver_id TEXT REFERENCES users(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medication_check_ins (
  id TEXT PRIMARY KEY,
  home_id TEXT NOT NULL REFERENCES homes(id) ON DELETE CASCADE,
  plan_id TEXT NOT NULL REFERENCES medication_plans(id) ON DELETE CASCADE,
  subject_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  scheduled_for TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'taken', 'skipped', 'missed')),
  note TEXT NOT NULL DEFAULT '',
  marked_by TEXT REFERENCES users(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(plan_id, scheduled_for)
);
