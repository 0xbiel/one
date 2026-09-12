-- Persistent, passwordless email identity for the local MVP.
-- Plaintext verification codes are never stored; the development outbox
-- returns one code once because no external mail provider is configured.
CREATE TABLE IF NOT EXISTS email_verifications (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  home_id TEXT NOT NULL REFERENCES homes(id) ON DELETE CASCADE,
  purpose TEXT NOT NULL CHECK (purpose IN ('create', 'login')),
  code_hash TEXT NOT NULL UNIQUE,
  expires_at TEXT NOT NULL,
  used_at TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS email_verifications_lookup_idx
  ON email_verifications(email, expires_at);

CREATE UNIQUE INDEX IF NOT EXISTS users_normalized_email_idx
  ON users(lower(trim(email)))
  WHERE email IS NOT NULL;
