ALTER TABLE summaries
    ADD COLUMN IF NOT EXISTS care_recipient_id TEXT REFERENCES care_recipients(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS summaries_recipient_idx
    ON summaries(home_id, care_recipient_id, created_at);
