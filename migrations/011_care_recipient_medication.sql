ALTER TABLE consents
  ADD COLUMN IF NOT EXISTS care_recipient_id TEXT REFERENCES care_recipients(id) ON DELETE CASCADE;

ALTER TABLE medication_plans
  ADD COLUMN IF NOT EXISTS care_recipient_id TEXT REFERENCES care_recipients(id) ON DELETE CASCADE;

ALTER TABLE medication_check_ins
  ADD COLUMN IF NOT EXISTS care_recipient_id TEXT REFERENCES care_recipients(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS medication_plans_recipient_idx
  ON medication_plans(home_id, care_recipient_id, active);

CREATE INDEX IF NOT EXISTS medication_check_ins_recipient_idx
  ON medication_check_ins(home_id, care_recipient_id, scheduled_for);

CREATE INDEX IF NOT EXISTS consents_recipient_idx
  ON consents(home_id, care_recipient_id, purpose, granted_at);
