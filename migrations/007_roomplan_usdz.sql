-- Persisted USDZ attachments for validated native RoomPlan maps.
ALTER TABLE room_maps ADD COLUMN IF NOT EXISTS usdz_artifact_key TEXT;
