-- Camera/map provenance and calibration validity. Existing records remain usable.
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS resolution_width INTEGER;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS resolution_height INTEGER;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS metadata_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE room_maps ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'manual';
ALTER TABLE room_maps ADD COLUMN IF NOT EXISTS approximate INTEGER NOT NULL DEFAULT 0;
ALTER TABLE room_maps ADD COLUMN IF NOT EXISTS localization_status TEXT NOT NULL DEFAULT 'unlocalized';
ALTER TABLE room_maps ADD COLUMN IF NOT EXISTS metadata_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS resolution_width INTEGER;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS resolution_height INTEGER;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS camera_metadata_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'manual';
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS invalidated_at TEXT;
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS invalidation_reason TEXT;
