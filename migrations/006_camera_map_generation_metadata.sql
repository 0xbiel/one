-- Persist the non-sensitive sweep context needed by the host geometry service.
ALTER TABLE camera_map_generation_jobs ADD COLUMN IF NOT EXISTS room_label TEXT NOT NULL DEFAULT 'Room';
ALTER TABLE camera_map_generation_jobs ADD COLUMN IF NOT EXISTS orientation TEXT NOT NULL DEFAULT 'portrait';
