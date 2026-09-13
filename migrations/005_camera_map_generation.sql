-- Persistent camera map-generation jobs and explicit map provenance.
ALTER TABLE room_maps ADD COLUMN IF NOT EXISTS dimension TEXT NOT NULL DEFAULT '2d';
ALTER TABLE calibrations ADD COLUMN IF NOT EXISTS metrics_json TEXT NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS camera_map_generation_jobs (
  id TEXT PRIMARY KEY,
  home_id TEXT NOT NULL REFERENCES homes(id) ON DELETE CASCADE,
  camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
  room_id TEXT,
  room_label TEXT NOT NULL DEFAULT 'Room',
  orientation TEXT NOT NULL DEFAULT 'portrait',
  status TEXT NOT NULL CHECK (status IN ('collecting', 'processing', 'ready', 'needs_rescan', 'unavailable', 'failed')),
  frame_count INTEGER NOT NULL DEFAULT 0,
  resolution_width INTEGER NOT NULL,
  resolution_height INTEGER NOT NULL,
  map_id TEXT REFERENCES room_maps(id) ON DELETE SET NULL,
  error_code TEXT,
  error_message TEXT,
  metrics_json TEXT NOT NULL DEFAULT '{}',
  model_version TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE INDEX IF NOT EXISTS camera_map_generation_jobs_camera_idx
  ON camera_map_generation_jobs(home_id, camera_id, updated_at);

-- Maps created by the old camera-zone and generic RoomPlan paths remain
-- readable, but are explicitly legacy and must be rescanned.
UPDATE room_maps
   SET source='legacy-2d', approximate=1,
       localization_status='rescan-required', dimension='2d'
 WHERE source IN ('manual', 'camera-provisional', 'roomplan-normalized');
