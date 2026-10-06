"""Read-only collection prerequisites; saved sessions and observations are distinct."""
import json
from datetime import datetime, timezone

STALE_AFTER_SECONDS = 300


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.astimezone(timezone.utc) if result.tzinfo else result.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _metadata(value):
    try:
        result = json.loads(value or '{}')
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        return {}


def collection_readiness(db, home_id: str, now: datetime | None = None) -> dict:
    """Report prerequisites without enabling capture or changing any state.

    A publisher session is only a connection proxy. Historical synthetic data
    is included in observation freshness but never establishes connectivity.
    """
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    runtime = db.one('SELECT paused FROM home_runtime WHERE home_id=?', (home_id,))
    paused = bool(runtime and runtime['paused'])
    consents = db.many("SELECT granted_at,revoked_at FROM consents WHERE home_id=? AND purpose='video_capture'", (home_id,))
    eligible = [row for row in consents if _time(row['granted_at']) and _time(row['granted_at']) <= current]
    latest = max(eligible, key=lambda row: _time(row['granted_at']), default=None)
    consent = bool(latest and latest['revoked_at'] is None)
    global_blockers = ([] if consent else ['video_capture_consent_required']) + (['home_paused'] if paused else [])
    cameras = []
    for camera in db.many('SELECT * FROM cameras WHERE home_id=? ORDER BY created_at,id', (home_id,)):
        cid = camera['id']
        blockers = list(global_blockers)
        if not camera['enabled']:
            blockers.append('camera_disabled')
        sessions = db.many('SELECT created_at,expires_at FROM sessions WHERE home_id=? AND user_id=?', (home_id,cid))
        session = max(sessions, key=lambda row: _time(row['created_at']) or datetime.min.replace(tzinfo=timezone.utc), default=None)
        expiry = _time(session['expires_at']) if session else None
        connected = bool(camera['enabled'] and expiry and expiry > current)
        if not connected:
            blockers.append('camera_offline')
        room = db.one('SELECT id,name FROM rooms WHERE home_id=? AND id=?', (home_id,camera.get('room_id')))
        placements = db.many("SELECT c.id,c.map_id,c.source,c.created_at FROM calibrations c JOIN room_maps m ON m.id=c.map_id AND m.home_id=c.home_id WHERE c.home_id=? AND c.camera_id=? AND c.status='active' AND c.invalidated_at IS NULL", (home_id,cid))
        placement = max(placements, key=lambda row: _time(row['created_at']) or datetime.min.replace(tzinfo=timezone.utc), default=None)
        if not room:
            blockers.append('room_required')
        if not placement:
            blockers.append('placement_required')
        observations = db.many('SELECT observed_at,detector_version FROM observations WHERE home_id=? AND camera_id=?', (home_id,cid))
        valid = [(row, _time(row['observed_at'])) for row in observations]
        valid = [(row, stamp) for row,stamp in valid if stamp and stamp <= current]
        observation, stamp = max(valid, key=lambda item:item[1], default=(None,None))
        age = (current-stamp).total_seconds() if stamp else None
        freshness = 'unknown' if age is None else ('fresh' if age <= STALE_AFTER_SECONDS else 'stale')
        if freshness != 'fresh':
            blockers.append('observations_missing' if freshness == 'unknown' else 'observations_stale')
        metadata = _metadata(camera.get('metadata_json'))
        synthetic = bool(observation and 'synthetic' in (observation.get('detector_version') or '').lower())
        cameras.append({'id':cid,'name':camera['name'],'enabled':bool(camera['enabled']),
            'status':'ready' if not blockers else ('offline' if not connected else 'blocked'),
            'connection':{'status':'online' if connected else 'offline','basis':'session_validity','last_session_at':session['created_at'] if session else None},
            'placement':{'room_id':room['id'] if room else None,'room_name':room['name'] if room else None,'map_id':placement['map_id'] if placement else None,'calibration_id':placement['id'] if placement else None,'source':placement['source'] if placement else None,'status':'positioned' if placement else ('room_only' if room else 'unplaced')},
            'freshness':{'status':freshness,'last_observed_at':stamp.isoformat() if stamp else None,'age_seconds':age,'basis':'derived_observation','stale_after_seconds':STALE_AFTER_SECONDS,'synthetic':synthetic,'label':'Simulated observations' if synthetic else 'Recorded observations'},
            'provenance':{'synthetic':bool(metadata.get('synthetic') or metadata.get('simulated') or metadata.get('demo')),'metadata':{key:metadata[key] for key in ('synthetic','simulated','fixture_schema','provenance','seed_version') if key in metadata}},
            'blockers':blockers})
    blockers = global_blockers + ([] if cameras else ['camera_required'])
    return {'home_id':home_id,'evaluated_at':current.isoformat(),'paused':paused,'video_capture_consent':consent,'ready':bool(cameras) and any(not camera['blockers'] for camera in cameras),'blockers':blockers,'cameras':cameras,'retention':{'observations_days':30,'live_frames':'processed_in_memory'},'limitations':['A valid publisher session does not prove frame arrival. Observation freshness describes derived records, not camera connectivity.','Placement reflects saved active calibration; automatic visual localization availability is a separate prerequisite.']}
