"""Read-only, explicit simulation status; never a publisher connection or capture gate."""
import json
from datetime import datetime, timezone

SIMULATION_METADATA_KEY = 'fixture_simulation_v1'


def _time(value):
    try:
        stamp=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if stamp.tzinfo is None:return None
        return stamp.astimezone(timezone.utc)
    except (ValueError,TypeError):return None


def simulation_state(camera: dict, now: datetime | None = None) -> dict:
    """Only scoped, opted-in, bounded synthetic camera metadata can be online.

    The result is display provenance. It cannot satisfy consent, publisher session,
    placement, or real collection readiness. Future or expired fixtures are offline.
    """
    current=now or datetime.now(timezone.utc)
    current=current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current.astimezone(timezone.utc)
    metadata=camera.get('metadata')
    if not isinstance(metadata,dict):
        try:metadata=json.loads(camera.get('metadata_json') or '{}')
        except (ValueError,TypeError):metadata={}
    if not isinstance(metadata,dict):metadata={}
    marker=metadata.get(SIMULATION_METADATA_KEY)
    result={'data_origin':'recorded','status':None,'label':None,'last_simulated_at':None,'expires_at':None,'map_id':None,'care_recipient_id':None,'real_capture':False,'ready':False}
    if not metadata.get('simulated') or not isinstance(marker,dict) or marker.get('synthetic') is not True:
        return result
    if marker.get('home_id')!=camera.get('home_id') or marker.get('camera_id')!=camera.get('id') or not marker.get('map_id') or not marker.get('care_recipient_id'):
        return result
    start=_time(marker.get('started_at'));latest=_time(marker.get('last_simulated_at'));expiry=_time(marker.get('expires_at'))
    if not start or not latest or not expiry or not start<=latest<expiry or (expiry-start).total_seconds()>86400:
        return result
    online=bool(camera.get('enabled') and marker.get('enabled') is True and start<=latest<=current<expiry)
    result.update(data_origin='simulated',status='simulated_online' if online else 'simulated_offline',label='Simulated online' if online else 'Simulated offline',last_simulated_at=latest.isoformat(),expires_at=expiry.isoformat(),map_id=marker['map_id'],care_recipient_id=marker['care_recipient_id'])
    return result
