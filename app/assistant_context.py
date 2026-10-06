"""Read-only, evidence-linked caregiver context; no provider or media access."""
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from zoneinfo import ZoneInfo

from .analytics import _days, _day, bounded_rows


def _metadata(value):
    try:
        result = json.loads(value or '{}')
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        return {}


def build_assistant_context(db, home_id: str, *, care_recipient_id=None,
                            subject_user_id=None, window_days=30, now=None,
                            timezone_name='UTC', max_rows=100, include_medication=False, max_bytes=16384):
    """Caller authorizes scope/consent; this helper only reads bounded metadata.

    Observation counts measure retained records, never motion or camera uptime.
    Stored explanations and arbitrary evidence payloads are deliberately excluded.
    Medication metadata requires explicit caller authorization via include_medication.
    """
    now = ((now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now) if now is not None else datetime.now(timezone.utc)).astimezone(ZoneInfo(timezone_name))
    window_days = max(7, min(int(window_days), 90))
    max_rows = max(1, min(int(max_rows), 250))
    start = (now - timedelta(days=window_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    clause = ' AND care_recipient_id=?' if care_recipient_id else ''
    params = (home_id, care_recipient_id) if care_recipient_id else (home_id,)
    observations, inferences, provenance = [], [], []
    truncated = False
    tables = [('observations', 'observed_at', 'observation'), ('events', 'last_seen_at', 'event'), ('summaries', 'created_at', 'daily_check_in')]
    if include_medication:
        tables.append(('medication_check_ins', 'scheduled_for', 'medication_check_in'))
    included = []
    observation_truncated = False
    for table, stamp, kind in tables:
        target_clause, target_params = clause, params
        if subject_user_id and not care_recipient_id:
            if table in ('observations', 'events'):
                # Camera rows have no authenticated subject_user_id attribution.
                continue
            target_clause = ' AND care_recipient_id IS NULL AND subject_user_id=?'
            target_params = (home_id, subject_user_id)
        included.append(kind + "_metadata" if kind != "observation" else "observation_metadata")
        try:
            rows = db.many(f'SELECT * FROM {table} WHERE home_id=?{target_clause} ORDER BY {stamp} DESC LIMIT 2001', target_params)
        except sqlite3.OperationalError as exc:
            # Older isolated fixtures may omit additive data tables.
            if 'no such table' not in str(exc):
                raise
            rows = []
        read_truncated = len(rows) > 2000
        rows = bounded_rows(rows[:2000], stamp, start.isoformat(), now)
        if table == 'events':
            # Daily check-in events mirror the canonical summaries written by
            # the same check-in request; they are not independent observations.
            rows = [row for row in rows if row.get('event_type') != 'daily_check_in']
        table_truncated = read_truncated or len(rows) > max_rows
        truncated = truncated or table_truncated
        if kind == 'observation':
            observation_truncated = table_truncated
        for row in rows[:max_rows]:
            item = {'source_id': row['id'], 'source_type': kind, 'observed_at': row[stamp], 'camera_id': row.get('camera_id'), 'room_id': None}
            if row.get('camera_id'):
                camera = db.one('SELECT room_id FROM cameras WHERE home_id=? AND id=?', (home_id, row['camera_id']))
                item['room_id'] = camera.get('room_id') if camera else None
            if kind == 'observation':
                observations.append(item)
            else:
                item.update({key: row.get(key) for key in ('status', 'event_type', 'plan_id') if key in row})
                inferences.append(item)
            metadata = _metadata(row.get('evidence_json'))
            if metadata.get('synthetic') is True or metadata.get('provenance') == 'synthetic' or str(row.get('detector_version') or row.get('model_version') or '').startswith('synthetic') or str(row.get('event_group_key') or '').startswith('synthetic-'):
                provenance.append({'source_id': row['id'], 'synthetic': True, 'claim': 'Fictional generated record; not a real-person observation.'})
    days = _days(window_days, now)
    observed = {_day(row['observed_at'], timezone_name) for row in observations}
    missing = [day for day in days if day not in observed]
    cameras = db.many('SELECT id,room_id,enabled FROM cameras WHERE home_id=? ORDER BY id LIMIT 250', (home_id,)) if hasattr(db, 'one') else []
    freshness = [{'camera_id': camera['id'], 'room_id': camera.get('room_id'), 'enabled': bool(camera.get('enabled')), 'last_scoped_observation_at': next((row['observed_at'] for row in observations if row['camera_id'] == camera['id']), None)} for camera in cameras]
    from .context_budget import bound_context_packet
    packet = {
        'scope': {'home_id': home_id, 'care_recipient_id': care_recipient_id, 'subject_user_id': subject_user_id},
        'window': {'start_at': start.isoformat(), 'end_at': now.isoformat(), 'timezone': timezone_name, 'window_days': window_days},
        'observations': observations, 'inferences': inferences,
        'missing_coverage': {'status': 'partial' if observations else 'unobserved', 'observed_days': len(observed), 'unobserved_days': missing, 'truncated': truncated, 'truncation_scope': 'context_records', 'observation_records_truncated': observation_truncated, 'cameras': freshness, 'limitations': ['No historical camera uptime is available.', 'Zero observations do not establish inactivity.', 'Activity trends remain unknown because observation coverage is not continuous.', 'Daily check-in summaries are canonical records; matching event mirrors are omitted.']},
        'synthetic_provenance': provenance,
        'privacy_boundary': {'includes': included, 'excludes': ['raw_frames', 'event_snapshot_bytes', 'transcripts', 'face_templates', 'biometric_material']},
    }

    return bound_context_packet(packet, max_bytes=max_bytes)
