"""A read-only caregiver day story grounded in scoped retained metadata."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

from .analytics import bounded_rows, _parse_timestamp
from .context_budget import bound_context_packet

READ_CAP = 2000
SOURCE_CAP = 3


def _synthetic(row):
    version = str(row.get('detector_version') or row.get('model_version') or row.get('event_group_key') or '')
    try:
        metadata = json.loads(row.get('evidence_json') or '{}')
    except (ValueError, TypeError):
        metadata = {}
    return version.startswith('synthetic') or (isinstance(metadata, dict) and (metadata.get('synthetic') is True or metadata.get('provenance') == 'synthetic'))


def _sources(rows, timestamp_key, source_type):
    return [{'source_id': row['id'], 'source_type': source_type, 'observed_at': row[timestamp_key], 'synthetic': _synthetic(row)} for row in rows[:SOURCE_CAP]]


def build_day_story(db, home_id, care_recipient_id, timezone_name='UTC', now=None, *, include_medication=False, reminders=None):
    """Caller authorizes analytics consent. No collection or provider calls.

    Observed-hour/baseline eligibility is deliberately unavailable: the current
    schema has no historical processing coverage. Review items are stored review
    statuses, never new routine-change alerts or medical assessments.
    """
    recipient = db.one('SELECT id,display_name,relationship FROM care_recipients WHERE home_id=? AND id=?', (home_id, care_recipient_id))
    if not recipient:
        raise ValueError('Care recipient is not in the selected household')
    zone = ZoneInfo(timezone_name)
    current = now or datetime.now(timezone.utc)
    current = (current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current).astimezone(zone)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    scope = {'home_id': home_id, 'care_recipient_id': care_recipient_id}
    window = {'start_at': start.isoformat(), 'end_at': current.isoformat(), 'timezone': timezone_name, 'partial_day': True}
    grouped, caps = {}, {}
    for table, timestamp_key in [('observations', 'observed_at'), ('summaries', 'created_at'), ('events', 'last_seen_at')]:
        rows = db.many(f'SELECT * FROM {table} WHERE home_id=? AND care_recipient_id=? ORDER BY {timestamp_key} DESC,id DESC LIMIT {READ_CAP + 1}', (home_id, care_recipient_id))
        caps[table] = len(rows) > READ_CAP
        grouped[table] = bounded_rows(rows[:READ_CAP], timestamp_key, start.isoformat(), current)
    observations, summaries, events = (grouped[table] for table in ('observations', 'summaries', 'events'))
    # Canonical check-ins are summaries; observation/check-in mirror events are
    # not counted again as independent observations or incidents.
    reviews = [row for row in events if row.get('event_type') not in {'object_observed', 'daily_check_in'} and row.get('status') in {'needs_review', 'attention', 'open'}]
    reviews = reviews[:3]
    latest_checks = summaries[:3]
    samples = _sources(observations, 'observed_at', 'observation')
    check_sources = _sources(latest_checks, 'created_at', 'daily_check_in')
    review_sources = _sources(reviews, 'last_seen_at', 'event')
    synthetic = str(recipient.get('relationship') or '').startswith('Fictional synthetic') or any(_synthetic(row) for rows in grouped.values() for row in rows)
    count = len(observations)
    count_phrase = f'at least {count}' if caps['observations'] else str(count)
    observation_label = 'simulated observation' if observations and all(_synthetic(row) for row in observations) else 'observation'
    name = str(recipient['display_name'])[:200]
    headline = f'{name}’s day so far' if observations or summaries else 'A check-in would help today'
    check_status = latest_checks[0].get('status') if latest_checks else None
    check_message = {
        'ok': 'Today’s check-in is marked okay.',
        'attention': 'Today’s check-in needs caregiver attention.',
        'unknown': 'Today’s check-in needs more information.',
    }.get(check_status, 'A daily check-in is still needed.' if not latest_checks else 'Today’s check-in is ready to review.')
    summary = check_message
    if not observations:
        summary += ' There are no camera updates for today; this does not establish inactivity.'
    if reviews:
        summary += ' Review the safety item below.' if len(reviews) == 1 else f' Review the {len(reviews)} safety items below.'
    elif observations:
        summary += ' No safety items are awaiting review in today’s available history.'
    attention_items = []
    for row in reviews:
        fall = row.get('event_type') == 'fall_suspected'
        attention_items.append({'id': row['id'], 'title': 'Check a possible fall' if fall else 'Review a care alert',
                                'detail': 'Check with the person or caregiver and review the saved signal. A signal does not confirm an incident.',
                                'source_ids': [row['id']], 'observed_at': row['last_seen_at'], 'severity': 'review', 'synthetic': _synthetic(row)})
    coverage = {'simple_label': 'Camera updates cover only parts of the day.' if observations else 'There are no camera updates for today.',
                'observed_hours': None, 'continuous_coverage_known': False, 'observation_count': count,
                'count_is_lower_bound': caps['observations'], 'last_observed_at': observations[0]['observed_at'] if observations else None,
                'missing_means': 'Unknown evidence coverage; not proof of inactivity.'}
    routine = {'status': 'unknown', 'reason': 'There is not enough comparable coverage to judge a change in routine.',
               'baseline_eligible': False, 'thresholds': {'status': 'not_validated', 'minimum_comparable_days': 7, 'minimum_observed_fraction': 0.8}}
    facts = []
    if latest_checks:
        latest = latest_checks[0]
        facts.append({'text': check_message, 'source_ids': [latest['id']], 'times': [latest['created_at']], 'synthetic': _synthetic(latest)})
    cameras = db.many('SELECT id,room_id,enabled FROM cameras WHERE home_id=? ORDER BY id LIMIT 250', (home_id,))
    room_lookup = {row['id']: row.get('room_id') for row in cameras}
    room_counts = Counter(room_lookup.get(row.get('camera_id')) for row in observations if room_lookup.get(row.get('camera_id')))
    rooms = db.many('SELECT id,name FROM rooms WHERE home_id=? ORDER BY id LIMIT 250', (home_id,))
    room_names = {row['id']: str(row['name'])[:120] for row in rooms}
    if observations:
        latest = observations[0]
        room_name = room_names.get(room_lookup.get(latest.get('camera_id')))
        clock = _parse_timestamp(latest['observed_at']).astimezone(zone).strftime('%H:%M')
        update = f'Latest camera update: {room_name} at {clock}.' if room_name else f'Latest camera update: {clock}.'
        facts.append({'text': update, 'source_ids': [latest['id']], 'times': [latest['observed_at']], 'synthetic': _synthetic(latest)})
    care_plan, medication_sources = _daily_care_plan(db, home_id, care_recipient_id, start, current, include_medication, synthetic, reminders)
    if include_medication and care_plan['scheduled_count']:
        due = care_plan['pending_count'] + care_plan['missed_count']
        if due:
            summary += f' {due} care-plan reminder' + (' needs' if due == 1 else 's need') + ' an acknowledgement.'
            due_sources = [row for row in medication_sources if row['status'] in {'pending', 'missed'} and _parse_timestamp(row['scheduled_for']) <= current]
            attention_items.append({'id': 'care-plan-reminders', 'title': 'Check today’s care-plan reminders',
                'detail': 'Ask the caregiver to acknowledge or update the outstanding reminders.',
                'source_ids': [row['source_id'] for row in due_sources[:SOURCE_CAP]],
                'observed_at': due_sources[0]['observed_at'] if due_sources else None, 'severity': 'reminder', 'synthetic': synthetic})
        elif care_plan['upcoming_count']:
            summary += ' A care-plan reminder is scheduled for later today.'
        elif care_plan['marked_taken_count']:
            summary += ' Today’s care-plan acknowledgement is saved.'
        if medication_sources:
            facts.append({'text': ("One reminder is marked taken." if care_plan["marked_taken_count"] == 1 else f"{care_plan['marked_taken_count']} reminders are marked taken.") if care_plan['marked_taken_count'] else (f"{due} reminder" + (" is" if due == 1 else "s are") + " awaiting acknowledgement." if due else "A care-plan reminder is scheduled for later today."),
                'source_ids': [row['source_id'] for row in medication_sources[:SOURCE_CAP]],
                'times': [row['observed_at'] for row in medication_sources[:SOURCE_CAP]], 'synthetic': synthetic})
    limitations = ['Missing observations do not establish inactivity.', 'Camera room attribution uses current saved placement; sampled sightings do not establish room visits or dwell time.', 'Observed hours and comparable baseline coverage are unavailable, so routine changes remain unknown.', 'Recorded care signals and check-ins are observations for caregiver review, not diagnoses.']
    if include_medication:
        limitations.append(care_plan['disclaimer'])
    if synthetic:
        limitations.insert(0, 'This story includes fictional synthetic records; they do not establish real recognition or live camera connectivity.')
    evidence_packet = bound_context_packet({'scope': scope, 'window': window,
        'privacy_boundary': {'excludes': ['raw_detections', 'raw_frames', 'transcripts', 'biometric_material', *(['medication_records'] if not include_medication else [])]},
        'missing_coverage': coverage,
        'aggregates': {'observation_count': count, 'check_in_count': len(summaries), 'review_item_count': len(reviews), 'count_is_lower_bound': caps['observations'], 'room_attribution': 'current_camera_assignment_not_visits_or_dwell'},
        'source_index': samples + check_sources + review_sources + medication_sources,
        'care_plan': care_plan,
        'care_plan_examples': medication_sources,
        'observation_sources': samples,
        'canonical_check_ins': [dict(source, status=row.get('status')) for source, row in zip(check_sources, latest_checks)],
        'review_events': [dict(source, event_type=row.get('event_type'), status=row.get('status')) for source, row in zip(review_sources, reviews)],
        'synthetic': synthetic}, max_bytes=16384)
    return {'recipient': {'id': recipient['id'], 'display_name': name}, 'window': window, 'headline': headline, 'summary': summary,
            'attention_items': attention_items, 'meaningful_changes': [], 'routine': routine, 'coverage': coverage,
            'facts': facts, 'care_plan': care_plan, 'evidence_packet': evidence_packet, 'synthetic': synthetic, 'limitations': limitations,
            'details': {'scope': scope, 'read_caps': caps, 'source_id_limit': SOURCE_CAP, 'current_room_sighting_counts': dict(room_counts),
                        'room_semantics': 'Sampled records attributed to currently assigned camera rooms; no continuous 3D trajectory, visits, or dwell is established.'}}


def evaluate_routine_change(current, baseline):
    """Future scaffold for explicitly validated processing-coverage aggregates.

    Runtime day stories do not invoke this: current schema cannot supply these
    denominators. Inputs must be validated, comparable, source-linked aggregates,
    not hours reconstructed from sightings. Thresholds are conservative proposal
    values for disposable testing, not medically validated criteria.
    """
    import math
    thresholds = {'status': 'not_validated', 'minimum_comparable_days': 7,
                  'minimum_observed_fraction': 0.8, 'minimum_relative_rate_change': 0.3}
    unknown = {'status': 'unknown', 'baseline_eligible': False,
               'reason': 'Validated comparable processing coverage is required.', 'thresholds': thresholds}
    for sample in (current, baseline):
        if not isinstance(sample, dict) or sample.get('coverage_validated') is not True or sample.get('comparable') is not True or not isinstance(sample.get('source_ids'), list) or not sample['source_ids'] or not all(isinstance(source_id, str) and source_id for source_id in sample['source_ids']):
            return unknown
        try:
            hours, fraction, days, count = (float(sample[key]) for key in ('observed_hours', 'observed_fraction', 'complete_days', 'observation_count'))
        except (KeyError, ValueError, TypeError):
            return unknown
        if not all(math.isfinite(value) for value in (hours, fraction, days, count)) or hours <= 0 or not .8 <= fraction <= 1 or days < 7 or count < 0 or hours > days * 25:
            return unknown
    previous = float(baseline['observation_count']) / float(baseline['observed_hours'])
    latest = float(current['observation_count']) / float(current['observed_hours'])
    if previous == 0:
        return unknown
    change = (latest - previous) / previous
    return {'status': 'review' if abs(change) >= .3 else 'no_supported_change',
            'baseline_eligible': True, 'reason': 'Comparable retained-record rates differ enough for caregiver review.' if abs(change) >= .3 else 'No change crosses the proposed retained-record rate threshold.',
            'thresholds': thresholds, 'relative_record_rate_change': change,
            'source_ids': list(dict.fromkeys(current['source_ids'] + baseline['source_ids']))}


def build_window_evidence(db, home_id, care_recipient_id, window_days=14, timezone_name='UTC', now=None):
    """Curated historical aggregates and at most three examples per source type."""
    from .analytics import care_analytics
    analytics = care_analytics(db, home_id, window_days=window_days, care_recipient_id=care_recipient_id, timezone_name=timezone_name, now=now)
    window = analytics['window']
    end = datetime.fromisoformat(window['end_at'])
    examples = {}
    for table, key, kind in [('observations', 'observed_at', 'observation'), ('summaries', 'created_at', 'daily_check_in'), ('events', 'last_seen_at', 'event')]:
        rows = db.many(f'SELECT * FROM {table} WHERE home_id=? AND care_recipient_id=? ORDER BY {key} DESC,id DESC LIMIT {READ_CAP + 1}', (home_id, care_recipient_id))
        rows = bounded_rows(rows[:READ_CAP], key, window['start_at'], end)
        if table == 'events':
            rows = [row for row in rows if row.get('event_type') not in {'object_observed', 'daily_check_in'} and row.get('status') in {'needs_review', 'attention', 'open'}]
        examples[kind] = [dict(source, **({'status': row.get('status')} if table != 'observations' else {})) for source, row in zip(_sources(rows, key, kind), rows[:SOURCE_CAP])]
    source_index = [source for rows in examples.values() for source in rows]
    return bound_context_packet({'scope': {'home_id': home_id, 'care_recipient_id': care_recipient_id}, 'window': window,
        'privacy_boundary': {'excludes': ['raw_detections', 'raw_frames', 'transcripts', 'biometric_material', 'medication_records']},
        'missing_coverage': {'observed_hours': None, 'baseline_eligible': False, 'limitations': analytics['coverage']['limitations'], 'truncated': analytics['coverage']['truncated']},
        'aggregates': {'observation_count': analytics['activity']['total_observations'], 'count_is_lower_bound': analytics['activity']['count_is_lower_bound'], 'observations_by_day': analytics['activity']['by_day'], 'daily_check_in_count': analytics['daily_check_in']['total'], 'check_ins_by_day': analytics['daily_check_in']['by_day'], 'routine_change': 'unknown'},
        'relevant_examples': examples, 'source_index': source_index,
        'synthetic': any(source['synthetic'] for source in source_index)}, max_bytes=16384)


def _daily_care_plan(db, home_id, recipient_id, start, now, included, synthetic, reminders=None):
    """Use existing scheduled acknowledgements only; never invent a schedule."""
    result = {'included': bool(included), 'status': 'not_included', 'scheduled_count': 0,
              'pending_count': 0, 'marked_taken_count': 0, 'skipped_count': 0,
              'missed_count': 0, 'upcoming_count': 0,
              'disclaimer': 'Reminders and acknowledgements do not confirm medication was taken.'}
    if not included:
        return result, []
    plans = db.many('SELECT * FROM medication_plans WHERE home_id=? AND care_recipient_id=? AND active=1 ORDER BY id LIMIT 250', (home_id, recipient_id))
    allowed = {row['id'] for row in plans}
    plan_lookup = {row['id']: row for row in plans}
    rows = db.many('SELECT id,plan_id,scheduled_for,status,created_at,updated_at FROM medication_check_ins WHERE home_id=? AND care_recipient_id=? ORDER BY scheduled_for DESC,id DESC LIMIT 2001', (home_id, recipient_id))
    tomorrow = start + timedelta(days=1)
    seen, valid = set(), []
    for row in rows[:2000]:
        scheduled = _parse_timestamp(row.get('scheduled_for'))
        recorded = _parse_timestamp(row.get('updated_at') or row.get('created_at'))
        if row['id'] in seen or row['plan_id'] not in allowed or not scheduled or not start <= scheduled < tomorrow or not recorded or recorded > now:
            continue
        seen.add(row['id']); valid.append(dict(row, source_type='medication_check_in', source_id=row['id']))
    # Reuse the ordinary API's parsed reminder slots; never parse free-form dose
    # schedules here. Scope and local-day checks remain mandatory.
    keys = {(row['plan_id'], _parse_timestamp(row['scheduled_for'])) for row in valid}
    for reminder in (reminders or [])[:250]:
        plan_id = reminder.get('plan_id')
        stamp = _parse_timestamp(reminder.get('scheduled_for'))
        if plan_id not in allowed or reminder.get('care_recipient_id') != recipient_id or not stamp or not start <= stamp < tomorrow or (plan_id, stamp) in keys or reminder.get('status') not in {'pending','taken','skipped','missed'}:
            continue
        plan = plan_lookup[plan_id]
        recorded = reminder.get('updated_at') or plan.get('updated_at') or plan.get('created_at')
        captured = _parse_timestamp(recorded)
        if not captured or captured > now:
            continue
        keys.add((plan_id, stamp))
        valid.append({'id': plan_id, 'source_id': plan_id, 'source_type': 'medication_plan', 'plan_id': plan_id, 'scheduled_for': reminder['scheduled_for'], 'status': reminder['status'], 'created_at': recorded, 'updated_at': recorded})
    valid.sort(key=lambda row: (_parse_timestamp(row['scheduled_for']), str(row['id'])), reverse=True)
    result.update(status='available' if valid else 'no_scheduled_reminders', scheduled_count=len(valid), count_is_lower_bound=len(rows)>2000 or len(plans)>=250 or len(reminders or [])>250)
    for row in valid:
        status = row.get('status')
        if status == 'pending':
            key = 'upcoming_count' if _parse_timestamp(row['scheduled_for']) > now else 'pending_count'
            result[key] += 1
        elif status in ('taken', 'skipped', 'missed'):
            key = 'marked_taken_count' if status == 'taken' else status + '_count'
            result[key] += 1
    # Outstanding reminders first, then newest acknowledgements; only three examples.
    valid.sort(key=lambda row: (row['status'] in {'pending','missed'} and _parse_timestamp(row['scheduled_for']) <= now, _parse_timestamp(row['scheduled_for']), str(row['id'])), reverse=True)
    sources = [{'source_id': row['source_id'], 'source_type': row['source_type'], 'plan_id': row['plan_id'],
                'observed_at': row.get('updated_at') or row['created_at'], 'scheduled_for': row['scheduled_for'], 'status': row['status'], 'synthetic': synthetic} for row in valid[:SOURCE_CAP]]
    return result, sources
