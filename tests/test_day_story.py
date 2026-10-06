import json
from datetime import datetime, timezone
import pytest
from test_analytics_context import DB
from app.day_story import build_day_story, build_window_evidence, evaluate_routine_change
from app.context_budget import serialized_size

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def database():
    db = DB()
    db.conn.executescript("CREATE TABLE rooms(id,home_id,name); INSERT INTO rooms VALUES ('room','home','Kitchen'); CREATE TABLE care_recipients(id,home_id,display_name,relationship); INSERT INTO care_recipients VALUES ('bobby','home','Bobby','Fictional synthetic care recipient');")
    return db


def checkin(db, id='summary', stamp='2026-10-01T10:00:00Z'):
    db.conn.execute('INSERT INTO summaries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (id,'home','bobby',None,'ok','unknown','PRIVATE instructions','[]',stamp,'2026-11-01T00:00:00Z','synthetic-v1','[]'))
    event(db, 'mirror-'+id, 'daily_check_in', stamp, 'ok')


def event(db, id='review', kind='fall_suspected', stamp='2026-10-01T10:30:00Z', status='needs_review'):
    db.conn.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (id,'home','bobby',kind,status,.8,'PRIVATE instructions',stamp,stamp,'2026-11-01T00:00:00Z','{"synthetic":true}'))


def test_empty_today_is_unknown_not_inactivity():
    db = database(); db.observe('old','2026-09-30T10:00:00Z')
    story = build_day_story(db,'home','bobby',now=NOW)
    assert story['headline'] == 'A check-in would help today'
    assert 'does not establish inactivity' in story['summary']
    assert story['coverage']['observed_hours'] is None
    assert story['routine']['status'] == 'unknown'
    assert story['routine']['baseline_eligible'] is False
    assert story['meaningful_changes'] == []
    assert story['facts'] == []
    assert story['synthetic'] is True


def test_populated_story_curated_sources_and_canonical_checkins():
    db = database()
    for i in range(12): db.observe(str(i),f'2026-10-01T{7+i//3:02}:00:00Z')
    db.observe('foreign','2026-10-01T10:00:00Z',recipient='other')
    db.observe('future','2026-10-01T15:00:00Z')
    checkin(db); event(db); event(db,'object','object_observed',status='needs_review')
    before = db.conn.total_changes
    story = build_day_story(db,'home','bobby',now=NOW)
    assert db.conn.total_changes == before
    assert story['coverage']['observation_count'] == 12
    assert story['details']['current_room_sighting_counts'] == {'room':12}
    assert len(story['attention_items']) == 1
    assert story['attention_items'][0]['source_ids'] == ['review']
    assert story['meaningful_changes'] == []
    packet = story['evidence_packet']
    assert len(packet['observation_sources']) == 3
    assert len(packet['canonical_check_ins']) == 1
    assert len(packet['source_index']) == 5
    assert 'PRIVATE' not in json.dumps(story)
    assert 'raw_frames' not in packet.get('aggregates', {})
    assert serialized_size(packet) <= 16384
    assert story == build_day_story(db,'home','bobby',now=NOW)


def test_local_midnight_and_scope():
    db = database()
    db.observe('first','2026-09-30T22:00:00Z')
    db.observe('old','2026-09-30T21:59:59Z')
    db.observe('first','2026-10-01T00:00:00+02:00')
    story = build_day_story(db,'home','bobby',timezone_name='Europe/Madrid',now=NOW)
    assert story['coverage']['observation_count'] == 1
    assert story['window']['start_at'] == '2026-10-01T00:00:00+02:00'
    with pytest.raises(ValueError): build_day_story(db,'foreign','bobby',now=NOW)


def test_historical_question_counts_are_not_today_counts():
    db = database()
    for i in range(138): db.observe(str(i),'2026-09-30T10:00:00Z')
    for i in range(14): checkin(db,str(i),f'2026-09-{17+i:02}T20:00:00Z')
    packet = build_window_evidence(db,'home','bobby',window_days=14,now=NOW)
    assert packet['aggregates']['observation_count'] == 138
    assert packet['aggregates']['daily_check_in_count'] == 13  # window Sep18-Oct1
    assert len(packet['relevant_examples']['observation']) == 3
    assert len(packet['relevant_examples']['daily_check_in']) == 3
    assert packet['relevant_examples']['event'] == []
    assert packet['missing_coverage']['observed_hours'] is None
    assert packet['aggregates']['routine_change'] == 'unknown'
    assert serialized_size(packet) <= 16384


def test_baseline_scaffold_requires_validated_comparable_hours():
    baseline = {'coverage_validated':True,'comparable':True,'complete_days':7,'observed_hours':100,'observed_fraction':.9,'observation_count':100,'source_ids':['synthetic-baseline-coverage']}
    current = dict(baseline, observation_count=150,source_ids=['synthetic-current-coverage'])
    assert evaluate_routine_change(current,baseline)['status'] == 'review'
    assert evaluate_routine_change(baseline,baseline)['status'] == 'no_supported_change'
    for changed in ({'coverage_validated':False},{'comparable':False},{'observed_hours':None},{'observed_fraction':.5},{'source_ids':[]},{'complete_days':3}):
        assert evaluate_routine_change(dict(current,**changed),baseline)['status'] == 'unknown'
    # No fabricated database attribute can make runtime eligibility true.
    db=database(); db.processing_coverage=baseline
    assert build_day_story(db,'home','bobby',now=NOW)['routine']['baseline_eligible'] is False


def medication_tables(db):
    db.conn.executescript("CREATE TABLE medication_plans(id,home_id,care_recipient_id,active); ALTER TABLE medication_check_ins ADD COLUMN created_at; ALTER TABLE medication_check_ins ADD COLUMN updated_at; INSERT INTO medication_plans VALUES ('plan','home','bobby',1); INSERT INTO medication_plans VALUES ('otherplan','home','other',1);")


def medication(db, id, stamp, status='pending', recipient='bobby', plan='plan'):
    db.conn.execute('INSERT INTO medication_check_ins VALUES (?,?,?,?,?,?,?,?,?)', (id,'home',recipient,None,stamp,status,plan,'2026-10-01T06:00:00Z','2026-10-01T06:00:00Z'))


def test_care_plan_default_excluded_and_consent_opt_in_scoped_today():
    db=database(); medication_tables(db)
    medication(db,'due','2026-10-01T09:00:00Z')
    medication(db,'taken','2026-10-01T08:00:00Z','taken')
    medication(db,'upcoming','2026-10-01T15:00:00Z')
    medication(db,'yesterday','2026-09-30T09:00:00Z')
    medication(db,'other','2026-10-01T09:00:00Z',recipient='other',plan='otherplan')
    medication(db,'crossplan','2026-10-01T09:00:00Z',plan='otherplan')
    default=build_day_story(db,'home','bobby',now=NOW)
    assert default['care_plan']['included'] is False
    assert all(row['source_type'] != 'medication_check_in' for row in default['evidence_packet']['source_index'])
    before=db.conn.total_changes
    story=build_day_story(db,'home','bobby',now=NOW,include_medication=True)
    assert db.conn.total_changes == before
    assert story['care_plan']['scheduled_count'] == 3
    assert story['care_plan']['pending_count'] == 1
    assert story['care_plan']['marked_taken_count'] == 1
    assert story['care_plan']['upcoming_count'] == 1
    assert 'needs an acknowledgement' in story['summary']
    assert story['attention_items'][0]['source_ids'] == ['due']
    assert {row['source_id'] for row in story['evidence_packet']['care_plan_examples']} == {'due','taken','upcoming'}
    assert 'do not confirm medication was taken' in story['care_plan']['disclaimer']
    assert 'medication_records' not in story['evidence_packet']['privacy_boundary']['excludes']
    assert serialized_size(story['evidence_packet']) <= 16384


def test_plain_story_prose_has_actions_without_ids_or_raw_count_wording():
    db=database(); db.observe('opaque-observation-id','2026-10-01T11:00:00Z'); checkin(db,id='opaque-checkin-id')
    story=build_day_story(db,'home','bobby',now=NOW)
    prose=story['headline']+' '+story['summary']+' '+' '.join(row['text'] for row in story['facts'])
    assert 'Bobby’s day so far' in prose
    assert 'check-in is marked okay' in prose
    assert 'Kitchen at 11:00' in prose
    assert 'record(s)' not in prose
    assert 'opaque-' not in prose
    assert story['facts'][1]['source_ids'] == ['opaque-observation-id']
    assert story['coverage']['observed_hours'] is None


def test_scheduled_reminder_does_not_invent_acknowledgement_or_ingestion():
    db=database(); medication_tables(db)
    medication(db,'later','2026-10-01T15:00:00Z')
    story=build_day_story(db,'home','bobby',now=NOW,include_medication=True)
    assert story['care_plan']['pending_count'] == 0
    assert story['care_plan']['marked_taken_count'] == 0
    assert story['care_plan']['upcoming_count'] == 1
    assert 'scheduled for later today' in story['summary']
    assert story['attention_items'] == []


def test_existing_api_reminder_slots_reused_without_names_doses_or_fake_checkin():
    db=database(); medication_tables(db)
    db.conn.executescript("ALTER TABLE medication_plans ADD COLUMN created_at; UPDATE medication_plans SET created_at='2026-10-01T06:00:00Z';")
    reminders=[{'plan_id':'plan','care_recipient_id':'bobby','scheduled_for':'2026-10-01T09:00:00Z','status':'pending','updated_at':None,'name':'PRIVATE medicine','dose':'PRIVATE dose','instructions':'PRIVATE instructions'},
               {'plan_id':'otherplan','care_recipient_id':'other','scheduled_for':'2026-10-01T09:00:00Z','status':'pending'}]
    story=build_day_story(db,'home','bobby',now=NOW,include_medication=True,reminders=reminders)
    assert story['care_plan']['scheduled_count'] == 1
    assert story['care_plan']['pending_count'] == 1
    assert story['care_plan']['marked_taken_count'] == 0
    source=story['evidence_packet']['care_plan_examples'][0]
    assert source['source_id'] == 'plan'
    assert source['source_type'] == 'medication_plan'
    assert source['observed_at'] == '2026-10-01T06:00:00Z'
    assert source['scheduled_for'] == '2026-10-01T09:00:00Z'
    assert 'PRIVATE' not in json.dumps(story)
    default=build_day_story(db,'home','bobby',now=NOW,reminders=reminders)
    assert default['care_plan']['included'] is False


def test_api_slots_do_not_double_count_existing_acknowledgement():
    db=database(); medication_tables(db)
    medication(db,'taken','2026-10-01T09:00:00Z','taken')
    reminders=[{'plan_id':'plan','care_recipient_id':'bobby','scheduled_for':'2026-10-01T11:00:00+02:00','status':'taken'}]
    story=build_day_story(db,'home','bobby',now=NOW,include_medication=True,reminders=reminders)
    assert story['care_plan']['scheduled_count'] == 1
    assert story['care_plan']['marked_taken_count'] == 1
    assert story['evidence_packet']['care_plan_examples'][0]['source_id'] == 'taken'
