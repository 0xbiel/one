import sqlite3
from datetime import datetime, timezone

from app.collection_readiness import collection_readiness

NOW = datetime(2026,10,1,12,tzinfo=timezone.utc)

class DB:
    def __init__(self):
        self.conn=sqlite3.connect(':memory:'); self.conn.row_factory=sqlite3.Row
        self.conn.executescript('''CREATE TABLE home_runtime(home_id,paused);
        CREATE TABLE consents(home_id,purpose,granted_at,revoked_at);
        CREATE TABLE cameras(id,home_id,name,room_id,enabled,created_at,metadata_json);
        CREATE TABLE sessions(home_id,user_id,created_at,expires_at);
        CREATE TABLE rooms(id,home_id,name);
        CREATE TABLE room_maps(id,home_id);
        CREATE TABLE calibrations(id,home_id,camera_id,map_id,source,status,created_at,invalidated_at);
        CREATE TABLE observations(home_id,camera_id,observed_at,detector_version);''')
        self.conn.execute("INSERT INTO cameras VALUES ('cam','home','Living','room',1,'2026-09-01T00:00:00Z','{}')")
        self.conn.execute("INSERT INTO rooms VALUES ('room','home','Living')")
        self.conn.execute("INSERT INTO room_maps VALUES ('map','home')")
        self.conn.execute("INSERT INTO calibrations VALUES ('cal','home','cam','map','manual-geometry-registration','active','2026-09-01T00:00:00Z',NULL)")
    def many(self,sql,params):return [dict(row) for row in self.conn.execute(sql,params)]
    def one(self,sql,params):return next(iter(self.many(sql,params)),None)
    def consent(self):self.conn.execute("INSERT INTO consents VALUES ('home','video_capture','2026-09-01T00:00:00Z',NULL)")
    def online(self):self.conn.execute("INSERT INTO sessions VALUES ('home','cam','2026-10-01T11:00:00Z','2026-10-02T00:00:00Z')")
    def observe(self,stamp='2026-10-01T11:59:00Z',home='home',version='local-cv-v1'):
        self.conn.execute('INSERT INTO observations VALUES (?,?,?,?)',(home,'cam',stamp,version))

def test_no_consent_and_read_only():
    db=DB();db.online();db.observe()
    before=db.conn.total_changes
    result=collection_readiness(db,'home',NOW)
    assert not result['ready']
    assert 'video_capture_consent_required' in result['cameras'][0]['blockers']
    assert db.conn.total_changes == before

def test_manual_placement_ready_with_fresh_online_consented_data():
    db=DB();db.consent();db.online();db.observe()
    result=collection_readiness(db,'home',NOW)
    assert result['ready']
    assert result['cameras'][0]['placement']['source']=='manual-geometry-registration'

def test_synthetic_history_cannot_make_offline_camera_online():
    db=DB();db.consent();db.observe(version='synthetic-bobby-v1')
    camera=collection_readiness(db,'home',NOW)['cameras'][0]
    assert camera['status']=='offline'
    assert camera['freshness']['synthetic']
    assert camera['freshness']['label']=='Simulated observations'

def test_out_of_order_timezone_and_future_records():
    db=DB();db.consent();db.online()
    db.observe('2026-10-01T13:59:00+02:00')
    db.observe('2026-09-01T00:00:00Z')
    db.observe('2027-01-01T00:00:00Z')
    freshness=collection_readiness(db,'home',NOW)['cameras'][0]['freshness']
    assert freshness['age_seconds']==60
    assert freshness['status']=='fresh'

def test_cross_home_session_map_room_and_observations_never_count():
    db=DB();db.consent()
    db.conn.execute("INSERT INTO sessions VALUES ('other','cam','2026-10-01T11:00:00Z','2026-10-02T00:00:00Z')")
    db.observe(home='other')
    db.conn.execute("UPDATE rooms SET home_id='other'")
    db.conn.execute("UPDATE room_maps SET home_id='other'")
    camera=collection_readiness(db,'home',NOW)['cameras'][0]
    assert set(camera['blockers']) == {'camera_offline','room_required','placement_required','observations_missing'}
    assert not collection_readiness(db,'other',NOW)['cameras']

def test_stale_paused_and_latest_revocation():
    db=DB();db.consent();db.online();db.observe('2026-10-01T11:00:00Z')
    db.conn.execute("INSERT INTO home_runtime VALUES ('home',1)")
    db.conn.execute("INSERT INTO consents VALUES ('home','video_capture','2026-09-30T00:00:00Z','2026-09-30T01:00:00Z')")
    result=collection_readiness(db,'home',NOW)
    assert not result['ready']
    assert set(result['cameras'][0]['blockers'])=={'video_capture_consent_required','home_paused','observations_stale'}

def test_disabled_and_missing_never_ready():
    db=DB();db.consent();db.online();db.observe();db.conn.execute('UPDATE cameras SET enabled=0')
    assert not collection_readiness(db,'home',NOW)['ready']
    assert collection_readiness(db,'missing',NOW)['blockers']==['video_capture_consent_required','camera_required']
