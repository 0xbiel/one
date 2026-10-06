import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import app.main as main
from app.config import Settings

class NoProvider:
    last_error='disabled'
    def __init__(self): self.contexts=[]
    def summarize(self, context): self.contexts.append(context); return None
    def family_summary(self, context): self.contexts.append(context); return None

@pytest.fixture
def setup(tmp_path,monkeypatch):
    lm=NoProvider();monkeypatch.setattr(main,'LMStudioAdapter',lambda settings:lm)
    c=TestClient(main.make_app(Settings(database_url='sqlite:///:memory:',env='test',object_store_path=tmp_path/'objects',bootstrap_secret='test')))
    def account(role='admin'):
        s=c.post('/api/v1/pairing/start',json={'display_name':'Disposable fixture','home_name':'Disposable home','role':role}).json()
        a=c.post('/api/v1/pairing/complete',json={'code':s['pairing_code']}).json()
        return '/api/v1/homes/'+a['home_id'],{'Authorization':'Bearer '+a['access_token']},a['home_id']
    return c,lm,account

def recipient(c,base,h,name):
    r=c.post(base+'/care-recipients',headers=h,json={'display_name':name,'relationship':'Fictional fixture'}).json()['data']
    assert c.post(base+'/consents',headers=h,json={'purpose':'analytics','policy_version':'test','care_recipient_id':r['id']}).status_code==200
    return r['id']

def event(c,home,recipient,id):
    now=datetime.now(timezone.utc).replace(microsecond=0);stamp=(now-timedelta(seconds=5)).isoformat()
    c.app.state.db.execute('INSERT INTO events(id,home_id,event_type,status,explanation,evidence_json,first_seen_at,last_seen_at,expires_at,care_recipient_id) VALUES (?,?,?,?,?,?,?,?,?,?)',(id,home,'object_observed','new','Untrusted raw transcript not for context',json.dumps({'synthetic':True,'token':'excluded-fixture'}),stamp,stamp,(now+timedelta(days=1)).isoformat(),recipient))

def test_recipient_context_and_checkin_never_include_other_person_or_home(setup):
    c,lm,account=setup;base,h,home=account();other_base,other_h,other_home=account()
    bobby=recipient(c,base,h,'Bobby');other=recipient(c,base,h,'Other fictional person');foreign=recipient(c,other_base,other_h,'Foreign fictional person')
    event(c,home,bobby,'bobby-event');event(c,home,other,'other-event');event(c,other_home,foreign,'foreign-event')
    r=c.get(base+'/assistant-context',headers=h,params={'care_recipient_id':bobby,'window_days':14,'timezone_name':'Europe/Madrid'})
    assert r.status_code==200,r.text
    data=r.json()['data']['bounded_context'];assert data['relevant_examples']['event']==[]
    assert 'Untrusted raw transcript' not in r.text and 'excluded-fixture' not in r.text
    assert c.get(base+'/assistant-context',headers=h,params={'care_recipient_id':foreign}).status_code==404
    assert c.get(other_base+'/assistant-context',headers=h,params={'care_recipient_id':foreign}).status_code==403
    assert c.get(base+'/analytics',headers=h,params={'timezone_name':'Not/AZone'}).status_code==422
    check=c.post(base+'/check-ins',headers=h,json={'care_recipient_id':bobby,'transcript':'Fixture check-in'})
    assert check.status_code==200,check.text
    assert check.json()['evidence_ids']==[]
    context=lm.contexts[-1];assert context['events']==[]  # observation mirror is not an independent care signal
    assert 'other-event' not in json.dumps(context) and 'foreign-event' not in json.dumps(context)
    family=c.post(base+'/family-assistant',headers=h,json={'care_recipient_id':bobby,'message':'Review retained observations'})
    assert family.status_code==200,family.text
    assert lm.contexts[-1]['plans']==[] and lm.contexts[-1]['check_ins']==[]
    assert 'other-event' not in json.dumps(lm.contexts[-1])

def test_owner_export_excludes_credentials_and_foreign_data(setup):
    c,lm,account=setup;base,h,home=account();foreign_base,foreign_h,foreign_home=account()
    c.post(base+'/consents',headers=h,json={'purpose':'family_mode','policy_version':'test'})
    invite=c.post(base+'/family/invites',headers=h,json={'display_name':'Disposable invited caregiver','role':'caregiver'}).json()
    assert 'code' in invite
    caregiver=c.post('/api/v1/family/invites/accept',json={'code':invite['code']}).json();ch={'Authorization':'Bearer '+caregiver['access_token']}
    c.post(base+'/cameras',headers=h,json={'name':'Synthetic camera','metadata':{'provenance':'synthetic','nested':{'access_token':'secret-fixture','invite_code':'secret-fixture','embedding':[123],'transcript':'secret-fixture'}}})
    r=c.post(base+'/privacy/export',headers=h);assert r.status_code==200,r.text
    assert r.headers['cache-control']=='no-store'
    assert 'secret-fixture' not in r.text and 'code_hash' not in r.text and 'token_hash' not in r.text and foreign_home not in r.text
    assert 'care_recipients' in r.json()['data'] and 'face_profiles' in r.json()['data']
    assert r.json()['data']['cameras'][0]['metadata_json'].find('synthetic')>=0
    assert c.post(base+'/privacy/export',headers=ch).status_code==403
    assert c.post(foreign_base+'/privacy/export',headers=h).status_code==403
    assert c.post(base+'/privacy/delete',headers=ch,json={'confirmed':True,'confirmation_home_id':home}).status_code==403

def test_delete_requires_strict_matching_confirmation_before_any_mutation(setup):
    c,lm,account=setup;base,h,home=account();foreign_base,foreign_h,foreign_home=account()
    for body in (None,{}, {'confirmed':False,'confirmation_home_id':home},{'confirmed':'true','confirmation_home_id':home},{'confirmed':True,'confirmation_home_id':foreign_home}):
        r=c.post(base+'/privacy/delete',headers=h,json=body)
        assert r.status_code==422,r.text
        assert c.app.state.db.one('SELECT id FROM homes WHERE id=?',(home,))
        assert c.app.state.db.many('SELECT id FROM deletion_requests WHERE home_id=?',(home,))==[]
    done=c.post(base+'/privacy/delete',headers=h,json={'confirmed':True,'confirmation_home_id':home});assert done.status_code==200,done.text
    assert not c.app.state.db.one('SELECT id FROM homes WHERE id=?',(home,))
    assert c.app.state.db.one('SELECT id FROM homes WHERE id=?',(foreign_home,))


def test_readiness_is_authorized_readonly_and_does_not_activate_capture(setup):
    c,lm,account=setup;base,h,home=account();foreign_base,_,foreign_home=account()
    before=c.app.state.db.many('SELECT * FROM consents WHERE home_id=?',(home,))
    r=c.get(base+'/collection-readiness',headers=h);assert r.status_code==200,r.text
    assert r.json()['data']['ready'] is False
    assert c.app.state.db.many('SELECT * FROM consents WHERE home_id=?',(home,))==before
    assert c.get(foreign_base+'/collection-readiness',headers=h).status_code==403


def test_revoked_recipient_analytics_blocks_context_and_provider_before_retrieval(setup):
    c,lm,account=setup;base,h,home=account();bobby=recipient(c,base,h,'Bobby');other=recipient(c,base,h,'Other')
    event(c,home,bobby,'bobby-event');event(c,home,other,'other-event')
    allowed=c.get(base+'/assistant-context',headers=h,params={'care_recipient_id':bobby})
    assert allowed.status_code==200 and 'other-event' not in allowed.text
    revoked=c.post(base+'/consents',headers=h,json={'purpose':'analytics','policy_version':'test','care_recipient_id':bobby,'granted':False});assert revoked.status_code==200
    before=len(lm.contexts)
    for path,body in [('/assistant-context',None),('/family-assistant',{'care_recipient_id':bobby,'message':'Review'}),('/check-ins',{'care_recipient_id':bobby,'transcript':'Review'})]:
        r=c.get(base+path,headers=h,params={'care_recipient_id':bobby}) if body is None else c.post(base+path,headers=h,json=body)
        assert r.status_code==403
        assert 'bobby-event' not in r.text
    assert len(lm.contexts)==before
    assert c.get(base+'/assistant-context',headers=h,params={'care_recipient_id':other}).status_code==200


def test_day_story_scoped_readonly_and_consent_guard(setup):
    c,lm,account=setup;base,h,home=account();other_base,other_h,other_home=account()
    person=recipient(c,base,h,'Bobby');foreign=recipient(c,other_base,other_h,'Foreign')
    event(c,home,person,'local-mirror');event(c,other_home,foreign,'foreign-mirror')
    before=len(lm.contexts)
    r=c.get(base+'/day-story',headers=h,params={'care_recipient_id':person})
    assert r.status_code==200,r.text
    assert r.json()['data']['routine']['status']=='unknown'
    assert r.json()['data']['attention_items']==[]
    assert 'foreign-mirror' not in r.text and len(lm.contexts)==before
    assert c.get(base+'/day-story',headers=h,params={'care_recipient_id':foreign}).status_code==404
    c.post(base+'/consents',headers=h,json={'purpose':'analytics','policy_version':'test','care_recipient_id':person,'granted':False})
    assert c.get(base+'/day-story',headers=h,params={'care_recipient_id':person}).status_code==403

def test_simulation_metadata_preserves_calibration_and_rejects_foreign_scope(setup):
    from test_imported_geometry import payload
    c,lm,account=setup;base,h,home=account();person=recipient(c,base,h,'Bobby')
    imported=payload();mapped=c.post(base+'/maps/imported-3d',headers=h,json=imported).json()
    cam=c.post(base+'/cameras',headers=h,json={'name':'Synthetic camera','metadata':{'simulated':True}}).json()['id']
    pose={'camera_id':cam,'map_id':mapped['id'],'room_id':imported['rooms'][0]['id'],'confirmed':True,'camera_to_world':[[1,0,0,0],[0,1,0,1.5],[0,0,1,0],[0,0,0,1]]}
    assert c.post(base+'/camera-registrations/manual',headers=h,json=pose).status_code==200
    db=c.app.state.db;before=db.many('SELECT * FROM calibrations WHERE camera_id=?',(cam,))
    now=datetime.now(timezone.utc);marker={'synthetic':True,'enabled':True,'home_id':home,'camera_id':cam,'map_id':mapped['id'],'care_recipient_id':person,'started_at':(now-timedelta(minutes=5)).isoformat(),'last_simulated_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    metadata={'simulated':True,'fixture_simulation_v1':marker}
    result=c.patch(base+'/cameras/'+cam,headers=h,json={'metadata':metadata})
    assert result.status_code==200,result.text
    assert result.json()['calibrations_invalidated'] is False
    assert db.many('SELECT * FROM calibrations WHERE camera_id=?',(cam,))==before
    assert c.get(base+'/collection-readiness',headers=h).json()['data']['ready'] is False
    invalid={**marker,'home_id':'foreign'}
    assert c.patch(base+'/cameras/'+cam,headers=h,json={'metadata':{'simulated':True,'fixture_simulation_v1':invalid}}).status_code==422
    assert db.many('SELECT * FROM calibrations WHERE camera_id=?',(cam,))==before
    changed=c.patch(base+'/cameras/'+cam,headers=h,json={'metadata':{**metadata,'intrinsics_changed':True}})
    assert changed.status_code==200 and changed.json()['calibrations_invalidated'] is True


def test_preview_and_invocation_share_curated_packet_and_privacy_boundary(setup):
    c,lm,account=setup;base,h,home=account();person=recipient(c,base,h,'Bobby')
    event(c,home,person,'observation-mirror')
    question='Review retained activity'
    preview=c.get(base+'/assistant-context',headers=h,params={'care_recipient_id':person,'message':question,'timezone_name':'UTC'}).json()['data']
    assert preview['request']==question and preview['bounded_context']['relevant_examples']['event']==[]
    assert len(lm.contexts)==0
    assert preview['privacy_boundary']['external_provider_authorized'] is False
    result=c.post(base+'/family-assistant',headers=h,json={'care_recipient_id':person,'message':question,'timezone_name':'UTC'})
    assert result.status_code==200,result.text
    packet=lm.contexts[-1]
    # Both routes build the same packet; independently refreshed end times can differ.
    assert packet.keys()==preview.keys()
    for key in ['subject','plans','check_ins','daily_check_ins','request','evidence_scope','privacy_boundary']:
        assert packet[key]==preview[key]
    for key in ['scope','aggregates','relevant_examples','source_index','privacy_boundary']:
        assert packet['bounded_context'][key]==preview['bounded_context'][key]
