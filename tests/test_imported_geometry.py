import copy, json, uuid
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import make_app
from app.imported_geometry import ImportedMapIn, CameraPlacementIn


def payload():
    return dict(schema_version='imported-geometry.v1',provenance='synthetic',units='m',up_axis='Y',coordinate_frame='import-local',rooms=[dict(id=str(uuid.uuid4()),name='Living room',center=dict(x=0,y=1.5,z=0),dimensions=dict(x=5,y=3,z=4)),dict(id=str(uuid.uuid4()),name='Kitchen',center=dict(x=4,y=1.5,z=0),dimensions=dict(x=3,y=3,z=3))],objects=[])

@pytest.fixture
def session(tmp_path):
    client=TestClient(make_app(Settings(database_url='sqlite:///:memory:', object_store_path=tmp_path/'objects',bootstrap_secret='test',env='test',lm_studio_url='http://127.0.0.1:9/v1')))
    start=client.post('/api/v1/pairing/start',json={'display_name':'Test Caregiver','home_name':'Imported Home','role':'caregiver'}).json()
    done=client.post('/api/v1/pairing/complete',json={'code':start['pairing_code']}).json()
    return client,'/api/v1/homes/'+done['home_id'],{'Authorization':'Bearer '+done['access_token']}

def test_import_provenance_geometry_usdz_and_manual_save_safety(session):
    c,base,h=session; p=payload()
    result=c.post(base+'/maps/imported-3d',headers=h,json=p)
    assert result.status_code==200,result.text
    model=result.json(); assert model['source']=='imported-3d' and model['provenance']=='synthetic'
    scene=c.get(base+'/scene',headers=h).json()
    assert scene['dimension']=='3d' and scene['visualIndexStatus']=='unavailable'
    assert scene['canonicalGeometry']['framework']=='ImportedGeometry'
    usdz=c.get(base+'/maps/'+model['id']+'/usdz',headers=h)
    assert usdz.status_code==200 and usdz.content.startswith(b'PK')
    cam=c.post(base+'/cameras',headers=h,json={'name':'Test camera'}).json()['id']
    pose=dict(camera_id=cam,map_id=model['id'],room_id=p['rooms'][0]['id'],confirmed=True,camera_to_world=[[1,0,0,0],[0,1,0,1.5],[0,0,1,0],[0,0,0,1]])
    saved=c.post(base+'/camera-registrations/manual',headers=h,json=pose)
    assert saved.status_code==200,saved.text
    before=c.get(base+'/scene',headers=h).json()['cameraRegistrations']
    assert len(before)==1
    for change in ({'confirmed':False},{'room_id':p['rooms'][1]['id']},{'room_id':str(uuid.uuid4())},{'map_id':str(uuid.uuid4())},{'camera_to_world':[[2,0,0,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]}):
        rejected=c.post(base+'/camera-registrations/manual',headers=h,json={**pose,**change})
        assert rejected.status_code in (404,422),rejected.text
        assert c.get(base+'/scene',headers=h).json()['cameraRegistrations']==before
    second=c.post(base+'/maps/imported-3d',headers=h,json=p).json()
    assert c.post(base+'/camera-registrations/manual',headers=h,json=pose).status_code==409
    assert second['id'] != model['id']

def test_import_rejects_invalid_and_keeps_generic_guard(session):
    c,base,h=session;p=payload()
    for key,value in [('units','cm'),('up_axis','Z'),('provenance','native-roomplan'),('rooms',[])]:
        assert c.post(base+'/maps/imported-3d',headers=h,json={**p,key:value}).status_code==422
    overlapping=copy.deepcopy(p);overlapping['rooms'][1]['center']['x']=0
    assert c.post(base+'/maps/imported-3d',headers=h,json=overlapping).status_code==422
    assert c.post(base+'/maps/imported-3d',json=p).status_code==401
    assert c.post('/api/v1/homes/'+str(uuid.uuid4())+'/maps/imported-3d',headers=h,json=p).status_code==403
    generic=c.post(base+'/maps',headers=h,json={'coordinate_frame':'import-local','map_data':p})
    assert generic.status_code==200 and generic.json()['dimension']=='2d'
    native=c.post(base+'/maps/roomplan',headers=h,json={'normalized_scan':p,'scan_metadata':{'provenance':'synthetic'}})
    assert native.status_code==422

def test_concurrent_visual_batches_accumulate_and_retry_is_idempotent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from app.imported_geometry import normalized_geometry
    class Geometry:
        barrier=Barrier(3)
        def build_visual_landmarks(self, *, map_id, frames):
            self.barrier.wait(timeout=10)
            n=int(frames[0]['frame_base64'])
            return {'status':'ready','landmarks':[{'point':[n+i/10,1,-1],'descriptor_base64':'AA==','response':1} for i in range(8)],'diagnostics':{'source_frame_count':1}}
    geometry=Geometry()
    c=TestClient(make_app(Settings(database_url='sqlite:///:memory:',object_store_path=tmp_path/'objects',bootstrap_secret='test',env='test'),geometry_service=geometry))
    start=c.post('/api/v1/pairing/start',json={'display_name':'Tester','home_name':'Test','role':'caregiver'}).json()
    done=c.post('/api/v1/pairing/complete',json={'code':start['pairing_code']}).json();base='/api/v1/homes/'+done['home_id'];h={'Authorization':'Bearer '+done['access_token']}
    scan=normalized_geometry(ImportedMapIn.model_validate(payload()))
    scan.update(schema_version='roomplan-normalized.v1',producer='native-ios',framework='RoomPlan',coordinate_frame='roomplan-local')
    native={'normalized_scan':scan,'scan_metadata':{'provenance':'native-roomplan','device_model':'unit-test-fixture','lidar':True,'roomplan_version':'1','units':'m','up_axis':'Y','geometry_type':'3d'}}
    created=c.post(base+'/maps/roomplan',headers=h,json=native);assert created.status_code==200,created.text
    mid=created.json()['id'];endpoint=base+'/maps/'+mid+'/visual-landmarks'
    def body(n):return {'frames':[{'frame_base64':str(n),'width':100,'height':100,'intrinsics':{'values':[[50,0,50],[0,50,50],[0,0,1]]},'camera_to_world':[[1,0,0,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]}]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda n:c.post(endpoint,headers=h,json=body(n)),range(3)))
    assert all(r.status_code==200 for r in results),[r.text for r in results]
    current=c.get(base+'/maps/'+mid,headers=h).json()
    assert current['metadata']['visual_landmarks']['landmark_count']==24
    geometry.barrier=Barrier(1)
    retry=c.post(endpoint,headers=h,json=body(0));assert retry.status_code==200
    assert retry.json()['diagnostics']['idempotent_replay'] is True
    current=c.get(base+'/maps/'+mid,headers=h).json()
    assert current['metadata']['visual_landmarks']['landmark_count']==24
