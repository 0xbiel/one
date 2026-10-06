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


def door_payload():
    p=payload();a,b=p['rooms'];floor=0
    p['doors']=[{'id':str(uuid.uuid4()),'room_ids':[a['id'],b['id']],'center':{'x':2.5,'y':1.0,'z':0},'width':1,'height':2,'normal_axis':'x'}]
    return p


def test_doors_connections_actual_wall_apertures_and_usdz_neutral_floor_names():
    import io,zipfile,re
    from app.imported_geometry import normalized_geometry,build_usdz
    p=door_payload();p['connections']=[{'id':str(uuid.uuid4()),'room_ids':p['doors'][0]['room_ids'],'door_id':p['doors'][0]['id']}]
    body=ImportedMapIn.model_validate(p);scan=normalized_geometry(body)
    assert len(scan['doors'])==len(scan['openings'])==len(scan['connections'])==1
    door=scan['doors'][0]
    assert door['room_ids']==p['doors'][0]['room_ids'] and door['passable']
    assert door['transform'][0][3]==2.5 and door['transform'][1][3]==1
    # At the shared boundary, neither duplicated room wall fills the aperture.
    shared=[w for w in scan['walls'] if abs(w['center']['x']-2.5)<1e-6 and w['dimensions']['x']==.04]
    assert len(shared)==4
    for wall in shared:
        lo=wall['center']['z']-wall['dimensions']['z']/2
        hi=wall['center']['z']+wall['dimensions']['z']/2
        assert hi<=-.5+1e-6 or lo>=.5-1e-6
    with zipfile.ZipFile(io.BytesIO(build_usdz(scan))) as archive:
        usda=archive.read('ImportedHome.usda').decode()
    assert f'floor_{p["rooms"][0]["id"].replace("-","_")}' in usda
    assert '0.70, 0.72, 0.74' in usda and '0.35, 0.60, 0.75' not in usda
    cubes=re.findall(r'def Cube "([^"]+)"',usda)
    assert len(cubes)==len(scan['floors'])+len(scan['walls'])+len(scan['objects'])
    assert not any('door' in name for name in cubes)


@pytest.mark.parametrize('change',[{'room_ids':['unknown','unknown']},{'normal_axis':'z'},{'width':4},{'height':4},{'center':{'x':2.4,'y':1,'z':0}},{'center':{'x':2.5,'y':1.1,'z':0}},{'center':{'x':2.5,'y':1,'z':1.5}},{'width':float('nan')},{'transform':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}])
def test_invalid_doors_rejected(change):
    p=door_payload();p['doors'][0].update(change)
    with pytest.raises(ValueError):ImportedMapIn.model_validate(p)


def test_invalid_connections_duplicate_overlapping_doors_and_floor_levels():
    p=door_payload();door=p['doors'][0]
    for connection in [{'id':str(uuid.uuid4()),'room_ids':door['room_ids'],'door_id':str(uuid.uuid4())},{'id':str(uuid.uuid4()),'room_ids':[door['room_ids'][0]]*2,'door_id':door['id']}]:
        with pytest.raises(ValueError):ImportedMapIn.model_validate({**p,'connections':[connection]})
    overlap=copy.deepcopy(door);overlap['id']=str(uuid.uuid4())
    with pytest.raises(ValueError):ImportedMapIn.model_validate({**p,'doors':[door,overlap]})
    p['rooms'][1]['center']['y']=1.6
    with pytest.raises(ValueError):ImportedMapIn.model_validate(p)


def test_backward_compatible_no_doors_and_short_door_lintel():
    from app.imported_geometry import normalized_geometry
    old=normalized_geometry(ImportedMapIn.model_validate(payload()))
    assert old['doors']==old['openings']==old.get('connections',[])==[]
    p=door_payload();p['doors'][0]['height']=.4;p['doors'][0]['center']['y']=.2
    scan=normalized_geometry(ImportedMapIn.model_validate(p))
    lintels=[w for w in scan['walls'] if abs(w['center']['x']-2.5)<1e-6 and abs(w['center']['z'])<1e-6 and abs(w['dimensions']['z']-1)<1e-6]
    assert len(lintels)==2
    assert all(abs(w['center']['y']-w['dimensions']['y']/2-.4)<1e-6 for w in lintels)
