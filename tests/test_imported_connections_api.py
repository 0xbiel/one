import copy,uuid
from test_imported_geometry import session,payload

def door_payload(original):
    return {**original,'doors':[{'id':str(uuid.uuid4()),'room_ids':[r['id'] for r in original['rooms']],'center':{'x':2.5,'y':1.,'z':0.},'width':.9,'height':2.,'normal_axis':'x'}]}

def test_connection_revision_preserves_placements_and_rejects_geometry_edits(session):
    c,base,h=session;original=payload();mapped=c.post(base+'/maps/imported-3d',headers=h,json=original).json()
    camera=c.post(base+'/cameras',headers=h,json={'name':'Scoped camera'}).json()['id']
    pose={'camera_id':camera,'map_id':mapped['id'],'room_id':original['rooms'][0]['id'],'confirmed':True,'camera_to_world':[[1,0,0,0],[0,1,0,1.5],[0,0,1,0],[0,0,0,1]]}
    assert c.post(base+'/camera-registrations/manual',headers=h,json=pose).status_code==200
    db=c.app.state.db;before=db.many('SELECT * FROM calibrations WHERE home_id=?',(mapped['home_id'],));cam_before=c.get(base+'/cameras',headers=h).json()
    endpoint=base+'/maps/'+mapped['id']+'/connections';changed=door_payload(original)
    invalid=copy.deepcopy(changed);invalid['rooms'][0]['center']['x']=.1
    assert c.patch(endpoint,headers={**h,'If-Match':str(mapped['revision'])},json=invalid).status_code==422
    assert c.patch(endpoint,headers={**h,'If-Match':'999'},json=changed).status_code==409
    assert c.patch(endpoint,headers=h,json=changed).status_code==422
    result=c.patch(endpoint,headers={**h,'If-Match':str(mapped['revision'])},json=changed)
    assert result.status_code==200,result.text
    revised=result.json();assert revised['id']==mapped['id'] and revised['revision']>mapped['revision']
    assert revised['map_data']['import']['rooms']==mapped['map_data']['import']['rooms']
    assert len(revised['map_data']['normalized_scan']['doors'])==1
    assert revised['metadata']['visual_landmarks']['status']=='unavailable'
    assert db.many('SELECT * FROM calibrations WHERE home_id=?',(mapped['home_id'],))==before
    assert c.get(base+'/cameras',headers=h).json()==cam_before
    repeated=c.patch(endpoint,headers={**h,'If-Match':str(revised['revision'])},json=changed)
    assert repeated.status_code==200 and repeated.json()['revision']==revised['revision']
    assert c.patch(endpoint,headers={**h,'If-Match':str(mapped['revision'])},json=changed).status_code==409
    usdz=c.get(base+'/maps/'+mapped['id']+'/usdz',headers=h);assert usdz.status_code==200 and usdz.content.startswith(b'PK')
    assert c.patch('/api/v1/homes/'+str(uuid.uuid4())+'/maps/'+mapped['id']+'/connections',headers={**h,'If-Match':str(revised['revision'])},json=changed).status_code==403
