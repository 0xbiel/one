"""Coordinate, calibration, split, payload-leakage and cuboid consistency checks."""
import json,hashlib,base64
from pathlib import Path
import numpy as np
root=Path(__file__).parent
expected=[(-1.9,1.55,1.85),(-.65,1.7,2.05),(.65,1.55,2.05),(1.85,1.7,1.9),(-1.25,1.85,2.35),(1.2,1.3,2.35)]
for d in sorted((root/'fixtures').iterdir()):
 j=json.loads((d/'evaluator_groundtruth.json').read_text());assert len(j['records'])==6
 for r,pos in zip(j['records'],expected):
  C=np.asarray(r['camera_to_world']);K=np.asarray(r['intrinsics']);assert np.allclose(C[:3,3],pos,atol=1e-6);assert np.allclose(C[:3,:3].T@C[:3,:3],np.eye(3),atol=1e-6);assert np.isclose(np.linalg.det(C[:3,:3]),1,atol=1e-6);assert C[1,1]>.9
  assert np.allclose(K,[[320,0,240],[0,320,180],[0,0,1]])
  # Backprojection->projection convention roundtrip for noncentral pixel and 3m axial depth.
  uv=np.asarray([123.,98.]);v=np.asarray([(uv[0]-240)*3/320,-(uv[1]-180)*3/320,-3,1]);world=C@v;cv=np.diag([1,-1,-1,1])@np.linalg.inv(C)@world;pr=K@cv[:3];assert np.allclose(pr[:2]/pr[2],uv)
  if r['role']=='reference':assert 'depth' in r
  else:assert 'depth' not in r
 scan=json.loads((d/'scan.json').read_text())
 for obj in scan['room_objects']:
  center=np.asarray(list(obj['center'].values()));dim=np.asarray(list(obj['dimensions'].values()));assert dim.min()>0;assert center[1]>0;assert np.allclose(np.asarray(obj['transform']['values'])[:3,3],center)
 for a in j['records'][:4]:
  for b in j['records'][4:]:assert hashlib.sha256((d/a['image']).read_bytes()).digest()!=hashlib.sha256((d/b['image']).read_bytes()).digest()
detector={(x['layout'],x['query'],x['frame_index']):x for x in json.loads((root/'detector_diagnostics.json').read_text())}
for m in json.loads((root/'manifest.json').read_text()):
 p=root/m['input'];assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256'];payload=json.loads(p.read_text());assert 'search_prior' not in payload;assert 'person_anchors' not in payload
 for i,frame in enumerate(payload['frames']):
  assert set(frame)=={'frame_base64','width','height'}
  det=detector[m['layout'],m['query'],i]
  assert hashlib.sha256(base64.b64decode(frame['frame_base64'])).hexdigest()==det['sha256']
  assert [{k:v for k,v in x.items() if k!='frame_index'} for x in payload['object_detections'] if x['frame_index']==i]==det['detections']
 assert len(payload['landmarks'])<=8000
print('PASS: 3 layouts,18 calibrated renders, distinct reference/query frames,48 input pose/depth exclusion and detector-provenance checks, cuboid transform consistency')
