"""Build scan-only inputs. Ground truth camera matrices for queries never enter payloads."""
import os,json,base64,hashlib
from pathlib import Path
import numpy as np,cv2
from geometry_service.localization import build_visual_landmarks
from geometry_service.contracts import VisualLandmarkBuildRequest
root=Path(__file__).parent
b64=lambda p:base64.b64encode(p).decode()
manifest=[]
for d in sorted((root/'fixtures').iterdir()):
 truth=json.loads((d/'evaluator_groundtruth.json').read_text());scan=json.loads((d/'scan.json').read_text());refs=[r for r in truth['records'] if r['role']=='reference']
 for condition in ['clean','noisy_depth_missing_boxes','no_depth','unknown_intrinsics']:
  landmarks=[];diagnostics=[];frames=[];rng=np.random.default_rng(624319)
  for r in refs:
   frame=dict(frame_base64=b64((d/r['image']).read_bytes()),width=r['width'],height=r['height'],intrinsics={'values':r['intrinsics']},camera_to_world={'values':r['camera_to_world']})
   if condition!='no_depth':
    depth=np.load(d/r['depth'])
    if condition=='noisy_depth_missing_boxes':
     depth=depth+rng.normal(0,.025,depth.shape);depth[rng.random(depth.shape)<.1]=0
    frame.update(depth_base64=b64(depth.astype('<f4').tobytes()),depth_width=depth.shape[1],depth_height=depth.shape[0])
   frames.append(frame)
   if condition!='no_depth':
    result=build_visual_landmarks(VisualLandmarkBuildRequest(map_id=d.name,frames=[frame]));diagnostics.append(result['diagnostics']);landmarks.extend([{**l,'view_id':r['id']} for l in result['landmarks']])
  if condition=='no_depth':
   # Pairs mimic separate scan batches and retain distinct view IDs for triangulation.
   for i in range(len(frames)-1):
    result=build_visual_landmarks(VisualLandmarkBuildRequest(map_id=d.name,frames=frames[i:i+2]));diagnostics.append(result['diagnostics']);landmarks.extend([{**l,'view_id':f'pair_{i}'} for l in result['landmarks']])
  # Deterministic round-robin cap preserves all scan views, no query information used.
  by_view={}
  for l in landmarks:by_view.setdefault(l['view_id'],[]).append(l)
  for group in by_view.values():group.sort(key=lambda l:l['response'],reverse=True)
  landmarks=[]
  for offset in range(max(map(len,by_view.values()),default=0)):
   for view in sorted(by_view):
    if offset<len(by_view[view]) and len(landmarks)<8000:landmarks.append(by_view[view][offset])
  (d/f'landmarks_{condition}.json').write_text(json.dumps(dict(landmarks=landmarks,diagnostics=diagnostics)))
  for r in [r for r in truth['records'] if r['role']=='query']:
   image=cv2.imread(str(d/r['image']));# fixed camera second exposure, no pose change
   second=np.clip(image.astype(float)*.94+1,0,255).astype('uint8');ok,jpg=cv2.imencode('.jpg',second,[cv2.IMWRITE_JPEG_QUALITY,92]);assert ok
   payload=dict(landmarks=landmarks,frames=[dict(frame_base64=b64((d/r['image']).read_bytes()),width=r['width'],height=r['height']),dict(frame_base64=b64(jpg.tobytes()),width=r['width'],height=r['height'])],intrinsics={'values':r['intrinsics']},**scan)
   if condition=='unknown_intrinsics':payload.pop('intrinsics')
   if condition=='noisy_depth_missing_boxes':payload['room_objects']=payload['room_objects'][::2]
   detections=d/(r['id']+'_detections.json')
   if detections.exists():payload['object_detections']=json.loads(detections.read_text())
   name=f'{d.name}_{r["id"]}_{condition}';p=root/'inputs'/f'{name}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(payload,separators=(',',':')))
   manifest.append(dict(case=name,layout=d.name,split=truth['split'],condition=condition,query=r['id'],input=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  print(d.name,condition,len(landmarks),flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
