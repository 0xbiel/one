"""Reference-only feature derivation; query truth kept in separate scoring process."""
import os,json,base64,hashlib,textwrap,cv2,numpy as np
from pathlib import Path
from geometry_service.localization import build_visual_landmarks
from geometry_service.contracts import VisualLandmarkBuildRequest,CameraLocalizationRequest
cv2.setNumThreads(1);root=Path(__file__).parent;d=root/'fixtures/connected_home';scan=json.loads((d/'scan.json').read_text());all_records=json.loads((d/'evaluator_groundtruth.json').read_text())['records'];refs=[r for r in all_records if r['role']=='reference'];queries=[r for r in all_records if r['role']=='query'];b64=lambda b:base64.b64encode(b).decode();groups={};diagnostics=[]
# Execute the actual pinned backend's per-view voxel merge block unchanged.
source_root=Path(os.environ.get('ONE_SOURCE_ROOT',str(root.parents[2])))
source=(source_root/'app/main.py').read_text();start=source.index('        merged_landmarks = previous_landmarks');end=source.index('        source_frame_count =',start);merge_code=compile(textwrap.dedent(source[start:end]),'app/main.py:2409-2445','exec')
def merge_native(previous_landmarks,landmarks):
 ns={'previous_landmarks':previous_landmarks,'landmarks':landmarks};exec(merge_code,{},ns);return ns['merged_landmarks']
merged=[]
# Only reference scan transforms/K/depth used to construct features.
for r in refs:
 depth=np.load(d/r['depth']);frame=dict(frame_base64=b64((d/r['image']).read_bytes()),width=r['width'],height=r['height'],intrinsics={'values':r['intrinsics']},camera_to_world={'values':r['camera_to_world']},depth_base64=b64(depth.astype('<f4').tobytes()),depth_width=depth.shape[1],depth_height=depth.shape[0]);result=build_visual_landmarks(VisualLandmarkBuildRequest(map_id='connected_home',frames=[frame]));groups[r['id']]=[{**l,'view_id':f'scan-view-{len(groups)+1}'} for l in result['landmarks']];merged=merge_native(merged,groups[r['id']]);diagnostics.append(dict(reference=r['id'],diagnostics=result['diagnostics'],features=len(groups[r['id']])));print(r['id'],len(groups[r['id']]),flush=True)
landmarks=merged
(root/'reference_landmarks.json').write_text(json.dumps(dict(landmarks=landmarks,diagnostics=diagnostics),separators=(',',':')))
if os.environ.get('FEATURES_ONLY')=='1':raise SystemExit(0)
# Explicit query accessor avoids ever forwarding world poses or GT room labels.
manifest=[];truth=[]
for r in queries:
 image=cv2.imread(str(d/r['image']));second=np.clip(image.astype(float)*.94+1,0,255).astype('uint8');ok,jpg=cv2.imencode('.jpg',second,[cv2.IMWRITE_JPEG_QUALITY,92]);assert ok
 for condition in ['clean','unknown_intrinsics']:
  payload=dict(landmarks=landmarks,frames=[dict(frame_base64=b64((d/r['image']).read_bytes()),width=r['width'],height=r['height']),dict(frame_base64=b64(jpg.tobytes()),width=r['width'],height=r['height'])],**scan)
  if condition=='clean':payload['intrinsics']={'values':r['intrinsics']}
  det=d/(r['id']+'_detections.json');assert det.exists(),det;payload['object_detections']=json.loads(det.read_text());CameraLocalizationRequest.model_validate(payload)
  name=r['id']+'_'+condition;p=root/'inputs'/f'{name}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(payload,separators=(',',':')))
  manifest.append(dict(case=name,split=r['split'],condition=condition,query=r['id'],input=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 truth.append({k:r[k] for k in ['id','room','split','camera_to_world']})
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'scoring_only.json').write_text(json.dumps(truth,indent=2));print('FROZEN',len(manifest),len(landmarks),flush=True)
