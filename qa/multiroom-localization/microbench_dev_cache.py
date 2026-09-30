"""DEV-only speed/equivalence check of actual extracted descriptor arrays."""
import os,json,time,base64,cv2,numpy as np,ast
from pathlib import Path
from geometry_service import localization as new
from geometry_service.contracts import CameraLocalizationRequest
root=Path(__file__).parent;source=Path(os.environ.get('REFERENCE_ROOT','/workspace/shared/one-baseline'))/'geometry_service/localization.py';fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_pose_guided_matches');env=dict(vars(new));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env);old=env['_pose_guided_matches'];cv2.setNumThreads(1)
payload=CameraLocalizationRequest.model_validate_json((root/'inputs/dev_living_query_0_unknown_intrinsics.json').read_text());image=cv2.imdecode(np.frombuffer(base64.b64decode(payload.frames[0].frame_base64),np.uint8),cv2.IMREAD_COLOR);gray,_=new.prepare_feature_gray(image);keypoints,descriptors=cv2.ORB_create(nfeatures=3400,scaleFactor=1.2,nlevels=8,fastThreshold=5,edgeThreshold=17).detectAndCompute(gray,None);points,landmarks,_=new._landmark_arrays(payload)
# This ground truth is DEV-only, purely to generate plausible microbench poses.
gt=next(r for r in json.loads((root/'fixtures/dev_living/evaluator_groundtruth.json').read_text())['records'] if r['id']=='query_0');W=new._world_to_cv(np.asarray(gt['camera_to_world']));rvec,_=cv2.Rodrigues(W[:3,:3]);results=[]
new._SOLVER_THREAD_STATE.guided_descriptor_pairs=None
for index,offset in enumerate([0,.1,-.2,.35,.6]):
 pose=dict(camera_matrix=np.asarray(gt['intrinsics'],float),rvec=rvec+np.array([[0],[offset/8],[0]]),tvec=W[:3,3]+np.array([offset,0,0]));kw=dict(frame_width=480,frame_height=360,keypoints=keypoints,descriptors=descriptors,landmark_points=points,landmark_descriptors=landmarks,pose=pose,radius_px=72)
 start=time.perf_counter();a=old(**kw);before=time.perf_counter()-start;start=time.perf_counter();b=new._pose_guided_matches(**kw);after=time.perf_counter()-start;assert [(m.queryIdx,m.trainIdx,m.distance) for m in a]==[(m.queryIdx,m.trainIdx,m.distance) for m in b]
 results.append(dict(pose=index,old_seconds=before,new_seconds=after,exact_matches=len(a),speedup=before/after,cache_cold=index==0))
cache=new._SOLVER_THREAD_STATE.guided_descriptor_pairs;data=cache['query'].nbytes+cache['landmarks'].nbytes+sum(a.nbytes+b.nbytes for item in cache['pairs'] if item is not None for a,b in [item]);out=dict(development_only=True,queries=len(descriptors),landmarks=len(landmarks),retained_array_bytes=data,fallback_rows=sum(x is None for x in cache['pairs']),results=results);(root/'cache-microbenchmark.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
