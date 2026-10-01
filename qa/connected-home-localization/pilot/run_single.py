"""Exclusive solver subprocess. No locks, no cache reuse, no ground truth reads."""
import json,sys,time,hashlib,os,faulthandler
from pathlib import Path
import numpy as np,cv2,torch
np.random.seed(104729);cv2.setRNGSeed(104729);cv2.setNumThreads(1);torch.manual_seed(104729);torch.set_num_threads(1)
from geometry_service.localization import localize_camera, _pose_scene_prior
from geometry_service.contracts import CameraLocalizationRequest
p=Path(sys.argv[1]);out=Path(sys.argv[2]);source=Path(os.environ['PYTHONPATH'])/'geometry_service'
fp={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(source.glob('*.py'))}
metadata=dict(input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),geometry_fingerprint=fp,geometry_fingerprint_sha256=hashlib.sha256(json.dumps(fp,sort_keys=True).encode()).hexdigest(),compute_started_monotonic=time.monotonic(),pid=os.getpid())
request=CameraLocalizationRequest.model_validate_json(p.read_text());start=time.monotonic();metadata['compute_started_monotonic']=start
out.with_suffix('.started.json').write_text(json.dumps(metadata,indent=2));faulthandler.dump_traceback_later(60,repeat=True)
result=localize_camera(request,progress_callback=lambda percent,stage:print(f'{time.monotonic()-start:.3f}s {percent}% {stage}',flush=True));faulthandler.cancel_dump_traceback_later()

if result.get('status')=='positioned' and result.get('camera_to_world') is not None:result['final_pose_scene_prior']=_pose_scene_prior(np.asarray(result['camera_to_world'],dtype=float),request)
result.update(metadata);result['wall_seconds']=time.monotonic()-start;out.write_text(json.dumps(result,indent=2));print(result['status'],flush=True)

