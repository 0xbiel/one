"""Actual pretrained YOLO-World detections; never projected/oracle bounding boxes."""
import os,json,time,hashlib
from pathlib import Path
for key,value in dict(YOLO_CONFIG_DIR='/workspace/shared/one-models',ONE_GEOMETRY_MODEL_PATH='/workspace/shared/one-models/yolov8s-worldv2.pt',ONE_GEOMETRY_DEVICE='cpu',ONE_GEOMETRY_ALLOW_CPU='1',XDG_CACHE_HOME='/workspace/shared/one-models/cache').items():os.environ.setdefault(key,value)
import cv2,numpy as np,torch
import ultralytics.utils.torch_utils as tu
tu.NUM_THREADS=2;torch.set_num_threads(2)
from geometry_service.config import ServiceSettings
from geometry_service.runtime import RoomLayoutRuntime
runtime=RoomLayoutRuntime(ServiceSettings.from_env());assert runtime.ready,runtime.health()
root=Path(__file__).parent;diagnostics=[]
alias={'bed':['bed'],'chair':['chair'],'table':['table','desk','dining table'],'storage':['cabinet','shelf','bookcase','wardrobe','dresser','nightstand'],'sofa':['sofa','couch']}
for d in sorted((root/'fixtures').iterdir()):
 scan=json.loads((d/'scan.json').read_text());labels=list(dict.fromkeys(['person','window','mirror','glass door','sliding glass door']+[l for obj in scan['room_objects'] for l in alias.get(obj['label'],[obj['label']])]))[:32]
 for p in sorted(d.glob('query_*.jpg')):
  im=cv2.imread(str(p));second=np.clip(im.astype(float)*.94+1,0,255).astype('uint8');ok,jpg=cv2.imencode('.jpg',second,[cv2.IMWRITE_JPEG_QUALITY,92]);assert ok
  detections=[]
  for idx,payload in enumerate([p.read_bytes(),jpg.tobytes()]):
   torch.set_num_threads(2);start=time.perf_counter();raw=runtime.detect_jpeg(payload,im.shape[1],im.shape[0],labels,minimum_confidence=.10);detections.extend([{**r,'frame_index':idx} for r in raw]);diagnostics.append(dict(layout=d.name,query=p.stem,frame_index=idx,sha256=hashlib.sha256(payload).hexdigest(),labels=labels,detections=raw,seconds=time.perf_counter()-start));print(d.name,p.stem,idx,len(raw),flush=True)
  (d/(p.stem+'_detections.json')).write_text(json.dumps(detections,indent=2))
(root/'detector_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))

