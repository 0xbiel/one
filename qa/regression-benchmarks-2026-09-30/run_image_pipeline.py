"""Real RGB inference -> production API pipeline, with deterministic replay clock.
Never reads ground-truth boxes. CPU processing speed does not change capture time.
"""
import argparse, os, json, time, hashlib, statistics, tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--clips',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True);p.add_argument('--expected-fps',type=float,default=10);a=p.parse_args()
os.environ.setdefault('YOLO_CONFIG_DIR','/workspace/shared/one-models')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/shared/one-models/cache')
import cv2, torch, base64
import ultralytics.utils.torch_utils as tu
tu.NUM_THREADS=2;torch.set_num_threads(2);torch.set_num_interop_threads(1)
from geometry_service.config import ServiceSettings
from geometry_service.runtime import RoomLayoutRuntime
from app.config import Settings
import app.main as main
import app.vision as vision
from fastapi.testclient import TestClient
runtime=RoomLayoutRuntime(ServiceSettings.from_env())
if not runtime.ready:raise RuntimeError(runtime.reason)
class Clock(datetime):
 current=datetime(2026,9,30,12,tzinfo=timezone.utc)
 @classmethod
 def now(cls,tz=None):return cls.current if tz else cls.current.replace(tzinfo=None)
vision.datetime=Clock
class RealService:
 last=[]
 cache={}
 calls=0
 def detect(self,*,frame_base64,width,height,candidate_labels,include_faces=False):
  torch.set_num_threads(2)
  frame_bytes=base64.b64decode(frame_base64)
  key=(hashlib.sha256(frame_bytes).hexdigest(),tuple(candidate_labels),width,height)
  if key not in self.cache:
   self.cache[key]=runtime.detect_jpeg(frame_bytes,width,height,candidate_labels)
   self.calls+=1
  self.last=self.cache[key]
  return {'status':'ready','model_version':runtime.model_version,'detections':self.last}
service=RealService();output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
latencies=[];hashes={}
with output.open('w') as log, tempfile.TemporaryDirectory() as tmp:
 for clip in a.clips:
  path=Path(a.assets)/(clip+'.mp4');hashes[clip]=hashlib.sha256(path.read_bytes()).hexdigest()
  main.datetime=datetime
  c=TestClient(main.make_app(Settings(database_url='sqlite:///:memory:',object_store_path=Path(tmp)/clip,bootstrap_secret='test',env='test',lm_studio_url='http://127.0.0.1:9/v1'),vision_detector=vision.LocalServiceDetector(service)))
  main.datetime=Clock
  started=c.post('/api/v1/pairing/start',json={'display_name':'Synthetic','home_name':'Synthetic benchmark'}).json()
  auth=c.post('/api/v1/pairing/complete',json={'code':started['pairing_code']}).json();home=auth['home_id'];h={'Authorization':'Bearer '+auth['access_token']}
  camera=c.post(f'/api/v1/homes/{home}/cameras',headers=h,json={'name':clip}).json()['id']
  c.post(f'/api/v1/homes/{home}/consents',headers=h,json={'purpose':'video_capture','policy_version':'2026-09-01','granted':True})
  cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);i=0
  if abs(fps-a.expected_fps)>1e-6:raise ValueError('Clip FPS does not match declared benchmark FPS')
  while True:
   ok,image=cap.read()
   if not ok:break
   Clock.current=datetime(2026,9,30,12,tzinfo=timezone.utc)+timedelta(seconds=i/fps)
   _,data=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95]);start=time.perf_counter()
   response=c.post(f'/api/v1/homes/{home}/vision/frames',headers=h,json={'camera_id':camera,'frame_base64':base64.b64encode(data).decode(),'width':image.shape[1],'height':image.shape[0],'captured_at':Clock.current.isoformat(),'candidate_labels':['person']})
   duration=time.perf_counter()-start;latencies.append(duration)
   if response.status_code!=200:raise RuntimeError(response.text)
   result=response.json();row={'clip_id':clip,'frame_index':i,'timestamp_s':i/fps,'person_detections':sum(d.get('label')=='person' for d in service.last),'stable_person_tracks':sum(d.get('label')=='person' for d in result['data']),'events':[dict(e,event_type=e['type']) for e in result['safety_events']],'raw_detections':service.last,'stable_detections':result['data'],'elapsed_s':duration}
   log.write(json.dumps(row)+'\n');log.flush()
   if i%10==0:print(clip,i,round(duration,3),len(service.last),len(result['safety_events']),flush=True)
   i+=1
  cap.release();print('COMPLETE',clip,i,flush=True)
meta={'code_commit':a.commit,'model_name':runtime.model_version,'model_sha256':hashlib.sha256(runtime.settings.checkpoint_path.read_bytes()).hexdigest(),'input_kind':'rendered_rgb_frames','inference_fps':a.expected_fps,'memoized_harness_processing_fps':len(latencies)/sum(latencies),'memoized_harness_latency_median_s':statistics.median(latencies),'memoized_harness_latency_p95_s':sorted(latencies)[int(.95*(len(latencies)-1))],'clip_sha256':hashes,'clock':'production clocks use decoded clip FPS capture timestamps to separate playback from CPU throughput','thresholds':{'detection':.2,'fall_min_confidence':.55,'stability_hits':3,'fall_confirmation_hits':3},'inference_calls':service.calls,'replayed_frames':len(latencies),'memoization':'only exact JPEG bytes + dimensions + labels; every frame still traverses production tracker/API','device':'cpu','threads':2,'image_input':'decoded MP4 frames, re-encoded JPEG quality95; no annotation inputs'}
output.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
