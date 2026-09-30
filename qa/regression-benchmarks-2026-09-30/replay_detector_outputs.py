"""Replay recorded image-model outputs through candidate production API.
This is a paired downstream regression comparison, not fresh image inference.
Ground truth is never loaded. Frozen source video hashes must match provenance.
"""
import argparse,base64,json,hashlib,tempfile
from pathlib import Path
from datetime import datetime,timezone,timedelta
import cv2
from fastapi.testclient import TestClient
from app.config import Settings
import app.main as main
import app.vision as vision
p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True);a=p.parse_args()
source=Path(a.results);rows=[json.loads(s) for s in source.read_text().splitlines()];meta=json.loads(source.with_suffix('.metadata.json').read_text())
class Clock(datetime):
 current=datetime(2026,9,30,12,tzinfo=timezone.utc)
 @classmethod
 def now(cls,tz=None):return cls.current if tz else cls.current.replace(tzinfo=None)
vision.datetime=Clock
class RecordedImageDetector:
 model_version=meta['model_name']+'-recorded-output-replay'
 current=[]
 def detect(self,frame,candidate_labels):
  return [vision.Detection(d['label'],d['confidence'],tuple(d['bbox']),frame.captured_at) for d in self.current]
detector=RecordedImageDetector();out=[]
with tempfile.TemporaryDirectory() as tmp:
 for clip in dict.fromkeys(r['clip_id'] for r in rows):
  path=Path(a.assets)/(clip+'.mp4')
  assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['clip_sha256'][clip], 'Source video changed'
  main.datetime=datetime
  c=TestClient(main.make_app(Settings(database_url='sqlite:///:memory:',object_store_path=Path(tmp)/clip,bootstrap_secret='test',env='test',lm_studio_url='http://127.0.0.1:9/v1'),vision_detector=detector));main.datetime=Clock
  start=c.post('/api/v1/pairing/start',json={'display_name':'Synthetic','home_name':'Replay'}).json();auth=c.post('/api/v1/pairing/complete',json={'code':start['pairing_code']}).json();home=auth['home_id'];h={'Authorization':'Bearer '+auth['access_token']}
  camera=c.post(f'/api/v1/homes/{home}/cameras',headers=h,json={'name':clip}).json()['id'];c.post(f'/api/v1/homes/{home}/consents',headers=h,json={'purpose':'video_capture','policy_version':'2026-09-01','granted':True})
  cap=cv2.VideoCapture(str(path));cliprows=[r for r in rows if r['clip_id']==clip]
  for original in cliprows:
   ok,img=cap.read();assert ok
   Clock.current=datetime(2026,9,30,12,tzinfo=timezone.utc)+timedelta(seconds=original['timestamp_s']);_,data=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,95]);detector.current=original['raw_detections']
   r=c.post(f'/api/v1/homes/{home}/vision/frames',headers=h,json={'camera_id':camera,'frame_base64':base64.b64encode(data).decode(),'width':img.shape[1],'height':img.shape[0],'captured_at':Clock.current.isoformat(),'candidate_labels':['person']});assert r.status_code==200,r.text
   result=r.json();out.append(dict(original,person_detections=sum(d['label']=='person' for d in detector.current),stable_person_tracks=sum(d['label']=='person' for d in result['data']),events=[dict(e,event_type=e['type']) for e in result['safety_events']],stable_detections=result['data'],evaluation_kind='recorded_real_image_detector_outputs_through_candidate_API'))
  cap.release()
Path(a.output).write_text(''.join(json.dumps(r)+'\n' for r in out))
meta.update(code_commit=a.commit,input_kind='recorded_real_image_predictions',source_inference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),comparison='Exact same real-model detections; fresh downstream API/tracker/event execution. Not independent second image-model inference.')
Path(a.output).with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2));print('Replayed',len(out),'frames')
