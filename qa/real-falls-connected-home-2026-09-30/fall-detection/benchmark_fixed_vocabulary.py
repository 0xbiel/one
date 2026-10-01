"""Predeclared diagnostic: production runtime vs fixed-vocabulary offline inference.
No production repository edits, training, thresholds, or evaluation-frame selection.
"""
import os, sys, json, time, pathlib, statistics, hashlib
from datetime import datetime, timezone

R=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(R/'one'))
os.environ.update(ONE_GEOMETRY_MODEL_PATH=str(R/'models/yolov8s-worldv2.pt'),ONE_GEOMETRY_MODEL_CONFIG=str(R/'one/geometry_service/model_config.yolo-world.json'),ONE_GEOMETRY_DEVICE='cpu',ONE_GEOMETRY_ALLOW_CPU='1',YOLO_CONFIG_DIR=str(R/'models'))
import torch, cv2, numpy as np
import ultralytics, ultralytics.utils.torch_utils as tu
tu.NUM_THREADS=2;torch.set_num_threads(2);torch.set_num_interop_threads(1)
from geometry_service.config import ServiceSettings
from geometry_service.runtime import RoomLayoutRuntime
from geometry_service.real_vision import decode_jpeg,detections_from_result
from geometry_service.low_light import enhance_low_light_image

LABELS=['person','keys','glasses','mobile phone','remote control','cup','bottle','book','medication box','cane','walker']
SELECTION=[('s1_fall_03',0.0),('s1_fall_05',2.0),('s1_fall_09',2.0),('s1_adl_08',3.0),('s1_adl_15',4.0)]
frames=json.loads((R/'frames.json').read_text())
selected=[min((f for f in frames if f['split']=='development' and f['clip']==clip),key=lambda f:abs(f['time_s']-t)) for clip,t in SELECTION]
declaration={'declared_at_utc':datetime.now(timezone.utc).isoformat(),'selection':selected,'timing_frame':selected[0],'timing_repetitions':10,'warmup_each_path':2,'threads':2,'scope':'CPU offline diagnostic only; simultaneous other jobs may affect latency'}
(R/'fixed_vocabulary_predeclared.json').write_text(json.dumps(declaration,indent=2))
def prep(item):
 im=cv2.imread(str(R/item['frame']));h,w=im.shape[:2];s=min(1.,640/max(h,w));im=cv2.resize(im,(round(w*s),round(h*s)));h,w=im.shape[:2]
 jpeg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,58])[1].tobytes()
 return jpeg,w,h
inputs=[prep(f) for f in selected]
runtime=RoomLayoutRuntime(ServiceSettings.from_env());assert runtime.ready,runtime.reason
original=runtime._model.model.get_text_pe;cache={}
def cached(text,*args,**kwargs):
 key=tuple(text)
 if key not in cache:cache[key]=original(text,*args,**kwargs)
 return cache[key]
runtime._model.model.get_text_pe=cached

def baseline(jpg,w,h):return runtime.detect_jpeg(jpg,w,h,LABELS)
reference=[baseline(*x) for x in inputs]
def timed(fn):
 for _ in range(2):fn(*inputs[0])
 times=[]
 for _ in range(10):
  start=time.perf_counter();fn(*inputs[0]);times.append(time.perf_counter()-start)
 return {'seconds':times,'mean_seconds':statistics.mean(times),'median_seconds':statistics.median(times),'p95_seconds':float(np.percentile(times,95)),'reciprocal_mean_fps':1/statistics.mean(times)}
baseline_timing=timed(baseline)
# This is valid only while the prompt vocabulary, device, and weights stay fixed.
# Decode/enhance/predict/parse are unchanged. Skip redundant per-frame prompt resets.
runtime._model.set_classes(LABELS)
def fixed(jpg,w,h):
 im=decode_jpeg(jpg,w,h);im,_=enhance_low_light_image(im)
 rr=runtime._model.predict(source=im,device='cpu',imgsz=(384,640),conf=.20,max_det=100,verbose=False)
 return detections_from_result(rr[0],width=w,height=h,minimum_confidence=.20) if rr else []
candidate=[fixed(*x) for x in inputs]
comparison=[{'frame':f,'equal':a==b,'baseline':a,'fixed':b} for f,a,b in zip(selected,reference,candidate)]
fixed_timing=timed(fixed)
report=dict(declaration,completed_at_utc=datetime.now(timezone.utc).isoformat(),exact_equal_all=all(x['equal'] for x in comparison),comparison=comparison,baseline_timing=baseline_timing,fixed_timing=fixed_timing,labels=LABELS,model_input=[384,640],confidence=.20,max_det=100,model_sha256=hashlib.sha256((R/'models/yolov8s-worldv2.pt').read_bytes()).hexdigest(),versions={'torch':torch.__version__,'ultralytics':ultralytics.__version__,'opencv':cv2.__version__},caveats=['Only five predeclared development frames checked; no universal numerical-equivalence proof','Baseline timings include exact-vocabulary CLIP embedding memoization already used by evaluation harness','Both timing paths include JPEG decode, identical low-light enhancement, image inference, and identical parsed output; exclude file read, resize, and JPEG encode','No camera capture, HTTP, tracking, persistence, UI, or multi-camera scheduling measured','CPU timings are not MPS/CUDA deployment latency; shared CPU contention may affect them','Fixed vocabulary must not be used when candidate labels change unless rebuilt'])
(R/'fixed_vocabulary_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
