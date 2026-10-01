import os,sys,json,pathlib,time,argparse
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'one'))
os.environ.update(ONE_GEOMETRY_MODEL_PATH=str(R/'models/yolov8s-worldv2.pt'),ONE_GEOMETRY_MODEL_CONFIG=str(R/'one/geometry_service/model_config.yolo-world.json'),ONE_GEOMETRY_DEVICE='cpu',ONE_GEOMETRY_ALLOW_CPU='1',YOLO_CONFIG_DIR=str(R/'models'))
import torch,cv2
from geometry_service.config import ServiceSettings
from geometry_service.runtime import RoomLayoutRuntime
p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--mode',default='world');a=p.parse_args()
import ultralytics.utils.torch_utils as tu
tu.NUM_THREADS=2
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
frames=json.load(open(R/'frames.json'));frames=[f for f in frames if f['split']==a.split]
labels=['person','keys','glasses','mobile phone','remote control','cup','bottle','book','medication box','cane','walker']
if a.mode=='world':
 model=RoomLayoutRuntime(ServiceSettings.from_env());print(model.health(),flush=True);assert model.ready,model.reason
 # Memoize identical CLIP prompt embeddings only; compare numerical outputs on first real frame.
 first=frames[0];im0=cv2.imread(str(R/first['frame']));h0,w0=im0.shape[:2];sc=min(1.,640/max(h0,w0));im0=cv2.resize(im0,(round(w0*sc),round(h0*sc)));h0,w0=im0.shape[:2];jpg0=cv2.imencode('.jpg',im0,[cv2.IMWRITE_JPEG_QUALITY,58])[1].tobytes()
 before=model.detect_jpeg(jpg0,w0,h0,labels);orig=model._model.model.get_text_pe;cache={}
 def cached(text,*args,**kwargs):
  key=tuple(text)
  if key not in cache:cache[key]=orig(text,*args,**kwargs)
  return cache[key]
 model._model.model.get_text_pe=cached
 after=model.detect_jpeg(jpg0,w0,h0,labels);assert before==after,(before,after)
 json.dump(dict(equal=True,before=before,after=after,change='Only identical CLIP prompt embeddings memoized; production runtime source unchanged'),open(R/f'cache_equivalence_{a.split}.json','w'),indent=2)
else:
 from ultralytics import YOLO
 model=YOLO(str(R/'models/yolo11n-pose.pt'))
from geometry_service.low_light import enhance_low_light_image
out=R/f'inference_{a.mode}_{a.split}.jsonl';done=set()
if out.exists():
 for l in out.read_text().splitlines():done.add(json.loads(l)['frame'])
with out.open('a') as f:
 for n,item in enumerate(frames):
  if item['frame'] in done:continue
  im=cv2.imread(str(R/item['frame']));h,w=im.shape[:2];scale=min(1.,640/max(h,w));im=cv2.resize(im,(round(w*scale),round(h*scale)));h,w=im.shape[:2];jpeg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,58])[1].tobytes();im=cv2.imdecode(__import__('numpy').frombuffer(jpeg,dtype='uint8'),cv2.IMREAD_COLOR);start=time.perf_counter()
  if a.mode=='world':dets=model.detect_jpeg(jpeg,w,h,labels)
  else:
   im,_=enhance_low_light_image(im);r=model.predict(im,device='cpu',imgsz=(384,640),conf=.20,max_det=100,verbose=False)[0];dets=[]
   for b,k in zip(r.boxes,r.keypoints.data):dets.append(dict(label='person',confidence=float(b.conf[0]),bbox=b.xyxy[0].tolist(),keypoints=k.tolist()))
  elapsed=time.perf_counter()-start;f.write(json.dumps(dict(**item,width=w,height=h,detections=dets,inference_seconds=elapsed))+'\n');f.flush()
  if n%25==0:print(a.mode,a.split,n,'/',len(frames),item['clip'],round(elapsed,3),flush=True)
print('DONE',out,flush=True)
