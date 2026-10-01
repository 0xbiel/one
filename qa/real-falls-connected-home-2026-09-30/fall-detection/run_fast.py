import os,sys,json,pathlib,time,argparse
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'one'))
os.environ.update(YOLO_CONFIG_DIR=str(R/'models'),XDG_CACHE_HOME=str(R/'cache'))
import torch,cv2,numpy as np
import ultralytics.utils.torch_utils as tu
tu.NUM_THREADS=2;torch.set_num_threads(2);torch.set_num_interop_threads(1)
from ultralytics import YOLOWorld,YOLO
from geometry_service.low_light import enhance_low_light_image
from geometry_service.real_vision import detections_from_result
p=argparse.ArgumentParser();p.add_argument('--split',default='evaluation');p.add_argument('--mode',default='world');p.add_argument('--frames',default='full_frames.json');p.add_argument('--prefix',default='full_');a=p.parse_args()
frames=[x for x in json.load(open(R/a.frames)) if x['split']==a.split]
labels=['person','keys','glasses','mobile phone','remote control','cup','bottle','book','medication box','cane','walker']
model=YOLOWorld(str(R/'models/yolov8s-worldv2.pt')) if a.mode=='world' else YOLO(str(R/'models/yolo11n-pose.pt'))
if a.mode=='world':model.set_classes(labels)
model.to('cpu')
out=R/f'{a.prefix}inference_{a.mode}_{a.split}.jsonl';done=set()
if out.exists():
 for l in out.read_text().splitlines():done.add(json.loads(l)['frame'])
with out.open('a') as f:
 for n,item in enumerate(frames):
  if item['frame'] in done:continue
  im=cv2.imread(str(R/item['frame']));h,w=im.shape[:2];sc=min(1.,640/max(h,w));im=cv2.resize(im,(round(w*sc),round(h*sc)));h,w=im.shape[:2]
  jpeg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,58])[1].tobytes();im=cv2.imdecode(np.frombuffer(jpeg,dtype='uint8'),cv2.IMREAD_COLOR)
  start=time.perf_counter();im,light=enhance_low_light_image(im);res=model.predict(im,device='cpu',imgsz=(384,640),conf=.2,max_det=100,verbose=False)[0]
  if a.mode=='world':dets=detections_from_result(res,width=w,height=h,minimum_confidence=.2)
  else:dets=[dict(label='person',confidence=float(b.conf[0]),bbox=b.xyxy[0].tolist(),keypoints=k.tolist()) for b,k in zip(res.boxes,res.keypoints.data)]
  elapsed=time.perf_counter()-start;f.write(json.dumps(dict(**item,width=w,height=h,detections=dets,inference_seconds=elapsed,low_light=light['low_light'],fixed_vocabulary_offline=True))+'\n');f.flush()
  if n%100==0:print(a.prefix,a.mode,a.split,n,'/',len(frames),item['clip'],round(elapsed,3),flush=True)
print('DONE',out,flush=True)
