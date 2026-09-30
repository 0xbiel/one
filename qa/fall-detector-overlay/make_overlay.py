"""Draw recorded person predictions on their exact decoded source frames.
No model inference, ground-truth boxes, interpolation or invented detections.
"""
from pathlib import Path
import argparse,json,hashlib,subprocess
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--trace',required=True);p.add_argument('--out',default=str(Path(__file__).parent));a=p.parse_args()
O=Path(a.out);(O/'decoded').mkdir(parents=True,exist_ok=True);(O/'frames').mkdir(exist_ok=True)
rows=[json.loads(x) for x in Path(a.trace).read_text().splitlines() if x.strip()];rows=sorted([r for r in rows if r['clip_id']=='fall_side'],key=lambda r:r['frame_index'])
assert [r['frame_index'] for r in rows]==list(range(80))
assert all(abs(r['timestamp_s']-r['frame_index']/10)<1e-8 for r in rows)
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',a.video,'-start_number','0',str(O/'decoded/%04d.png')],check=True)
try:font=ImageFont.truetype('DejaVuSans.ttf',20);small=ImageFont.truetype('DejaVuSans.ttf',16)
except OSError:font=small=ImageFont.load_default()
no_person=[];event_count=0
for r in rows:
 i=r['frame_index'];im=Image.new('RGB',(640,576),'#10202d');frame=Image.open(O/'decoded'/f'{i:04d}.png').convert('RGB');assert frame.size==(640,480);im.paste(frame,(0,64));d=ImageDraw.Draw(im)
 person=[x for x in r['raw_detections'] if x['label'].strip().lower()=='person'];assert len(person)==r['person_detections']
 event_count+=sum(e.get('event_type')=='fall_suspected' for e in r['events'])
 d.text((14,7),f'ACTUAL DETECTOR OUTPUT   |   {r["timestamp_s"]:.1f} s',font=font,fill='#f2f7fa')
 status=(f'{len(person)} person prediction' if person else 'NO PERSON DETECTED')+f'   |   Fall signals: {event_count}'
 d.text((14,34),status,font=font,fill='#60e2e9' if person else '#ffbf69')
 if not person:no_person.append(r['timestamp_s'])
 for x in person:
  x1,y1,x2,y2=x['bbox'];assert all(isinstance(v,(float,int)) for v in [x1,y1,x2,y2]);d.rectangle((x1,y1+64,x2,y2+64),outline='#32f0ff',width=3)
  label=f'person {x["confidence"]:.2f}';box=d.textbbox((0,0),label,font=small);w=box[2]-box[0]+10;ly=max(64,y1+64-25);d.rectangle((x1,ly,x1+w,ly+25),fill='#10202d');d.text((x1+5,ly+3),label,font=small,fill='#32f0ff')
 d.text((14,552),'Recorded person predictions only. No ground-truth boxes.',font=small,fill='#c7d4df')
 im.save(O/'frames'/f'{i:04d}.png')
(O/'fall_side_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
summary={'source_video_sha256':hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),'source_full_trace_sha256':hashlib.sha256(Path(a.trace).read_bytes()).hexdigest(),'clip_id':'fall_side','fps':10,'frames':80,'person_prediction_frames':80-len(no_person),'no_person_prediction_frames':len(no_person),'first_no_person_prediction_s':min(no_person,default=None),'fall_signals':event_count,'overlay':'raw_detections filtered to label person; unchanged pixel coordinates with64px header offset','uses_ground_truth_boxes':False}
(O/'provenance.json').write_text(json.dumps(summary,indent=2))
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','10','-i',str(O/'frames/%04d.png'),'-vf','split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse','-loop','0',str(O/'actual_detector_side_fall.gif')],check=True)
print(json.dumps(summary,indent=2))
