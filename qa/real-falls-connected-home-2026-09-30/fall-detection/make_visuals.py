import json,pathlib,collections,cv2,numpy as np,subprocess
from PIL import Image,ImageDraw,ImageFont
R=pathlib.Path(__file__).parent
M={x['id']:x for x in json.load(open(R/'manifest_downloaded.json'))};S=json.load(open(R/'scores.json'))
models={}
for model in ['world','pose']:
 models[model]=collections.defaultdict(list)
 for split in ['development','evaluation']:
  for l in (R/f'inference_{model}_{split}.jsonl').read_text().splitlines():
   r=json.loads(l);models[model][r['clip']].append(r)
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';F=lambda n:ImageFont.truetype(font,n)
score={(x['clip'],x['method'],round(x['hz'],2)):x for x in S['clips']}
ids=['s2_fall_04','s3_fall_13','s3_fall_16','s4_fall_10','s4_adl_05','s3_adl_08']
W,H=1280,650;writer=cv2.VideoWriter(str(R/'pilot_diagnostics_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(W,H))
thumbs=[]
for cid in ids:
 c=M[cid];lefts=models['world'][cid];rights=models['pose'][cid]
 for idx,(a,b) in enumerate(zip(lefts,rights)):
  canvas=Image.new('RGB',(W,H),'#101b28');d=ImageDraw.Draw(canvas)
  d.text((20,10),f"REAL VIDEO PILOT  |  {cid}  |  {c['label']} label",font=F(23),fill='white')
  d.text((20,43),c['description'][:110],font=F(16),fill='#bccbda')
  for k,(r,model,title,color,method) in enumerate([(a,'world','ONE: YOLO-World + existing fall rule','#ffca58','world_baseline'),(b,'pose','YOLO11-pose boxes + same fall rule','#69d5ff','pose_box_baseline')]):
   im=Image.open(R/r['frame']);im=im.resize((r['width'],r['height']));dd=ImageDraw.Draw(im)
   for det in r['detections']:
    if det['label']!='person':continue
    box=det['bbox'];dd.rectangle(box,outline=color,width=3);dd.text((box[0],max(0,box[1]-16)),f"person {det['confidence']:.2f}",font=F(12),fill=color)
    if model=='pose':
     pts=det['keypoints']
     for x,y,cf in pts:
      if cf>=.5:dd.ellipse((x-2,y-2,x+2,y+2),fill=color)
     for i,j in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
      if pts[i][2]>=.5 and pts[j][2]>=.5:dd.line((pts[i][0],pts[i][1],pts[j][0],pts[j][1]),fill=color,width=2)
   im.thumbnail((620,450));canvas.paste(im,(k*640+(640-im.width)//2,112+(450-im.height)//2))
   d.text((k*640+16,78),title,font=F(17),fill=color)
   for rate,y in [(5,570),(.71,598)]:
    s=score[(cid,method,rate)];alert=any(t['time_s']<=r['time_s']+.01 for t in s['signals']);rate_name='5 Hz diagnostic' if rate==5 else '1.4 s sampling'
    d.text((k*640+16,y),f'{rate_name}: '+('ALERT' if alert else 'no alert')+f"   |  final {s['outcome']}",font=F(17),fill='#ff7f82' if alert else '#c0d0da')
  d.text((20,630),'GMDCSA-24 / Ekram Alam et al. (2024), MIT repository license | Staged clips; cold-start pilot, not safety validation',font=F(13),fill='#8ba3b5')
  writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
  if idx==len(lefts)//2:thumbs.append(canvas.copy())
writer.release();subprocess.run(['ffmpeg','-v','error','-y','-i',str(R/'pilot_diagnostics_raw.mp4'),'-c:v','libx264','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(R/'ONE-real-fall-pilot-diagnostics.mp4')],check=True)
board=Image.new('RGB',(1280,3*340),'#101b28')
for i,im in enumerate(thumbs):im.thumbnail((640,325));board.paste(im,((i%2)*640,(i//2)*340))
board.save(R/'ONE-real-fall-pilot-diagnostics.jpg',quality=95)
print('made visuals')
