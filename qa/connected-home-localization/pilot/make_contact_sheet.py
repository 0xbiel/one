from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
import json
r=Path(__file__).parent;d=r/'fixtures/connected_home';truth=json.loads((d/'evaluator_groundtruth.json').read_text());qs=[x for x in truth['records'] if x['role']=='query'];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',25)
w,h=1440,1670;im=Image.new('RGB',(w,h),'#eef1f4');dr=ImageDraw.Draw(im);dr.text((22,18),'Connected home: 12 distinct camera views',font=bold,fill='#182638');dr.text((22,55),'Synthetic RGB renders • four similar rooms + hallway • full-home localization inputs',font=font,fill='#40566a')
for i,q in enumerate(qs):
 x=(i%3)*480;y=92+(i//3)*390;photo=Image.open(d/q['image']).convert('RGB');im.paste(photo,(x,y));label=q.get('kind',q['room']).replace('_',' ');dr.text((x+9,y+361),f"{i+1:02d}  {label}  |  {q['split']}",font=small,fill='#182638')
dr.text((22,h-20),'Each query has two exposures at one fixed pose. Room truth is the camera-center room; it is never supplied to the solver.',font=small,fill='#40566a');im.save(r/'camera_samples.png')
