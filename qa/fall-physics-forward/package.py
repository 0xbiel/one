from pathlib import Path
import json,hashlib,zipfile
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).parent;A=P/'assets'
try:font=ImageFont.truetype('DejaVuSans.ttf',20)
except OSError:font=ImageFont.load_default()
out=Image.new('RGB',(1280,1060),'#12212b');d=ImageDraw.Draw(out)
for i,(f,title) in enumerate([('0000.png','SUPPORTED INITIAL POSE | 0.0 s'),('0019.png','GRAVITY-DRIVEN COLLAPSE | 1.58 s'),('0026.png','FORWARD FLOOR LANDING | 2.17 s'),('0071.png','SETTLED | 5.92 s')]):
 x=(i%2)*640;y=(i//2)*530;out.paste(Image.open(A/'frames'/f),(x,y+50));d.text((x+15,y+15),title,fill='white',font=font)
out.save(A/'forward_physics_contact_sheet.jpg',quality=92)
files=[P/n for n in ['README.md','generate_physics.py','room_base.py','render_replay.py','verify_simulation.py','encode.py','package.py']]+[A/n for n in ['fall_physics_forward.mp4','fall_physics_forward_preview.gif','forward_physics_contact_sheet.jpg','physics_source.blend','physics_replay.blend','physics_samples_24fps.json','ground_truth_frames.json','manifest.json','validation.json','video_validation.json']]
(A/'sha256.json').write_text(json.dumps({str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2));files.append(A/'sha256.json')
with zipfile.ZipFile(P/'one_forward_physics_fall_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(P)))
print('Packaged',P/'one_forward_physics_fall_bundle.zip')
