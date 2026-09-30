from pathlib import Path
import hashlib,json,zipfile,subprocess
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).parent;A=P/'assets';m=json.loads((A/'manifest.json').read_text());checks={};panels=[]
try:font=ImageFont.truetype('DejaVuSans.ttf',20)
except OSError:font=ImageFont.load_default()
for c in m['clips']:
 f=A/c['file'];probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(f)]))['streams'][0]
 assert int(probe['nb_read_frames'])==80 and probe['width']==640 and probe['height']==480 and probe['r_frame_rate']=='10/1',(c['id'],probe)
 assert len(json.loads((A/c['id']/'ground_truth_frames.json').read_text()))==80
 checks[c['file']]=hashlib.sha256(f.read_bytes()).hexdigest()
 panel=Image.new('RGB',(640,530),'#12212b');panel.paste(Image.open(A/c['id']/('0035.png' if c['id']=='bend' else '0065.png')),(0,50));d=ImageDraw.Draw(panel);d.text((15,12),c['id'].replace('_',' ').upper()+(' | FALL' if c['expected_fall'] else ' | NON-FALL CONTROL'),fill='white',font=font);panels.append(panel)
board=Image.new('RGB',(1280,530*4),'#12212b')
for i,x in enumerate(panels):board.paste(x,((i%2)*640,(i//2)*530))
ImageDraw.Draw(board).text((660,1620),'3D synthetic inputs only\n10 fps | 640 x 480 | 8 seconds\nDetector results reported separately',fill='white',font=font);board.save(A/'scenario_contact_sheet.jpg',quality=90)
(A/'sha256.json').write_text(json.dumps(checks,indent=2))
with zipfile.ZipFile(P/'ONE-fall-animation-fixtures.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['README.md','generate_animations.py','room_base.py','encode.py','score_results.py','test_score_results.py','package.py']:z.write(P/name,name)
 for name in ['manifest.json','sha256.json','scenario_contact_sheet.jpg','animated_room.blend']:z.write(A/name,'assets/'+name)
 for c in m['clips']:
  z.write(A/c['file'],'assets/'+c['file']);z.write(A/c['id']/'ground_truth_frames.json','assets/'+c['id']+'/ground_truth_frames.json')
print(json.dumps(checks,indent=2));print('All seven encoded clips verified, packaged')
