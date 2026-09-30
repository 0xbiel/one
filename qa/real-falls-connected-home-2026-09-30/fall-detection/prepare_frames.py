import pathlib,json,subprocess
from PIL import Image,ImageDraw
R=pathlib.Path(__file__).parent
m=json.load(open(R/'manifest_downloaded.json'))
frames=[];thumbs=[]
for c in m:
 p=R/'dataset'/c['path'];d=R/'frames'/c['id'];d.mkdir(parents=True,exist_ok=True)
 v=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,duration,nb_frames','-of','json',str(p)]))['streams'][0]
 c['video']=v
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(p),'-vf','fps=5','-q:v','2',str(d/'%05d.jpg')],check=True)
 paths=sorted(d.glob('*.jpg'));c['sampled_frames']=len(paths)
 # ffmpeg fps selects at the center of each 0.2 second interval; use nominal bin times.
 for i,fp in enumerate(paths):frames.append(dict(clip=c['id'],split=c['split'],time_s=i/5,frame=str(fp.relative_to(R))))
 for i in [0,len(paths)//2,len(paths)-1]:
  im=Image.open(paths[i]);im.thumbnail((256,160));card=Image.new('RGB',(256,188),'#17212b');card.paste(im,(0,20));ImageDraw.Draw(card).text((4,3),f'{c["id"]} {i/5:.1f}s',fill='white');thumbs.append(card)
json.dump(m,open(R/'manifest_downloaded.json','w'),indent=2);json.dump(frames,open(R/'frames.json','w'),indent=2)
for s in range(4):
 canvas=Image.new('RGB',(768,8*188),'white')
 for i,im in enumerate(thumbs[s*24:(s+1)*24]):canvas.paste(im,((i%3)*256,(i//3)*188))
 canvas.save(R/f'contact_subject_{s+1}.jpg')
print('frames',len(frames))
