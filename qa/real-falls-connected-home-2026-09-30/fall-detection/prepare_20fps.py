import json,pathlib,subprocess
R=pathlib.Path(__file__).parent;M={c['id']:c for c in json.load(open(R/'manifest_downloaded.json'))}
ids=['s1_fall_05','s1_fall_09','s2_fall_04','s4_fall_10','s3_fall_13','s4_fall_11','s1_adl_15','s2_adl_04','s2_adl_12','s3_adl_08','s4_adl_05','s4_adl_10']
(R/'cadence20_predeclared.json').write_text(json.dumps(dict(timestamp_utc='2026-09-30T13:31:30Z',selection='Metadata-stratified before20Hz inference. Six falls: side/front/back/chair/seated/partialbody; original night clip s2_fall_12 excluded because native14.985fps, replaced before20Hz inference by chair fall s4_fall_10. Six controls: bending/yoga/bed/floorlying/pushups/bed. No heldout outcome-based selection.',clips=ids,sampling='Native video to20Hz without interpolation; paired5Hz and1.4s use same20Hz cache; no claim20Hz clinicalvalidation or deploymentthroughput'),indent=2))
frames=[];manifest=[]
for cid in ids:
 c=M[cid];v=c['video'];num,den=map(int,v['r_frame_rate'].split('/'));assert num/den>=20,(cid,v)
 p=R/'dataset'/c['path'];d=R/'frames20'/cid;d.mkdir(parents=True,exist_ok=True)
 
 if not list(d.glob('*.jpg')):subprocess.run(['ffmpeg','-v','error','-y','-i',str(p),'-vf','fps=20','-q:v','2',str(d/'%05d.jpg')],check=True)
 paths=sorted(d.glob('*.jpg'));c['sampled_frames']=len(paths);manifest.append(c)
 for i,fp in enumerate(paths):frames.append(dict(clip=cid,split=c['split'],time_s=i/20,frame=str(fp.relative_to(R))))
json.dump(frames,open(R/'frames20.json','w'),indent=2);json.dump(manifest,open(R/'manifest20.json','w'),indent=2);print('native20fpsframes',len(frames),flush=True)
