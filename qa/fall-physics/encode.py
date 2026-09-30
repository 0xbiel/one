from pathlib import Path
import subprocess,json,hashlib
P=Path(__file__).resolve().parent;O=P/'assets';frames=sorted((O/'frames').glob('*.png'))
assert len(frames)==72,f'Expected 72 frames, found {len(frames)}'
out=O/'fall_physics_ragdoll.mp4'
subprocess.run(['ffmpeg','-y','-framerate','12','-i',str(O/'frames'/'%04d.png'),'-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(out)]))
assert probe['streams'][0]['nb_frames']=='72';assert abs(float(probe['format']['duration'])-6)<.001
(O/'video_validation.json').write_text(json.dumps({'probe':probe,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))
subprocess.run(['ffmpeg','-y','-i',str(out),'-vf','fps=8,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse','-loop','0',str(O/'fall_physics_preview.gif')],check=True)
print(out)
