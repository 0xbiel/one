"""Encode deterministic PNG timelines and create a compact labeled preview."""
from pathlib import Path
import json,subprocess
P=Path(__file__).parent/'assets';m=json.loads((P/'manifest.json').read_text())
for clip in m['clips']:
 d=P/clip['id']
 if len(list(d.glob('*.png')))!=m['frames_per_clip']:continue
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(m['fps']),'-i',str(d/'%04d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(P/clip['file'])],check=True)
