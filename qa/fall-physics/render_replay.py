import bpy,sys
from pathlib import Path
P=Path(__file__).resolve().parent/'assets';s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=16;s.cycles.use_denoising=False;s.render.threads_mode='FIXED';s.render.threads=4
if '--eevee' in sys.argv:s.render.engine='CYCLES' if '--cycles' in sys.argv else 'BLENDER_EEVEE_NEXT';s.eevee.taa_render_samples=8
out=P/('preview_eevee' if '--preview' in sys.argv else 'frames');out.mkdir(exist_ok=True)
for f in ([1,20,27,72] if '--preview' in sys.argv else range(1,73)):
 s.frame_set(f);s.render.filepath=str(out/f'{f-1:04d}.png');bpy.ops.render.render(write_still=True)
