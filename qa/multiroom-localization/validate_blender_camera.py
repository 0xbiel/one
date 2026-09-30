"""Independent check of exported K against Blender's actual camera projection."""
import bpy,json
from pathlib import Path
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
root=Path(__file__).parent;B=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)));s=bpy.context.scene
bpy.ops.object.camera_add();c=bpy.context.object;s.camera=c;c.data.lens=24;c.data.sensor_width=36;c.data.sensor_fit='HORIZONTAL';s.render.resolution_x=480;s.render.resolution_y=360;s.render.resolution_percentage=100
checks=[]
for d in sorted((root/'fixtures').iterdir()):
 for r in json.loads((d/'evaluator_groundtruth.json').read_text())['records']:
  c.matrix_world=B@Matrix(r['camera_to_world']);bpy.context.view_layer.update();max_error=0
  for x,y,z in [(.31,-.22,-3),(-.4,.7,-2),(0,0,-5)]:
   ndc=world_to_camera_view(s,c,c.matrix_world@Vector((x,y,z)));actual=Vector((ndc.x*480,(1-ndc.y)*360));expected=Vector((320*x/-z+240,-320*y/-z+180));error=(actual-expected).length;assert error<.001,(r['id'],error);max_error=max(max_error,error)
  checks.append(dict(layout=d.name,frame=r['id'],max_projection_discrepancy_px=max_error))
(root/'blender_camera_validation.json').write_text(json.dumps(checks,indent=2));print('PASS54off-axis Blender/K projection comparisons')
