"""Comparison images only; never used by landmark builder or solver."""
import bpy,json
from pathlib import Path
from mathutils import Matrix,Vector
root=Path(__file__).parent;B=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
for d in sorted((root/'fixtures').iterdir()):
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=32;s.cycles.use_denoising=False;s.world.color=(.3,.3,.3);s.view_settings.view_transform='AgX'
 data=json.loads((d/'scan.json').read_text());boxes=data['room_objects']+[
 dict(id='floor',center=dict(x=0,y=-.05,z=0),dimensions=dict(x=6,y=.1,z=5)),dict(id='rear_wall',center=dict(x=0,y=1.4,z=-2.55),dimensions=dict(x=6.1,y=2.8,z=.1)),dict(id='left_wall',center=dict(x=-3.05,y=1.4,z=0),dimensions=dict(x=.1,y=2.8,z=5)),dict(id='right_wall',center=dict(x=3.05,y=1.4,z=0),dimensions=dict(x=.1,y=2.8,z=5))]
 for i,box in enumerate(boxes):
  c=box['center'];di=box['dimensions'];bpy.ops.mesh.primitive_cube_add(size=1,location=(c['x'],-c['z'],c['y']));o=bpy.context.object;o.name=box['id'];o.dimensions=(di['x'],di['z'],di['y']);m=bpy.data.materials.new(o.name);m.diffuse_color=(.3+(i%3)*.13,.4+(i%2)*.08,.55,1);o.data.materials.append(m)
 for loc,power,size in [((0,2.65,.4),450,4),((-2,2.4,1.8),300,2),((2,2.4,-1),200,2)]:
  bpy.ops.object.light_add(type='AREA',location=(loc[0],-loc[2],loc[1]));o=bpy.context.object;o.data.energy=power;o.data.shape='DISK';o.data.size=size
 r=next(r for r in json.loads((d/'evaluator_groundtruth.json').read_text())['records'] if r['id']=='query_0');bpy.ops.object.camera_add();c=bpy.context.object;c.matrix_world=B@Matrix(r['camera_to_world']);s.camera=c;c.data.lens=24;c.data.sensor_width=36;c.data.sensor_fit='HORIZONTAL';s.render.resolution_x=480;s.render.resolution_y=360;s.render.resolution_percentage=100;s.render.image_settings.file_format='JPEG';s.render.filepath=str(d/'boxscan_preview.jpg');bpy.ops.render.render(write_still=True)
