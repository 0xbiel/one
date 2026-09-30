"""Frozen connected-home render generator. No localization outputs are inputs."""
import bpy,runpy,json,math,numpy as np
from pathlib import Path
from mathutils import Vector,Matrix
root=Path(__file__).parent
runpy.run_path(str(root/'base_scene.py'))
s=bpy.context.scene
# Detailed furnishing templates inherited from the prior procedural furniture family.
templates=[o.copy() for o in s.objects if o.type=='MESH' and not o.name.startswith(('floor','wall_','plank_','rug'))]
for o in templates:o.data=o.data.copy()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
M={m.name:m for m in bpy.data.materials};B=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
def pos(v):return(v[0],-v[2],v[1])
boxes=[]
def box(n,c,d,mat='wall',structural=True):
 bpy.ops.mesh.primitive_cube_add(size=1,location=pos(c));o=bpy.context.object;o.name=n;o.dimensions=(d[0],d[2],d[1]);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(M[mat])
 if structural:boxes.append(dict(id=n,center=dict(zip('xyz',c)),dimensions=dict(zip('xyz',d))))
 return o
box('floor',(0,-.075,0),(14,.15,10),'floor')
for n,c,d in [('west',(-7.05,1.4,0),(.1,2.8,10.2)),('east',(7.05,1.4,0),(.1,2.8,10.2)),('north',(0,1.4,-5.05),(14,2.8,.1)),('south',(0,1.4,5.05),(14,2.8,.1))]:box(n,c,d)
# Real 1.2m door openings into a continuous 2m-wide hallway, with lintels.
for x in [-1,1]:
 for a,b in [(-5,-3.1),(-1.9,1.9),(3.1,5)]:box(f'partition_{x}_{a}',(x,1.4,(a+b)/2),(.12,2.8,b-a))
 for z in [-2.5,2.5]:box(f'lintel_{x}_{z}',(x,2.5,z),(.12,.6,1.2))
for x in [-4,4]:box(f'middle_{x}',(x,1.4,0),(5.88,2.8,.12))
# Fine RGB-only floor detail, deliberately absent from the cuboid scan.
for i in range(-15,16):box(f'plank_{i}',(i*.45,.003,0),(.007,.004,10),'wood',False)
rooms=[('west_south',-4,2.5),('west_north',-4,-2.5),('east_south',4,2.5),('east_north',4,-2.5)]
objects=[];zones=[]
for ri,(name,ox,oz) in enumerate(rooms):
 zones.append(dict(id=name,floor_y=0,polygon=[dict(x=ox+x,z=oz+z) for x,z in [(-3,-2.5),(3,-2.5),(3,2.5),(-3,2.5)]]))
 for template in templates:
  o=template.copy();o.data=template.data.copy();s.collection.objects.link(o);o.name=name+'_'+template.name;o.location+=Vector(pos((ox,0,oz)));o.pass_index=template.pass_index+ri*10 if template.pass_index else 0
  # East rooms: television belongs on the outer wall, never across the hallway doorway.
  if ox>0 and template.name.startswith('tv_'):
   o.location.x=2*ox-o.location.x;o.scale.x*=-1
  # Similar-looking room confounders share shape, with bounded appearance differences.
  if ri and ('pillow' in o.name or 'art_shape' in o.name or 'cabinet_door' in o.name):
   m=o.data.materials[0].copy();m.name=name+'_'+m.name;node=m.node_tree.nodes.get('Principled BSDF');col=node.inputs['Base Color'].default_value;node.inputs['Base Color'].default_value=(col[ri%3],col[(ri+1)%3],col[(ri+2)%3],1);o.data.materials[0]=m
 bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
 sem=json.loads((root/'assets/semantic_objects.json').read_text())['objects']
 for item in sem:
  idx=item['instance_index']+ri*10;pts=[]
  for o in s.objects:
   if o.type=='MESH' and o.pass_index==idx:
    ev=o.evaluated_get(deps);pts.extend([B.inverted()@ev.matrix_world@Vector(v) for v in ev.bound_box])
  lo=np.min(np.asarray(pts),axis=0);hi=np.max(np.asarray(pts),axis=0);c=(lo+hi)/2;t=np.eye(4);t[:3,3]=c
  objects.append(dict(id=name+'_'+item['id'],label=item['category'],center=dict(zip('xyz',map(float,c))),dimensions=dict(zip('xyz',map(float,hi-lo))),transform={'values':t.tolist()},confidence=1.0))
 for px,pz in [(ox,oz),(ox,oz+1.6)]:
  bpy.ops.object.light_add(type='AREA',location=pos((px,2.7,pz)));o=bpy.context.object;o.data.energy=300;o.data.size=3
zones.append(dict(id='hallway',floor_y=0,polygon=[dict(x=x,z=z) for x,z in [(-1,-5),(1,-5),(1,5),(-1,5)]]))
for z in [-3,0,3]:
 bpy.ops.object.light_add(type='AREA',location=pos((0,2.7,z)));o=bpy.context.object;o.data.energy=150;o.data.size=1.5
# Hallway art provides genuine RGB feature evidence rather than ID labels.
for z in [-4,-1,1,4]:
 box(f'hall_frame_{z}',(-.925,1.6,z),(.03,.7,.65),'wood',False)
 for j in range(5):box(f'hall_art_{z}_{j}',(-.9,1.4+j*.09,z+(j%3-1)*.14),(.025,.04,.08),'sofa' if (j+int(z))%2 else 'cushion',False)
bpy.ops.object.camera_add();c=bpy.context.object;s.camera=c;c.data.lens=24;c.data.sensor_width=36;c.data.sensor_fit='HORIZONTAL';c.data.clip_start=.05;c.data.clip_end=50
s.render.engine='CYCLES';s.cycles.samples=32;s.cycles.use_denoising=False;s.render.threads_mode='FIXED';s.render.threads=2;s.world.color=(.25,.25,.25);s.view_settings.view_transform='AgX'
W,H=480,360;f=320.;K=[[f,0,W/2],[0,f,H/2],[0,0,1]]
records=[]
for name,ox,oz in rooms:
 for j,(x,y,z) in enumerate([(-1.9,1.55,1.85),(-.65,1.7,2.05),(.65,1.55,2.05),(1.85,1.7,1.9)]):records.append(dict(id=f'ref_{name}_{j}',role='reference',position=[ox+x,y,oz+z],target=[ox,.75,oz-1],room=name))
 for j,(x,y,z) in enumerate([(-1.25,1.85,2.3),(1.2,1.3,2.25)]):records.append(dict(id=f'query_{name}_{j}',role='query',position=[ox+x,y,oz+z],target=[ox,.75,oz-1],room=name,split='dev' if name=='west_south' else 'heldout'))
for j,(p,t) in enumerate([([0,1.6,4.4],[0,1,-3]),([0,1.6,-4.4],[0,1,3]),([.2,1.65,0],[0,1,4]),([-.2,1.65,0],[0,1,-4])]):records.append(dict(id=f'ref_hall_{j}',role='reference',position=p,target=t,room='hallway'))
for j,(p,t,room,kind) in enumerate([([0,1.7,3.9],[0,1,-3],'hallway','hallway'),([-.75,1.6,2.5],[-4,1,1],'hallway','door_before'),([-1.25,1.6,2.5],[-4,1,1],'west_south','door_after'),([.75,1.6,-2.5],[4,1,-4],'hallway','door_before')]):records.append(dict(id=f'query_transition_{j}',role='query',position=p,target=t,room=room,split='heldout',kind=kind))
out=root/'fixtures/connected_home';out.mkdir(parents=True,exist_ok=True)
(out/'scan.json').write_text(json.dumps(dict(room_objects=objects,room_zones=zones),indent=2));(out/'structural_boxes.json').write_text(json.dumps(boxes,indent=2))
(root/'frozen_design.json').write_text(json.dumps(dict(seed=104729,frames=records,rooms=zones,scope='One connected four-room home plus hallway; heldout camera poses, not heldout architecture',thresholds={'tight_m':.1,'tight_deg':2,'loose_m':.25,'loose_deg':5},conditions=['clean','unknown_intrinsics'],timeout_seconds=180),indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(root/'connected_home.blend'))
# Preview is a cutaway only; benchmark camera renders retain all walls.
c.location=Vector(pos((18,18,22)));c.rotation_euler=(Vector((0,0,.8))-c.location).to_track_quat('-Z','Y').to_euler();c.data.type='ORTHO';c.data.ortho_scale=20
for name in ['south','east']:bpy.data.objects[name].hide_render=True
s.render.resolution_x=1100;s.render.resolution_y=850;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.filepath=str(root/'connected_home_preview.png');bpy.ops.render.render(write_still=True)
for name in ['south','east']:bpy.data.objects[name].hide_render=False
c.data.type='PERSP';s.render.resolution_x=W;s.render.resolution_y=H
for r in records:
 c.location=Vector(pos(r['position']));c.rotation_euler=(Vector(pos(r['target']))-c.location).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update();stem=r['id'];s.render.image_settings.file_format='JPEG';s.render.image_settings.quality=94;s.render.filepath=str(out/(stem+'.jpg'));bpy.ops.render.render(write_still=True)
 matrix=np.asarray(B.inverted()@c.matrix_world);assert np.allclose(matrix[:3,3],r['position'],atol=1e-5);r.update(image=stem+'.jpg',width=W,height=H,intrinsics=K,camera_to_world=matrix.tolist())
 if r['role']=='reference':
  dw,dh=160,120;depth=np.zeros((dh,dw),dtype='<f4');deps=bpy.context.evaluated_depsgraph_get();rot=c.matrix_world.to_3x3()
  for y in range(dh):
   for x in range(dw):
    u=x*(W-1)/(dw-1);v=y*(H-1)/(dh-1);direction=Vector(((u-W/2)/f,-(v-H/2)/f,-1));direction.normalize();hit,loc,*_=s.ray_cast(deps,c.location,rot@direction,distance=30)
    if hit:depth[y,x]=-(c.matrix_world.inverted()@loc).z
  np.save(out/(stem+'_depth.npy'),depth);r['depth']=stem+'_depth.npy'
 (out/'evaluator_groundtruth.json').write_text(json.dumps(dict(layout='connected_home',records=records),indent=2));print('FRAME_DONE',stem,flush=True)
