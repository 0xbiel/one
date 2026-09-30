"""Deterministic detailed RGB scene versus separately emitted cuboid scan fixture."""
import bpy,json,sys,math
import numpy as np
from pathlib import Path
from mathutils import Vector,Matrix
root=Path(__file__).parent; out=root/'fixtures';out.mkdir(exist_ok=True)
source=Path(sys.argv[sys.argv.index('--')+1])
sem=json.loads((source.parent/'semantic_objects.json').read_text())['objects']
# Fixed before any localization run. Layout split is immutable; no fitting on heldout poses.
layouts=[('dev_living','dev',0),('heldout_rearranged','heldout',1),('heldout_mirrored','heldout',2)]
refs=[(-1.9,1.55,1.85),(-.65,1.7,2.05),(.65,1.55,2.05),(1.85,1.7,1.9)]
queries=[(-1.25,1.85,2.35),(1.2,1.3,2.35)]
W,H=480,360; f=320.; K=[[f,0,W/2],[0,f,H/2],[0,0,1]]
B=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1))) # world Y-up to Blender Z-up
for name,split,variant in layouts:
 bpy.ops.wm.open_mainfile(filepath=str(source));s=bpy.context.scene;c=s.camera
 s.render.engine='CYCLES';s.cycles.samples=64;s.cycles.use_denoising=False;s.render.threads_mode='FIXED';s.render.threads=4
 s.render.resolution_x=W;s.render.resolution_y=H;s.render.resolution_percentage=100
 c.data.type='PERSP';c.data.lens=24;c.data.sensor_width=36;c.data.sensor_fit='HORIZONTAL'
 d=out/name;d.mkdir(exist_ok=True);objects=[]
 for item in sem:
  item=json.loads(json.dumps(item)); idx=item['instance_index']; delta=Vector((0,0,0))
  if variant==1:delta=Vector((.28 if idx%2 else -.25,0,.12 if idx%3 else -.18))
  if variant==2:delta=Vector((-2*item['center']['x'],0,.16 if idx%2 else -.12))
  # Mirroring positions only preserves detailed object shape and metric dimensions.
  for o in s.objects:
   if o.type=='MESH' and o.pass_index==idx:o.location+=Vector((delta.x,-delta.z,delta.y))
  for j,axis in enumerate('xyz'):item['center'][axis]+=delta[j];item['transform'][j][3]+=delta[j]
  bpy.context.view_layer.update()
  points=[]
  for ob in s.objects:
   if ob.type=='MESH' and ob.pass_index==idx:
    ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points.extend([B.inverted()@ev.matrix_world@Vector(corner) for corner in ev.bound_box])
  lo=np.min(np.asarray(points),axis=0);hi=np.max(np.asarray(points),axis=0);center=(lo+hi)/2
  item['center']=dict(zip('xyz',map(float,center)));item['dimensions']=dict(zip('xyz',map(float,hi-lo)))
  for j in range(3):item['transform'][j][3]=float(center[j])
  objects.append(dict(id=item['id'],label=item['category'],center=item['center'],dimensions=item['dimensions'],transform={'values':item['transform']},confidence=1.0))
 # Independent room appearance, texture remains real geometry detail in RGB, never USDZ.
 if variant:
  for mat in bpy.data.materials:
   if mat.use_nodes:
    p=mat.node_tree.nodes.get('Principled BSDF')
    if p:
     col=p.inputs['Base Color'].default_value; p.inputs['Base Color'].default_value=(col[(variant)%3],col[(variant+1)%3],col[(variant+2)%3],1)
 scan=dict(room_objects=objects,room_zones=[dict(id=name,floor_y=0,polygon=[{'x':-3,'z':-2.5},{'x':3,'z':-2.5},{'x':3,'z':2.5},{'x':-3,'z':2.5}])])
 (d/'scan.json').write_text(json.dumps(scan,indent=2))
 records=[]
 for role,poses in [('reference',refs),('query',queries)]:
  for i,pose in enumerate(poses):
   c.location=Vector((pose[0],-pose[2],pose[1]));target=Vector((0,1,.75));c.rotation_euler=(target-c.location).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update()
   stem=f'{role}_{i}';s.render.image_settings.file_format='JPEG';s.render.image_settings.quality=94;s.render.filepath=str(d/(stem+'.jpg'));bpy.ops.render.render(write_still=True)
   matrix=np.asarray(B.inverted()@c.matrix_world).tolist()
   assert np.allclose(np.asarray(matrix)[:3,3],pose,atol=1e-6), (matrix,pose)
   assert np.isclose(np.linalg.det(np.asarray(matrix)[:3,:3]),1,atol=1e-6)
   records.append(dict(id=stem,role=role,image=stem+'.jpg',width=W,height=H,intrinsics=K,camera_to_world=matrix))
   if role=='reference':
    # Axial camera depth (not ray range), generated ONLY for scan reference views.
    dw,dh=160,120;depth=np.zeros((dh,dw),dtype='<f4');deps=bpy.context.evaluated_depsgraph_get();rot=c.matrix_world.to_3x3()
    for y in range(dh):
     for x in range(dw):
      u=x*(W-1)/(dw-1);v=y*(H-1)/(dh-1);direction=Vector(((u-W/2)/f,-(v-H/2)/f,-1));direction.normalize()
      hit,loc,*_=s.ray_cast(deps,c.location,rot@direction,distance=15)
      if hit:depth[y,x]=-(c.matrix_world.inverted()@loc).z
    np.save(d/(stem+'_depth.npy'),depth);records[-1]['depth']=stem+'_depth.npy'
 (d/'evaluator_groundtruth.json').write_text(json.dumps(dict(layout=name,split=split,records=records),indent=2))
 print('FINISHED_LAYOUT',name,flush=True)
