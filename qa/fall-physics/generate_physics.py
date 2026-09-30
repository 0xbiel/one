"""Bullet articulated rigid-body fall. All post-release motion comes from physics."""
import bpy, math, json, sys, argparse
from pathlib import Path
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(); ap.add_argument('--render',action='store_true'); ap.add_argument('--preview',action='store_true');args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
exec((P/'room_base.py').read_text().split('s.render.resolution_x=480;')[0])
O=P/'assets';O.mkdir(exist_ok=True)
s.render.engine='CYCLES';s.cycles.samples=16;s.cycles.use_denoising=False;s.render.threads_mode='FIXED';s.render.threads=4
s.render.resolution_x=640;s.render.resolution_y=480;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.fps=24;s.frame_start=1;s.frame_end=145
s.gravity=(0,0,-9.81)
c.location=(3.6,-6.0,2.8);c.rotation_euler=(Vector((-.1,-.9,.65))-c.location).to_track_quat('-Z','Y').to_euler();c.data.lens=36
# Room furniture is collision-enabled. Decorative seams and rug are visual only.
for ob in list(s.objects):
 if ob.type=='MESH' and (ob.name=='floor' or ob.name.startswith(('wall_','table_','sofa_','dining_','chair_','cabinet'))):
  bpy.context.view_layer.objects.active=ob;bpy.ops.rigidbody.object_add();ob.rigid_body.type='PASSIVE';ob.rigid_body.collision_shape='BOX';ob.rigid_body.friction=.7;ob.rigid_body.restitution=0;ob.rigid_body.use_margin=True;ob.rigid_body.collision_margin=.002
# Clear fall area in front of the unchanged furnished room.
for n,col in [('skin',(.61,.32,.18)),('shirt',(.13,.35,.65)),('trousers',(.07,.075,.085)),('hair',(.025,.016,.012)),('shoe',(.045,.028,.018))]:
 m=bpy.data.materials.new(n);m.diffuse_color=(*col,1);m.use_nodes=True;m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(*col,1);M[n]=m
parts={}; extras=[]; specs={}
R=Matrix.Rotation(math.radians(-10),4,'Y'); shift=Vector((.50,-1.65,.027))
def world(v):return R.to_3x3()@Vector(v)+shift
def body(name,loc,scale,mat,mass):
 bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,location=world(loc));o=bpy.context.object;o.name='ragdoll_'+name;o.scale=scale;o.rotation_euler=R.to_euler();bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(M[mat])
 for p in o.data.polygons:p.use_smooth=True
 bpy.ops.rigidbody.object_add();rb=o.rigid_body;rb.mass=mass;rb.collision_shape='CONVEX_HULL';rb.friction=.65;rb.restitution=.03;rb.linear_damping=.15;rb.angular_damping=.25;rb.use_margin=True;rb.collision_margin=.003;rb.use_deactivation=False
 rb.kinematic=True;rb.keyframe_insert('kinematic',frame=1);rb.keyframe_insert('kinematic',frame=24);rb.kinematic=False;rb.keyframe_insert('kinematic',frame=25)
 parts[name]=o;specs[name]={'mass_kg':mass,'collision_shape':'CONVEX_HULL'};return o
body('pelvis',(0,0,.91),(.18,.125,.15),'trousers',11)
body('torso',(0,0,1.23),(.23,.14,.27),'shirt',27)
body('head',(0,0,1.64),(.115,.105,.15),'skin',5)
for side,x in [('l',-.12),('r',.12)]:
 body(side+'_thigh',(x,0,.685),(.10,.10,.22),'trousers',7)
 body(side+'_shin',(x,0,.285),(.072,.075,.20),'trousers',3.3)
 body(side+'_foot',(x,-.065,.067),(.083,.15,.065),'shoe',1)
 ax=-.285 if side=='l' else .285
 body(side+'_upperarm',(ax,0,1.235),(.071,.071,.17),'shirt',2.2)
 body(side+'_forearm',(ax,-.012,.935),(.055,.055,.155),'skin',1.4)
 body(side+'_hand',(ax,-.016,.735),(.06,.045,.08),'skin',.5)
constraints=[]
def joint(name,a,b,loc,angles):
 bpy.ops.object.empty_add(type='PLAIN_AXES',location=world(loc));o=bpy.context.object;o.name='joint_'+name;o.rotation_euler=R.to_euler();bpy.ops.rigidbody.constraint_add();con=o.rigid_body_constraint;con.type='GENERIC';con.object1=parts[a];con.object2=parts[b];con.disable_collisions=True
 for axis in 'xyz':
  setattr(con,'use_limit_lin_'+axis,True);setattr(con,'limit_lin_'+axis+'_lower',0);setattr(con,'limit_lin_'+axis+'_upper',0)
 for axis,(lo,hi) in zip('xyz',angles):
  setattr(con,'use_limit_ang_'+axis,True);setattr(con,'limit_ang_'+axis+'_lower',math.radians(lo));setattr(con,'limit_ang_'+axis+'_upper',math.radians(hi))
 con.use_override_solver_iterations=True;con.solver_iterations=80
 constraints.append({'name':name,'a':a,'b':b,'angular_limits_degrees_xyz':angles})
joint('spine','pelvis','torso',(0,0,1.04),[(-25,35),(-25,25),(-25,25)])
joint('neck','torso','head',(0,0,1.49),[(-30,40),(-35,35),(-50,50)])
for side,x in [('l',-.12),('r',.12)]:
 ax=-.285 if side=='l' else .285
 joint(side+'_hip','pelvis',side+'_thigh',(x,0,.86),[(-25,90),(-35,35),(-25,25)])
 joint(side+'_knee',side+'_thigh',side+'_shin',(x,0,.485),[(-135,3),(-2,2),(-2,2)])
 joint(side+'_ankle',side+'_shin',side+'_foot',(x,0,.09),[(-30,30),(-15,15),(-10,10)])
 joint(side+'_shoulder','torso',side+'_upperarm',(ax,0,1.385),[(-80,80),(-80,80),(-60,60)])
 joint(side+'_elbow',side+'_upperarm',side+'_forearm',(ax,0,1.08),[(-3,135),(-2,2),(-5,5)])
 joint(side+'_wrist',side+'_forearm',side+'_hand',(ax,-.016,.79),[(-35,35),(-25,25),(-20,20)])
# Cosmetic details rigidly follow head and introduce no forces.
head=parts['head']
for name,loc,scale,mat in [('hair',(0,0,.1),(.117,.107,.067),'hair'),('eye_l',(-.043,-.095,.022),(.014,.012,.014),'hair'),('eye_r',(.043,-.095,.022),(.014,.012,.014),'hair'),('nose',(0,-.107,-.01),(.024,.027,.03),'skin')]:
 bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8);o=bpy.context.object;o.name=name;o.scale=scale;o.data.materials.append(M[mat]);o.parent=head;o.location=loc;extras.append(o)
worldrb=s.rigidbody_world;worldrb.substeps_per_frame=10;worldrb.solver_iterations=80;worldrb.point_cache.frame_start=1;worldrb.point_cache.frame_end=145
s.frame_set(1);bpy.context.view_layer.update();bpy.ops.wm.save_as_mainfile(filepath=str(O/'physics_source.blend'))
# Advance sequentially through Bullet; sample matrices before removing physics for render replay.
samples=[]
for f in range(1,146):
 s.frame_set(f);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get();record={'blender_frame':f,'timestamp_s':(f-1)/24,'bodies':{}}
 for name,o in parts.items():
  mat=o.evaluated_get(deps).matrix_world.copy();vs=[mat@v.co for v in o.data.vertices];minz=min(v.z for v in vs)
  record['bodies'][name]={'matrix_world':[[float(x) for x in row] for row in mat],'lowest_surface_z_m':float(minz),'floor_contact_proxy':minz<=.015}
 samples.append(record)
(O/'physics_samples_24fps.json').write_text(json.dumps(samples))
# Actual body contact estimate derived from sampled world-space collision mesh; not solver impulses.
first=next((r for r in samples if r['blender_frame']>=25 and any(v['floor_contact_proxy'] for k,v in r['bodies'].items() if k in ('head','torso','pelvis','l_upperarm','r_upperarm','l_forearm','r_forearm'))),None)
meta={'schema':'one-bullet-ragdoll/v1','expected_fall':True,'physics_engine':'Blender 4.3.2 Bullet rigid bodies','gravity_m_s2':[0,0,-9.81],'simulation_fps':24,'output_fps':12,'duration_s':6,'release_s':1.0,'initial_condition':'10 degree lateral lean, kinematic support through 0.9583s; released at 1s; no motion keyframes or applied forces after release','first_upper_body_floor_contact_proxy_s':None if first is None else first['timestamp_s'],'contact_definition':'Minimum transformed collision-mesh vertex z <= 0.015m, sampled at 24Hz. Geometric proxy, not solver contact impulse. Foot contact intentionally excluded from first upper-body contact.','bodies':specs,'joints':constraints,'limitations':['Stylized segmented adult; not photorealistic','Passive ragdoll; no active muscles, protective reflexes, cloth simulation, soft tissue or injury model','Deterministic sequential replay on this Blender build; cross-platform bitwise identity not guaranteed','Physical fall event is synthetic ground truth; detector predictions must be independently inferred from video']}
(O/'manifest.json').write_text(json.dumps(meta,indent=2))
# Replay computed rigid transforms, keeping dynamics source in separate .blend.
for o in list(s.objects):
 if o.rigid_body_constraint:
  bpy.context.view_layer.objects.active=o;bpy.ops.rigidbody.constraint_remove()
for o in parts.values():
 bpy.context.view_layer.objects.active=o;bpy.ops.rigidbody.object_remove();o.animation_data_clear()
if s.rigidbody_world:bpy.ops.rigidbody.world_remove()
annotations=[]
for i,r in enumerate(samples[::2]):
 for name,ob in parts.items():ob.matrix_world=Matrix(r['bodies'][name]['matrix_world']);ob.keyframe_insert('location',frame=i+1);ob.keyframe_insert('rotation_euler',frame=i+1)
 bpy.context.view_layer.update();pts=[world_to_camera_view(s,c,ob.matrix_world@Vector(v)) for ob in parts.values() for v in ob.bound_box]
 annotations.append({'frame_index':i,'timestamp_s':i/12,'bbox_gt_xyxy':[min(p.x for p in pts)*640,(1-max(p.y for p in pts))*480,max(p.x for p in pts)*640,(1-min(p.y for p in pts))*480]})
s.render.fps=12;s.frame_end=72
(O/'ground_truth_frames.json').write_text(json.dumps(annotations[:72],indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(O/'physics_replay.blend'))
if args.preview or args.render:
 frames=[1,15,20,27,48,72] if args.preview else range(1,73)
 out=O/('preview' if args.preview else 'frames');out.mkdir(exist_ok=True)
 for f in frames:s.frame_set(f);s.render.filepath=str(out/f'{f-1:04d}.png');bpy.ops.render.render(write_still=True)
print('PHYSICS_COMPLETE',json.dumps(meta))
