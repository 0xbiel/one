"""Deterministic procedural 3D video fixtures; no detections are synthesized."""
import bpy, math, json, sys, argparse, hashlib, shutil
from pathlib import Path
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--only',default='all');ap.add_argument('--preview',action='store_true');ap.add_argument('--threads',type=int,default=8);args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
# Reuse the committed room construction, without running its export/render section.
exec((P/'room_base.py').read_text().split('s.render.resolution_x=480;')[0])
O=P/'assets';O.mkdir(exist_ok=True)
s.render.engine='CYCLES';s.cycles.samples=32;s.cycles.use_denoising=False;s.render.threads_mode='FIXED';s.render.threads=args.threads
s.render.resolution_x=640;s.render.resolution_y=480;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG'
s.render.fps=10;s.frame_start=1;s.frame_end=80
# Camera sees the unobstructed foreground; original room and furnishing retained.
c.location=(3.6,-6.0,2.8);c.rotation_euler=(Vector((0,-.55,.8))-c.location).to_track_quat('-Z','Y').to_euler();c.data.lens=36
for n,color in [('skin',(.61,.32,.18)),('shirt',(.13,.35,.65)),('trousers',(.07,.075,.085)),('hair',(.025,.016,.012)),('shoe',(.045,.028,.018))]:
 m=bpy.data.materials.new(n);m.diffuse_color=(*color,1);m.use_nodes=True;m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(*color,1);M[n]=m
actor={}
def ellipsoid(name,scale,mat):
 bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12);o=bpy.context.object;o.name='actor_'+name;o.scale=scale;o.data.materials.append(M[mat]);actor[name]=o
 for p in o.data.polygons:p.use_smooth=True
 return o
ellipsoid('torso',(.23,.135,.29),'shirt');ellipsoid('pelvis',(.185,.13,.15),'trousers');ellipsoid('head',(.115,.105,.145),'skin');ellipsoid('hair',(.117,.107,.07),'hair');ellipsoid('neck',(.065,.065,.08),'skin')
for side in ['l','r']:
 for name,rad,mat in [('upperarm',.075,'shirt'),('forearm',.055,'skin'),('thigh',.10,'trousers'),('shin',.075,'trousers')]:ellipsoid(side+'_'+name,(rad,rad,.2),mat)
 ellipsoid(side+'_hand',(.06,.045,.09),'skin');ellipsoid(side+'_foot',(.082,.15,.06),'shoe')
# Facial details face toward foreground camera.
for side in ['l','r']:ellipsoid(side+'_eye',(.018,.012,.016),'hair')
ellipsoid('nose',(.026,.03,.032),'skin')
def blend(a,b,u):return Vector(a)*(1-u)+Vector(b)*u
def smooth(x):x=max(0,min(1,x));return x*x*(3-2*x)
def pose(kind,t):
 j={'hip':(0,0,.91),'shoulder':(0,0,1.40),'neck':(0,0,1.52),'head':(0,0,1.68),'lk':(-.12,0,.48),'rk':(.12,0,.48),'la':(-.12,0,.09),'ra':(.12,0,.09),'ls':(-.235,0,1.40),'rs':(.235,0,1.40),'le':(-.28,-.015,1.11),'re':(.28,-.015,1.11),'lw':(-.29,-.03,.90),'rw':(.29,-.03,.90)}
 j={k:Vector(v) for k,v in j.items()}
 if kind in ['fall_side','fall_occluded','lie_down']:
  u=smooth((t-2)/(.7 if kind!='lie_down' else 3.0));a=-math.pi*.5*u
  # Rotation about ankles with a floor-safe final side-lying pose.
  R=Matrix.Rotation(a,3,'Y');j={k:R@v+Vector((.35*u,0,.10*u)) for k,v in j.items()}
 elif kind=='fall_forward':
  u=smooth((t-2)/.7);R=Matrix.Rotation(math.pi*.5*u,3,'X');j={k:R@v+Vector((0,.6*u,.15*u)) for k,v in j.items()}
 elif kind=='sit':
  u=smooth((t-2)/1.3);target={'hip':(0,.22,.57),'shoulder':(0,.18,1.06),'neck':(0,.18,1.18),'head':(0,.18,1.34),'lk':(-.13,-.23,.50),'rk':(.13,-.23,.50),'la':(-.13,-.25,.09),'ra':(.13,-.25,.09),'ls':(-.235,.18,1.06),'rs':(.235,.18,1.06),'le':(-.27,-.02,.80),'re':(.27,-.02,.80),'lw':(-.15,-.22,.59),'rw':(.15,-.22,.59)};j={k:blend(v,target[k],u) for k,v in j.items()}
 elif kind=='bend':
  u=smooth((t-2)/1.1)*(1-smooth((t-5)/1.2));pivot=j['hip'];R=Matrix.Rotation(math.radians(75)*u,3,'X')
  for k in ['shoulder','neck','head','ls','rs','le','re','lw','rw']:j[k]=pivot+R@(j[k]-pivot)
 return {k:v+Vector((.55,-1.25,0)) for k,v in j.items()}
def set_pose(j):
 def point(n,p):actor[n].location=p
 def bone(n,a,b):
  o=actor[n];o.location=(a+b)/2;o.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler();o.scale.z=(b-a).length/2+.035
 bone('torso',j['hip'],j['shoulder']);point('pelvis',j['hip']);bone('neck',j['shoulder'],j['neck']);point('head',j['head'])
 up=(j['head']-j['neck']).normalized();rot=up.to_track_quat('Z','Y');
 for n in ['head','hair','l_eye','r_eye','nose']:actor[n].rotation_euler=rot.to_euler()
 point('hair',j['head']+up*.10)
 for side,x in [('l',-.045),('r',.045)]:point(side+'_eye',j['head']+rot@Vector((x,-.095,.025)))
 point('nose',j['head']+rot@Vector((0,-.108,-.015)))
 for side in ['l','r']:
  bone(side+'_upperarm',j[side+'s'],j[side+'e']);bone(side+'_forearm',j[side+'e'],j[side+'w']);point(side+'_hand',j[side+'w']);bone(side+'_thigh',j['hip']+rot@Vector((-.12 if side=='l' else .12,0,0)),j[side+'k']);bone(side+'_shin',j[side+'k'],j[side+'a']);point(side+'_foot',j[side+'a']+Vector((0,-.045,0)))
# Seat only visible for sitting scenario; occluder only in partial-occlusion scenario.
chair=cube('test_seat',(.55,.50,1.03),(.55,.08,.55),'wood');chair.location=(.55,-1.03,.48)
chair_parts=[chair]
for x in [.33,.77]:
 for y in [-1.24,-.82]:
  o=cube('test_chair_leg',(0,0,0),(.05,.46,.05),'wood');o.location=(x,y,.23);chair_parts.append(o)
o=cube('test_chair_back',(0,0,0),(.55,.5,.06),'wood');o.location=(.55,-.79,.75);chair_parts.append(o)
occluder=cube('foreground_occluder',(0,0,0),(1.1,.7,.12),'wood');occluder.location=(.1,-2.1,.35)
dropped=ellipsoid('dropped_object',(.16,.12,.20),'cushion');actor.pop('dropped_object')
scenarios=['fall_side','fall_forward','fall_occluded','sit','bend','lie_down','drop_object']
manifest={'schema':'one-synthetic-fall-clips/v1','fps':10,'width':640,'height':480,'frames_per_clip':80,'duration_seconds':8,'frame_timestamp':'zero-based frame index / fps','source_room_commit':'a89ff32f1024f83b925c5cccdc0dc29bd91ec370','ground_truth_origin':'procedural choreography; not detector outputs','limitations':['Stylized procedural adult actor, not photorealistic','Ground-truth intent labels do not establish medical or emergency validity','Projected ground-truth boxes are annotations only and must not substitute for image detections'],'clips':[]}
render_cache={}
for kind in scenarios:
 expected=kind.startswith('fall_');entry={'id':kind,'file':kind+'.mp4','expected_fall':expected,'motion_onset_s':2.0,'fall_onset_s':2.0 if expected else None,'floor_contact_s':2.7 if expected else (5.0 if kind=='lie_down' else None),'observation_end_s':7.9,'confounder':None if expected else kind};manifest['clips'].append(entry)
 if args.only!='all' and kind not in args.only.split(','):continue
 
 for part in chair_parts:part.hide_render=kind!='sit'
 occluder.hide_render=kind!='fall_occluded';dropped.hide_render=kind!='drop_object'
 out=O/kind;out.mkdir(exist_ok=True);annotations=[]
 for i in ([0,27,65] if args.preview else range(80)):
  t=i/10;set_pose(pose(kind,t));dropped.location=(1.05,-1.15,1.3-1.05*smooth((t-2)/.5));bpy.context.view_layer.update()
  pts=[world_to_camera_view(s,c,o.matrix_world@Vector(v)) for o in actor.values() for v in o.bound_box]
  annotations.append({'frame_index':i,'timestamp_s':t,'bbox_gt_xyxy':[round(min(p.x for p in pts)*640,3),round((1-max(p.y for p in pts))*480,3),round(max(p.x for p in pts)*640,3),round((1-min(p.y for p in pts))*480,3)]})
  s.render.filepath=str(out/f'{i:04d}.png')
  key=hashlib.sha256(str(([tuple(round(x,6) for row in o.matrix_world for x in row) for o in actor.values()],chair.hide_render,occluder.hide_render,dropped.hide_render,tuple(dropped.location) if not dropped.hide_render else None)).encode()).hexdigest()
  if key in render_cache:shutil.copyfile(render_cache[key],s.render.filepath)
  else:bpy.ops.render.render(write_still=True);render_cache[key]=s.render.filepath
 (out/'ground_truth_frames.json').write_text(json.dumps(annotations,indent=2))
(O/'manifest.json').write_text(json.dumps(manifest,indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(O/'animated_room.blend'))
