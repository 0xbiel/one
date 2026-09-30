import bpy,json
from pathlib import Path
P=Path(__file__).resolve().parent/'assets';old=json.loads((P/'physics_samples_24fps.json').read_text());s=bpy.context.scene;err=0
for f in range(1,146):
 s.frame_set(f);bpy.context.view_layer.update();d=bpy.context.evaluated_depsgraph_get()
 for name,v in old[f-1]['bodies'].items():
  m=bpy.data.objects['ragdoll_'+name].evaluated_get(d).matrix_world
  err=max(err,max(abs(m[i][j]-v['matrix_world'][i][j]) for i in range(4) for j in range(4)))
report={'replay_of_source_max_matrix_abs_error':err,'same_build_deterministic_tolerance_1e_6_pass':err<=1e-6,'min_sampled_collision_mesh_z_m':min(v['lowest_surface_z_m'] for r in old for v in r['bodies'].values())}
(P/'validation.json').write_text(json.dumps(report,indent=2));print(report)
