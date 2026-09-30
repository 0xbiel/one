"""Independent final aggregation of every frozen paired case, retaining failures."""
import json,math,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).parent;manifest=json.loads((root/'manifest.json').read_text());rows=[]
provenance=json.loads((root/'provenance.json').read_text())
for m in manifest:
 truth=json.loads((root/'fixtures'/m['layout']/'evaluator_groundtruth.json').read_text());G=np.asarray(next(x for x in truth['records'] if x['id']==m['query'])['camera_to_world'])
 assert hashlib.sha256((root/m['input']).read_bytes()).hexdigest()==m['sha256']
 for variant in ['baseline','optimized']:
  value=json.loads((root/'final-results'/variant/(m['case']+'.json')).read_text())
  if value['status'] not in {'startup_timeout','process_error'} or 'input_sha256' in value:
   assert value['input_sha256']==m['sha256'],m['case']
   expected={k.split('/')[-1]:v for k,v in provenance['sources'][variant].items()}
   assert value['geometry_fingerprint']==expected,m['case']
  else:
   assert value.get('camera_to_world') is None
  pose=value.get('camera_to_world');t=r=None
  if pose is not None:
   P=np.asarray(pose);t=float(np.linalg.norm(P[:3,3]-G[:3,3]));r=math.degrees(math.acos(float(np.clip((np.trace(P[:3,:3].T@G[:3,:3])-1)/2,-1,1))))
  diag=value.get('diagnostics',{});rows.append({**m,'variant':variant,'status':value['status'],'translation_m':t,'rotation_deg':r,'within_10cm_2deg':value['status']=='positioned' and t is not None and t<=.1 and r<=2,'within_25cm_5deg':value['status']=='positioned' and t is not None and t<=.25 and r<=5,'seconds':value.get('wall_seconds'),'selected_source':diag.get('selected_estimate_source'),'intrinsics_source':value.get('intrinsics_source'),'selected_fov':diag.get('selected_fov_degrees'),'inlier_count':value.get('inlier_count'),'match_count':value.get('match_count'),'reprojection_px':value.get('reprojection_error_px')})
summary=[]
for variant in ['baseline','optimized']:
 for split in ['dev','heldout','all']:
  for condition in ['clean','noisy_depth_missing_boxes','no_depth','unknown_intrinsics','all']:
   group=[r for r in rows if r['variant']==variant and (r['split']==split or split=='all') and (r['condition']==condition or condition=='all')];positioned=[r for r in group if r['status']=='positioned'];stats={}
   for key in ['translation_m','rotation_deg']:
    vals=[r[key] for r in positioned if r[key] is not None];stats[key]={str(q):float(np.percentile(vals,q)) if vals else None for q in [50,90,95,100]}
   summary.append(dict(variant=variant,split=split,condition=condition,total=len(group),positioned=len(positioned),failed=len(group)-len(positioned),timeouts=sum(r['status']=='timeout' for r in group),within_10cm_2deg=sum(r['within_10cm_2deg'] for r in group),within_25cm_5deg=sum(r['within_25cm_5deg'] for r in group),positioned_only_percentiles=stats))
(root/'results.json').write_text(json.dumps(rows,indent=2));(root/'summary.json').write_text(json.dumps(summary,indent=2));assert len(rows)==48
for r in summary:
 if r['variant']=='optimized' and r['split']=='heldout':print(r)
