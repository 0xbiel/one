import argparse,json,math,numpy as np
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--label',default='current');args=parser.parse_args()
root=Path(__file__).parent;manifest=json.loads((root/'manifest.json').read_text());truth={x['id']:x for x in json.loads((root/'scoring_only.json').read_text())};zones=[x['id'] for x in json.loads((root/'fixtures/connected_home/scan.json').read_text())['room_zones']];rows=[]
for m in manifest:
 p=root/'final-results'/args.label/f"{m['case']}.json"
 if not p.exists():continue
 r=json.loads(p.read_text());t=truth[m['query']];row=dict(case=m['case'],split=m['split'],condition=m['condition'],true_room=t['room'],predicted_room='abstain',status=r['status'],seconds=r.get('wall_seconds'),confidence=r.get('confidence'))
 if r['status']=='positioned' and r.get('camera_to_world') is not None:
  a=np.array(r['camera_to_world']);b=np.array(t['camera_to_world']);row.update(predicted_room=r.get('final_pose_scene_prior',{}).get('zone_id','abstain'),position_error_m=float(np.linalg.norm(a[:3,3]-b[:3,3])),rotation_error_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))),predicted_position=a[:3,3].tolist(),true_position=b[:3,3].tolist())
 row['room_correct']=row['predicted_room']==row['true_room'];row['tight_success']=row.get('position_error_m',999)<.10 and row.get('rotation_error_deg',999)<2;row['loose_success']=row.get('position_error_m',999)<.25 and row.get('rotation_error_deg',999)<5;rows.append(row)
def summary(rs):
 positioned=[r for r in rs if r['status']=='positioned'];conf={a:{b:0 for b in zones+['abstain']} for a in zones}
 for r in rs:conf[r['true_room']][r['predicted_room']]+=1
 out=dict(n=len(rs),positioned=len(positioned),room_correct=sum(r['room_correct'] for r in rs),tight_success=sum(r['tight_success'] for r in rs),loose_success=sum(r['loose_success'] for r in rs),statuses={s:sum(r['status']==s for r in rs) for s in sorted(set(r['status'] for r in rs))},room_confusion=conf,metric_denominator='Positioned only for errors; all requested cases for success and room rates')
 if positioned:
  for metric in ['position_error_m','rotation_error_deg']:
   out[metric]={q:float(np.percentile([r[metric] for r in positioned],v)) for q,v in [('median',50),('p90',90),('max',100)]}
 return out
report=dict(expected_cases=len(manifest),completed_cases=len(rows),all=summary(rows),by_condition={k:summary([r for r in rows if r['condition']==k]) for k in sorted(set(r['condition'] for r in rows))},by_split={k:summary([r for r in rows if r['split']==k]) for k in sorted(set(r['split'] for r in rows))},rows=rows)
(root/('results_summary.json' if args.label=='current' else args.label+'_summary.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
