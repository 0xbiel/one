"""Strict paired image-prediction replay summary, separate from oracle tests."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',nargs='+',required=True);p.add_argument('--candidate',nargs='+',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert len(a.baseline)==len(a.candidate)
manifest=json.loads(Path(a.manifest).read_text());cases={c['id']:dict(c,frames=80,fps=10,source='controlled_choreography') for c in manifest['clips']}
cases.update({name:dict(id=name,expected_fall=True,fall_onset_s=1,frames=72,fps=12,source='Bullet_physics') for name in ['fall_physics_ragdoll','fall_physics_forward']})
all_rows={};out=[];shared_model_sha256=None
for before_path,after_path in zip(a.baseline,a.candidate):
 b=Path(before_path);c=Path(after_path);before=[json.loads(s) for s in b.read_text().splitlines()];after=[json.loads(s) for s in c.read_text().splitlines()];meta=json.loads(c.with_suffix('.metadata.json').read_text());bm=json.loads(b.with_suffix('.metadata.json').read_text())
 assert meta['source_inference_sha256']==hashlib.sha256(b.read_bytes()).hexdigest()
 assert bm['input_kind']=='rendered_rgb_frames' and meta['input_kind']=='recorded_real_image_predictions'
 assert meta['model_sha256']==bm['model_sha256']
 if shared_model_sha256 is None:shared_model_sha256=bm['model_sha256']
 assert bm['model_sha256']==shared_model_sha256, 'All run pairs must use one frozen checkpoint'
 assert len(before)==len(after)
 for br,cr in zip(before,after):
  assert (br['clip_id'],br['frame_index'],br['timestamp_s'])==(cr['clip_id'],cr['frame_index'],cr['timestamp_s'])
  assert br['raw_detections']==cr['raw_detections']
  key=br['clip_id'];assert key in cases
  all_rows.setdefault(key,[]).append((br,cr))
assert set(all_rows)==set(cases)
for key,case in cases.items():
 rows=all_rows[key];assert len(rows)==case['frames']
 assert [r[0]['frame_index'] for r in rows]==list(range(case['frames']))
 assert all(abs(br['timestamp_s']-i/case['fps'])<1e-8 for i,(br,_) in enumerate(rows))
 stats={}
 for label,index in [('baseline',0),('candidate_replay',1)]:
  items=[r[index] for r in rows];times=[r['timestamp_s'] for r in items if any(e.get('event_type')=='fall_suspected' for e in r['events'])]
  after_onset=[t for t in times if t>=(case.get('fall_onset_s') or 0)]
  signaled=bool(after_onset if case['expected_fall'] else times)
  stats[label]={'outcome':('TP' if signaled else 'FN') if case['expected_fall'] else ('FP' if signaled else 'TN'),'fall_event_count':len(times),'signal_times_s':times,'delay_from_onset_s':min(after_onset)-case['fall_onset_s'] if case['expected_fall'] and after_onset else None,'pre_onset_events':sum(t<(case.get('fall_onset_s') or 0) for t in times),'person_detection_frames':sum(any(d.get('label')=='person' for d in r['raw_detections']) for r in items),'stable_person_frames':sum(any(d.get('label')=='person' for d in r['stable_detections']) for r in items)}
 out.append({'clip':key,'source':case['source'],'expected_fall':case['expected_fall'],'frames':case['frames'],'fps':case['fps'],**stats})
counts={}
for group in ['controlled_choreography','Bullet_physics','all']:
 selected=[r for r in out if group=='all' or r['source']==group];counts[group]={version:{k:sum(r[version]['outcome']==k for r in selected) for k in ['TP','FN','FP','TN']} for version in ['baseline','candidate_replay']}
result={'method':'Genuine baseline YOLO-World RGB inference through production API; exact same inferred predictions replayed through candidate API. Not a second independent image-model run. No GT boxes loaded. Source video hashes checked by replay.','model_sha256':shared_model_sha256,'claim_scope':'Nine synthetic fixtures only; no clinical/real-world validation, no safety improvement or deployment throughput claim.','counts':counts,'clips':out}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(counts,indent=2))
