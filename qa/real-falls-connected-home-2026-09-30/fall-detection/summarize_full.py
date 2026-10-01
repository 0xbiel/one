import pathlib,json,collections,statistics
R=pathlib.Path(__file__).parent;J=json.load(open(R/'full_scores.json'));M={c['id']:c for c in json.load(open(R/'full_manifest_downloaded.json'))};rows=J['clips'];report={'summary':J['summary'],'strata':[],'phase_sensitivity':[],'caveats':['Offline RGB detector-tracker-rule replay, not HTTP deployment','Cold-start reset per clip; short/missing pre-fall context penalizes low-rate tracker','Subject-folder split; distinct person identity not verified','No weight training or held-out tuning; expanded sample overlaps prior pilot','Fixed initial default 11-prompt profile; actual live object registrations change vocabulary','Event-level labels and coarse onset times; no actor-specific bounding-box ground truth','Frame-presence statistics count any person prediction, not true-person detection recall']}
for method in ['world_baseline','pose_box_baseline']:
 for hz in [5,.7142857142857143]:
  base=[r for r in rows if r['split']=='evaluation' and r['method']==method and r['hz']==hz]
  for name,pred in [('night',lambda c:'night' in c.get('time_of_recording','').lower()),('day',lambda c:'night' not in c.get('time_of_recording','').lower()),('fall_onset_under_2.8s',lambda c:c['label']=='Fall' and c['fall_onset_s'] is not None and c['fall_onset_s']<2.8),('fall_onset_at_least_2.8s',lambda c:c['label']=='Fall' and c['fall_onset_s'] is not None and c['fall_onset_s']>=2.8),('new_beyond_pilot',lambda c:c['id'] not in {x['id'] for x in json.load(open(R/'manifest_downloaded.json'))})]:
   rr=[r for r in base if pred(M[r['clip']])];cm=collections.Counter(r['outcome'] for r in rr);report['strata'].append(dict(method=method,hz=hz,stratum=name,clips=len(rr),**{k:cm[k] for k in ['TP','FN','FP','TN']}))
phasefile=R/'full_phase_sensitivity.json'
if phasefile.exists():
 pr=json.load(open(phasefile))
 for method in ['world_baseline','pose_box_baseline']:
  rr=[x for x in pr if x['split']=='evaluation' and x['method']==method];report['phase_sensitivity'].append(dict(method=method,phases=len(rr),min_tp=min(x['TP'] for x in rr),max_tp=max(x['TP'] for x in rr),mean_tp=statistics.mean(x['TP'] for x in rr),min_fp=min(x['FP'] for x in rr),max_fp=max(x['FP'] for x in rr)))
report['early_signals_on_fall_clips']=[dict(clip=r['clip'],method=r['method'],hz=r['hz'],signals=r['early_signals']) for r in rows if r['split']=='evaluation' and r['method'] in ['world_baseline','pose_box_baseline'] and r['early_signals']]
report['latencies']={}
for method in ['world_baseline','pose_box_baseline']:
 for hz in [5,.7142857142857143]:
  ls=[r['latency_from_annotation_s'] for r in rows if r['split']=='evaluation' and r['method']==method and r['hz']==hz and r['outcome']=='TP' and r['latency_from_annotation_s'] is not None]
  report['latencies'][method+'_'+str(hz)]=dict(count=len(ls),min=min(ls) if ls else None,max=max(ls) if ls else None,median=statistics.median(ls) if ls else None)
json.dump(report,open(R/'full_analysis_summary.json','w'),indent=2);print(json.dumps(report,indent=2))
