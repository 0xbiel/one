import json,pathlib,sys,collections,datetime,statistics,argparse
p=argparse.ArgumentParser();p.add_argument("--prefix",default="");p.add_argument("--phase",type=int,default=0);p.add_argument("--manifest",default="manifest_downloaded.json");p.add_argument("--base-fps",type=int,default=5);p.add_argument("--rates",default="5,0.7142857142857143");args=p.parse_args();rates=list(map(float,args.rates.split(",")))
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'one'))
import app.vision as vision
from app.vision import Detection,TemporalStabilityTracker
from app.fall import FallDetectionTracker
REAL_DT=datetime.datetime
class Clock(REAL_DT):
 current=REAL_DT(2026,9,30,tzinfo=datetime.timezone.utc)
 @classmethod
 def now(cls,tz=None):return cls.current
vision.datetime=Clock
manifest={c['id']:c for c in json.load(open(R/args.manifest))}
methods=['world_baseline','world_single_identity','world_confidence_ablation','pose_box_baseline','pose_single_identity']
if args.prefix=='cadence20_':methods+=['world_duration_matched','pose_duration_matched']
output=[]
for model in ['world','pose']:
 grouped=collections.defaultdict(list)
 for split in ['development','evaluation']:
  f=R/f'{args.prefix}inference_{model}_{split}.jsonl'
  if f.exists():
   for l in f.read_text().splitlines():v=json.loads(l);grouped[v['clip']].append(v)
 for cid,rows in grouped.items():
  c=manifest[cid]
  if len(rows)!=c['sampled_frames']:continue
  for method in [x for x in methods if x.startswith(model)]:
   for hz in rates:
    hits=round(.4*hz)+1 if 'duration_matched' in method and hz>=5 else 3
    tracker=TemporalStabilityTracker(min_hits=hits);fall=FallDetectionTracker(min_hits=hits);identity_hits=0;signals=[];raw_count=high_count=stable_count=eligible_count=0;trackids=set();postures=collections.Counter();traces=[]
    for i,row in enumerate(rows):
     if (i-args.phase)%round(args.base_fps/hz):continue
     t=row['time_s'];Clock.current=REAL_DT(2026,9,30,tzinfo=datetime.timezone.utc)+datetime.timedelta(seconds=t)
     ds=[Detection(d['label'],d['confidence'],tuple(d['bbox']),Clock.current) for d in row['detections']]
     people=[d for d in ds if d.label=='person'];raw_count+=bool(people);high_count+=any(d.confidence>=.55 for d in people)
     if 'single_identity' in method:
      identity_hits+=bool(people)
      stable=[Detection(d.label,d.confidence,d.bbox,d.frame_at,1) for d in sorted(people,key=lambda x:x.confidence,reverse=True)[:1]] if identity_hits>=3 else []
     else:stable=tracker.update(ds)
     stable_people=[d for d in stable if d.label=='person'];stable_count+=bool(stable_people)
     eligible=[]
     for d in stable_people:
      item=dict(label=d.label,confidence=1.0 if 'confidence_ablation' in method else d.confidence,bbox=list(d.bbox),track_id=d.track_id)
      w=d.bbox[2]-d.bbox[0];h=d.bbox[3]-d.bbox[1]
      if item['confidence']>=.55 and w>=row['width']*.08 and h>=row['height']*.08:
       eligible.append(d);postures[FallDetectionTracker._posture(w,h)[0]]+=1
      trackids.add(d.track_id)
      sig=fall.update('pilot',item,frame_width=row['width'],frame_height=row['height'],observed_at=Clock.current)
      if sig:signals.append(dict(time_s=t,confidence=sig.confidence,metrics=sig.metrics))
     eligible_count+=bool(eligible)
     traces.append(dict(time_s=t,raw_person=len(people),high_person=sum(x.confidence>=.55 for x in people),stable_person=len(stable_people),eligible_person=len(eligible),track_ids=[x.track_id for x in stable_people]))
    onset=c['fall_onset_s'];valid_signals=[s for s in signals if onset is None or s['time_s']>=onset-.2]
    first=valid_signals[0]['time_s'] if valid_signals else None
    isfall=c['label']=='Fall';positive=bool(valid_signals if isfall else signals)
    output.append(dict(clip=cid,split=c['split'],method=method,hz=hz,label=c['label'],outcome=('TP' if positive else 'FN') if isfall else ('FP' if positive else 'TN'),sampled_frames=len(traces),raw_person_frames=raw_count,high_person_frames=high_count,stable_person_frames=stable_count,eligible_person_frames=eligible_count,track_ids=sorted(trackids),eligible_postures=dict(postures),signals=signals,latency_from_annotation_s=round(first-onset,3) if first is not None and onset is not None else None,early_signals=[s for s in signals if onset is not None and s['time_s']<onset-.2],trace=traces))
summary=[]
for split in ['development','evaluation']:
 for method in methods:
  for hz in rates:
   rr=[r for r in output if r['split']==split and r['method']==method and r['hz']==hz]
   if not rr:continue
   cm=collections.Counter(r['outcome'] for r in rr);tp=cm['TP'];fp=cm['FP'];fn=cm['FN'];lat=[r['latency_from_annotation_s'] for r in rr if r['outcome']=='TP' and r['latency_from_annotation_s'] is not None];n=sum(r['sampled_frames'] for r in rr)
   false_events=sum(len(r['signals']) if r['label']=='ADL' else len(r['early_signals']) for r in rr)
   summary.append(dict(split=split,method=method,hz=hz,clips=len(rr),timed_event_false_alerts=false_events,timed_event_precision=tp/(tp+false_events) if tp+false_events else None,early_alerts_on_fall_clips=sum(len(r['early_signals']) for r in rr),latency_sample_count=len(lat),**{k:cm[k] for k in ['TP','FN','FP','TN']},precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,median_latency=statistics.median(lat) if lat else None,person_raw_coverage=sum(r['raw_person_frames'] for r in rr)/n,person_eligible_coverage=sum(r['eligible_person_frames'] for r in rr)/n,frames=n))
json.dump(dict(summary=summary,clips=output),open(R/f'{args.prefix}scores{"_phase"+str(args.phase) if args.phase else ""}.json','w'),indent=2)
print(json.dumps(summary,indent=2))
