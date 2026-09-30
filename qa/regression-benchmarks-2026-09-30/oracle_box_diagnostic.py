"""Explicit ideal-box diagnostic ONLY. This is not image-detector evidence."""
import json,argparse
from pathlib import Path
from datetime import datetime,timedelta,timezone
import app.vision as vision
from app.fall import FallDetectionTracker
p=argparse.ArgumentParser();p.add_argument('--controlled-assets',required=True);p.add_argument('--physics-assets',nargs=2,required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.controlled_assets);m=json.loads((root/'manifest.json').read_text());cases=[(c['id'],root/c['id']/'ground_truth_frames.json',c['expected_fall']) for c in m['clips']]+[(name,Path(path)/'ground_truth_frames.json',True) for name,path in zip(['fall_physics_ragdoll','fall_physics_forward'],a.physics_assets)]
class Clock(datetime):
 current=datetime(2026,9,30,12,tzinfo=timezone.utc)
 @classmethod
 def now(cls,tz=None):return cls.current
vision.datetime=Clock;result=[]
for name,path,expected in cases:
 rows=json.loads(path.read_text());direct=FallDetectionTracker();tracked=FallDetectionTracker();stability=vision.TemporalStabilityTracker();direct_events=[];tracked_events=[]
 for r in rows:
  Clock.current=datetime(2026,9,30,12,tzinfo=timezone.utc)+timedelta(seconds=r['timestamp_s']);bbox=r['bbox_gt_xyxy'];item={'label':'person','confidence':1.0,'bbox':bbox,'track_id':1}
  sig=direct.update(name,item,frame_width=640,frame_height=480,observed_at=Clock.current)
  if sig:direct_events.append(r['timestamp_s'])
  for det in stability.update([vision.Detection('person',1.0,tuple(bbox),Clock.current)]):
   sig=tracked.update(name,dict(item,track_id=det.track_id),frame_width=640,frame_height=480,observed_at=Clock.current)
   if sig:tracked_events.append(r['timestamp_s'])
 result.append({'clip':name,'expected_fall':expected,'direct_fall_rule_signal_times_s':direct_events,'stability_plus_fall_rule_signal_times_s':tracked_events})
Path(a.output).write_text(json.dumps({'input_kind':'oracle_ground_truth_boxes','warning':'Perfect projected annotation boxes at confidence1, not inferred images; never combine these scores with real detector evaluation. Diagnostic exposes heuristic limitations only.','clips':result},indent=2));print(json.dumps(result,indent=2))
