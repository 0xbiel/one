"""Score actual pipeline per-frame results. Never substitutes annotation boxes.
Input JSONL each {clip_id, frame_index, timestamp_s, person_detections, stable_person_tracks, events:[{event_type:'fall_suspected'}]}.
Run metadata JSON must document detector model/checksum, code commit, thresholds and inference FPS.
"""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('--manifest',default=str(Path(__file__).parent/'assets/manifest.json'));p.add_argument('--metadata',required=True);p.add_argument('--output',default='benchmark.json');a=p.parse_args()
m=json.loads(Path(a.manifest).read_text());meta=json.loads(Path(a.metadata).read_text());rows=[json.loads(x) for x in Path(a.results).read_text().splitlines() if x.strip()]
required=['code_commit','model_name','model_sha256','input_kind','inference_fps']
for k in required:
 if not meta.get(k):raise SystemExit('Missing provenance: '+k)
if meta['input_kind']!='rendered_rgb_frames':raise SystemExit('Only rendered_rgb_frames are valid for image-pipeline scores; label oracle controls separately')
summary=[]
for c in m['clips']:
 r=sorted([x for x in rows if x['clip_id']==c['id']],key=lambda x:x['frame_index'])
 if len(r)!=m['frames_per_clip'] or [x['frame_index'] for x in r]!=list(range(m['frames_per_clip'])):raise SystemExit('Missing/duplicate frames: '+c['id'])
 if any(abs(x['timestamp_s']-x['frame_index']/m['fps'])>1e-6 for x in r):raise SystemExit('Incorrect capture timestamps: '+c['id'])
 times=[x['timestamp_s'] for x in r if any(e.get('event_type')=='fall_suspected' for e in x.get('events',[]))]
 detected=any(t>=c['fall_onset_s'] for t in times) if c['expected_fall'] else bool(times)
 delay=min([t-c['fall_onset_s'] for t in times if t>=c['fall_onset_s']],default=None) if c['expected_fall'] else None
 summary.append({'clip':c['id'],'expected_fall':c['expected_fall'],'outcome':('TP' if detected else 'FN') if c['expected_fall'] else ('FP' if detected else 'TN'),'first_signal_s':min(times,default=None),'delay_from_onset_s':delay,'person_detection_frame_rate':sum(x.get('person_detections',0)>0 for x in r)/len(r),'stable_track_frame_rate':sum(x.get('stable_person_tracks',0)>0 for x in r)/len(r),'event_count':len(times),'premature_event_count':sum(t<c['fall_onset_s'] for t in times) if c['expected_fall'] else 0})
counts={k:sum(x['outcome']==k for x in summary) for k in ['TP','FN','FP','TN']};ds=[x['delay_from_onset_s'] for x in summary if x['delay_from_onset_s'] is not None]
out={'provenance':meta,'counts':counts,'fall_recall':counts['TP']/max(1,counts['TP']+counts['FN']),'negative_clip_false_positive_rate':counts['FP']/max(1,counts['FP']+counts['TN']),'median_delay_s':statistics.median(ds) if ds else None,'clips':summary,'claim_scope':'These seven synthetic fixtures only; not clinical or real-world validation'}
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
