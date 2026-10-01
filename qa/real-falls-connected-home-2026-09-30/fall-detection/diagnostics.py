import pathlib,json,collections,statistics,argparse
p=argparse.ArgumentParser();p.add_argument("--prefix",default="");p.add_argument("--manifest",default="manifest_downloaded.json");a=p.parse_args()
R=pathlib.Path(__file__).parent;M={x['id']:x for x in json.load(open(R/a.manifest))};out=[]
for model in ['world','pose']:
 for split in ['development','evaluation']:
  p=R/f'{a.prefix}inference_{model}_{split}.jsonl'
  if not p.exists():continue
  group=collections.defaultdict(list)
  for l in p.read_text().splitlines():r=json.loads(l);group[r['clip']].append(r)
  for cid,rows in group.items():
   if len(rows)!=M[cid]['sampled_frames']:continue
   onset=M[cid]['fall_onset_s'];phases=['all','pre','post'] if onset is not None else ['all']
   for phase in phases:
    rr=[x for x in rows if phase=='all' or (phase=='pre' and x['time_s']<onset) or (phase=='post' and x['time_s']>=onset)]
    if not rr:continue
    people=[[d for d in r['detections'] if d['label']=='person'] for r in rr]
    d=dict(clip=cid,model=model,split=split,phase=phase,frames=len(rr),person_at_020=sum(bool(x) for x in people),person_at_055=sum(any(x['confidence']>=.55 for x in ds) for ds in people))
    if model=='pose':
     d['four_torso_points_conf_050']=sum(any(all(p['keypoints'][k][2]>=.5 for k in [5,6,11,12]) for p in ds) for ds in people)
    out.append(d)
json.dump(out,open(R/f'{a.prefix}diagnostics.json','w'),indent=2)
