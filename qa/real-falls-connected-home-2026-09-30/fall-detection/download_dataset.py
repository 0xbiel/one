import urllib.request,urllib.parse,json,pathlib,concurrent.futures,hashlib,csv,re
ROOT=pathlib.Path(__file__).parent; base=ROOT/'dataset';tree=json.load(open(ROOT/'dataset_tree.json'))
# Frozen metadata-stratified pilot before inference. S1 development, S2-4 held out.
sel={1:{'Fall':[3,5,9,12],'ADL':[1,8,13,15]},2:{'Fall':[1,4,12,17],'ADL':[4,9,12,22]},3:{'Fall':[5,13,16,20],'ADL':[5,8,9,19]},4:{'Fall':[3,7,10,11],'ADL':[2,5,10,15]}}
manifest=[]
for subj,groups in sel.items():
 for group,ids in groups.items():
  rows=list(csv.reader((base/f'Subject {subj}'/f'{group}.csv').read_text().splitlines()))[1:]
  for i in ids:
   name=f'{i:02}.mp4';row=next(r for r in rows if r[0].strip()==name);p=f'Subject {subj}/{group}/{name}'
   ann=','.join(row[5:]);m=re.search(r'Fall(?:ing)?[^\[]*\[\s*(\d+(?:\.\d+)?)',ann)
   manifest.append(dict(id=f's{subj}_{group.lower()}_{i:02}',subject=subj,split='development' if subj==1 else 'evaluation',label=group,path=p,description=row[4].strip(),annotation=ann,fall_onset_s=float(m.group(1)) if m else None,source=f'https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/blob/{tree["sha"]}/'+urllib.parse.quote(p)))
(ROOT/'manifest_preinference.json').write_text(json.dumps(dict(dataset_revision=tree['sha'],selection='Metadata-stratified coverage selected before inference; no random population estimate; subject 1 development, subjects 2-4 evaluation. 5 Hz sampling planned. Staged episodes, not clinical validation.',clips=manifest),indent=2))
def f(item):
 p=item['path'];u='https://raw.githubusercontent.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/'+tree['sha']+'/'+urllib.parse.quote(p)
 out=base/p;out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists():out.write_bytes(urllib.request.urlopen(u,timeout=120).read())
 item['sha256']=hashlib.sha256(out.read_bytes()).hexdigest();item['bytes']=out.stat().st_size
 print(item['id'],item['bytes'],flush=True);return item
result=list(concurrent.futures.ThreadPoolExecutor(6).map(f,manifest));(ROOT/'manifest_downloaded.json').write_text(json.dumps(result,indent=2))
