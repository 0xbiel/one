"""Restore exact frozen requests/results from compact checked-in parts, without model inference."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json
ROOT=Path(__file__).resolve().parent

def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args();r=ROOT/'replay'
 if not (r/'replay_manifest.json').is_file():
  raise SystemExit('Exact no-model replay is unavailable in this compact checkout: payloads were omitted. See omitted_payloads.json and FRESH_RUN.md. A fresh model run is a different workflow.')
 m=json.loads((r/'replay_manifest.json').read_text())
 required=[r/'run_metadata.json']+[r/x['file'] for x in m['chunks']]+[r/f['file'] for c in m['cases'] for f in c['frames']]+[r/'raw_results'/f"{c['case']}.json.gz" for c in m['cases']]
 missing=sorted({str(x.relative_to(ROOT)) for x in required if not x.is_file()})
 if missing:
  raise SystemExit('Exact no-model replay payloads are missing ('+str(len(missing))+' files). See omitted_payloads.json and FRESH_RUN.md. First missing: '+missing[0])
 pieces=[]
 for item in m['chunks']:
  data=(r/item['file']).read_bytes();assert digest(data)==item['sha256'];assert len(data)==item['bytes'];pieces.append(data)
 packed=b''.join(pieces);assert digest(packed)==m['common_gzip_sha256'];common=json.loads(gzip.decompress(packed));metadata=json.loads((r/'run_metadata.json').read_text())
 for case in m['cases']:
  frames=[]
  for f in case['frames']:
   raw=(r/f['file']).read_bytes();assert digest(raw)==f['sha256'];values={'frame_base64':base64.b64encode(raw).decode(),'width':f['width'],'height':f['height']};frames.append({k:values[k] for k in f['keys']})
  values={**common,'frames':frames,**case['details']};request={k:values[k] for k in case['keys']};data=json.dumps(request,separators=(',',':')).encode();assert digest(data)==case['sha256'],case['case']
  result=gzip.decompress((r/'raw_results'/f"{case['case']}.json.gz").read_bytes());meta=metadata[case['case']];assert digest(result)==meta['result_sha256']
  if not args.check_only:
   dest=ROOT/'pilot'/case['input'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
   dest=ROOT/'pilot/final-results/current'/f"{case['case']}.json";dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(result);dest.with_suffix('.started.json').write_text(json.dumps(meta['started'],indent=2));dest.with_suffix('.log').write_text(meta['log'])
 print(json.dumps({'request_sha256_verified':len(m['cases']),'raw_results_verified':len(m['cases']),'model_inference_run':False,'written':not args.check_only}))
if __name__=='__main__':main()
