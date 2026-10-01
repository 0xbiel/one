"""Single-owner serial benchmark; watchdog budget begins after solver-ready record.

Use --case for a developer preflight; the final run omits it. Outputs are never
reused. Driver does not load ground truth. Summarization is a separate process.
"""
import json,sys,time,os,subprocess,argparse,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--source',required=True);p.add_argument('--label',required=True);p.add_argument('--output',default='final-results');args=p.parse_args();root=Path(__file__).parent
manifest=json.loads((root/'manifest.json').read_text());records=[]
for m in manifest:
 if args.case and m['case']!=args.case:continue
 assert hashlib.sha256((root/m['input']).read_bytes()).hexdigest()==m['sha256']
 out=root/args.output/args.label/(m['case']+'.json');out.parent.mkdir(parents=True,exist_ok=True);ready=out.with_suffix('.started.json')
 if out.exists() or ready.exists():raise RuntimeError(f'Fresh output required: {out}')
 env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':args.source,'ONE_GEOMETRY_SOLVER_DEVICE':'cpu'}
 with open(out.with_suffix('.log'),'w') as log:
  proc=subprocess.Popen([sys.executable,str(root/'run_single.py'),str(root/m['input']),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT);launch=time.monotonic();meta=None;status=None
  while proc.poll() is None:
   if meta is None and ready.exists():
    try:meta=json.loads(ready.read_text())
    except json.JSONDecodeError:pass
   if meta is None and time.monotonic()-launch>60:status='startup_timeout';proc.kill();break
   if meta is not None and time.monotonic()-meta['compute_started_monotonic']>=180:status='timeout';proc.kill();break
   time.sleep(.05)
  proc.wait()
 if status is None and not out.exists():status='process_error'
 if status is not None:
  out.write_text(json.dumps(dict(**(meta or {}),status=status,camera_to_world=None,time_budget_seconds=180,wall_seconds=time.monotonic()-(meta['compute_started_monotonic'] if meta else launch),status_origin='exclusive benchmark watchdog; no lock waiting'),indent=2))
 value=json.loads(out.read_text());records.append(dict(case=m['case'],status=value['status'],seconds=value.get('wall_seconds')));print(args.label,m['case'],value['status'],value.get('wall_seconds'),flush=True)
(root/args.output/(args.label+'_run.json')).write_text(json.dumps(records,indent=2))

