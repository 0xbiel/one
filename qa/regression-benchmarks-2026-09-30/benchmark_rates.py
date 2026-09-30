"""Paired exact-input microbenchmark; correctness counts precede speed claims."""
import argparse, importlib.util, json, statistics, time, random
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
mods={}
for name,path in [('baseline',a.baseline),('candidate',a.candidate)]:
 spec=importlib.util.spec_from_file_location(name,Path(path)/'app/analytics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[name]=m
cases=[([7]*7,'stable'),([7]*9,'stable'),([7]*30,'stable'),([7,7,7,6,6,6,6],'decreasing'),([6,6,6,7,7,7,7],'increasing'),([0]*7,'unknown'),([1]*3,'unknown'),([1,1,2,2],'increasing')]
runs={n:[] for n in mods};random.seed(42)
for repeat in range(21):
 names=list(mods);random.shuffle(names)
 for name in names:
  start=time.perf_counter_ns()
  for i in range(10000):mods[name]._trend(cases[i%len(cases)][0])
  if repeat:runs[name].append((time.perf_counter_ns()-start)/10000)
out={'method':'20 paired randomized-order batches × 10,000 calls after warmup, same process and inputs; nanoseconds/call; noisy shared CPU, no speedup claim','results':{name:{'correct_cases':sum(m._trend(v)==e for v,e in cases),'case_count':len(cases),'median_ns_per_call':statistics.median(runs[name]),'p95_batch_ns_per_call':sorted(runs[name])[18],'batch_ns_per_call':runs[name]} for name,m in mods.items()}}
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
