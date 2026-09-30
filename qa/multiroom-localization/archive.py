"""Retain all immutable inputs and only complete output files in a verifiable ZIP."""
import json,zipfile,hashlib,sys
from pathlib import Path
root=Path(__file__).parent;out=Path(sys.argv[1]);complete={}
for p in (root/'final-results').glob('*/*.json'):
 if p.name.endswith('.started.json'):continue
 try:complete[str(p.relative_to(root))]=json.loads(p.read_text())['status']
 except (ValueError,KeyError):pass
status=dict(expected_outputs=48,completed_outputs=len(complete),in_progress=len(complete)!=48,completed_cases=complete)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or 'provisional' in p.parts or p.name=='library_receipt.json':continue
  relative=str(p.relative_to(root))
  if relative.startswith('final-results/'):
   if p.suffix=='.log' and str(p.with_suffix('.json').relative_to(root)) not in complete:continue
   if p.suffix=='.json' and not p.name.endswith('.started.json') and relative not in complete and not p.name.endswith('_run.json'):continue
  z.write(p,'camera-localization-benchmark/'+relative)
 z.writestr('camera-localization-benchmark/RUN_STATUS.json',json.dumps(status,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for n in z.namelist():
  if n.endswith('.json'):json.loads(z.read(n))
print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),**status),indent=2))
