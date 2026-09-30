"""Feed all generated USDZ bytes through the actual backend upload validator."""
import json,hashlib,os
from pathlib import Path
from app.roomplan import validate_roomplan_usdz
root=Path(__file__).parent;rows=[]
for p in sorted((root/'fixtures').glob('*/scan.usdz')):
 data=p.read_bytes();validate_roomplan_usdz(data);rows.append(dict(layout=p.parent.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),backend_package_validation='passed'))
output=dict(source_sha256=hashlib.sha256((Path(os.environ['PYTHONPATH'])/'app/roomplan.py').read_bytes()).hexdigest(),results=rows)
(root/('backend_usdz_validation_'+os.environ['VALIDATION_LABEL']+'.json')).write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
