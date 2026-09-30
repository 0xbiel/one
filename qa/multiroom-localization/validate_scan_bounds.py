"""Independently reopen each packaged USDZ and compare world bounds to scan JSON."""
import json
from pathlib import Path
from pxr import Usd,UsdGeom
root=Path(__file__).parent;rows=[]
for d in sorted((root/'fixtures').iterdir()):
 stage=Usd.Stage.Open(str(d/'scan.usdz'));cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]);assert UsdGeom.GetStageMetersPerUnit(stage)==1;assert UsdGeom.GetStageUpAxis(stage)=='Y'
 for obj in json.loads((d/'scan.json').read_text())['room_objects']:
  prim=stage.GetPrimAtPath('/Room/'+obj['id']);assert prim.IsA(UsdGeom.Cube);bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange();low=bounds.GetMin();high=bounds.GetMax();center=[(a+b)/2 for a,b in zip(low,high)];dimensions=[b-a for a,b in zip(low,high)];error=max([abs(center[i]-obj['center'][k]) for i,k in enumerate('xyz')]+[abs(dimensions[i]-obj['dimensions'][k]) for i,k in enumerate('xyz')]);assert error<1e-6,(obj['id'],error);rows.append(dict(layout=d.name,object=obj['id'],max_bound_error_m=error))
(root/'scan_bbox_validation.json').write_text(json.dumps(rows,indent=2));print('PASS24USDZ cuboid world-bound comparisons within1micrometre')
