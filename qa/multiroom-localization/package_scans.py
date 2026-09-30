"""Create metric Y-up simple box USDZ artifacts, independently of detailed RGB scene."""
import json,hashlib
from pathlib import Path
from pxr import Usd,UsdGeom,UsdUtils,UsdValidation,Sdf,Gf
root=Path(__file__).parent
for d in sorted((root/'fixtures').iterdir()):
 data=json.loads((d/'scan.json').read_text());p=d/'scan.usdc';p.unlink(missing_ok=True);s=Usd.Stage.CreateNew(str(p));UsdGeom.SetStageUpAxis(s,'Y');UsdGeom.SetStageMetersPerUnit(s,1);prim=UsdGeom.Xform.Define(s,'/Room');s.SetDefaultPrim(prim.GetPrim())
 boxes=list(data['room_objects'])+[
 dict(id='floor',center=dict(x=0,y=-.05,z=0),dimensions=dict(x=6,y=.1,z=5)),
 dict(id='rear_wall',center=dict(x=0,y=1.4,z=-2.55),dimensions=dict(x=6.1,y=2.8,z=.1)),
 dict(id='left_wall',center=dict(x=-3.05,y=1.4,z=0),dimensions=dict(x=.1,y=2.8,z=5)),
 dict(id='right_wall',center=dict(x=3.05,y=1.4,z=0),dimensions=dict(x=.1,y=2.8,z=5))]
 for b in boxes:
  cube=UsdGeom.Cube.Define(s,'/Room/'+b['id']);cube.CreateSizeAttr(1);cube.AddTranslateOp().Set(Gf.Vec3d(*[b['center'][k] for k in 'xyz']));cube.AddScaleOp().Set(Gf.Vec3f(*[b['dimensions'][k] for k in 'xyz']));cube.CreateDisplayColorAttr([Gf.Vec3f(.65,.68,.72)])
 s.GetRootLayer().Save();z=d/'scan.usdz';z.unlink(missing_ok=True);assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(p)),str(z));stage=Usd.Stage.Open(str(z));assert sum(x.IsA(UsdGeom.Cube) for x in stage.Traverse())==len(boxes);assert not any(x.IsA(UsdGeom.Mesh) for x in stage.Traverse())
 reg=UsdValidation.ValidationRegistry();validators=[x.name for x in reg.GetAllValidatorMetadata() if not x.isSuite];issues=UsdValidation.ValidationContext(reg.GetOrLoadValidatorsByName(validators)).Validate(stage);assert not issues,[str(x) for x in issues]
 (d/'scan_validation.json').write_text(json.dumps(dict(boxes=len(boxes),meshes=0,meters_per_unit=1,up_axis='Y',validators=len(validators),errors=[],sha256=hashlib.sha256(z.read_bytes()).hexdigest()),indent=2))
 print(d.name,len(boxes))
