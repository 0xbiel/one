import json,hashlib
from pathlib import Path
from pxr import Usd,UsdGeom,UsdUtils,UsdValidation,Sdf,Gf
root=Path(__file__).parent;d=root/'fixtures/connected_home';data=json.loads((d/'scan.json').read_text());boxes=data['room_objects']+json.loads((d/'structural_boxes.json').read_text());p=d/'connected_home_simple_boxes.usdc';p.unlink(missing_ok=True)
s=Usd.Stage.CreateNew(str(p));UsdGeom.SetStageUpAxis(s,'Y');UsdGeom.SetStageMetersPerUnit(s,1);prim=UsdGeom.Xform.Define(s,'/Home');s.SetDefaultPrim(prim.GetPrim())
for i,b in enumerate(boxes):
 cube=UsdGeom.Cube.Define(s,f'/Home/box_{i:03d}');cube.GetPrim().SetDisplayName(b['id']);cube.CreateSizeAttr(1);cube.AddTranslateOp().Set(Gf.Vec3d(*[b['center'][k] for k in 'xyz']));cube.AddScaleOp().Set(Gf.Vec3f(*[b['dimensions'][k] for k in 'xyz']));cube.CreateDisplayColorAttr([Gf.Vec3f(*((.35,.55,.68) if i<len(data['room_objects']) else (.7,.72,.7)))])
s.GetRootLayer().Save();z=d/'connected_home_simple_boxes.usdz';z.unlink(missing_ok=True);assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(p)),str(z));stage=Usd.Stage.Open(str(z));assert sum(x.IsA(UsdGeom.Cube) for x in stage.Traverse())==len(boxes);assert not any(x.IsA(UsdGeom.Mesh) for x in stage.Traverse())
reg=UsdValidation.ValidationRegistry();validators=[x.name for x in reg.GetAllValidatorMetadata() if not x.isSuite];issues=UsdValidation.ValidationContext(reg.GetOrLoadValidatorsByName(validators)).Validate(stage);assert not issues,[str(x) for x in issues]
report=dict(boxes=len(boxes),semantic_boxes=len(data['room_objects']),structural_boxes=len(boxes)-len(data['room_objects']),meshes=0,meters_per_unit=1,up_axis='Y',validators=len(validators),errors=[],sha256=hashlib.sha256(z.read_bytes()).hexdigest());(d/'scan_validation.json').write_text(json.dumps(report,indent=2));print(report)
