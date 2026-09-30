"""Package evaluated Blender meshes as a metric Y-up USDZ and validate it."""
import json,re,zipfile,struct,hashlib
from pathlib import Path
from pxr import Usd,UsdGeom,UsdShade,UsdValidation,UsdUtils,Sdf,Gf
O=Path(__file__).parent/'assets';p=O/'room.usdc';p.unlink(missing_ok=True);(O/'room.usdz').unlink(missing_ok=True)
s=Usd.Stage.CreateNew(str(p));UsdGeom.SetStageUpAxis(s,UsdGeom.Tokens.y);UsdGeom.SetStageMetersPerUnit(s,1);root=UsdGeom.Xform.Define(s,'/Room');s.SetDefaultPrim(root.GetPrim());root.GetPrim().SetCustomDataByKey('regeneratedSyntheticFixture',True)
for m in json.loads((O/'meshes.json').read_text()):
 path='/Room/'+re.sub('[^a-zA-Z0-9_]','_',m['name']);mesh=UsdGeom.Mesh.Define(s,path);mesh.CreatePointsAttr(m['points']);mesh.CreateFaceVertexCountsAttr(m['counts']);mesh.CreateFaceVertexIndicesAttr(m['indices']);mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDisplayColorAttr([Gf.Vec3f(*m['color'])]);mat=UsdShade.Material.Define(s,path+'/Material');shader=UsdShade.Shader.Define(s,path+'/Material/Shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*m['color']));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65);mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
s.GetRootLayer().Save();assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(p)),str(O/'room.usdz'));v=Usd.Stage.Open(str(O/'room.usdz'));r=UsdValidation.ValidationRegistry();names=[x.name for x in r.GetAllValidatorMetadata() if not x.isSuite];issues=UsdValidation.ValidationContext(r.GetOrLoadValidatorsByName(names)).Validate(v);assert not issues,[str(e) for e in issues]
with zipfile.ZipFile(O/'room.usdz') as z:
 for item in z.infolist():
  assert item.compress_type==0
  with open(O/'room.usdz','rb') as f:f.seek(item.header_offset+26);nl,el=struct.unpack('<HH',f.read(4));assert(item.header_offset+30+nl+el)%64==0
result=dict(regenerated=True,meshes=sum(p.IsA(UsdGeom.Mesh) for p in v.Traverse()),usd_validators=len(names),validator_names=names,errors=[],meters_per_unit=UsdGeom.GetStageMetersPerUnit(v),up_axis=UsdGeom.GetStageUpAxis(v),sha256=hashlib.sha256((O/'room.usdz').read_bytes()).hexdigest());(O/'validation.json').write_text(json.dumps(result,indent=2));print(result)
