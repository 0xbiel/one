"""Bounded metric cuboid imports; never claims sensor capture or visual readiness.

USDZ is generated from the validated geometry, so renderer and placement use
identical coordinates. Arbitrary USDZ/mesh ingestion needs a separate parser.
"""
import io
import math
import struct
import zipfile
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Vec3(Strict):
    x: float = Field(ge=-100, le=100)
    y: float = Field(ge=-100, le=100)
    z: float = Field(ge=-100, le=100)

class Size3(Strict):
    x: float = Field(gt=0, le=100)
    y: float = Field(gt=0, le=20)
    z: float = Field(gt=0, le=100)

class ImportedRoom(Strict):
    id: UUID
    name: str = Field(min_length=1, max_length=120)
    center: Vec3
    dimensions: Size3

    def contains(self, point):
        return all(abs(point[i] - getattr(self.center, a)) <= getattr(self.dimensions, a)/2 + 1e-6 for i,a in enumerate('xyz'))

class ImportedObject(Strict):
    id: UUID
    room_id: UUID
    category: Literal['table', 'sofa', 'bed', 'storage', 'fixture']
    center: Vec3
    dimensions: Size3

class ImportedMapIn(Strict):
    schema_version: Literal['imported-geometry.v1']
    provenance: Literal['synthetic', 'user-authored', 'survey-import']
    units: Literal['m']
    up_axis: Literal['Y']
    coordinate_frame: Literal['import-local']
    rooms: list[ImportedRoom] = Field(min_length=1, max_length=100)
    objects: list[ImportedObject] = Field(default_factory=list, max_length=1000)

    @model_validator(mode='after')
    def coherent(self):
        rooms={r.id:r for r in self.rooms}
        if len(rooms)!=len(self.rooms) or len({o.id for o in self.objects})!=len(self.objects):
            raise ValueError('Geometry IDs must be unique')
        for i,r in enumerate(self.rooms):
            for a in 'xyz':
                if abs(getattr(r.center,a))+getattr(r.dimensions,a)/2 > 100:
                    raise ValueError('Room bounds exceed the 100 meter import envelope')
            for other in self.rooms[i+1:]:
                if all(abs(getattr(r.center,a)-getattr(other.center,a)) < (getattr(r.dimensions,a)+getattr(other.dimensions,a))/2-1e-6 for a in 'xyz'):
                    raise ValueError('Room interiors must not overlap')
        for obj in self.objects:
            room=rooms.get(obj.room_id)
            if room is None: raise ValueError('Object references an unknown room')
            for a in 'xyz':
                if abs(getattr(obj.center,a)-getattr(room.center,a))+getattr(obj.dimensions,a)/2 > getattr(room.dimensions,a)/2+1e-6:
                    raise ValueError('Object must lie inside its room')
        return self

class CameraPlacementIn(Strict):
    camera_id: UUID
    map_id: UUID
    room_id: UUID
    camera_to_world: list[list[float]] = Field(min_length=4,max_length=4)
    confirmed: Literal[True]

    @model_validator(mode='after')
    def rigid(self):
        m=self.camera_to_world
        if any(len(row)!=4 for row in m) or any(not math.isfinite(v) for row in m for v in row):
            raise ValueError('Pose must be a finite 4x4 matrix')
        if any(abs(m[3][i]-v)>1e-6 for i,v in enumerate([0,0,0,1])):
            raise ValueError('Pose must be affine')
        for i in range(3):
            for j in range(3):
                if abs(sum(m[k][i]*m[k][j] for k in range(3))-(1 if i==j else 0))>1e-4:
                    raise ValueError('Pose must have orthonormal rotation without scale')
        det=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
        if abs(det-1)>1e-4: raise ValueError('Pose rotation must be right handed')
        return self

def element(id, category, center, dimensions):
    x,y,z=center
    return dict(id=str(id),category=category,confidence='low',center=dict(zip('xyz',center)),dimensions=dict(zip('xyz',dimensions)),transform=[[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]],vertices=[],attributes=['imported'])

def normalized_geometry(body):
    floors=[]; walls=[]; sections=[]; objects=[]
    for r in body.rooms:
        x,y,z=r.center.x,r.center.y,r.center.z
        w,h,d=r.dimensions.x,r.dimensions.y,r.dimensions.z
        floor=element(r.id,'floor',[x,y-h/2,z],[w,.02,d])
        floor['vertices']=[dict(x=px,y=y-h/2,z=pz) for px,pz in [(x-w/2,z-d/2),(x+w/2,z-d/2),(x+w/2,z+d/2),(x-w/2,z+d/2)]]
        floors.append(floor)
        # Cutaway bounding-box walls keep the imported room layout legible.
        for k,(cx,cz,sx,sz) in enumerate([(x,z-d/2,w,.04),(x,z+d/2,w,.04),(x-w/2,z,.04,d),(x+w/2,z,.04,d)]):
            walls.append(element(f'{r.id}-wall-{k}','wall',[cx,y-h/2+.35,cz],[sx,.7,sz]))
        sections.append(dict(id=str(r.id),label=r.name,center=r.center.model_dump(),story=0))
    for o in body.objects: objects.append(element(o.id,o.category,list(o.center.model_dump().values()),list(o.dimensions.model_dump().values())))
    return dict(schema_version='imported-geometry.v1',producer='import',framework='ImportedGeometry',units='m',up_axis='Y',coordinate_frame='import-local',geometry_type='3d',walls=walls,floors=floors,objects=objects,doors=[],windows=[],openings=[],sections=sections)

def build_usdz(scan):
    lines=['#usda 1.0','(defaultPrim = "ImportedHome"\n metersPerUnit = 1\n upAxis = "Y")','def Xform "ImportedHome" {']
    for i,e in enumerate(scan['floors']+scan['walls']+scan['objects']):
        center=', '.join(str(e['center'][a]) for a in 'xyz'); size=', '.join(str(e['dimensions'][a]/2) for a in 'xyz')
        color='0.35, 0.60, 0.75' if e['category']=='floor' else '0.78, 0.80, 0.84'
        lines += [f'def Cube "part_{i}" {{',f'double3 xformOp:translate = ({center})',f'double3 xformOp:scale = ({size})','uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',f'color3f[] primvars:displayColor = [({color})]','}']
    lines.append('}')
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as archive:
        info=zipfile.ZipInfo('ImportedHome.usda'); padding=(-(30+len(info.filename)+4))%64
        info.extra=struct.pack('<HH',0x1986,padding)+b'\0'*padding
        archive.writestr(info,'\n'.join(lines))
    return out.getvalue()
