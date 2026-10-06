"""Bounded metric cuboid imports; never claims sensor capture or visual readiness.

USDZ is generated from the validated geometry, so renderer and placement use
identical coordinates. Arbitrary USDZ/mesh ingestion needs a separate parser.
"""
import io
import math
import struct
import zipfile
from typing import Literal
from uuid import UUID, uuid5, NAMESPACE_URL
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

class ImportedDoor(Strict):
    id: UUID
    room_ids: list[UUID] = Field(min_length=2, max_length=2)
    center: Vec3
    width: float = Field(gt=0, le=10)
    height: float = Field(gt=0, le=10)
    normal_axis: Literal['x', 'z']

class ImportedConnection(Strict):
    id: UUID
    room_ids: list[UUID] = Field(min_length=2, max_length=2)
    door_id: UUID


def shared_boundary(a, b, axis):
    tangent = 'z' if axis == 'x' else 'x'
    amin=getattr(a.center,axis)-getattr(a.dimensions,axis)/2
    amax=getattr(a.center,axis)+getattr(a.dimensions,axis)/2
    bmin=getattr(b.center,axis)-getattr(b.dimensions,axis)/2
    bmax=getattr(b.center,axis)+getattr(b.dimensions,axis)/2
    if abs(amax-bmin)<=1e-6: plane=(amax+bmin)/2
    elif abs(bmax-amin)<=1e-6: plane=(bmax+amin)/2
    else: raise ValueError('Door rooms must share the declared boundary')
    low=max(getattr(r.center,tangent)-getattr(r.dimensions,tangent)/2 for r in (a,b))
    high=min(getattr(r.center,tangent)+getattr(r.dimensions,tangent)/2 for r in (a,b))
    if high-low<=1e-6: raise ValueError('Door rooms need a shared boundary aperture')
    floors=[r.center.y-r.dimensions.y/2 for r in (a,b)]
    if abs(floors[0]-floors[1])>1e-6: raise ValueError('Door rooms must share floor height')
    ceiling=min(r.center.y+r.dimensions.y/2 for r in (a,b))
    return plane,tangent,low,high,floors[0],ceiling

class ImportedMapIn(Strict):
    schema_version: Literal['imported-geometry.v1']
    provenance: Literal['synthetic', 'user-authored', 'survey-import']
    units: Literal['m']
    up_axis: Literal['Y']
    coordinate_frame: Literal['import-local']
    rooms: list[ImportedRoom] = Field(min_length=1, max_length=100)
    objects: list[ImportedObject] = Field(default_factory=list, max_length=1000)

    doors: list[ImportedDoor] = Field(default_factory=list, max_length=500)
    connections: list[ImportedConnection] = Field(default_factory=list, max_length=500)

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
        all_ids=[r.id for r in self.rooms]+[o.id for o in self.objects]+[d.id for d in self.doors]+[c.id for c in self.connections]
        if len(all_ids)!=len(set(all_ids)): raise ValueError('Geometry IDs must be globally unique')
        for i,door in enumerate(self.doors):
            if len(set(door.room_ids))!=2 or any(rid not in rooms for rid in door.room_ids):
                raise ValueError('Door must reference two distinct known rooms')
            a,b=(rooms[rid] for rid in door.room_ids)
            plane,tangent,low,high,floor,ceiling=shared_boundary(a,b,door.normal_axis)
            if abs(getattr(door.center,door.normal_axis)-plane)>1e-6:
                raise ValueError('Door center must lie on shared boundary')
            if getattr(door.center,tangent)-door.width/2<low-1e-6 or getattr(door.center,tangent)+door.width/2>high+1e-6:
                raise ValueError('Door aperture exceeds shared boundary')
            if abs(door.center.y-door.height/2-floor)>1e-6 or door.center.y+door.height/2>ceiling+1e-6:
                raise ValueError('Door aperture must start at common floor and fit both rooms')
            for prior in self.doors[:i]:
                if prior.normal_axis==door.normal_axis and abs(getattr(prior.center,door.normal_axis)-plane)<=1e-6 and set(prior.room_ids)&set(door.room_ids):
                    if abs(getattr(prior.center,tangent)-getattr(door.center,tangent))<(prior.width+door.width)/2-1e-6:
                        raise ValueError('Door apertures must not overlap')
        doors={d.id:d for d in self.doors}
        used=set()
        for connection in self.connections:
            door=doors.get(connection.door_id)
            if not door or set(connection.room_ids)!=set(door.room_ids) or len(set(connection.room_ids))!=2:
                raise ValueError('Connection must reference its door and exact room pair')
            if connection.door_id in used: raise ValueError('Door connection must be unique')
            used.add(connection.door_id)
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
    floors=[]; walls=[]; sections=[]; objects=[]; doors=[]; openings=[]
    for r in body.rooms:
        x,y,z=r.center.x,r.center.y,r.center.z
        w,h,d=r.dimensions.x,r.dimensions.y,r.dimensions.z
        floor_y=y-h/2
        floor=element(r.id,'floor',[x,floor_y,z],[w,.02,d])
        floor['vertices']=[dict(x=px,y=floor_y,z=pz) for px,pz in [(x-w/2,z-d/2),(x+w/2,z-d/2),(x+w/2,z+d/2),(x-w/2,z+d/2)]]
        floors.append(floor)
        # Cutaway walls contain real apertures. Both duplicated shared walls
        # are segmented identically, so neither room seals the doorway.
        for k,(axis,plane,tangent,low,high) in enumerate([('z',z-d/2,'x',x-w/2,x+w/2),('z',z+d/2,'x',x-w/2,x+w/2),('x',x-w/2,'z',z-d/2,z+d/2),('x',x+w/2,'z',z-d/2,z+d/2)]):
            apertures=sorted([door for door in body.doors if r.id in door.room_ids and door.normal_axis==axis and abs(getattr(door.center,axis)-plane)<=1e-6],key=lambda door:getattr(door.center,tangent))
            wall_height=min(.7,h)
            pieces=[];cursor=low
            for door in apertures:
                left=getattr(door.center,tangent)-door.width/2;right=getattr(door.center,tangent)+door.width/2
                if left>cursor+1e-6:pieces.append((cursor,left,floor_y,floor_y+wall_height))
                # Short apertures retain only the lintel above the opening.
                if door.height<wall_height-1e-6:pieces.append((left,right,floor_y+door.height,floor_y+wall_height))
                cursor=right
            if cursor<high-1e-6:pieces.append((cursor,high,floor_y,floor_y+wall_height))
            for segment,(a,b,bottom,top) in enumerate(pieces):
                center={'x':x,'y':(bottom+top)/2,'z':z};size={'x':.04,'y':top-bottom,'z':.04}
                center[axis]=plane;center[tangent]=(a+b)/2;size[tangent]=b-a
                wall=element(f'{r.id}-wall-{k}-{segment}','wall',[center[a] for a in 'xyz'],[size[a] for a in 'xyz'])
                walls.append(wall)
        sections.append(dict(id=str(r.id),label=r.name,center=r.center.model_dump(),story=0))
    for o in body.objects: objects.append(element(o.id,o.category,list(o.center.model_dump().values()),list(o.dimensions.model_dump().values())))
    for door in body.doors:
        sizes=[.04,door.height,door.width] if door.normal_axis=='x' else [door.width,door.height,.04]
        item=element(door.id,'door',list(door.center.model_dump().values()),sizes)
        item.update(room_ids=[str(rid) for rid in door.room_ids],width=door.width,height=door.height,normal_axis=door.normal_axis,passable=True)
        # Door is aperture metadata, never a filled cube in USDZ.
        item['attributes']+=['aperture'];doors.append(item)
        openings.append({**item,'category':'opening'})
    connections=[c.model_dump(mode='json') for c in body.connections] if body.connections else [dict(id=str(uuid5(NAMESPACE_URL,f'imported-geometry-connection/{door.id}')),room_ids=[str(rid) for rid in door.room_ids],door_id=str(door.id)) for door in body.doors]
    result=dict(schema_version='imported-geometry.v1',producer='import',framework='ImportedGeometry',units='m',up_axis='Y',coordinate_frame='import-local',geometry_type='3d',walls=walls,floors=floors,objects=objects,doors=doors,windows=[],openings=openings,sections=sections)
    if connections:result['connections']=connections
    return result

def build_usdz(scan):
    lines=['#usda 1.0','(defaultPrim = "ImportedHome"\n metersPerUnit = 1\n upAxis = "Y")','def Xform "ImportedHome" {']
    for i,e in enumerate(scan['floors']+scan['walls']+scan['objects']):
        center=', '.join(str(e['center'][a]) for a in 'xyz'); size=', '.join(str(e['dimensions'][a]/2) for a in 'xyz')
        color='0.70, 0.72, 0.74' if e['category']=='floor' else '0.78, 0.80, 0.84'
        name=('floor_'+e['id'].replace('-', '_')) if e['category']=='floor' else (('wall_' if e['category']=='wall' else 'object_')+str(i))
        lines += [f'def Cube "{name}" {{',f'double3 xformOp:translate = ({center})',f'double3 xformOp:scale = ({size})','uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',f'color3f[] primvars:displayColor = [({color})]','}']
    lines.append('}')
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as archive:
        info=zipfile.ZipInfo('ImportedHome.usda'); padding=(-(30+len(info.filename)+4))%64
        info.extra=struct.pack('<HH',0x1986,padding)+b'\0'*padding
        archive.writestr(info,'\n'.join(lines))
    return out.getvalue()
