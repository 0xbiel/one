"""Compare every ordered DMatch tuple to the untouched reference function."""
import ast,json,sys,os,time
from pathlib import Path
import cv2,numpy as np
from geometry_service import localization as new
source=Path(os.environ.get('REFERENCE_ROOT','/workspace/shared/one-replacement'))/'geometry_service/localization.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pose_guided_matches');env=dict(vars(new));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env);old=env['_pose_guided_matches']
class Scorer:
 threshold=.55
 def score_pairs(self,a,b,qa=None,la=None):return .1+.8*(np.bitwise_xor(a,b).sum(axis=1)%101)/100
count=0

def compare(kw):
 global count
 a=old(**kw);b=new._pose_guided_matches(**kw);ta=[(m.queryIdx,m.trainIdx,m.distance) for m in a];tb=[(m.queryIdx,m.trainIdx,m.distance) for m in b];assert ta==tb,(count,ta,tb);count+=1

start=time.perf_counter()
for seed in range(30):
 rng=np.random.default_rng(seed);nq=160;nl=240;xy=rng.uniform([0,0],[480,360],(nq,2));desc=rng.integers(0,256,(nq,32),dtype=np.uint8);lds=rng.integers(0,256,(nl,32),dtype=np.uint8);points=np.column_stack((rng.uniform(-2,2,nl),rng.uniform(-1.5,1.5,nl),rng.uniform(.3,6,nl)))
 # Correlated pairs straddle all original Hamming and ratio thresholds.
 for i,bits in enumerate([0,1,16,31,32,33,39,40,41,48,55,56,57,64]*4):
  lds[i]=desc[i%nq];flat=np.unpackbits(lds[i]);flat[:bits]^=1;lds[i]=np.packbits(flat);points[i]=[(xy[i,0]-240)*3/320,(xy[i,1]-180)*3/320,3]
 # Deliberate repeated descriptors, exact pixel ties, near-duplicate 3D points.
 desc[60:66]=desc[0];xy[60:63]=xy[0];xy[63:66]=xy[0]+[4,0];lds[60:66]=desc[0];points[60:63]=points[0];points[63:66]=points[0]+[.03,0,0]
 keypoints=[cv2.KeyPoint(float(x),float(y),7) for x,y in xy];common=dict(frame_width=480,frame_height=360,keypoints=keypoints,descriptors=desc,landmark_points=points,landmark_descriptors=lds,query_responses=np.arange(nq)/nq,landmark_responses=np.arange(nl)/nl)
 for radius in [0,8,36,72,96]:
  for learned in [None,Scorer()]:
   pose=dict(camera_matrix=np.array([[320.,0,240],[0,320,180],[0,0,1]]),rvec=np.array([.03,-.02,0.]) if seed%2 else np.zeros(3),tvec=np.array([.02,0,-.03]) if seed%3 else np.zeros(3));compare(dict(**common,pose=pose,radius_px=radius,learned_matcher=learned))
 # In-place mutation must invalidate the cached shortlist.
 desc[0]^=255;lds[1]^=255;compare(dict(**common,pose=pose,radius_px=36))
 # Different array objects with identical content remain exactly equivalent.
 compare(dict(**{**common,'descriptors':desc.copy(),'landmark_descriptors':lds.copy()},pose=pose,radius_px=36))
for n in [0,1]:
 compare(dict(frame_width=480,frame_height=360,keypoints=[cv2.KeyPoint(2.,2.,7)]*n,descriptors=np.zeros((n,32),np.uint8),landmark_points=np.zeros((n,3)),landmark_descriptors=np.zeros((n,32),np.uint8),pose={}))
assert all(x is None for x in new._guided_descriptor_pair_candidates(np.zeros((1500,32),np.uint8),np.zeros((1500,32),np.uint8)))
assert new._guided_descriptor_pair_candidates(np.zeros((3401,32),np.uint8),np.zeros((2,32),np.uint8)) is None
print(json.dumps(dict(exact_ordered_match_comparisons=count,passed=True,dense_pair_budget_fallback=True,in_place_mutation_checked=True,seconds=time.perf_counter()-start)))
