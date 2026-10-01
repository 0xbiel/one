import unittest
import numpy as np
from geometry_service.localization import _depth_at, _consensus_stats

class IntegrityTests(unittest.TestCase):
    def test_scaled_intrinsics_grid(self):
        d=np.full((4,4),2.0,dtype=np.float32);d[2,2]=4.0
        self.assertEqual(_depth_at(d,60,60,100,100),4.0)
        self.assertIsNone(_depth_at(d,-1,20,100,100))
    def test_depth_edge_preserves_center_surface(self):
        d=np.full((3,3),5.0,dtype=np.float32);d[1,1]=1.0
        self.assertEqual(_depth_at(d,1,1,3,3),1.0)
        d[1,1]=np.nan
        self.assertIsNone(_depth_at(d,1,1,3,3))
    def test_pooled_hypothesis_is_not_an_independent_frame(self):
        def candidate(i):
            return {'_pose_matrix':np.eye(4),'diagnostics':{'frame_index':i,'landmark_view_id':'view','inlier_landmark_view_ids':['view']}}
        a=candidate(0); b=candidate(1); pooled=candidate(-1)
        stats=_consensus_stats(a,[a,b,pooled])
        self.assertEqual(stats['frame_count'],2)
        self.assertEqual(stats['pose_only_frame_count'],2)
    def test_split_excludes_held_out_partner(self):
        from geometry_service.learned_matcher import _view_split_masks
        train, validation = _view_split_masks([(0,1),(1,0),(1,2),(2,1)], ['held','train-a','train-b'], ['held'])
        self.assertEqual(train.tolist(), [False,False,True,True])
        self.assertEqual(validation.tolist(), [True,False,False,False])

    def test_triangulation_rejects_contradicting_depth(self):
        import cv2
        from geometry_service.localization import _triangulated_candidates, _world_to_cv, _project_point
        rng=np.random.default_rng(2)
        descriptors=rng.integers(0,256,(8,32),dtype=np.uint8)
        points=[np.array([x,y,-3.0]) for x in [-.6,-.2,.2,.6] for y in [-.3,.3]]
        frames=[]
        k=np.array([[120.,0,100],[0,120,100],[0,0,1]])
        for x in [0.,.3]:
            pose=np.eye(4);pose[0,3]=x
            cv=_world_to_cv(pose);proj=k@cv[:3,:]
            keys=[cv2.KeyPoint(*[float(v) for v in _project_point(proj,p)],3) for p in points]
            frames.append(dict(keypoints=keys,descriptors=descriptors,intrinsics=k,camera_to_world=pose,world_to_cv=cv,width=200,height=200))
        self.assertEqual(len(_triangulated_candidates(frames)),8)
        frames[0]['depth']=np.full((20,20),1.,dtype=np.float32)
        self.assertEqual(len(_triangulated_candidates(frames)),0)
        frames[0]['depth']=np.full((20,20),3.,dtype=np.float32)
        self.assertEqual(len(_triangulated_candidates(frames)),8)
if __name__=='__main__':unittest.main()

