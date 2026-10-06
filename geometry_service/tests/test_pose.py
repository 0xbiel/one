from geometry_service.pose import attach_pose_detections


def test_pose_is_attached_only_to_the_matching_person_box():
    detections = [
        {'label': 'person', 'confidence': 0.8, 'bbox': [10, 10, 110, 210]},
        {'label': 'chair', 'confidence': 0.9, 'bbox': [200, 20, 300, 150]},
    ]
    poses = [
        {'bbox': [12, 12, 108, 208], 'keypoints': [[20, 20, 0.9]] * 17, 'keypoint_confidence': 0.9},
    ]

    result = attach_pose_detections(detections, poses)

    assert result[0]['pose_keypoints'] == poses[0]['keypoints']
    assert 'pose_keypoints' not in result[1]


def test_one_pose_cannot_be_attached_to_two_overlapping_people():
    detections = [
        {'label': 'person', 'confidence': 0.8, 'bbox': [0, 0, 120, 220]},
        {'label': 'person', 'confidence': 0.8, 'bbox': [20, 10, 110, 210]},
    ]
    poses = [
        {'bbox': [22, 12, 108, 208], 'keypoints': [[30, 30, 0.9]] * 17, 'keypoint_confidence': 0.9},
    ]

    result = attach_pose_detections(detections, poses)

    assert sum('pose_keypoints' in item for item in result) == 1
    assert 'pose_keypoints' not in result[0]
    assert result[1]['pose_keypoints'] == poses[0]['keypoints']
