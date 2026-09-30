import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from geometry_service.contracts import (
    CameraLocalizationObjectDetection,
    CameraLocalizationRequest,
    CameraLocalizationRoomObject,
    CameraLocalizationRoomZone,
    VisualLandmarkBuildRequest,
)
from geometry_service.localization import (
    _camera_matrix_candidates,
    _camera_to_world,
    _candidate_is_positioned,
    _consensus_stats,
    _descriptor_matches,
    _guided_person_calibration,
    _merge_descriptor_matches,
    _pose_from_fixed_center_matches,
    _pose_from_matches,
    _pose_scene_prior,
    _pose_with_camera_center,
    _person_feature_mask,
    _pose_guided_matches,
    _provisional_pose_from_guided_matches,
    _project_room_object_bbox,
    _refine_pose_with_guided_matches,
    _refine_semantic_cuboid_pose,
    _room_bounded_landmark_indices,
    _room_perimeter_search_centers,
    _support_surface_search_centers,
    _semantic_cuboid_has_robust_support,
    _semantic_cuboid_consensus,
    _semantic_cuboid_rank,
    _semantic_cuboid_score,
    _semantic_object_assignments,
    _poses_agree,
    _project_point,
    _semantic_object_pose_seeds,
    _stable_movable_object_detection,
    _temporal_consensus_matches,
    _world_point,
    _world_to_cv,
    build_visual_landmarks,
    localize_camera,
)
from geometry_service.low_light import enhance_low_light_image


def _request(landmark_count: int, width: int, height: int) -> CameraLocalizationRequest:
    descriptor = base64.b64encode(bytes(32)).decode("ascii")
    return CameraLocalizationRequest(
        landmarks=[
            {"point": [float(index), 0.0, 0.0], "descriptor_base64": descriptor, "response": 1.0}
            for index in range(landmark_count)
        ],
        frames=[{"frame_base64": "x", "width": width, "height": height}],
        fov_degrees=60.0,
    )


def _candidate(frame_index: int, view_id: str, *, fov: float = 60.0, inliers: int = 7) -> dict:
    return {
        "status": "needs_rescan",
        "camera_to_world": None,
        "confidence": 0.75,
        "inlier_count": inliers,
        "reprojection_error_px": 2.0,
        "diagnostics": {
            "frame_index": frame_index,
            "landmark_view_id": view_id,
            "inlier_landmark_view_ids": [view_id],
            "inlier_landmark_view_count": 1,
            "global_inlier_ratio": 0.10,
            "image_coverage_ratio": 0.04,
            "world_spread_m": 1.2,
            "selected_fov_degrees": fov,
        },
        "_pose_matrix": np.eye(4, dtype=np.float64),
    }


def _semantic_candidate(
    frame_index: int,
    center: tuple[float, float, float],
    *,
    yaw_degrees: float = 0.0,
    fov: float = 96.0,
    guided_matches: int = 7,
    labels: tuple[str, ...] = ("bed", "storage"),
    mean_iou: float = 0.82,
) -> dict:
    yaw = np.deg2rad(yaw_degrees)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )
    matrix[:3, 3] = center
    return {
        "kind": "semantic-cuboid",
        "frame_index": frame_index,
        "camera_center": list(center),
        "candidate_camera_to_world": matrix.tolist(),
        "selected_fov_degrees": fov,
        "supported_object_count": len(labels),
        "supported_group_count": len(labels),
        "supported_mean_iou": mean_iou,
        "supported_minimum_iou": 0.62,
        "matched_object_count": len(labels),
        "semantic_group_count": len(labels),
        "guided_match_count": guided_matches,
        "matches": [
            {
                "label": label,
                "iou": mean_iou,
                "positive_depth_ratio": 1.0,
            }
            for label in labels
        ],
    }


def _semantic_center_scene(fov: float = 60.0) -> tuple[CameraLocalizationRequest, np.ndarray]:
    width, height = 640, 360
    focal = 0.5 * width / np.tan(np.radians(fov) / 2.0)
    camera_matrix = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    objects = [
        ("bed-1", "bed", (0.0, -1.0, 0.0), (1.2, 0.6, 2.0)),
        ("chair-1", "chair", (1.3, -0.8, -0.2), (0.6, 1.0, 0.6)),
        ("table-1", "table", (-1.0, -1.0, 0.3), (1.0, 0.7, 0.7)),
        ("storage-1", "storage", (0.5, -0.3, -1.0), (1.0, 2.0, 0.5)),
    ]
    true_center = np.asarray([0.2, 0.2, 2.0], dtype=np.float64)
    true_rotation = np.diag([1.0, -1.0, -1.0])
    true_rvec, _ = cv2.Rodrigues(true_rotation)
    true_tvec = -(true_rotation @ true_center.reshape(3, 1))
    world_points = np.asarray([item[2] for item in objects], dtype=np.float64)
    projected, _ = cv2.projectPoints(world_points, true_rvec, true_tvec, camera_matrix, None)

    request = _request(8, width, height)
    request.fov_degrees = fov
    request.room_zones = [
        CameraLocalizationRoomZone.model_validate(
            {
                "id": "room",
                "floor_y": -1.3,
                "polygon": [
                    {"x": -3.0, "z": -3.0},
                    {"x": 3.0, "z": -3.0},
                    {"x": 3.0, "z": 3.0},
                    {"x": -3.0, "z": 3.0},
                ],
            }
        )
    ]
    request.room_objects = [
        CameraLocalizationRoomObject(
            id=identifier,
            label=label,
            center={"x": center[0], "y": center[1], "z": center[2]},
            dimensions={"x": dimensions[0], "y": dimensions[1], "z": dimensions[2]},
            confidence=1.0,
        )
        for identifier, label, center, dimensions in objects
    ]
    request.object_detections = [
        CameraLocalizationObjectDetection(
            frame_index=0,
            label=label,
            confidence=0.9,
            bbox=[float(u - 20.0), float(v - 20.0), float(u + 20.0), float(v + 20.0)],
        )
        for (_, label, _, _), ((u, v),) in zip(objects, projected)
    ]
    return request, true_center


def _guided_match_scene(outlier_index: int | None = 6) -> dict:
    rng = np.random.default_rng(23)
    width, height = 640, 360
    camera_matrix = np.asarray(
        [[510.0, 0.0, width / 2.0], [0.0, 510.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    landmark_points = np.column_stack(
        [
            rng.uniform(-1.0, 1.0, 7),
            rng.uniform(-0.7, 0.7, 7),
            rng.uniform(-0.3, 0.3, 7),
        ]
    ).astype(np.float32)
    true_rvec = np.asarray([[0.02], [-0.06], [0.01]], dtype=np.float64)
    true_tvec = np.asarray([[0.05], [-0.03], [4.4]], dtype=np.float64)
    projected, _ = cv2.projectPoints(landmark_points, true_rvec, true_tvec, camera_matrix, None)
    pixels = projected.reshape(-1, 2)
    pixels[:6] += rng.normal(0.0, 0.35, (6, 2))
    if outlier_index is not None:
        pixels[outlier_index] += np.asarray([95.0, -70.0])
    keypoints = [SimpleNamespace(pt=tuple(pixel)) for pixel in pixels]
    matches = [cv2.DMatch(_queryIdx=index, _trainIdx=index, _distance=8.0) for index in range(7)]
    seed_pose = {
        "rvec": true_rvec + np.asarray([[0.01], [-0.01], [0.008]]),
        "tvec": true_tvec + np.asarray([[0.04], [-0.02], [0.06]]),
        "camera_matrix": camera_matrix,
        "fov_degrees": 64.0,
    }
    return {
        "frame_width": width,
        "frame_height": height,
        "keypoints": keypoints,
        "landmark_points": landmark_points,
        "pose": seed_pose,
        "matches": matches,
    }


class LocalizationTests(unittest.TestCase):
    def test_low_light_preprocessing_is_bounded_and_observable(self) -> None:
        image = np.full((120, 160, 3), 24, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (135, 95), (52, 52, 52), 2)
        cv2.line(image, (25, 80), (130, 30), (78, 78, 78), 2)

        enhanced, diagnostics = enhance_low_light_image(image)

        self.assertTrue(bool(diagnostics["low_light"]))
        self.assertGreater(float(np.mean(enhanced)), float(np.mean(image)))
        self.assertLessEqual(float(np.max(enhanced)), 255.0)
        self.assertGreaterEqual(float(np.min(enhanced)), 0.0)
        self.assertGreaterEqual(float(diagnostics["gamma"]), 0.55)
        self.assertLessEqual(float(diagnostics["gamma"]), 0.88)

    def test_visual_landmark_batch_attaches_scale_robust_sift_descriptors(self) -> None:
        rng = np.random.default_rng(91)
        image = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        for index in range(18):
            center = (20 + (index * 37) % 280, 20 + (index * 53) % 200)
            cv2.circle(image, center, 7 + index % 5, (255, 255, 255), 2)
            cv2.putText(image, str(index), (center[0] - 6, center[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertTrue(encoded_ok)
        depth = np.full((60, 80), 3.0, dtype="<f4")
        intrinsics = [[260.0, 0.0, 160.0], [0.0, 260.0, 120.0], [0.0, 0.0, 1.0]]
        payload = VisualLandmarkBuildRequest.model_validate({
            "map_id": "sift-test-map",
            "frames": [{
                "frame_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "width": 320,
                "height": 240,
                "depth_base64": base64.b64encode(depth.tobytes()).decode("ascii"),
                "depth_width": 80,
                "depth_height": 60,
                "intrinsics": {"values": intrinsics},
                "camera_to_world": {"values": np.eye(4).tolist()},
            }],
        })

        result = build_visual_landmarks(payload)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["detector"], "opencv-orb+sift")
        self.assertGreater(int(result["diagnostics"]["sift_depth_feature_counts"][0]), 0)
        self.assertGreater(int(result["diagnostics"]["sift_attached_count"]), 0)
        self.assertTrue(any(item.get("sift_descriptor_base64") for item in result["landmarks"]))

    def test_orb_and_sift_matches_are_merged_without_duplicate_correspondences(self) -> None:
        orb_matches = [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=40.0)]
        sift_matches = [
            cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=2.0),
            cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=20.0),
        ]

        merged = _merge_descriptor_matches(orb_matches, sift_matches, sift_query_offset=4)

        self.assertEqual([(match.queryIdx, match.trainIdx) for match in merged], [(4, 0), (5, 1)])

    def test_fresh_scale_robust_index_reaches_high_confidence_only_with_independent_evidence(self) -> None:
        rng = np.random.default_rng(19)
        image = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
        for index in range(120):
            x = int(rng.integers(20, 620))
            y = int(rng.integers(20, 340))
            cv2.circle(image, (x, y), int(rng.integers(2, 9)), tuple(int(value) for value in rng.integers(0, 256, 3)), -1)
            cv2.putText(image, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertTrue(encoded_ok)
        depth = np.full((90, 160), 3.0, dtype="<f4")
        frame = {
            "frame_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "width": 640,
            "height": 360,
            "depth_base64": base64.b64encode(depth.tobytes()).decode("ascii"),
            "depth_width": 160,
            "depth_height": 90,
            "intrinsics": {"values": [[520.0, 0.0, 320.0], [0.0, 520.0, 180.0], [0.0, 0.0, 1.0]]},
            "camera_to_world": {"values": np.eye(4).tolist()},
        }
        built = build_visual_landmarks(
            VisualLandmarkBuildRequest.model_validate({"map_id": "confidence-test", "frames": [frame]})
        )
        self.assertEqual(built["detector"], "opencv-orb+sift")
        self.assertGreater(int(built["diagnostics"]["sift_attached_count"]), 100)
        landmarks = [
            {**landmark, "view_id": view_id}
            for view_id in ("scan-view-a", "scan-view-b")
            for landmark in built["landmarks"]
        ]
        request = CameraLocalizationRequest.model_validate({
            "landmarks": landmarks,
            "frames": [
                {"frame_base64": frame["frame_base64"], "width": 640, "height": 360},
                {"frame_base64": frame["frame_base64"], "width": 640, "height": 360},
            ],
            "fov_degrees": 60.0,
            "room_zones": [{
                "id": "synthetic-room",
                "floor_y": -1.5,
                "polygon": [
                    {"x": -10.0, "z": -10.0},
                    {"x": 10.0, "z": -10.0},
                    {"x": 10.0, "z": 10.0},
                    {"x": -10.0, "z": 10.0},
                ],
            }],
        })

        result = localize_camera(request)

        self.assertEqual(result["status"], "positioned")
        self.assertGreaterEqual(float(result["confidence"]), 0.90)
        self.assertGreaterEqual(int(result["inlier_count"]), 12)
        self.assertEqual(result["diagnostics"]["consensus_scan_view_count"], 2)

    def test_room_bounded_landmark_filter_removes_far_background_but_keeps_room_margin(self) -> None:
        request = _request(60, 640, 480)
        request.room_zones = [
            CameraLocalizationRoomZone.model_validate({
                "id": "room",
                "floor_y": 0.0,
                "polygon": [
                    {"x": -2.0, "z": -2.0},
                    {"x": 2.0, "z": -2.0},
                    {"x": 2.0, "z": 2.0},
                    {"x": -2.0, "z": 2.0},
                ],
            })
        ]
        points = np.asarray(
            [[-1.5 + (index % 9) * 0.35, 1.2, -1.5 + (index // 9) * 0.5] for index in range(45)]
            + [[5.0 + index * 0.1, 1.1, 5.0] for index in range(15)],
            dtype=np.float32,
        )

        indices, diagnostics = _room_bounded_landmark_indices(request, points)

        self.assertTrue(diagnostics["applied"])
        self.assertEqual(len(indices), 45)
        self.assertEqual(diagnostics["filtered_landmark_count"], 15)
        self.assertAlmostEqual(float(diagnostics["retained_ratio"]), 0.75, places=3)

    def test_room_bounded_landmark_filter_falls_back_when_room_bounds_remove_most_points(self) -> None:
        request = _request(60, 640, 480)
        request.room_zones = [
            CameraLocalizationRoomZone.model_validate({
                "id": "room",
                "floor_y": 0.0,
                "polygon": [
                    {"x": -1.0, "z": -1.0},
                    {"x": 1.0, "z": -1.0},
                    {"x": 1.0, "z": 1.0},
                    {"x": -1.0, "z": 1.0},
                ],
            })
        ]
        points = np.asarray(
            [[0.0, 1.0, 0.0] for _ in range(10)]
            + [[8.0 + index * 0.1, 1.0, 8.0] for index in range(50)],
            dtype=np.float32,
        )

        indices, diagnostics = _room_bounded_landmark_indices(request, points)

        self.assertFalse(diagnostics["applied"])
        self.assertEqual(len(indices), 60)
        self.assertEqual(diagnostics["reason"], "fallback_preserved_full_landmark_set")

    def test_temporal_consensus_matches_keeps_stationary_landmarks_and_rejects_wandering_matches(self) -> None:
        observations = {
            index: [
                (0, 100.0 + index * 12.0, 150.0 + index * 5.0, 28.0 + index),
                (1, 100.8 + index * 12.0, 149.4 + index * 5.0, 29.0 + index),
                (2, 99.5 + index * 12.0, 150.6 + index * 5.0, 27.0 + index),
            ]
            for index in range(6)
        }
        observations[99] = [
            (0, 80.0, 80.0, 24.0),
            (1, 130.0, 120.0, 25.0),
            (2, 180.0, 160.0, 23.0),
        ]
        observations[100] = [(0, 220.0, 200.0, 20.0)]

        keypoints, matches, diagnostics = _temporal_consensus_matches(observations)

        self.assertEqual(len(keypoints), 6)
        self.assertEqual(len(matches), 6)
        self.assertEqual({match.trainIdx for match in matches}, set(range(6)))
        self.assertEqual(diagnostics["support_frame_count"], 3)
        self.assertEqual(diagnostics["stable_landmark_count"], 6)
        self.assertLess(float(diagnostics["maximum_pixel_jitter"]), 2.0)

    def test_unknown_fov_uses_camera_model_sweep_without_implicit_default(self) -> None:
        request = _request(8, 1280, 720)
        request.fov_degrees = None

        candidates = _camera_matrix_candidates(request, 1280, 720)

        self.assertGreater(len(candidates), 3)
        self.assertTrue(all(source == "estimated-fov-sweep" for _matrix, source, _fov in candidates))
        self.assertGreater(len({round(float(matrix[0, 0]), 3) for matrix, _source, _fov in candidates}), 3)

    def test_person_feature_mask_accepts_low_confidence_occluder_and_rejects_blocked_frame(self) -> None:
        request = _request(8, 640, 480)
        request.object_detections = [
            CameraLocalizationObjectDetection(
                frame_index=0,
                label="person",
                confidence=0.06,
                bbox=[220.0, 80.0, 420.0, 470.0],
            )
        ]

        mask, ratio, occluded = _person_feature_mask(request, frame_index=0, width=640, height=480)

        self.assertIsNotNone(mask)
        self.assertIsNotNone(ratio)
        self.assertFalse(occluded)
        assert mask is not None
        self.assertEqual(int(mask[250, 320]), 0)

        request.object_detections = [
            CameraLocalizationObjectDetection(
                frame_index=0,
                label="person",
                confidence=0.06,
                bbox=[0.0, 0.0, 430.0, 480.0],
            )
        ]
        mask, ratio, occluded = _person_feature_mask(request, frame_index=0, width=640, height=480)
        self.assertIsNotNone(mask)
        self.assertIsNotNone(ratio)
        self.assertGreater(float(ratio), 0.68)
        self.assertFalse(occluded)

        # A close person can cover well over 92% once padding is applied while
        # still leaving a narrow strip of static room. Keep that strip usable
        # and let feature extraction decide whether it has enough evidence.
        request.object_detections = [
            CameraLocalizationObjectDetection(
                frame_index=0,
                label="person",
                confidence=0.39,
                bbox=[10.0, 1.0, 570.0, 480.0],
            )
        ]
        mask, ratio, occluded = _person_feature_mask(request, frame_index=0, width=640, height=480)
        self.assertIsNotNone(mask)
        self.assertIsNotNone(ratio)
        self.assertGreater(float(ratio), 0.92)
        self.assertFalse(occluded)

        request.object_detections = [
            CameraLocalizationObjectDetection(
                frame_index=0,
                label="person",
                confidence=0.06,
                bbox=[0.0, 0.0, 640.0, 480.0],
            )
        ]
        mask, ratio, occluded = _person_feature_mask(request, frame_index=0, width=640, height=480)
        self.assertIsNone(mask)
        self.assertIsNotNone(ratio)
        self.assertTrue(occluded)

    def test_reflective_window_is_masked_from_static_feature_matching(self) -> None:
        request = _request(8, 640, 480)
        request.object_detections = [
            CameraLocalizationObjectDetection(
                frame_index=0,
                label="window",
                confidence=0.82,
                bbox=[0.0, 20.0, 260.0, 460.0],
            )
        ]

        mask, ratio, occluded = _person_feature_mask(request, frame_index=0, width=640, height=480)

        self.assertIsNotNone(mask)
        self.assertIsNotNone(ratio)
        self.assertFalse(occluded)
        assert mask is not None
        self.assertEqual(int(mask[240, 120]), 0)
        self.assertEqual(int(mask[240, 500]), 255)
        self.assertGreater(float(ratio), 0.30)

    def test_movable_chair_is_masked_from_static_feature_matching(self) -> None:
        request = _request(8, 640, 480)
        request.object_detections = [
            CameraLocalizationObjectDetection(
                frame_index=0,
                label="chair",
                confidence=0.76,
                bbox=[180.0, 190.0, 430.0, 478.0],
            )
        ]

        mask, ratio, occluded = _person_feature_mask(request, frame_index=0, width=640, height=480)

        self.assertIsNotNone(mask)
        self.assertIsNotNone(ratio)
        self.assertFalse(occluded)
        assert mask is not None
        self.assertEqual(int(mask[300, 300]), 0)

    def test_guided_person_calibration_recovers_pose_with_bystander(self) -> None:
        request = _request(8, 640, 480)
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, 3] = [0.0, 1.5, 3.0]
        world_to_cv = _world_to_cv(camera_to_world)
        focal = 640.0 / (2.0 * np.tan(np.deg2rad(60.0) * 0.5))
        intrinsics = np.asarray([[focal, 0.0, 320.0], [0.0, focal, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        projection = intrinsics @ world_to_cv[:3, :]
        targets = [(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (-1.0, 0.0, -0.5), (1.0, 0.0, -0.5)]
        frames = [{"frame_base64": "x", "width": 640, "height": 480} for _ in range(8)]
        anchors = []
        detections = []
        for target_index, target in enumerate(targets):
            pixel = _project_point(projection, np.asarray(target, dtype=np.float64))
            self.assertIsNotNone(pixel)
            assert pixel is not None
            for offset in range(2):
                frame_index = target_index * 2 + offset
                anchors.append({"frame_index": frame_index, "x": target[0], "y": target[1], "z": target[2]})
                foot_x = float(pixel[0]) + (0.4 if offset else -0.4)
                foot_y = float(pixel[1]) + (0.3 if offset else -0.3)
                y2 = foot_y + 1.5
                detections.extend([
                    {"frame_index": frame_index, "label": "person", "confidence": 0.82, "bbox": [foot_x - 28.0, y2 - 100.0, foot_x + 28.0, y2]},
                    {"frame_index": frame_index, "label": "person", "confidence": 0.97, "bbox": [70.0, 120.0, 150.0, 360.0]},
                ])
        payload = request.model_dump()
        payload.update({
            "frames": frames,
            "person_anchors": anchors,
            "object_detections": detections,
            "room_zones": [{
                "id": "room",
                "floor_y": 0.0,
                "polygon": [{"x": -2.5, "z": -2.0}, {"x": 2.5, "z": -2.0}, {"x": 2.5, "z": 3.5}, {"x": -2.5, "z": 3.5}],
            }],
        })
        request = CameraLocalizationRequest.model_validate(payload)

        result, diagnostics = _guided_person_calibration(request)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["status"], "positioned")
        self.assertEqual(result["diagnostics"]["selected_estimate_source"], "guided-person-floor")
        self.assertEqual(diagnostics["status"], "accepted")
        recovered = np.asarray(result["camera_to_world"], dtype=np.float64)
        self.assertLess(float(np.linalg.norm(recovered[:3, 3] - camera_to_world[:3, 3])), 0.15)

    def test_guided_person_calibration_sweeps_unknown_fov_with_six_targets(self) -> None:
        request = _request(8, 640, 480)
        request.fov_degrees = None
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, 3] = [0.15, 1.45, 3.1]
        world_to_cv = _world_to_cv(camera_to_world)
        true_fov = 84.0
        focal = 640.0 / (2.0 * np.tan(np.deg2rad(true_fov) * 0.5))
        intrinsics = np.asarray([[focal, 0.0, 320.0], [0.0, focal, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        projection = intrinsics @ world_to_cv[:3, :]
        targets = [
            (-1.2, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (1.2, 0.0, 1.0),
            (-1.2, 0.0, -0.6),
            (0.0, 0.0, -0.6),
            (1.2, 0.0, -0.6),
        ]
        frames = [{"frame_base64": "x", "width": 640, "height": 480} for _ in range(12)]
        anchors = []
        detections = []
        for target_index, target in enumerate(targets):
            pixel = _project_point(projection, np.asarray(target, dtype=np.float64))
            self.assertIsNotNone(pixel)
            assert pixel is not None
            for offset in range(2):
                frame_index = target_index * 2 + offset
                anchors.append({"frame_index": frame_index, "x": target[0], "y": target[1], "z": target[2]})
                foot_x = float(pixel[0]) + (-0.8 if offset == 0 else 0.8)
                foot_y = float(pixel[1]) + (-0.5 if offset == 0 else 0.5)
                y2 = foot_y + 1.5
                detections.extend([
                    {"frame_index": frame_index, "label": "person", "confidence": 0.86, "bbox": [foot_x - 26.0, y2 - 96.0, foot_x + 26.0, y2]},
                    {"frame_index": frame_index, "label": "person", "confidence": 0.96, "bbox": [60.0, 115.0, 145.0, 355.0]},
                ])
        payload = request.model_dump()
        payload.update({
            "frames": frames,
            "person_anchors": anchors,
            "object_detections": detections,
            "room_zones": [{
                "id": "room",
                "floor_y": 0.0,
                "polygon": [{"x": -2.5, "z": -2.0}, {"x": 2.5, "z": -2.0}, {"x": 2.5, "z": 3.5}, {"x": -2.5, "z": 3.5}],
            }],
        })
        request = CameraLocalizationRequest.model_validate(payload)

        result, diagnostics = _guided_person_calibration(request)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["status"], "positioned")
        self.assertEqual(diagnostics["target_count"], 6)
        self.assertLess(abs(float(diagnostics["selected_fov_degrees"]) - true_fov), 8.0)
        recovered = np.asarray(result["camera_to_world"], dtype=np.float64)
        self.assertLess(float(np.linalg.norm(recovered[:3, 3] - camera_to_world[:3, 3])), 0.20)

    def test_semantic_cuboid_rank_rejects_extra_bad_assignment(self) -> None:
        strong_two_object = {
            "matched_object_count": 2,
            "semantic_group_count": 2,
            "score": 2.05,
            "mean_iou": 0.90,
            "minimum_iou": 0.86,
            "mean_center_score": 0.94,
        }
        weak_three_object = {
            "matched_object_count": 3,
            "semantic_group_count": 3,
            "score": 2.70,
            "mean_iou": 0.56,
            "minimum_iou": 0.018,
            "mean_center_score": 0.80,
        }

        self.assertGreater(_semantic_cuboid_rank(strong_two_object), _semantic_cuboid_rank(weak_three_object))

    def test_semantic_cuboid_search_hint_keeps_two_good_groups_with_one_bad_outlier(self) -> None:
        live_style_candidate = {
            "supported_object_count": 2,
            "supported_group_count": 2,
            "supported_mean_iou": 0.735,
            "supported_minimum_iou": 0.53,
            "matched_object_count": 3,
            "semantic_group_count": 2,
            "mean_iou": 0.495,
            "minimum_iou": 0.015,
        }

        self.assertTrue(_semantic_cuboid_has_robust_support(live_style_candidate))

    def test_semantic_cuboid_rejects_visible_assigned_object_fully_behind_camera(self) -> None:
        width, height, fov = 960, 540, 90.0
        focal = 0.5 * width / np.tan(np.radians(fov) / 2.0)
        camera_matrix = np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        request = _request(8, width, height)
        request.room_objects = [
            CameraLocalizationRoomObject(
                id="bed-visible",
                label="bed",
                center={"x": -0.8, "y": 0.0, "z": 4.0},
                dimensions={"x": 1.2, "y": 0.8, "z": 1.6},
                confidence=0.95,
            ),
            CameraLocalizationRoomObject(
                id="storage-visible",
                label="storage",
                center={"x": 1.0, "y": 0.0, "z": 4.5},
                dimensions={"x": 1.0, "y": 1.8, "z": 0.8},
                confidence=0.95,
            ),
            CameraLocalizationRoomObject(
                id="chair-behind",
                label="chair",
                center={"x": 0.2, "y": 0.0, "z": -2.0},
                dimensions={"x": 0.8, "y": 1.2, "z": 0.8},
                # RoomPlan can report low semantic confidence for furniture
                # whose metric cuboid is still useful. A strong live chair
                # detection fully behind the camera must remain contradictory.
                confidence=0.45,
            ),
        ]
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        detections: list[CameraLocalizationObjectDetection] = []
        for room_object in request.room_objects[:2]:
            projected = _project_room_object_bbox(
                room_object,
                rvec=rvec,
                tvec=tvec,
                camera_matrix=camera_matrix,
                frame_width=width,
                frame_height=height,
            )
            self.assertIsNotNone(projected)
            assert projected is not None
            bbox, _ = projected
            detections.append(
                CameraLocalizationObjectDetection(
                    frame_index=0,
                    label=room_object.label,
                    confidence=0.92,
                    bbox=bbox,
                )
            )
        detections.append(
            CameraLocalizationObjectDetection(
                frame_index=0,
                label="chair",
                confidence=0.86,
                bbox=[420.0, 220.0, 760.0, 535.0],
            )
        )
        request.object_detections = detections

        contradiction_free = _semantic_cuboid_score(
            request,
            frame_index=0,
            assignment=[(0, 0), (1, 1)],
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            frame_width=width,
            frame_height=height,
        )
        contradicted = _semantic_cuboid_score(
            request,
            frame_index=0,
            assignment=[(0, 0), (1, 1), (2, 2)],
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            frame_width=width,
            frame_height=height,
        )

        self.assertTrue(_semantic_cuboid_has_robust_support(contradiction_free))
        self.assertEqual(contradicted["supported_object_count"], 2)
        self.assertEqual(contradicted["contradictory_object_count"], 1)
        self.assertEqual(contradicted["contradictory_group_count"], 1)
        self.assertEqual(contradicted["contradictions"][0]["label"], "chair")
        self.assertFalse(_semantic_cuboid_has_robust_support(contradicted))
        self.assertGreater(_semantic_cuboid_rank(contradiction_free), _semantic_cuboid_rank(contradicted))

    def test_semantic_cuboid_does_not_mark_near_plane_partial_visibility_as_contradiction(self) -> None:
        width, height, fov = 640, 360, 60.0
        focal = 0.5 * width / np.tan(np.radians(fov) / 2.0)
        camera_matrix = np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        request = _request(8, width, height)
        room_object = CameraLocalizationRoomObject(
            id="storage-near-camera",
            label="storage",
            center={"x": 0.0, "y": 0.0, "z": 0.1},
            dimensions={"x": 0.4, "y": 0.4, "z": 0.6},
            confidence=1.0,
        )
        request.room_objects = [room_object]
        projected = _project_room_object_bbox(
            room_object,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.zeros((3, 1), dtype=np.float64),
            camera_matrix=camera_matrix,
            frame_width=width,
            frame_height=height,
        )
        self.assertIsNotNone(projected)
        assert projected is not None
        bbox, positive_ratio = projected
        self.assertEqual(positive_ratio, 0.5)
        request.object_detections = [
            CameraLocalizationObjectDetection(
                frame_index=0,
                label="storage",
                confidence=0.9,
                bbox=bbox,
            )
        ]

        score = _semantic_cuboid_score(
            request,
            frame_index=0,
            assignment=[(0, 0)],
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.zeros((3, 1), dtype=np.float64),
            camera_matrix=camera_matrix,
            frame_width=width,
            frame_height=height,
        )

        self.assertEqual(score["contradictory_object_count"], 0)
        self.assertEqual(score["contradictions"], [])

    def test_semantic_cuboid_rank_prefers_multi_object_evidence_over_single_perfect_box(self) -> None:
        strong_two_object = {
            "matched_object_count": 2,
            "semantic_group_count": 2,
            "score": 1.44,
            "mean_iou": 0.91,
            "minimum_iou": 0.83,
            "mean_center_score": 0.95,
        }
        perfect_single_object = {
            "matched_object_count": 1,
            "semantic_group_count": 1,
            "score": 0.80,
            "mean_iou": 0.99,
            "minimum_iou": 0.99,
            "mean_center_score": 0.99,
        }

        self.assertGreater(_semantic_cuboid_rank(strong_two_object), _semantic_cuboid_rank(perfect_single_object))

    def test_semantic_cuboid_consensus_accepts_tight_four_frame_cluster(self) -> None:
        candidates = [
            _semantic_candidate(0, (1.1385, -0.7725, 1.2027), yaw_degrees=0.0, guided_matches=7),
            _semantic_candidate(1, (1.0760, -0.8066, 1.2214), yaw_degrees=0.4, guided_matches=6),
            _semantic_candidate(2, (1.0722, -0.8054, 1.2386), yaw_degrees=-0.3, guided_matches=5),
            _semantic_candidate(3, (1.0846, -0.7800, 1.1846), yaw_degrees=0.7, guided_matches=4),
        ]

        consensus = _semantic_cuboid_consensus(candidates, burst_frame_count=6)

        self.assertIsNotNone(consensus)
        assert consensus is not None
        self.assertEqual(consensus["frame_count"], 4)
        self.assertEqual(consensus["visual_support_frame_count"], 2)
        self.assertEqual(consensus["common_labels"], ["bed", "storage"])
        self.assertEqual(consensus["fov_degrees"], 96.0)
        self.assertLess(consensus["max_center_residual_m"], 0.12)
        self.assertLess(consensus["max_rotation_residual_degrees"], 1.1)

    def test_semantic_cuboid_consensus_rejects_only_three_frames_in_six_frame_burst(self) -> None:
        candidates = [
            _semantic_candidate(index, (1.0 + 0.02 * index, -0.8, 1.2), guided_matches=7)
            for index in range(3)
        ]

        self.assertIsNone(_semantic_cuboid_consensus(candidates, burst_frame_count=6))

    def test_semantic_cuboid_consensus_accepts_three_strong_frames_when_dynamic_occlusion_is_high(self) -> None:
        candidates = [
            _semantic_candidate(index, (0.885 + 0.003 * index, -0.73, 0.65), guided_matches=7)
            for index in range(3)
        ]

        consensus = _semantic_cuboid_consensus(
            candidates,
            burst_frame_count=6,
            dynamic_mask_max_ratio=0.65,
        )

        self.assertIsNotNone(consensus)
        assert consensus is not None
        self.assertEqual(consensus["frame_count"], 3)
        self.assertEqual(consensus["visual_support_frame_count"], 3)
        self.assertTrue(consensus["occlusion_relaxed_frame_requirement"])

    def test_semantic_cuboid_consensus_high_occlusion_still_rejects_weak_three_frame_visual_support(self) -> None:
        candidates = [
            _semantic_candidate(index, (0.885 + 0.003 * index, -0.73, 0.65), guided_matches=5)
            for index in range(3)
        ]

        self.assertIsNone(
            _semantic_cuboid_consensus(
                candidates,
                burst_frame_count=6,
                dynamic_mask_max_ratio=0.65,
            )
        )

    def test_semantic_cuboid_consensus_rejects_single_shared_label(self) -> None:
        candidates = [
            _semantic_candidate(
                index,
                (1.0 + 0.01 * index, -0.8, 1.2),
                labels=("bed", "storage") if index < 2 else ("bed", "desk"),
                guided_matches=7,
            )
            for index in range(4)
        ]

        self.assertIsNone(_semantic_cuboid_consensus(candidates, burst_frame_count=6))

    def test_semantic_cuboid_consensus_rejects_inconsistent_fov(self) -> None:
        candidates = [
            _semantic_candidate(0, (1.00, -0.8, 1.20), fov=92.0),
            _semantic_candidate(1, (1.01, -0.8, 1.20), fov=92.0),
            _semantic_candidate(2, (1.02, -0.8, 1.20), fov=100.0),
            _semantic_candidate(3, (1.03, -0.8, 1.20), fov=100.0),
        ]

        self.assertIsNone(_semantic_cuboid_consensus(candidates, burst_frame_count=6))

    def test_semantic_cuboid_consensus_rejects_pose_scatter(self) -> None:
        candidates = [
            _semantic_candidate(0, (0.78, -0.8, 1.20), yaw_degrees=-4.0),
            _semantic_candidate(1, (0.93, -0.8, 1.20), yaw_degrees=-1.0),
            _semantic_candidate(2, (1.07, -0.8, 1.20), yaw_degrees=1.0),
            _semantic_candidate(3, (1.22, -0.8, 1.20), yaw_degrees=4.0),
        ]

        self.assertIsNone(_semantic_cuboid_consensus(candidates, burst_frame_count=6))

    def test_semantic_cuboid_consensus_requires_fresh_orb_support_in_two_frames(self) -> None:
        candidates = [
            _semantic_candidate(index, (1.0 + 0.01 * index, -0.8, 1.2), guided_matches=5)
            for index in range(4)
        ]

        self.assertIsNone(_semantic_cuboid_consensus(candidates, burst_frame_count=6))

    def test_semantic_cuboid_consensus_prefers_larger_competing_cluster(self) -> None:
        strong_cluster = [
            _semantic_candidate(index, (1.0 + 0.01 * index, -0.8, 1.2), guided_matches=7)
            for index in range(5)
        ]
        smaller_cluster = [
            _semantic_candidate(index + 5, (-0.6 + 0.01 * index, -0.8, -0.4), guided_matches=8, mean_iou=0.92)
            for index in range(4)
        ]

        consensus = _semantic_cuboid_consensus(strong_cluster + smaller_cluster, burst_frame_count=9)

        self.assertIsNotNone(consensus)
        assert consensus is not None
        self.assertEqual(consensus["frame_count"], 5)
        self.assertEqual(consensus["frame_indices"], [0, 1, 2, 3, 4])

    def test_room_perimeter_search_centers_are_bounded_and_inset(self) -> None:
        request = _request(8, 640, 360)
        request.room_zones = [
            CameraLocalizationRoomZone.model_validate({
                "id": "room",
                "floor_y": -1.3,
                "polygon": [
                    {"x": -2.0, "z": -1.5},
                    {"x": 2.0, "z": -1.5},
                    {"x": 2.0, "z": 1.5},
                    {"x": -2.0, "z": 1.5},
                ],
            })
        ]

        centers = _room_perimeter_search_centers(request, camera_y=-0.45, max_centers=12)

        self.assertEqual(len(centers), 12)
        self.assertTrue(all(abs(float(center[1]) + 0.45) < 1e-9 for center in centers))
        # Samples are inset from a wall rather than fixed to the polygon edge.
        self.assertTrue(any(1.15 <= abs(float(center[0])) <= 1.85 for center in centers))
        self.assertTrue(any(0.65 <= abs(float(center[2])) <= 1.35 for center in centers))

    def test_support_surface_search_centers_follow_rotated_roomplan_desk(self) -> None:
        request = _request(8, 640, 360)
        yaw = np.radians(30.0)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        request.room_objects = [
            CameraLocalizationRoomObject(
                id="desk-1",
                label="desk",
                center={"x": 3.0, "y": -1.0, "z": 5.0},
                dimensions={"x": 2.0, "y": 0.8, "z": 1.0},
                transform={
                    "values": [
                        [c, 0.0, s, 3.0],
                        [0.0, 1.0, 0.0, -1.0],
                        [-s, 0.0, c, 5.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                },
                confidence=0.95,
            )
        ]

        centers = _support_surface_search_centers(request, max_centers=15)

        self.assertEqual(len(centers), 15)
        self.assertEqual(
            {round(float(center[1]), 2) for center in centers},
            {-0.52, -0.36, -0.18},
        )
        self.assertTrue(all(np.isfinite(center).all() for center in centers))
        self.assertGreater(
            max(float(np.linalg.norm(center[[0, 2]] - np.asarray([3.0, 5.0]))) for center in centers),
            0.30,
        )
        self.assertLessEqual(
            max(float(np.linalg.norm(center[[0, 2]] - np.asarray([3.0, 5.0]))) for center in centers),
            1.05,
        )

    def test_support_surface_search_is_scene_derived_and_rejects_weak_or_non_support_objects(self) -> None:
        placements = (
            ("table", (-4.2, -0.9, 1.7), (1.4, 0.72, 0.8), 0.0),
            ("dining table", (2.6, -0.4, -3.8), (2.2, 0.76, 1.1), 45.0),
            ("desk", (8.0, 1.2, 6.5), (1.0, 0.7, 0.55), -25.0),
        )
        for label, location, size, yaw_degrees in placements:
            request = _request(8, 640, 360)
            yaw = np.radians(yaw_degrees)
            c, s = float(np.cos(yaw)), float(np.sin(yaw))
            request.room_objects = [
                CameraLocalizationRoomObject(
                    id=f"{label}-1",
                    label=label,
                    center={"x": location[0], "y": location[1], "z": location[2]},
                    dimensions={"x": size[0], "y": size[1], "z": size[2]},
                    transform={
                        "values": [
                            [c, 0.0, s, location[0]],
                            [0.0, 1.0, 0.0, location[1]],
                            [-s, 0.0, c, location[2]],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    },
                    confidence=0.90,
                )
            ]
            centers = _support_surface_search_centers(request, max_centers=15)
            self.assertTrue(centers)
            self.assertEqual({round(float(center[1]), 2) for center in centers}, {
                round(location[1] + size[1] / 2.0 + height, 2)
                for height in (0.08, 0.24, 0.42)
            })
            self.assertTrue(all(np.isfinite(center).all() for center in centers))
            # No center may depend on the prior room's coordinates or leave
            # the actual rotated support-surface footprint by a large margin.
            self.assertLessEqual(
                max(float(np.linalg.norm(center - np.asarray([*location[:1], location[1], location[2]]))) for center in centers),
                max(size[0], size[2]) * 0.65 + 0.50,
            )

        weak = _request(8, 640, 360)
        weak.room_objects = [
            CameraLocalizationRoomObject(
                id="weak-table",
                label="table",
                center={"x": 100.0, "y": 0.0, "z": -100.0},
                dimensions={"x": 2.0, "y": 0.8, "z": 1.0},
                confidence=0.44,
            ),
            CameraLocalizationRoomObject(
                id="cabinet",
                label="storage",
                center={"x": 0.0, "y": 0.0, "z": 0.0},
                dimensions={"x": 2.0, "y": 1.0, "z": 1.0},
                confidence=1.0,
            ),
        ]
        self.assertEqual(_support_surface_search_centers(weak), [])

    def test_duplicate_webcam_frames_do_not_promote_one_scan_view(self) -> None:
        first = _candidate(0, "scan-view-4", fov=54.0, inliers=6)
        second = _candidate(1, "scan-view-4", fov=54.0, inliers=6)
        stats = _consensus_stats(first, [first, second])
        first["_consensus_frame_count"] = stats["frame_count"]
        first["_consensus_scan_view_count"] = stats["scan_view_count"]

        self.assertEqual(stats["frame_count"], 2)
        self.assertEqual(stats["scan_view_count"], 1)
        self.assertFalse(_candidate_is_positioned(first))

    def test_search_prior_does_not_promote_single_view_candidate(self) -> None:
        candidate = _candidate(0, "scan-view-4", fov=96.0, inliers=12)
        candidate["diagnostics"]["search_prior_distance_m"] = 0.01
        candidate["diagnostics"]["search_prior_aligned"] = True
        candidate["_consensus_frame_count"] = 1
        candidate["_consensus_scan_view_count"] = 1

        self.assertFalse(_candidate_is_positioned(candidate))

    def test_independent_roomplan_views_can_promote_consistent_pose(self) -> None:
        first = _candidate(0, "scan-view-2", fov=60.0, inliers=7)
        second = _candidate(1, "scan-view-7", fov=60.0, inliers=7)
        stats = _consensus_stats(first, [first, second])
        first["_consensus_frame_count"] = stats["frame_count"]
        first["_consensus_scan_view_count"] = stats["scan_view_count"]

        self.assertEqual(stats["scan_view_count"], 2)
        self.assertEqual(stats["frame_count"], 2)
        self.assertTrue(_candidate_is_positioned(first))

    def test_conflicting_fov_hypotheses_do_not_form_independent_consensus(self) -> None:
        first = _candidate(0, "scan-view-2", fov=48.0, inliers=7)
        second = _candidate(1, "scan-view-7", fov=96.0, inliers=7)
        stats = _consensus_stats(first, [first, second])

        self.assertEqual(stats["pose_only_frame_count"], 2)
        self.assertEqual(stats["frame_count"], 1)
        self.assertEqual(stats["scan_view_count"], 1)

    def test_weak_all_views_candidate_does_not_count_its_landmarks_as_consensus(self) -> None:
        candidate = _candidate(0, "all-views", fov=96.0, inliers=7)
        candidate["diagnostics"]["inlier_landmark_view_ids"] = [
            "scan-view-2",
            "scan-view-3",
            "scan-view-4",
            "scan-view-5",
            "scan-view-9",
        ]
        candidate["diagnostics"]["inlier_landmark_view_count"] = 5
        stats = _consensus_stats(candidate, [candidate])
        candidate["_consensus_frame_count"] = stats["frame_count"]
        candidate["_consensus_scan_view_count"] = stats["scan_view_count"]

        self.assertEqual(stats["scan_view_count"], 0)
        self.assertFalse(_candidate_is_positioned(candidate))

    def test_pose_consensus_rejects_camera_motion(self) -> None:
        first = np.eye(4, dtype=np.float64)
        nearby = np.eye(4, dtype=np.float64)
        nearby[:3, 3] = [0.12, -0.04, 0.08]
        moved = np.eye(4, dtype=np.float64)
        moved[:3, 3] = [0.8, 0.0, 0.0]

        self.assertTrue(_poses_agree(first, nearby))
        self.assertFalse(_poses_agree(first, moved))

    def test_search_pose_can_snap_center_without_changing_rotation(self) -> None:
        rvec = np.asarray([[0.12], [-0.21], [0.07]], dtype=np.float64)
        pose = {"rvec": rvec, "tvec": np.asarray([[0.4], [-0.2], [3.1]], dtype=np.float64)}
        center = np.asarray([-0.3958, -0.8427, -1.0203], dtype=np.float64)

        snapped = _pose_with_camera_center(pose, center)
        matrix = np.asarray(_camera_to_world(snapped["rvec"], snapped["tvec"]), dtype=np.float64)

        np.testing.assert_allclose(matrix[:3, 3], center, atol=1e-7)
        np.testing.assert_allclose(snapped["rvec"], rvec, atol=0.0)

    def test_fixed_center_search_recovers_rotation_from_fresh_matches(self) -> None:
        rng = np.random.default_rng(11)
        width, height = 640, 360
        fov = 96.0
        focal = 0.5 * width / np.tan(np.radians(fov) / 2.0)
        camera_matrix = np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        center = np.asarray([-0.3958, -0.8427, -1.0203], dtype=np.float64)
        true_rvec = np.asarray([[0.08], [-0.32], [0.03]], dtype=np.float64)
        true_rotation, _ = cv2.Rodrigues(true_rvec)
        camera_points = np.column_stack(
            [
                rng.uniform(-1.2, 1.2, 14),
                rng.uniform(-0.7, 0.7, 14),
                rng.uniform(2.2, 4.5, 14),
            ]
        )
        true_world_points = center.reshape(1, 3) + (true_rotation.T @ camera_points.T).T
        true_tvec = -(true_rotation @ center.reshape(3, 1))
        projected, _ = cv2.projectPoints(true_world_points, true_rvec, true_tvec, camera_matrix, None)
        true_pixels = projected.reshape(-1, 2) + rng.normal(0.0, 0.5, (14, 2))

        false_world_points = rng.uniform([-2.0, -1.2, -2.0], [3.0, 1.0, 3.0], size=(18, 3))
        false_pixels = np.column_stack([rng.uniform(0, width, 18), rng.uniform(0, height, 18)])
        landmark_points = np.vstack([true_world_points, false_world_points]).astype(np.float32)
        keypoints = [SimpleNamespace(pt=tuple(pixel)) for pixel in np.vstack([true_pixels, false_pixels])]
        matches = [
            cv2.DMatch(_queryIdx=index, _trainIdx=index, _distance=float(18 + (index % 8)))
            for index in range(14)
        ]
        matches.extend(
            cv2.DMatch(_queryIdx=index, _trainIdx=index, _distance=float(25 + (index % 20)))
            for index in range(14, 32)
        )

        pose = _pose_from_fixed_center_matches(
            frame_width=width,
            frame_height=height,
            keypoints=keypoints,
            matches=matches,
            landmark_points=landmark_points,
            camera_center=center,
            fov_degrees=fov,
            random_seed=3,
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        estimated_rotation, _ = cv2.Rodrigues(pose["rvec"])
        relative = true_rotation.T @ estimated_rotation
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        self.assertLess(np.degrees(np.arccos(cosine)), 2.0)
        estimated_matrix = np.asarray(_camera_to_world(pose["rvec"], pose["tvec"]), dtype=np.float64)
        np.testing.assert_allclose(estimated_matrix[:3, 3], center, atol=1e-6)
        self.assertGreaterEqual(pose["inlier_count"], 10)

    def test_depth_landmark_reprojects_to_its_source_pixel(self) -> None:
        intrinsics = np.asarray([[920.0, 0.0, 640.0], [0.0, 915.0, 360.0], [0.0, 0.0, 1.0]])
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, 3] = [0.7, 1.4, -0.3]
        source_pixel = np.asarray([431.5, 277.25])

        point = _world_point(source_pixel[0], source_pixel[1], 2.6, intrinsics, camera_to_world)
        projection = intrinsics @ _world_to_cv(camera_to_world)[:3, :]

        np.testing.assert_allclose(_project_point(projection, point), source_pixel, atol=1e-6)

    def test_semantic_room_objects_provide_a_bounded_pose_seed(self) -> None:
        width, height = 640, 360
        request, true_center = _semantic_center_scene()

        seeds = _semantic_object_pose_seeds(request, frame_index=0, frame_width=width, frame_height=height)

        self.assertTrue(seeds)
        best = seeds[0]
        estimated = np.asarray(_camera_to_world(best["rvec"], best["tvec"]), dtype=np.float64)
        np.testing.assert_allclose(estimated[:3, 3], true_center, atol=0.30)
        self.assertEqual(best["semantic_object_match_count"], 3)
        self.assertNotIn("chair", best["semantic_object_labels"])
        self.assertEqual(best["semantic_object_anchor"], "center")

    def test_minimal_semantic_roots_preserve_pose_across_fov_and_assignment_order(self) -> None:
        # Three exact center correspondences have ambiguous minimal roots.
        # Recover the upright branch without depending on SQPnP's root/order
        # choices or using duplicated floor anchors as extra object evidence.
        for fov in (54.0, 60.0, 74.0):
            for reverse_order in (False, True):
                with self.subTest(fov=fov, reverse_order=reverse_order):
                    request, true_center = _semantic_center_scene(fov)
                    if reverse_order:
                        request.room_objects.reverse()
                        request.object_detections.reverse()
                    seeds = _semantic_object_pose_seeds(
                        request, frame_index=0, frame_width=640, frame_height=360,
                    )
                    self.assertTrue(seeds)
                    best = seeds[0]
                    estimated = np.asarray(_camera_to_world(best["rvec"], best["tvec"]))
                    np.testing.assert_allclose(estimated[:3, 3], true_center, atol=0.01)
                    self.assertEqual(best["fov_degrees"], fov)
                    self.assertEqual(best["semantic_object_inlier_count"], 3)
                    self.assertEqual(best["semantic_object_anchor"], "center")
                    self.assertTrue(_pose_scene_prior(estimated, request)["accepted"])

    def test_stable_movable_detection_can_seed_but_not_publish_semantic_pose(self) -> None:
        request = CameraLocalizationRequest.model_validate(
            {
                "landmarks": [
                    {
                        "point": [float(index), 0.0, 1.0],
                        "descriptor_base64": base64.b64encode(bytes(32)).decode("ascii"),
                    }
                    for index in range(8)
                ],
                "frames": [
                    {"frame_base64": "x", "width": 640, "height": 360}
                    for _ in range(4)
                ],
                "fov_degrees": 60.0,
                "room_objects": [
                    {
                        "id": "chair-1",
                        "label": "chair",
                        "center": {"x": 0.0, "y": -0.8, "z": 1.0},
                        "dimensions": {"x": 0.6, "y": 1.0, "z": 0.6},
                        "confidence": 0.45,
                    },
                    {
                        "id": "storage-1",
                        "label": "storage",
                        "center": {"x": 1.0, "y": -0.4, "z": 1.5},
                        "dimensions": {"x": 0.8, "y": 1.2, "z": 0.5},
                        "confidence": 0.95,
                    },
                ],
                "object_detections": [
                    {
                        "frame_index": frame_index,
                        "label": label,
                        "confidence": 0.90,
                        "bbox": bbox,
                    }
                    for frame_index in range(4)
                    for label, bbox in (
                        ("chair", [100.0, 120.0, 220.0, 300.0]),
                        ("shelf", [340.0, 80.0, 620.0, 350.0]),
                    )
                ],
            }
        )

        self.assertTrue(_stable_movable_object_detection(request))
        self.assertFalse(_semantic_object_assignments(request, 0))
        assignments = _semantic_object_assignments(request, 0, allow_movable_seed=True)
        self.assertTrue(assignments)
        self.assertTrue(any(0 in {index for index, _ in pairs} for pairs in assignments))

        score = _semantic_cuboid_score(
            request,
            frame_index=0,
            assignment=[(0, 0), (1, 1)],
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[0.0], [0.0], [3.0]], dtype=np.float64),
            camera_matrix=np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]]),
            frame_width=640,
            frame_height=360,
        )
        self.assertTrue(score["contains_movable_semantic_group"])
        self.assertFalse(_semantic_cuboid_has_robust_support(score))

    def test_stable_movable_detection_tracks_cluster_with_per_frame_distractor(self) -> None:
        request = CameraLocalizationRequest.model_validate(
            {
                "landmarks": [
                    {
                        "point": [float(index), 0.0, 1.0],
                        "descriptor_base64": base64.b64encode(bytes(32)).decode("ascii"),
                    }
                    for index in range(8)
                ],
                "frames": [
                    {"frame_base64": "x", "width": 640, "height": 360}
                    for _ in range(4)
                ],
                "room_objects": [
                    {
                        "id": "chair-1",
                        "label": "chair",
                        "center": {"x": 0.0, "y": -0.8, "z": 1.0},
                        "dimensions": {"x": 0.6, "y": 1.0, "z": 0.6},
                        "confidence": 0.45,
                    }
                ],
                "object_detections": [
                    {
                        "frame_index": frame_index,
                        "label": "chair",
                        "confidence": 0.80,
                        "bbox": [100.0, 120.0, 180.0, 300.0],
                    }
                    for frame_index in range(4)
                ]
                + [
                    {
                        "frame_index": 0,
                        "label": "chair",
                        "confidence": 0.90,
                        "bbox": [0.0, 40.0, 500.0, 350.0],
                    }
                ],
            }
        )

        self.assertTrue(_stable_movable_object_detection(request))

    def test_stable_movable_detection_handles_partial_low_confidence_burst(self) -> None:
        request = CameraLocalizationRequest.model_validate(
            {
                "landmarks": [
                    {
                        "point": [float(index), 0.0, 1.0],
                        "descriptor_base64": base64.b64encode(bytes(32)).decode("ascii"),
                    }
                    for index in range(8)
                ],
                "frames": [
                    {"frame_base64": "x", "width": 640, "height": 360}
                    for _ in range(6)
                ],
                "room_objects": [
                    {
                        "id": "chair-1",
                        "label": "chair",
                        "center": {"x": 0.0, "y": -0.8, "z": 1.0},
                        "dimensions": {"x": 0.6, "y": 1.0, "z": 0.6},
                        "confidence": 0.45,
                    }
                ],
                "object_detections": [
                    {
                        "frame_index": frame_index,
                        "label": "chair",
                        "confidence": confidence,
                        "bbox": bbox,
                    }
                    for frame_index, confidence, bbox in (
                        (0, 0.18, [104.0, 248.0, 149.0, 358.0]),
                        (1, 0.22, [105.0, 248.0, 149.0, 355.0]),
                        (2, 0.20, [102.0, 248.0, 149.0, 356.0]),
                        (3, 0.352, [106.0, 248.0, 149.0, 311.0]),
                        (4, 0.389, [105.0, 248.0, 149.0, 311.0]),
                        (5, 0.407, [105.0, 248.0, 149.0, 326.0]),
                    )
                ],
            }
        )

        self.assertTrue(_stable_movable_object_detection(request))

    def test_rotated_roomplan_cuboids_refine_semantic_camera_center(self) -> None:
        width, height, fov = 960, 540, 74.0
        focal = 0.5 * width / np.tan(np.radians(fov) / 2.0)
        camera_matrix = np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        request = _request(8, width, height)
        request.room_zones = [
            CameraLocalizationRoomZone.model_validate(
                {
                    "id": "room",
                    "floor_y": -1.3,
                    "polygon": [
                        {"x": -3.0, "z": -2.0},
                        {"x": 3.0, "z": -2.0},
                        {"x": 3.0, "z": 4.0},
                        {"x": -3.0, "z": 4.0},
                    ],
                }
            )
        ]

        def transform(yaw_degrees: float) -> dict:
            yaw = np.radians(yaw_degrees)
            c, s = float(np.cos(yaw)), float(np.sin(yaw))
            return {
                "values": [
                    [c, 0.0, s, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-s, 0.0, c, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }

        specs = [
            ("bed-1", "bed", (-0.8, -0.95, 0.1), (1.5, 0.6, 2.0), 18.0),
            ("table-1", "table", (0.8, -0.95, 0.2), (1.2, 0.75, 0.8), -24.0),
            ("storage-1", "storage", (1.6, -0.35, -0.6), (0.9, 1.9, 0.5), 8.0),
            ("chair-1", "chair", (-1.5, -0.75, -0.4), (0.7, 1.1, 0.7), 31.0),
        ]
        request.room_objects = [
            CameraLocalizationRoomObject(
                id=identifier,
                label=label,
                center={"x": center[0], "y": center[1], "z": center[2]},
                dimensions={"x": dimensions[0], "y": dimensions[1], "z": dimensions[2]},
                transform=transform(yaw),
                confidence=0.95,
            )
            for identifier, label, center, dimensions, yaw in specs
        ]

        true_center = np.asarray([0.15, -0.05, 3.15], dtype=np.float64)
        true_rotation = np.diag([1.0, -1.0, -1.0])
        true_rvec, _ = cv2.Rodrigues(true_rotation)
        true_tvec = -(true_rotation @ true_center.reshape(3, 1))
        detections = []
        for room_object in request.room_objects:
            projected = _project_room_object_bbox(
                room_object,
                rvec=true_rvec,
                tvec=true_tvec,
                camera_matrix=camera_matrix,
                frame_width=width,
                frame_height=height,
            )
            self.assertIsNotNone(projected)
            assert projected is not None
            bbox, _ = projected
            detections.append(
                CameraLocalizationObjectDetection(
                    frame_index=0,
                    label=room_object.label,
                    confidence=0.92,
                    bbox=bbox,
                )
            )
        request.object_detections = detections

        seed_center = true_center + np.asarray([0.45, 0.12, 0.30])
        seed_tvec = -(true_rotation @ seed_center.reshape(3, 1))
        seed = {
            "rvec": true_rvec.copy(),
            "tvec": seed_tvec,
            "camera_matrix": camera_matrix.copy(),
            "intrinsics_source": "semantic-object-seed",
            "fov_degrees": fov,
            "inlier_indices": np.arange(4),
            "inlier_count": 4,
            "pool_size": 4,
            "mean_error": 10.0,
            "coverage_ratio": 0.1,
            "world_spread_m": 2.0,
            "positive_depth_ratio": 1.0,
            "semantic_object_assignment": [(index, index) for index in range(4)],
            "semantic_object_match_count": 4,
            "semantic_object_labels": [item[1] for item in specs],
            "semantic_object_anchor": "center",
            "semantic_object_confidence": 0.9,
        }

        refined = _refine_semantic_cuboid_pose(
            request,
            frame_index=0,
            frame_width=width,
            frame_height=height,
            seed=seed,
        )
        refined_matrix = np.asarray(_camera_to_world(refined["rvec"], refined["tvec"]), dtype=np.float64)
        seed_error = float(np.linalg.norm(seed_center[[0, 2]] - true_center[[0, 2]]))
        refined_error = float(np.linalg.norm(refined_matrix[[0, 2], 3] - true_center[[0, 2]]))

        self.assertLess(refined_error, seed_error * 0.45)
        self.assertGreaterEqual(refined["semantic_cuboid"]["mean_iou"], 0.55)
        self.assertEqual(refined["semantic_cuboid"]["matched_object_count"], 4)
        self.assertGreaterEqual(refined["semantic_cuboid"]["semantic_group_count"], 3)

    def test_room_object_projection_keeps_partially_visible_near_plane_cuboid(self) -> None:
        width, height, fov = 640, 360, 60.0
        focal = 0.5 * width / np.tan(np.radians(fov) / 2.0)
        camera_matrix = np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        room_object = CameraLocalizationRoomObject(
            id="storage-near-camera",
            label="storage",
            center={"x": 0.0, "y": 0.0, "z": 0.1},
            dimensions={"x": 0.4, "y": 0.4, "z": 0.6},
            transform={"values": np.eye(4, dtype=np.float64).tolist()},
            confidence=1.0,
        )

        projected = _project_room_object_bbox(
            room_object,
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.zeros((3, 1), dtype=np.float64),
            camera_matrix=camera_matrix,
            frame_width=width,
            frame_height=height,
        )

        self.assertIsNotNone(projected)
        assert projected is not None
        bbox, positive_ratio = projected
        self.assertEqual(positive_ratio, 0.5)
        self.assertGreater(bbox[2] - bbox[0], 2.0)
        self.assertGreater(bbox[3] - bbox[1], 2.0)

    def test_pose_search_recovers_unknown_webcam_fov_with_ranked_outliers(self) -> None:
        rng = np.random.default_rng(42)
        width, height = 960, 540
        true_fov = 84.0
        focal = 0.5 * width / np.tan(np.radians(true_fov) / 2.0)
        camera_matrix = np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])

        inlier_count = 18
        outlier_count = 64
        inlier_world = np.column_stack(
            [
                rng.uniform(-2.0, 2.0, inlier_count),
                rng.uniform(-1.2, 1.2, inlier_count),
                rng.uniform(-0.6, 0.6, inlier_count),
            ]
        ).astype(np.float32)
        outlier_world = rng.uniform(-2.0, 2.0, (outlier_count, 3)).astype(np.float32)
        landmark_points = np.vstack([inlier_world, outlier_world])

        rvec = np.asarray([[0.04], [-0.08], [0.03]], dtype=np.float64)
        tvec = np.asarray([[0.2], [-0.1], [5.2]], dtype=np.float64)
        projected, _ = cv2.projectPoints(inlier_world, rvec, tvec, camera_matrix, None)
        inlier_pixels = projected.reshape(-1, 2) + rng.normal(0.0, 0.45, (inlier_count, 2))
        outlier_pixels = np.column_stack(
            [rng.uniform(0, width, outlier_count), rng.uniform(0, height, outlier_count)]
        )
        all_pixels = np.vstack([inlier_pixels, outlier_pixels])

        keypoints = [SimpleNamespace(pt=tuple(pixel)) for pixel in all_pixels]
        matches = [
            SimpleNamespace(
                queryIdx=index,
                trainIdx=index,
                distance=float(18 + index % 14 if index < inlier_count else 45 + index % 25),
            )
            for index in range(inlier_count + outlier_count)
        ]

        pose = _pose_from_matches(
            payload=_request(len(landmark_points), width, height),
            frame_width=width,
            frame_height=height,
            keypoints=keypoints,
            matches=matches,
            landmark_points=landmark_points,
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose["fov_degrees"], true_fov)
        self.assertGreaterEqual(pose["inlier_count"], 16)
        self.assertLess(pose["mean_error"], 1.0)
        self.assertGreater(pose["coverage_ratio"], 0.02)

    def test_pose_search_prefers_physically_plausible_seed_over_tighter_upside_down_fit(self) -> None:
        width, height = 640, 480
        payload = _request(6, width, height).model_dump()
        payload["room_zones"] = [{
            "id": "room",
            "floor_y": 0.0,
            "polygon": [
                {"x": -2.0, "z": -2.0},
                {"x": 2.0, "z": -2.0},
                {"x": 2.0, "z": 2.0},
                {"x": -2.0, "z": 2.0},
            ],
        }]
        request = CameraLocalizationRequest.model_validate(payload)

        landmark_points = np.asarray([
            [-0.9, 0.7, -3.0],
            [-0.5, 1.8, -3.2],
            [0.0, 0.9, -2.8],
            [0.4, 1.6, -3.3],
            [0.8, 0.8, -3.1],
            [1.0, 1.9, -2.9],
        ], dtype=np.float32)
        image_points = np.asarray([
            [120.0, 120.0],
            [180.0, 165.0],
            [250.0, 210.0],
            [330.0, 250.0],
            [410.0, 290.0],
            [500.0, 330.0],
        ], dtype=np.float32)
        keypoints = [SimpleNamespace(pt=tuple(point)) for point in image_points]
        matches = [SimpleNamespace(queryIdx=index, trainIdx=index, distance=18.0 + index) for index in range(6)]

        upright_camera = np.eye(4, dtype=np.float64)
        upright_camera[:3, 3] = [0.0, 1.5, 0.0]
        upside_down_camera = np.eye(4, dtype=np.float64)
        upside_down_camera[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        upside_down_camera[:3, 3] = [0.0, 1.5, 0.0]

        upright_world_to_cv = _world_to_cv(upright_camera)
        upside_down_world_to_cv = _world_to_cv(upside_down_camera)
        upright_rvec, _ = cv2.Rodrigues(upright_world_to_cv[:3, :3])
        upside_down_rvec, _ = cv2.Rodrigues(upside_down_world_to_cv[:3, :3])
        upright_tvec = upright_world_to_cv[:3, 3].reshape(3, 1)
        upside_down_tvec = upside_down_world_to_cv[:3, 3].reshape(3, 1)

        def fake_solve_pnp(_objects, _images, camera_matrix, *_args, **_kwargs):
            # The narrowest FOV receives a deceptively tighter upside-down
            # solution; all later FOVs expose the physically valid basin.
            if float(camera_matrix[0, 0]) > 800.0:
                return True, upside_down_rvec.copy(), upside_down_tvec.copy(), None
            return True, upright_rvec.copy(), upright_tvec.copy(), None

        def fake_refine(_objects, _images, _camera_matrix, _distortion, rvec, tvec):
            return rvec, tvec

        def fake_project(_objects, rvec, _tvec, _camera_matrix, _distortion):
            upside_down = float(np.linalg.norm(np.asarray(rvec) - upside_down_rvec)) < 1e-5
            offset = 0.25 if upside_down else 2.0
            projected = image_points + np.asarray([offset, 0.0], dtype=np.float32)
            return projected.reshape(-1, 1, 2), None

        with (
            patch("geometry_service.localization.cv2.solvePnPRansac", side_effect=fake_solve_pnp),
            patch("geometry_service.localization.cv2.solvePnPRefineLM", side_effect=fake_refine),
            patch("geometry_service.localization.cv2.projectPoints", side_effect=fake_project),
        ):
            pose = _pose_from_matches(
                payload=request,
                frame_width=width,
                frame_height=height,
                keypoints=keypoints,
                matches=matches,
                landmark_points=landmark_points,
            )

        self.assertIsNotNone(pose)
        assert pose is not None
        scene_prior = _pose_scene_prior(np.asarray(_camera_to_world(pose["rvec"], pose["tvec"])), request)
        self.assertTrue(scene_prior["accepted"])
        self.assertTrue(pose["seed_scene_prior"]["accepted"])

    def test_pose_guided_refinement_recovers_cross_view_correspondences(self) -> None:
        rng = np.random.default_rng(7)
        width, height = 640, 360
        focal = 0.5 * width / np.tan(np.radians(84.0) / 2.0)
        camera_matrix = np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        landmark_points = np.column_stack(
            [
                rng.uniform(-1.4, 1.4, 18),
                rng.uniform(-0.9, 0.9, 18),
                rng.uniform(-0.5, 0.5, 18),
            ]
        ).astype(np.float32)
        landmark_descriptors = rng.integers(0, 256, (18, 32), dtype=np.uint8)
        true_rvec = np.asarray([[0.03], [-0.07], [0.02]], dtype=np.float64)
        true_tvec = np.asarray([[0.1], [-0.08], [4.7]], dtype=np.float64)
        projected, _ = cv2.projectPoints(landmark_points, true_rvec, true_tvec, camera_matrix, None)
        pixels = projected.reshape(-1, 2) + rng.normal(0.0, 0.35, (18, 2))

        distractor_pixels = np.column_stack(
            [rng.uniform(0, width, 20), rng.uniform(0, height, 20)]
        )
        keypoints = [SimpleNamespace(pt=tuple(pixel)) for pixel in np.vstack([pixels, distractor_pixels])]
        descriptors = np.vstack(
            [landmark_descriptors.copy(), rng.integers(0, 256, (20, 32), dtype=np.uint8)]
        )
        seed_pose = {
            "rvec": true_rvec + np.asarray([[0.004], [-0.004], [0.003]]),
            "tvec": true_tvec + np.asarray([[0.015], [-0.01], [0.02]]),
            "camera_matrix": camera_matrix,
            "intrinsics_source": "estimated-fov",
            "fov_degrees": 84.0,
            "inlier_indices": np.arange(6),
            "inlier_count": 6,
            "pool_size": 6,
            "mean_error": 3.0,
            "coverage_ratio": 0.02,
            "world_spread_m": 1.0,
            "positive_depth_ratio": 1.0,
        }

        result = _refine_pose_with_guided_matches(
            frame_width=width,
            frame_height=height,
            keypoints=keypoints,
            descriptors=descriptors,
            landmark_points=landmark_points,
            landmark_descriptors=landmark_descriptors,
            pose=seed_pose,
        )

        self.assertIsNotNone(result)
        assert result is not None
        refined, matches = result
        self.assertGreaterEqual(len(matches), 16)
        self.assertGreaterEqual(refined["inlier_count"], 16)
        self.assertLess(refined["mean_error"], 1.0)

    def test_pose_guided_matching_keeps_near_duplicate_roomplan_landmarks(self) -> None:
        width, height = 640, 360
        camera_matrix = np.asarray(
            [[500.0, 0.0, width / 2.0], [0.0, 500.0, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # Two physical features, each recorded twice by overlapping RoomPlan
        # scan views. The duplicate records are only a few centimetres apart.
        landmark_points = np.asarray(
            [
                [-0.40, 0.00, 0.00],
                [-0.36, 0.01, 0.01],
                [0.45, 0.05, 0.10],
                [0.49, 0.04, 0.11],
            ],
            dtype=np.float32,
        )
        descriptor_a = np.zeros(32, dtype=np.uint8)
        descriptor_b = np.full(32, 255, dtype=np.uint8)
        landmark_descriptors = np.asarray(
            [descriptor_a, descriptor_a, descriptor_b, descriptor_b],
            dtype=np.uint8,
        )
        pose = {
            "rvec": np.zeros((3, 1), dtype=np.float64),
            "tvec": np.asarray([[0.0], [0.0], [4.0]], dtype=np.float64),
            "camera_matrix": camera_matrix,
        }
        projected, _ = cv2.projectPoints(
            landmark_points[[0, 2]], pose["rvec"], pose["tvec"], camera_matrix, None
        )
        keypoints = [SimpleNamespace(pt=tuple(point)) for point in projected.reshape(-1, 2)]
        descriptors = np.asarray([descriptor_a, descriptor_b], dtype=np.uint8)

        matches = _pose_guided_matches(
            frame_width=width,
            frame_height=height,
            keypoints=keypoints,
            descriptors=descriptors,
            landmark_points=landmark_points,
            landmark_descriptors=landmark_descriptors,
            pose=pose,
            radius_px=24.0,
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual({match.queryIdx for match in matches}, {0, 1})

    def test_pose_guided_matching_uses_learned_score_to_resolve_repeated_texture(self) -> None:
        width, height = 640, 360
        camera_matrix = np.asarray(
            [[500.0, 0.0, width / 2.0], [0.0, 500.0, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        landmark_points = np.asarray(
            [
                [0.00, 0.00, 0.00],
                [0.12, 0.00, 0.00],
                [0.70, 0.00, 0.00],
            ],
            dtype=np.float32,
        )
        correct_descriptor = np.asarray([255] * 5 + [0] * 27, dtype=np.uint8)
        repeated_texture_descriptor = np.asarray([255] * 4 + [0] * 28, dtype=np.uint8)
        other_descriptor = np.full(32, 255, dtype=np.uint8)
        landmark_descriptors = np.asarray(
            [correct_descriptor, repeated_texture_descriptor, other_descriptor],
            dtype=np.uint8,
        )
        pose = {
            "rvec": np.zeros((3, 1), dtype=np.float64),
            "tvec": np.asarray([[0.0], [0.0], [4.0]], dtype=np.float64),
            "camera_matrix": camera_matrix,
        }
        projected, _ = cv2.projectPoints(
            landmark_points[[0, 2]], pose["rvec"], pose["tvec"], camera_matrix, None
        )
        keypoints = [SimpleNamespace(pt=tuple(point)) for point in projected.reshape(-1, 2)]
        descriptors = np.asarray(
            [np.zeros(32, dtype=np.uint8), other_descriptor],
            dtype=np.uint8,
        )

        baseline = _pose_guided_matches(
            frame_width=width,
            frame_height=height,
            keypoints=keypoints,
            descriptors=descriptors,
            landmark_points=landmark_points,
            landmark_descriptors=landmark_descriptors,
            pose=pose,
            radius_px=32.0,
        )
        baseline_query_zero = next(match for match in baseline if match.queryIdx == 0)
        self.assertEqual(baseline_query_zero.trainIdx, 1)

        class FakeLearnedMatcher:
            threshold = 0.55

            def score_pairs(self, query, landmarks, query_responses=None, landmark_responses=None):
                del query, query_responses, landmark_responses
                nonzero_bytes = np.count_nonzero(landmarks, axis=1)
                return np.asarray(
                    [0.92 if count == 5 else 0.10 for count in nonzero_bytes],
                    dtype=np.float32,
                )

        learned = _pose_guided_matches(
            frame_width=width,
            frame_height=height,
            keypoints=keypoints,
            descriptors=descriptors,
            landmark_points=landmark_points,
            landmark_descriptors=landmark_descriptors,
            pose=pose,
            radius_px=32.0,
            learned_matcher=FakeLearnedMatcher(),
            query_responses=np.asarray([0.01, 0.01], dtype=np.float32),
            landmark_responses=np.asarray([0.01, 0.01, 0.01], dtype=np.float32),
        )
        learned_query_zero = next(match for match in learned if match.queryIdx == 0)
        self.assertEqual(learned_query_zero.trainIdx, 0)

    def test_descriptor_matching_keeps_ratio_pass_when_learned_score_disagrees(self) -> None:
        query = np.zeros((2, 32), dtype=np.uint8)
        query[1] = 255
        wrong = np.asarray([255] * 2 + [0] * 30, dtype=np.uint8)
        correct = np.asarray([255] * 3 + [0] * 29, dtype=np.uint8)
        far = np.full(32, 255, dtype=np.uint8)
        landmarks = np.asarray([wrong, correct, far], dtype=np.uint8)
        indices = np.arange(3, dtype=np.int32)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        baseline = _descriptor_matches(query, landmarks, indices, matcher)
        query_zero = next(match for match in baseline if match.queryIdx == 0)
        self.assertEqual(query_zero.trainIdx, 0)

        class FakeLearnedMatcher:
            threshold = 0.55

            def score_pairs(self, query_descriptors, landmark_descriptors, query_responses=None, landmark_responses=None):
                del query_descriptors, query_responses, landmark_responses
                nonzero_bytes = np.count_nonzero(landmark_descriptors, axis=1)
                return np.asarray(
                    [0.94 if count == 3 else 0.12 for count in nonzero_bytes],
                    dtype=np.float32,
                )

        learned = _descriptor_matches(
            query,
            landmarks,
            indices,
            matcher,
            learned_matcher=FakeLearnedMatcher(),
        )
        learned_query_zero = next(match for match in learned if match.queryIdx == 0)
        self.assertEqual(learned_query_zero.trainIdx, 0)

    def test_provisional_guided_pose_recovers_six_good_matches_with_one_outlier(self) -> None:
        scene = _guided_match_scene()

        provisional = _provisional_pose_from_guided_matches(**scene)

        self.assertIsNotNone(provisional)
        assert provisional is not None
        self.assertGreaterEqual(provisional["inlier_count"], 6)
        self.assertEqual(provisional["provisional_subset_size"], 6)
        np.testing.assert_array_equal(provisional["inlier_indices"], np.arange(6))
        self.assertGreaterEqual(provisional["provisional_hypothesis_pool_size"], 6)
        self.assertLess(provisional["mean_error"], 2.0)


    def test_provisional_fit_support_tracks_rejected_and_clean_correspondences(self) -> None:
        for outlier_index in (0, 3, 6, None):
            with self.subTest(outlier_index=outlier_index):
                provisional = _provisional_pose_from_guided_matches(
                    **_guided_match_scene(outlier_index),
                )
                self.assertIsNotNone(provisional)
                assert provisional is not None
                expected = [index for index in range(7) if index != outlier_index]
                np.testing.assert_array_equal(provisional["inlier_indices"], expected)
                self.assertEqual(provisional["provisional_subset_size"], len(expected))
                self.assertEqual(provisional["pool_size"], 7)
                self.assertLess(provisional["mean_error"], 2.0)


if __name__ == "__main__":
    unittest.main()

