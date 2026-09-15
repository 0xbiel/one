import base64
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from geometry_service.contracts import (
    CameraLocalizationObjectDetection,
    CameraLocalizationRequest,
    CameraLocalizationRoomObject,
    CameraLocalizationRoomZone,
)
from geometry_service.localization import (
    _camera_to_world,
    _candidate_is_positioned,
    _consensus_stats,
    _pose_from_fixed_center_matches,
    _pose_from_matches,
    _pose_with_camera_center,
    _project_room_object_bbox,
    _refine_pose_with_guided_matches,
    _refine_semantic_cuboid_pose,
    _room_perimeter_search_centers,
    _semantic_cuboid_rank,
    _poses_agree,
    _project_point,
    _semantic_object_pose_seeds,
    _world_point,
    _world_to_cv,
)


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


class LocalizationTests(unittest.TestCase):
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
        width, height, fov = 640, 360, 60.0
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

        seeds = _semantic_object_pose_seeds(request, frame_index=0, frame_width=width, frame_height=height)

        self.assertTrue(seeds)
        best = seeds[0]
        estimated = np.asarray(_camera_to_world(best["rvec"], best["tvec"]), dtype=np.float64)
        np.testing.assert_allclose(estimated[:3, 3], true_center, atol=0.05)
        self.assertEqual(best["semantic_object_match_count"], 4)
        self.assertEqual(best["semantic_object_anchor"], "center")

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


if __name__ == "__main__":
    unittest.main()
