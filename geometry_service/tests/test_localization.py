import base64
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from geometry_service.contracts import CameraLocalizationRequest
from geometry_service.localization import _pose_from_matches, _poses_agree, _project_point, _world_point, _world_to_cv


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


class LocalizationTests(unittest.TestCase):
    def test_pose_consensus_rejects_camera_motion(self) -> None:
        first = np.eye(4, dtype=np.float64)
        nearby = np.eye(4, dtype=np.float64)
        nearby[:3, 3] = [0.12, -0.04, 0.08]
        moved = np.eye(4, dtype=np.float64)
        moved[:3, 3] = [0.8, 0.0, 0.0]

        self.assertTrue(_poses_agree(first, nearby))
        self.assertFalse(_poses_agree(first, moved))

    def test_depth_landmark_reprojects_to_its_source_pixel(self) -> None:
        intrinsics = np.asarray([[920.0, 0.0, 640.0], [0.0, 915.0, 360.0], [0.0, 0.0, 1.0]])
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, 3] = [0.7, 1.4, -0.3]
        source_pixel = np.asarray([431.5, 277.25])

        point = _world_point(source_pixel[0], source_pixel[1], 2.6, intrinsics, camera_to_world)
        projection = intrinsics @ _world_to_cv(camera_to_world)[:3, :]

        np.testing.assert_allclose(_project_point(projection, point), source_pixel, atol=1e-6)

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


if __name__ == "__main__":
    unittest.main()
