"""RoomPlan visual landmarks and fixed-camera 6-DoF localization.

The scan side turns RGB + LiDAR depth + ARKit poses into ORB descriptors tied
to metric RoomPlan coordinates.  The fixed camera side matches ORB features
against those derived landmarks and estimates a real camera pose with
solvePnPRansac. Raw RGB/depth bytes are never returned or persisted here.
"""

from __future__ import annotations

import base64
import binascii
import math
from typing import Any

import cv2
import numpy as np

from .contracts import CameraLocalizationRequest, VisualLandmarkBuildRequest
from .real_vision import VisionInferenceError, decode_jpeg


class LocalizationInputError(ValueError):
    pass


def _jpeg_bytes(value: str) -> bytes:
    encoded = value.strip()
    if encoded.startswith("data:image/jpeg;base64,"):
        encoded = encoded.partition(",")[2]
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LocalizationInputError("frame_base64 is not valid base64") from exc
    if not payload or len(payload) > 3_000_000:
        raise LocalizationInputError("each JPEG must be between 1 byte and 3 MB")
    return payload


def _depth_array(value: str, width: int, height: int) -> np.ndarray:
    try:
        payload = base64.b64decode(value.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LocalizationInputError("depth_base64 is not valid base64") from exc
    expected = width * height * 4
    if len(payload) != expected:
        raise LocalizationInputError("depth_base64 must contain tightly packed float32 metres")
    depth = np.frombuffer(payload, dtype="<f4").reshape(height, width)
    if not np.isfinite(depth).any():
        raise LocalizationInputError("depth map has no finite samples")
    return depth


def _matrix(value: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise LocalizationInputError(f"matrix must be finite {shape[0]}x{shape[1]}")
    return result


def _depth_at(depth: np.ndarray, u: float, v: float, image_width: int, image_height: int) -> float | None:
    x = int(round(u / max(1, image_width - 1) * (depth.shape[1] - 1)))
    y = int(round(v / max(1, image_height - 1) * (depth.shape[0] - 1)))
    x0, x1 = max(0, x - 1), min(depth.shape[1], x + 2)
    y0, y1 = max(0, y - 1), min(depth.shape[0], y + 2)
    values = depth[y0:y1, x0:x1].reshape(-1)
    values = values[np.isfinite(values) & (values >= 0.20) & (values <= 15.0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _world_point(u: float, v: float, depth_m: float, intrinsics: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if fx <= 0 or fy <= 0:
        raise LocalizationInputError("camera intrinsics must contain positive focal lengths")
    # ARKit camera coordinates: +X right, +Y up, -Z forward.
    camera_point = np.asarray(
        [(u - cx) * depth_m / fx, -(v - cy) * depth_m / fy, -depth_m, 1.0],
        dtype=np.float64,
    )
    world = camera_to_world @ camera_point
    if not np.isfinite(world[:3]).all():
        raise LocalizationInputError("derived landmark is non-finite")
    return world[:3]


def _world_to_cv(camera_to_world: np.ndarray) -> np.ndarray:
    """Convert an ARKit camera pose into an OpenCV world-to-camera matrix."""
    cv_from_arkit = np.diag([1.0, -1.0, -1.0, 1.0])
    try:
        world_to_arkit = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError as exc:
        raise LocalizationInputError("camera_to_world must be invertible") from exc
    result = cv_from_arkit @ world_to_arkit
    if not np.isfinite(result).all():
        raise LocalizationInputError("camera pose produced a non-finite projection")
    return result


def _project_point(projection: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    homogeneous = projection @ np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
    if not np.isfinite(homogeneous).all() or homogeneous[2] <= 1e-6:
        return None
    return homogeneous[:2] / homogeneous[2]


def _triangulated_candidates(frames: list[dict[str, Any]]) -> list[tuple[np.ndarray, bytes, float]]:
    """Triangulate ORB features from known RoomPlan/ARKit scan poses.

    RoomPlan can finish a valid LiDAR capture while ARKit omits sceneDepth from
    some or all exposed ARFrames.  The camera poses are still metric and share
    the RoomPlan coordinate frame, so multi-view feature triangulation provides
    a safe fallback without persisting any raw scan frames.
    """
    if len(frames) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    candidates: list[tuple[np.ndarray, bytes, float]] = []
    for first_index in range(len(frames) - 1):
        first = frames[first_index]
        first_center = first["camera_to_world"][:3, 3]
        first_projection = first["intrinsics"] @ first["world_to_cv"][:3, :]
        for second_index in range(first_index + 1, len(frames)):
            second = frames[second_index]
            second_center = second["camera_to_world"][:3, 3]
            baseline = float(np.linalg.norm(second_center - first_center))
            if baseline < 0.08:
                continue

            pairs = matcher.knnMatch(first["descriptors"], second["descriptors"], k=2)
            ratio_matches = [
                best
                for pair in pairs
                if len(pair) == 2
                for best, alternate in [pair]
                if best.distance < 0.72 * alternate.distance
            ]
            unique: dict[int, Any] = {}
            for match in sorted(ratio_matches, key=lambda item: item.distance):
                unique.setdefault(match.trainIdx, match)
            matches = list(unique.values())[:400]
            if len(matches) < 6:
                continue

            first_points = np.asarray([first["keypoints"][match.queryIdx].pt for match in matches], dtype=np.float64)
            second_points = np.asarray([second["keypoints"][match.trainIdx].pt for match in matches], dtype=np.float64)
            second_projection = second["intrinsics"] @ second["world_to_cv"][:3, :]
            homogeneous_points = cv2.triangulatePoints(
                first_projection,
                second_projection,
                first_points.T,
                second_points.T,
            )

            for match, first_uv, second_uv, homogeneous in zip(matches, first_points, second_points, homogeneous_points.T):
                if abs(float(homogeneous[3])) <= 1e-8:
                    continue
                point = np.asarray(homogeneous[:3] / homogeneous[3], dtype=np.float64)
                if not np.isfinite(point).all():
                    continue
                point_h = np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
                first_depth = float((first["world_to_cv"] @ point_h)[2])
                second_depth = float((second["world_to_cv"] @ point_h)[2])
                if not (0.20 <= first_depth <= 15.0 and 0.20 <= second_depth <= 15.0):
                    continue

                ray_first = point - first_center
                ray_second = point - second_center
                norm_product = float(np.linalg.norm(ray_first) * np.linalg.norm(ray_second))
                if norm_product <= 1e-8:
                    continue
                cosine = float(np.clip(np.dot(ray_first, ray_second) / norm_product, -1.0, 1.0))
                if math.degrees(math.acos(cosine)) < 0.75:
                    continue

                projected_first = _project_point(first_projection, point)
                projected_second = _project_point(second_projection, point)
                if projected_first is None or projected_second is None:
                    continue
                reprojection_error = max(
                    float(np.linalg.norm(projected_first - first_uv)),
                    float(np.linalg.norm(projected_second - second_uv)),
                )
                if reprojection_error > 3.5:
                    continue

                keypoint = first["keypoints"][match.queryIdx]
                descriptor = first["descriptors"][match.queryIdx]
                candidates.append((point, bytes(descriptor.tolist()), float(keypoint.response)))
    return candidates


def build_visual_landmarks(payload: VisualLandmarkBuildRequest) -> dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8, fastThreshold=7, edgeThreshold=19)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    candidates: list[tuple[np.ndarray, bytes, float]] = []
    frame_feature_counts: list[int] = []
    depth_feature_counts: list[int] = []
    feature_frames: list[dict[str, Any]] = []
    for frame in payload.frames:
        jpeg = _jpeg_bytes(frame.frame_base64)
        try:
            image = decode_jpeg(jpeg, frame.width, frame.height)
        except VisionInferenceError as exc:
            raise LocalizationInputError(str(exc)) from exc
        gray = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or not keypoints:
            frame_feature_counts.append(0)
            depth_feature_counts.append(0)
            continue
        intrinsics = _matrix(frame.intrinsics.values, (3, 3))
        camera_to_world = _matrix(frame.camera_to_world.values, (4, 4))
        world_to_cv = _world_to_cv(camera_to_world)
        feature_frames.append(
            {
                "keypoints": keypoints,
                "descriptors": descriptors,
                "intrinsics": intrinsics,
                "camera_to_world": camera_to_world,
                "world_to_cv": world_to_cv,
            }
        )
        frame_feature_counts.append(len(keypoints))
        accepted = 0
        if frame.depth_base64 is not None and frame.depth_width is not None and frame.depth_height is not None:
            depth = _depth_array(frame.depth_base64, frame.depth_width, frame.depth_height)
            for keypoint, descriptor in zip(keypoints, descriptors):
                depth_m = _depth_at(depth, keypoint.pt[0], keypoint.pt[1], frame.width, frame.height)
                if depth_m is None:
                    continue
                point = _world_point(keypoint.pt[0], keypoint.pt[1], depth_m, intrinsics, camera_to_world)
                candidates.append((point, bytes(descriptor.tolist()), float(keypoint.response)))
                accepted += 1
        depth_feature_counts.append(accepted)

    triangulated = _triangulated_candidates(feature_frames)
    candidates.extend(triangulated)

    # Keep the strongest descriptor in each 3 cm voxel so repeated scan frames
    # do not create a huge near-duplicate landmark set.
    voxels: dict[tuple[int, int, int], tuple[np.ndarray, bytes, float]] = {}
    for item in candidates:
        point, _, response = item
        key = tuple(int(round(float(component) / 0.03)) for component in point)
        if key not in voxels or response > voxels[key][2]:
            voxels[key] = item
    selected = sorted(voxels.values(), key=lambda item: item[2], reverse=True)[:5_000]
    if len(selected) < 40:
        return {
            "status": "needs_rescan",
            "schema_version": "roomplan-visual-landmarks.v1",
            "detector": "opencv-orb",
            "landmarks": [],
            "diagnostics": {
                "reason": "insufficient_visual_features",
                "landmark_count": len(selected),
                "frame_feature_counts": frame_feature_counts,
                "depth_feature_counts": depth_feature_counts,
                "triangulated_feature_count": len(triangulated),
                "raw_frames_persisted": False,
            },
        }
    return {
        "status": "ready",
        "schema_version": "roomplan-visual-landmarks.v1",
        "detector": "opencv-orb",
        "landmarks": [
            {
                "point": [round(float(value), 6) for value in point],
                "descriptor_base64": base64.b64encode(descriptor).decode("ascii"),
                "response": round(response, 6),
            }
            for point, descriptor, response in selected
        ],
        "diagnostics": {
            "landmark_count": len(selected),
            "source_frame_count": len(payload.frames),
            "frame_feature_counts": frame_feature_counts,
            "depth_feature_counts": depth_feature_counts,
            "triangulated_feature_count": len(triangulated),
            "raw_frames_persisted": False,
        },
    }


def _landmark_arrays(payload: CameraLocalizationRequest) -> tuple[np.ndarray, np.ndarray, list[str]]:
    points: list[list[float]] = []
    descriptors: list[np.ndarray] = []
    view_ids: list[str] = []
    for landmark in payload.landmarks:
        try:
            descriptor = base64.b64decode(landmark.descriptor_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise LocalizationInputError("landmark descriptor is invalid base64") from exc
        if len(descriptor) != 32:
            raise LocalizationInputError("ORB landmark descriptors must be 32 bytes")
        point = np.asarray(landmark.point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise LocalizationInputError("landmark points must be finite xyz coordinates")
        points.append(point.tolist())
        descriptors.append(np.frombuffer(descriptor, dtype=np.uint8))
        view_ids.append(landmark.view_id or "legacy")
    return np.asarray(points, dtype=np.float32), np.vstack(descriptors).astype(np.uint8), view_ids


def _descriptor_matches(
    query_descriptors: np.ndarray,
    landmark_descriptors: np.ndarray,
    landmark_indices: np.ndarray,
    matcher: Any,
) -> list[Any]:
    if len(landmark_indices) < 2:
        return []
    subset = landmark_descriptors[landmark_indices]
    pairs = matcher.knnMatch(query_descriptors, subset, k=2)
    reverse = matcher.match(subset, query_descriptors)
    reverse_best = {match.queryIdx: match.trainIdx for match in reverse}
    good: list[Any] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        first, second = pair
        if (
            first.distance <= 72
            and first.distance < 0.84 * second.distance
            and reverse_best.get(first.trainIdx) == first.queryIdx
        ):
            good.append(
                cv2.DMatch(
                    _queryIdx=first.queryIdx,
                    _trainIdx=int(landmark_indices[first.trainIdx]),
                    _distance=first.distance,
                )
            )
    unique: dict[int, Any] = {}
    for match in sorted(good, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)
    return list(unique.values())


def _camera_matrix(payload: CameraLocalizationRequest, width: int, height: int) -> tuple[np.ndarray, str]:
    if payload.intrinsics is not None:
        matrix = _matrix(payload.intrinsics.values, (3, 3))
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise LocalizationInputError("camera intrinsics must contain positive focal lengths")
        return matrix, "provided"
    focal = 0.5 * width / math.tan(math.radians(payload.fov_degrees) / 2.0)
    return np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64), "estimated-fov"


def _camera_matrix_candidates(payload: CameraLocalizationRequest, width: int, height: int) -> list[tuple[np.ndarray, str, float | None]]:
    """Return a small focal-length search when browser intrinsics are unknown.

    Browser camera APIs do not expose calibrated intrinsics. Treating every
    webcam as exactly 60 degrees can move otherwise-correct correspondences
    outside the RANSAC threshold, so test a bounded set of common horizontal
    fields of view and report the selected value.
    """
    if payload.intrinsics is not None:
        matrix, source = _camera_matrix(payload, width, height)
        return [(matrix, source, None)]

    requested = float(payload.fov_degrees)
    values = [requested, 42.0, 48.0, 54.0, 60.0, 66.0, 74.0, 84.0, 96.0]
    unique = sorted({round(value, 3) for value in values if 30.0 <= value <= 120.0})
    result: list[tuple[np.ndarray, str, float | None]] = []
    for fov in unique:
        focal = 0.5 * width / math.tan(math.radians(fov) / 2.0)
        matrix = np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        result.append((matrix, "estimated-fov", fov))
    return result


def _ranked_match_pools(matches: list[Any]) -> list[list[Any]]:
    ranked = sorted(matches, key=lambda item: item.distance)
    pools: list[list[Any]] = []
    seen_sizes: set[int] = set()
    for distance_limit in (36, 44, 52, 60, 72):
        pool = [item for item in ranked if item.distance <= distance_limit]
        if len(pool) >= 6 and len(pool) not in seen_sizes:
            pools.append(pool)
            seen_sizes.add(len(pool))
    for size in (12, 24, 48, 96, len(ranked)):
        selected_size = min(size, len(ranked))
        if selected_size >= 6 and selected_size not in seen_sizes:
            pools.append(ranked[:selected_size])
            seen_sizes.add(selected_size)
    return pools


def _coverage_ratio(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3 or width <= 0 or height <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(abs(cv2.contourArea(hull)) / max(1.0, float(width * height)))


def _pose_from_matches(
    *,
    payload: CameraLocalizationRequest,
    frame_width: int,
    frame_height: int,
    keypoints: list[Any],
    matches: list[Any],
    landmark_points: np.ndarray,
) -> dict[str, Any] | None:
    """Find a stable pose from ranked descriptor matches and unknown webcam FOV."""
    best: dict[str, Any] | None = None
    all_object_points = np.asarray([landmark_points[item.trainIdx] for item in matches], dtype=np.float32)
    all_image_points = np.asarray([keypoints[item.queryIdx].pt for item in matches], dtype=np.float32)
    ranked_pools = _ranked_match_pools(matches)

    for camera_matrix, intrinsics_source, fov_degrees in _camera_matrix_candidates(payload, frame_width, frame_height):
        for pool in ranked_pools:
            object_points = np.asarray([landmark_points[item.trainIdx] for item in pool], dtype=np.float32)
            image_points = np.asarray([keypoints[item.queryIdx].pt for item in pool], dtype=np.float32)
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
                iterationsCount=2_000,
                reprojectionError=8.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_AP3P,
            )
            if not ok:
                continue

            projected, _ = cv2.projectPoints(all_object_points, rvec, tvec, camera_matrix, None)
            residuals = np.linalg.norm(projected.reshape(-1, 2) - all_image_points, axis=1)
            inlier_indices = np.flatnonzero(residuals <= 8.0)
            if len(inlier_indices) >= 6:
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        all_object_points[inlier_indices],
                        all_image_points[inlier_indices],
                        camera_matrix,
                        np.zeros((4, 1), dtype=np.float64),
                        rvec,
                        tvec,
                    )
                    projected, _ = cv2.projectPoints(all_object_points, rvec, tvec, camera_matrix, None)
                    residuals = np.linalg.norm(projected.reshape(-1, 2) - all_image_points, axis=1)
                    inlier_indices = np.flatnonzero(residuals <= 8.0)
                except cv2.error:
                    pass
            if len(inlier_indices) < 6:
                continue

            inlier_object_points = all_object_points[inlier_indices]
            inlier_image_points = all_image_points[inlier_indices]
            world_to_cv, _ = cv2.Rodrigues(rvec)
            camera_space = (world_to_cv @ inlier_object_points.T + tvec.reshape(3, 1)).T
            positive_depth_ratio = float(np.mean(camera_space[:, 2] > 0.05))
            if positive_depth_ratio < 0.9:
                continue

            mean_error = float(np.mean(residuals[inlier_indices]))
            coverage = _coverage_ratio(inlier_image_points, frame_width, frame_height)
            world_spread = float(np.linalg.norm(np.ptp(inlier_object_points, axis=0)))
            candidate = {
                "rvec": rvec,
                "tvec": tvec,
                "camera_matrix": camera_matrix,
                "intrinsics_source": intrinsics_source,
                "fov_degrees": fov_degrees,
                "inlier_indices": inlier_indices,
                "inlier_count": int(len(inlier_indices)),
                "pool_size": len(pool),
                "mean_error": mean_error,
                "coverage_ratio": coverage,
                "world_spread_m": world_spread,
                "positive_depth_ratio": positive_depth_ratio,
            }
            score = (candidate["inlier_count"], candidate["coverage_ratio"], -candidate["mean_error"])
            if best is None or score > best["score"]:
                candidate["score"] = score
                best = candidate
    return best


def _camera_to_world(rvec: np.ndarray, tvec: np.ndarray) -> list[list[float]]:
    world_to_cv, _ = cv2.Rodrigues(rvec)
    cv_to_world = world_to_cv.T
    camera_center = -(cv_to_world @ tvec.reshape(3, 1)).reshape(3)
    cv_from_arkit = np.diag([1.0, -1.0, -1.0])
    rotation = cv_to_world @ cv_from_arkit
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = camera_center
    return [[round(float(value), 8) for value in row] for row in matrix]


def _poses_agree(first: np.ndarray, second: np.ndarray) -> bool:
    translation_delta = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_delta_degrees = math.degrees(math.acos(cosine))
    return translation_delta <= 0.35 and rotation_delta_degrees <= 10.0


def _localization_candidate(frame_index: int, view_id: str, matches: list[Any], pose: dict[str, Any]) -> dict[str, Any]:
    rvec = pose["rvec"]
    tvec = pose["tvec"]
    camera_matrix = pose["camera_matrix"]
    error = float(pose["mean_error"])
    inlier_count = int(pose["inlier_count"])
    match_count = int(len(matches))
    pool_size = max(inlier_count, int(pose["pool_size"]))
    pool_ratio = inlier_count / max(1, pool_size)
    global_ratio = inlier_count / max(1, match_count)
    coverage = float(pose["coverage_ratio"])
    world_spread = float(pose["world_spread_m"])
    confidence = min(
        0.99,
        max(
            0.0,
            0.35 * min(1.0, inlier_count / 12.0)
            + 0.25 * min(1.0, global_ratio / 0.10)
            + 0.20 * max(0.0, 1.0 - error / 8.0)
            + 0.20 * min(1.0, coverage / 0.05),
        ),
    )
    return {
        "status": "needs_rescan",
        "coordinate_frame": "roomplan-local",
        "camera_to_world": None,
        "confidence": round(confidence, 6),
        "inlier_count": inlier_count,
        "match_count": match_count,
        "reprojection_error_px": round(error, 6),
        "intrinsics_source": pose["intrinsics_source"],
        "intrinsics": [[round(float(value), 8) for value in row] for row in camera_matrix],
        "diagnostics": {
            "frame_index": frame_index,
            "landmark_view_id": view_id,
            "inlier_ratio": round(pool_ratio, 6),
            "global_inlier_ratio": round(global_ratio, 6),
            "selected_match_pool_size": pool_size,
            "selected_fov_degrees": pose["fov_degrees"],
            "image_coverage_ratio": round(coverage, 6),
            "world_spread_m": round(world_spread, 6),
            "positive_depth_ratio": round(float(pose["positive_depth_ratio"]), 6),
            "raw_frames_persisted": False,
        },
        "_pose_matrix": np.asarray(_camera_to_world(rvec, tvec), dtype=np.float64),
    }


def localize_camera(payload: CameraLocalizationRequest) -> dict[str, Any]:
    landmark_points, landmark_descriptors, landmark_view_ids = _landmark_arrays(payload)
    orb = cv2.ORB_create(nfeatures=2600, scaleFactor=1.2, nlevels=8, fastThreshold=7, edgeThreshold=19)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    candidates: list[dict[str, Any]] = []
    best_match_count = 0
    best_feature_count = 0
    best_min_distance: float | None = None
    view_groups: dict[str, list[int]] = {}
    for index, view_id in enumerate(landmark_view_ids):
        view_groups.setdefault(view_id, []).append(index)
    grouped_landmarks = [
        (view_id, np.asarray(indices, dtype=np.int32))
        for view_id, indices in sorted(view_groups.items())
        if len(indices) >= 6
    ]

    for frame_index, frame in enumerate(payload.frames):
        jpeg = _jpeg_bytes(frame.frame_base64)
        try:
            image = decode_jpeg(jpeg, frame.width, frame.height)
        except VisionInferenceError as exc:
            raise LocalizationInputError(str(exc)) from exc
        gray = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 6:
            continue
        best_feature_count = max(best_feature_count, len(keypoints))
        for view_id, landmark_indices in grouped_landmarks:
            matches = _descriptor_matches(descriptors, landmark_descriptors, landmark_indices, matcher)
            best_match_count = max(best_match_count, len(matches))
            if matches:
                frame_min_distance = float(min(match.distance for match in matches))
                best_min_distance = frame_min_distance if best_min_distance is None else min(best_min_distance, frame_min_distance)
            if len(matches) < 6:
                continue
            pose = _pose_from_matches(
                payload=payload,
                frame_width=frame.width,
                frame_height=frame.height,
                keypoints=keypoints,
                matches=matches,
                landmark_points=landmark_points,
            )
            if pose is not None:
                candidates.append(_localization_candidate(frame_index, view_id, matches, pose))

    if not candidates:
        fallback_matrix, _ = _camera_matrix(payload, payload.frames[0].width, payload.frames[0].height)
        return {
            "status": "needs_rescan",
            "coordinate_frame": "roomplan-local",
            "camera_to_world": None,
            "confidence": 0.0,
            "inlier_count": 0,
            "match_count": best_match_count,
            "reprojection_error_px": None,
            "intrinsics_source": "provided" if payload.intrinsics is not None else "estimated-fov",
            "intrinsics": [[round(float(value), 8) for value in row] for row in fallback_matrix],
            "diagnostics": {
                "reason": "pnp_no_geometric_consensus" if best_match_count >= 6 else "insufficient_feature_matches",
                "best_feature_count": best_feature_count,
                "best_match_count": best_match_count,
                "best_min_descriptor_distance": best_min_distance,
                "raw_frames_persisted": False,
            },
        }
    for candidate in candidates:
        candidate["_consensus_frame_count"] = len(
            {
                int(other["diagnostics"]["frame_index"])
                for other in candidates
                if _poses_agree(candidate["_pose_matrix"], other["_pose_matrix"])
            }
        )
    best = max(
        candidates,
        key=lambda item: (
            item["_consensus_frame_count"],
            item["inlier_count"],
            item["confidence"],
            -item["reprojection_error_px"],
        ),
    )
    diagnostics = best["diagnostics"]
    consensus_frame_count = int(best.pop("_consensus_frame_count"))
    pose_matrix = best.pop("_pose_matrix")
    diagnostics["pose_candidate_frame_count"] = len({int(item["diagnostics"]["frame_index"]) for item in candidates})
    diagnostics["consensus_frame_count"] = consensus_frame_count
    strong_single_frame = (
        best["inlier_count"] >= 10
        and diagnostics["global_inlier_ratio"] >= 0.06
        and best["reprojection_error_px"] <= 5.0
        and diagnostics["image_coverage_ratio"] >= 0.01
        and diagnostics["world_spread_m"] >= 0.50
    )
    repeated_pose = (
        consensus_frame_count >= 2
        and best["inlier_count"] >= 6
        and diagnostics["global_inlier_ratio"] >= 0.05
        and best["reprojection_error_px"] <= 6.5
        and diagnostics["image_coverage_ratio"] >= 0.005
        and diagnostics["world_spread_m"] >= 0.30
    )
    if strong_single_frame or repeated_pose:
        best["status"] = "positioned"
        best["camera_to_world"] = [[round(float(value), 8) for value in row] for row in pose_matrix]
        if repeated_pose:
            best["confidence"] = round(max(float(best["confidence"]), min(0.95, 0.62 + 0.05 * consensus_frame_count)), 6)
    for candidate in candidates:
        candidate.pop("_pose_matrix", None)
        candidate.pop("_consensus_frame_count", None)
    return best
