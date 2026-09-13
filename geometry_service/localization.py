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


def build_visual_landmarks(payload: VisualLandmarkBuildRequest) -> dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    candidates: list[tuple[np.ndarray, bytes, float]] = []
    frame_feature_counts: list[int] = []
    for frame in payload.frames:
        jpeg = _jpeg_bytes(frame.frame_base64)
        try:
            image = decode_jpeg(jpeg, frame.width, frame.height)
        except VisionInferenceError as exc:
            raise LocalizationInputError(str(exc)) from exc
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or not keypoints:
            frame_feature_counts.append(0)
            continue
        depth = _depth_array(frame.depth_base64, frame.depth_width, frame.depth_height)
        intrinsics = _matrix(frame.intrinsics.values, (3, 3))
        camera_to_world = _matrix(frame.camera_to_world.values, (4, 4))
        accepted = 0
        for keypoint, descriptor in zip(keypoints, descriptors):
            depth_m = _depth_at(depth, keypoint.pt[0], keypoint.pt[1], frame.width, frame.height)
            if depth_m is None:
                continue
            point = _world_point(keypoint.pt[0], keypoint.pt[1], depth_m, intrinsics, camera_to_world)
            candidates.append((point, bytes(descriptor.tolist()), float(keypoint.response)))
            accepted += 1
        frame_feature_counts.append(accepted)

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
                "reason": "insufficient_depth_features",
                "landmark_count": len(selected),
                "frame_feature_counts": frame_feature_counts,
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
            "raw_frames_persisted": False,
        },
    }


def _landmark_arrays(payload: CameraLocalizationRequest) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    descriptors: list[np.ndarray] = []
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
    return np.asarray(points, dtype=np.float32), np.vstack(descriptors).astype(np.uint8)


def _camera_matrix(payload: CameraLocalizationRequest, width: int, height: int) -> tuple[np.ndarray, str]:
    if payload.intrinsics is not None:
        matrix = _matrix(payload.intrinsics.values, (3, 3))
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise LocalizationInputError("camera intrinsics must contain positive focal lengths")
        return matrix, "provided"
    focal = 0.5 * width / math.tan(math.radians(payload.fov_degrees) / 2.0)
    return np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64), "estimated-fov"


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


def localize_camera(payload: CameraLocalizationRequest) -> dict[str, Any]:
    landmark_points, landmark_descriptors = _landmark_arrays(payload)
    orb = cv2.ORB_create(nfeatures=1400, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best: dict[str, Any] | None = None

    for frame_index, frame in enumerate(payload.frames):
        jpeg = _jpeg_bytes(frame.frame_base64)
        try:
            image = decode_jpeg(jpeg, frame.width, frame.height)
        except VisionInferenceError as exc:
            raise LocalizationInputError(str(exc)) from exc
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 6:
            continue
        pairs = matcher.knnMatch(descriptors, landmark_descriptors, k=2)
        good = [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < 0.72 * second.distance]
        # One landmark can only support one 2D correspondence in a frame.
        unique: dict[int, Any] = {}
        for match in sorted(good, key=lambda item: item.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        if len(matches) < 6:
            continue
        object_points = np.asarray([landmark_points[item.trainIdx] for item in matches], dtype=np.float32)
        image_points = np.asarray([keypoints[item.queryIdx].pt for item in matches], dtype=np.float32)
        camera_matrix, intrinsics_source = _camera_matrix(payload, frame.width, frame.height)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            iterationsCount=250,
            reprojectionError=6.0,
            confidence=0.995,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 6:
            continue
        inlier_indices = inliers.reshape(-1)
        projected, _ = cv2.projectPoints(object_points[inlier_indices], rvec, tvec, camera_matrix, None)
        projected = projected.reshape(-1, 2)
        error = float(np.mean(np.linalg.norm(projected - image_points[inlier_indices], axis=1)))
        inlier_count = int(len(inlier_indices))
        match_count = int(len(matches))
        ratio = inlier_count / max(1, match_count)
        confidence = min(0.99, max(0.0, 0.45 * ratio + 0.35 * min(1.0, inlier_count / 24.0) + 0.20 * max(0.0, 1.0 - error / 8.0)))
        candidate = {
            "status": "positioned" if inlier_count >= 8 and ratio >= 0.35 and error <= 6.0 and confidence >= 0.55 else "needs_rescan",
            "coordinate_frame": "roomplan-local",
            "camera_to_world": _camera_to_world(rvec, tvec) if confidence >= 0.55 else None,
            "confidence": round(confidence, 6),
            "inlier_count": inlier_count,
            "match_count": match_count,
            "reprojection_error_px": round(error, 6),
            "intrinsics_source": intrinsics_source,
            "intrinsics": [[round(float(value), 8) for value in row] for row in camera_matrix],
            "diagnostics": {"frame_index": frame_index, "inlier_ratio": round(ratio, 6), "raw_frames_persisted": False},
        }
        if best is None or (candidate["inlier_count"], candidate["confidence"]) > (best["inlier_count"], best["confidence"]):
            best = candidate

    if best is None:
        return {
            "status": "needs_rescan",
            "coordinate_frame": "roomplan-local",
            "camera_to_world": None,
            "confidence": 0.0,
            "inlier_count": 0,
            "match_count": 0,
            "reprojection_error_px": None,
            "intrinsics_source": "provided" if payload.intrinsics is not None else "estimated-fov",
            "intrinsics": payload.intrinsics.values if payload.intrinsics is not None else None,
            "diagnostics": {"reason": "insufficient_feature_matches", "raw_frames_persisted": False},
        }
    return best
