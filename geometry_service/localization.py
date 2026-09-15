"""RoomPlan visual landmarks and fixed-camera 6-DoF localization.

The scan side turns RGB + LiDAR depth + ARKit poses into ORB descriptors tied
to metric RoomPlan coordinates.  The fixed camera side matches ORB features
against those derived landmarks and estimates a real camera pose with
solvePnPRansac. Raw RGB/depth bytes are never returned or persisted here.
"""

from __future__ import annotations

import base64
import binascii
from itertools import combinations, permutations, product
import math
from typing import Any

import cv2
import numpy as np

from .contracts import CameraLocalizationRequest, VisualLandmarkBuildRequest
from .learned_matcher import LearnedMatcher, get_or_fit_matcher
from .real_vision import VisionInferenceError, decode_jpeg


_HAMMING_POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


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
    *,
    query_responses: np.ndarray | None = None,
    landmark_responses: np.ndarray | None = None,
    learned_matcher: LearnedMatcher | None = None,
) -> list[Any]:
    if len(landmark_indices) < 2:
        return []
    subset = landmark_descriptors[landmark_indices]
    candidate_count = min(6, len(subset)) if learned_matcher is not None else 2
    pairs = matcher.knnMatch(query_descriptors, subset, k=candidate_count)
    reverse = matcher.match(subset, query_descriptors)
    reverse_best = {match.queryIdx: match.trainIdx for match in reverse}
    good: list[Any] = []
    learned_candidates: list[tuple[list[Any], int]] = []

    def append_match(selected: Any) -> None:
        good.append(
            cv2.DMatch(
                _queryIdx=selected.queryIdx,
                _trainIdx=int(landmark_indices[selected.trainIdx]),
                _distance=selected.distance,
            )
        )

    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair[0], pair[1]
        baseline_accept = (
            first.distance <= 72
            and first.distance < 0.84 * second.distance
            and reverse_best.get(first.trainIdx) == first.queryIdx
        )
        if baseline_accept:
            append_match(first)
        elif learned_matcher is not None:
            learned_candidates.append((pair, int(first.queryIdx)))

    # Torch inference is batched across all ambiguous query features. Calling
    # the tiny network once per keypoint made the first live request slower
    # than the geometric search it was meant to assist.
    if learned_matcher is not None and learned_candidates:
        flat_query_indices: list[int] = []
        flat_landmark_indices: list[int] = []
        slices: list[tuple[list[Any], int, int]] = []
        start = 0
        for pair, query_index in learned_candidates:
            flat_query_indices.extend([query_index] * len(pair))
            flat_landmark_indices.extend(int(candidate.trainIdx) for candidate in pair)
            end = start + len(pair)
            slices.append((pair, start, end))
            start = end
        flat_query = np.asarray(flat_query_indices, dtype=np.int32)
        flat_landmark = np.asarray(flat_landmark_indices, dtype=np.int32)
        scores = learned_matcher.score_pairs(
            query_descriptors[flat_query],
            landmark_descriptors[landmark_indices[flat_landmark]],
            query_responses[flat_query] if query_responses is not None else None,
            landmark_responses[landmark_indices[flat_landmark]] if landmark_responses is not None else None,
        )
        for pair, start, end in slices:
            pair_scores = scores[start:end]
            best_candidate_index = int(np.argmax(pair_scores))
            selected = pair[best_candidate_index]
            selected_score = float(pair_scores[best_candidate_index])
            # The learned model is a rescue path for an ambiguous Hamming
            # shortlist. It cannot turn a weak descriptor into evidence by
            # itself: keep a bounded distance ceiling and require a meaningful
            # learned score.
            if selected.distance > 80 or selected_score < learned_matcher.threshold:
                continue
            if len(pair_scores) > 1:
                alternate_scores = np.delete(pair_scores, best_candidate_index)
                if selected_score < float(np.max(alternate_scores)) + 0.015 and selected.distance > 48:
                    continue
            alternate_distances = [
                float(candidate.distance)
                for index, candidate in enumerate(pair)
                if index != best_candidate_index
            ]
            nearest_alternate_distance = min(alternate_distances, default=float("inf"))
            ratio_ok = selected.distance < 0.84 * nearest_alternate_distance
            learned_override = selected_score >= 0.78 and selected.distance <= 64
            # Reverse Hamming uniqueness is retained as a cheap ambiguity
            # guard. The final multi-view PnP checks remain unchanged.
            if reverse_best.get(selected.trainIdx) == selected.queryIdx and (
                ratio_ok or learned_override
            ):
                append_match(selected)
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


def _camera_matrix_candidates(
    payload: CameraLocalizationRequest,
    width: int,
    height: int,
    *,
    preferred_fov_degrees: float | None = None,
) -> list[tuple[np.ndarray, str, float | None]]:
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
    if preferred_fov_degrees is not None and 30.0 <= preferred_fov_degrees <= 120.0:
        preferred = round(float(preferred_fov_degrees), 3)
        if preferred not in unique:
            unique.append(preferred)
        # The stable fixed-camera FOV is only a search prior. Trying it first
        # avoids repeating the full focal sweep for every scan view/frame;
        # fresh correspondences still have to reconstruct and validate the
        # pose below. Keep every other FOV as a fallback if this hint fails.
        unique = [preferred, *sorted((value for value in unique if value != preferred), key=lambda value: abs(value - preferred))]
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
    # RANSAC is the expensive part of localization.  The old search ran every
    # descriptor threshold plus every prefix size for every FOV/view/frame,
    # which made a six-frame localization exceed the API timeout.  Keep the
    # strictest usable descriptor pool, then a few progressively wider pools.
    for distance_limit in (36, 44, 52, 60, 72):
        pool = [item for item in ranked if item.distance <= distance_limit]
        if len(pool) >= 6 and len(pool) not in seen_sizes:
            pools.append(pool)
            seen_sizes.add(len(pool))
            break
    for size in (18, 36, 72, len(ranked)):
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
    preferred_fov_degrees: float | None = None,
) -> dict[str, Any] | None:
    """Find a stable pose from ranked descriptor matches and unknown webcam FOV."""
    best: dict[str, Any] | None = None
    all_object_points = np.asarray([landmark_points[item.trainIdx] for item in matches], dtype=np.float32)
    all_image_points = np.asarray([keypoints[item.queryIdx].pt for item in matches], dtype=np.float32)
    ranked_pools = _ranked_match_pools(matches)

    for camera_matrix, intrinsics_source, fov_degrees in _camera_matrix_candidates(
        payload,
        frame_width,
        frame_height,
        preferred_fov_degrees=preferred_fov_degrees,
    ):
        for pool in ranked_pools:
            object_points = np.asarray([landmark_points[item.trainIdx] for item in pool], dtype=np.float32)
            image_points = np.asarray([keypoints[item.queryIdx].pt for item in pool], dtype=np.float32)
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
                iterationsCount=1_200,
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
            # Once RANSAC has found the same number of inliers, prefer the
            # camera model that actually reprojects them more tightly.  The
            # previous coverage-first ordering could select a much wider or
            # narrower FOV simply because its six points happened to span a
            # larger image area.
            score = (candidate["inlier_count"], -candidate["mean_error"], candidate["coverage_ratio"])
            if best is None or score > best["score"]:
                candidate["score"] = score
                best = candidate
        # A stable historical FOV can steer search order, but it cannot make a
        # pose valid. Once that FOV itself yields a usable fresh PnP seed, stop
        # spending time on alternate focal lengths and let the normal guided,
        # scene, and independent-view checks decide whether it is publishable.
        if (
            preferred_fov_degrees is not None
            and fov_degrees is not None
            and abs(float(fov_degrees) - float(preferred_fov_degrees)) <= 0.5
            and best is not None
            and best["fov_degrees"] is not None
            and abs(float(best["fov_degrees"]) - float(preferred_fov_degrees)) <= 0.5
            and best["inlier_count"] >= 6
            and best["mean_error"] <= 6.5
        ):
            return best
    return best


def _pose_guided_matches(
    *,
    frame_width: int,
    frame_height: int,
    keypoints: list[Any],
    descriptors: np.ndarray,
    landmark_points: np.ndarray,
    landmark_descriptors: np.ndarray,
    pose: dict[str, Any],
    radius_px: float = 36.0,
) -> list[Any]:
    """Expand a plausible PnP seed with spatially gated descriptor matches.

    Descriptor-only matching is deliberately conservative, but a physically
    plausible seed gives us extra information: a RoomPlan landmark should
    reproject close to the corresponding webcam feature.  Use that geometric
    gate to look across *all* scan views, while retaining mutual descriptor
    uniqueness so repeated room textures cannot simply confirm the seed.
    """
    if descriptors is None or len(keypoints) < 2 or len(landmark_points) < 2:
        return []
    camera_matrix = pose["camera_matrix"]
    rvec = pose["rvec"]
    tvec = pose["tvec"]
    projected, _ = cv2.projectPoints(landmark_points, rvec, tvec, camera_matrix, None)
    projected = projected.reshape(-1, 2)
    world_to_cv, _ = cv2.Rodrigues(rvec)
    camera_space = (world_to_cv @ landmark_points.T + tvec.reshape(3, 1)).T

    cell_size = max(8.0, radius_px)
    cell_radius = max(1, int(math.ceil(radius_px / cell_size)))
    grid: dict[tuple[int, int], list[int]] = {}
    keypoint_xy = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    for query_index, (x, y) in enumerate(keypoint_xy):
        grid.setdefault((int(x // cell_size), int(y // cell_size)), []).append(query_index)

    # landmark -> (query, descriptor distance, pixel distance)
    landmark_best: dict[int, tuple[int, float, float]] = {}
    for landmark_index, ((x, y), camera_point) in enumerate(zip(projected, camera_space)):
        if (
            camera_point[2] <= 0.05
            or x < -radius_px
            or y < -radius_px
            or x > frame_width + radius_px
            or y > frame_height + radius_px
        ):
            continue
        cell_x, cell_y = int(x // cell_size), int(y // cell_size)
        candidate_queries: list[int] = []
        for offset_x in range(-cell_radius, cell_radius + 1):
            for offset_y in range(-cell_radius, cell_radius + 1):
                candidate_queries.extend(grid.get((cell_x + offset_x, cell_y + offset_y), []))
        if not candidate_queries:
            continue

        # The old implementation called cv2.norm once per landmark/keypoint
        # pair from Python. A real scan can contain thousands of landmarks, so
        # the repeated Python/C++ boundary dominated the 45-second API budget.
        # Score each landmark's local candidates in one NumPy batch while
        # preserving the exact spatial gate and Hamming thresholds.
        query_indices = np.asarray(candidate_queries, dtype=np.int32)
        pixel_deltas = keypoint_xy[query_indices] - np.asarray([x, y], dtype=np.float32)
        pixel_distances = np.linalg.norm(pixel_deltas, axis=1)
        inside = pixel_distances <= radius_px
        if not np.any(inside):
            continue
        query_indices = query_indices[inside]
        pixel_distances = pixel_distances[inside]
        xor = np.bitwise_xor(descriptors[query_indices], landmark_descriptors[landmark_index])
        descriptor_distances = _HAMMING_POPCOUNT[xor].sum(axis=1)
        descriptor_ok = descriptor_distances <= 56
        if not np.any(descriptor_ok):
            continue
        query_indices = query_indices[descriptor_ok]
        pixel_distances = pixel_distances[descriptor_ok]
        descriptor_distances = descriptor_distances[descriptor_ok]
        order = np.lexsort((pixel_distances, descriptor_distances))
        best_index = int(order[0])
        best_distance = float(descriptor_distances[best_index])
        best_pixel_distance = float(pixel_distances[best_index])
        best_query = int(query_indices[best_index])
        if len(order) == 1:
            if best_distance > 40.0:
                continue
        else:
            second_distance = float(descriptor_distances[int(order[1])])
            if best_distance >= 0.86 * second_distance and best_distance > 32.0:
                continue
        landmark_best[landmark_index] = (best_query, best_distance, best_pixel_distance)

    # Enforce the reverse uniqueness as well: each image feature must prefer
    # this landmark over other projected landmarks in the same local region.
    by_query: dict[int, list[tuple[float, float, int]]] = {}
    for landmark_index, (query_index, descriptor_distance, pixel_distance) in landmark_best.items():
        by_query.setdefault(query_index, []).append((descriptor_distance, pixel_distance, landmark_index))

    result: list[Any] = []
    for query_index, options in by_query.items():
        options.sort(key=lambda item: (item[0], item[1]))
        best_distance, best_pixel_distance, landmark_index = options[0]
        if len(options) > 1 and best_distance >= 0.86 * options[1][0] and best_distance > 32.0:
            continue
        if landmark_best.get(landmark_index, (None,))[0] != query_index:
            continue
        result.append(
            cv2.DMatch(
                _queryIdx=query_index,
                _trainIdx=landmark_index,
                _distance=best_distance + 0.05 * best_pixel_distance,
            )
        )
    return sorted(result, key=lambda item: item.distance)


def _refine_pose_with_guided_matches(
    *,
    frame_width: int,
    frame_height: int,
    keypoints: list[Any],
    descriptors: np.ndarray,
    landmark_points: np.ndarray,
    landmark_descriptors: np.ndarray,
    pose: dict[str, Any],
    matches: list[Any] | None = None,
) -> tuple[dict[str, Any], list[Any]] | None:
    if matches is None:
        matches = _pose_guided_matches(
            frame_width=frame_width,
            frame_height=frame_height,
            keypoints=keypoints,
            descriptors=descriptors,
            landmark_points=landmark_points,
            landmark_descriptors=landmark_descriptors,
            pose=pose,
        )
    if len(matches) < 8:
        return None
    object_points = np.asarray([landmark_points[item.trainIdx] for item in matches], dtype=np.float32)
    image_points = np.asarray([keypoints[item.queryIdx].pt for item in matches], dtype=np.float32)
    camera_matrix = pose["camera_matrix"]

    def try_refine(use_extrinsic_guess: bool) -> dict[str, Any] | None:
        rvec = np.asarray(pose["rvec"], dtype=np.float64).copy() if use_extrinsic_guess else np.zeros((3, 1), dtype=np.float64)
        tvec = np.asarray(pose["tvec"], dtype=np.float64).copy() if use_extrinsic_guess else np.zeros((3, 1), dtype=np.float64)
        try:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=use_extrinsic_guess,
                iterationsCount=500,
                # This is only the RANSAC capture gate. A moved fixed camera
                # can start from a physically plausible per-view seed that is
                # several pixels off while still providing good cross-view
                # correspondences. Let RANSAC enter that basin, then keep the
                # real acceptance below at the existing <=6 px residual after
                # refinement.
                reprojectionError=8.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
        if not ok:
            return None
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        inlier_indices = np.flatnonzero(residuals <= 6.0)
        if len(inlier_indices) < 8:
            return None
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
                rvec,
                tvec,
            )
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
            residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            inlier_indices = np.flatnonzero(residuals <= 6.0)
        except cv2.error:
            pass
        if len(inlier_indices) < 8:
            return None

        inlier_object_points = object_points[inlier_indices]
        inlier_image_points = image_points[inlier_indices]
        world_to_cv, _ = cv2.Rodrigues(rvec)
        camera_space = (world_to_cv @ inlier_object_points.T + tvec.reshape(3, 1)).T
        positive_depth_ratio = float(np.mean(camera_space[:, 2] > 0.05))
        if positive_depth_ratio < 0.9:
            return None
        return {
            **pose,
            "rvec": rvec,
            "tvec": tvec,
            "inlier_indices": inlier_indices,
            "inlier_count": int(len(inlier_indices)),
            "pool_size": len(matches),
            "mean_error": float(np.mean(residuals[inlier_indices])),
            "coverage_ratio": _coverage_ratio(inlier_image_points, frame_width, frame_height),
            "world_spread_m": float(np.linalg.norm(np.ptp(inlier_object_points, axis=0))),
            "positive_depth_ratio": positive_depth_ratio,
            "guided_match_count": len(matches),
        }

    # A RoomPlan center probe supplies a useful pose guess when its coarse
    # direction alignment is close. Repeated textures can also make that
    # rotation a poor local minimum, so run one unconstrained RANSAC solve and
    # let the same strict residual and scene gates choose between the two.
    refined_candidates = [try_refine(True), try_refine(False)]
    refined_candidates = [candidate for candidate in refined_candidates if candidate is not None]
    if not refined_candidates:
        return None
    return (
        max(
            refined_candidates,
            key=lambda candidate: (
                int(candidate["inlier_count"]),
                -float(candidate["mean_error"]),
                float(candidate["coverage_ratio"]),
            ),
        ),
        matches,
    )


def _provisional_pose_from_guided_matches(
    *,
    frame_width: int,
    frame_height: int,
    keypoints: list[Any],
    landmark_points: np.ndarray,
    pose: dict[str, Any],
    matches: list[Any],
) -> dict[str, Any] | None:
    """Use 6-7 guided matches only to improve the next search projection.

    This intentionally has a lower floor than the real guided refinement, but
    its result is never returned as a localization candidate. It exists only
    to reproject all landmarks more accurately so a subsequent fresh matching
    pass can reach the ordinary eight-match refinement threshold.
    """
    if len(matches) < 6:
        return None
    object_points = np.asarray([landmark_points[item.trainIdx] for item in matches], dtype=np.float32)
    image_points = np.asarray([keypoints[item.queryIdx].pt for item in matches], dtype=np.float32)
    camera_matrix = np.asarray(pose["camera_matrix"], dtype=np.float64)
    rvec = np.asarray(pose["rvec"], dtype=np.float64).copy()
    tvec = np.asarray(pose["tvec"], dtype=np.float64).copy()
    try:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            iterationsCount=400,
            reprojectionError=8.0,
            confidence=0.995,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    inlier_indices = np.flatnonzero(residuals <= 8.0)
    if len(inlier_indices) < 6:
        return None
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_indices],
            image_points[inlier_indices],
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    inlier_object_points = object_points[inlier_indices]
    inlier_image_points = image_points[inlier_indices]
    world_to_cv, _ = cv2.Rodrigues(rvec)
    camera_space = (world_to_cv @ inlier_object_points.T + tvec.reshape(3, 1)).T
    if float(np.mean(camera_space[:, 2] > 0.05)) < 0.9:
        return None
    return {
        **pose,
        "rvec": rvec,
        "tvec": tvec,
        "inlier_indices": inlier_indices,
        "inlier_count": int(len(inlier_indices)),
        "pool_size": len(matches),
        "mean_error": float(np.mean(residuals[inlier_indices])),
        "coverage_ratio": _coverage_ratio(inlier_image_points, frame_width, frame_height),
        "world_spread_m": float(np.linalg.norm(np.ptp(inlier_object_points, axis=0))),
        "positive_depth_ratio": float(np.mean(camera_space[:, 2] > 0.05)),
        "search_prior_provisional_refinement": True,
    }


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


def _pose_with_camera_center(pose: dict[str, Any], camera_center: np.ndarray) -> dict[str, Any]:
    """Keep a seed rotation/intrinsics while moving only its search center."""
    world_to_cv, _ = cv2.Rodrigues(np.asarray(pose["rvec"], dtype=np.float64))
    center = np.asarray(camera_center, dtype=np.float64).reshape(3, 1)
    return {
        **pose,
        "tvec": -(world_to_cv @ center),
    }


def _rotation_from_direction_pairs(world_directions: np.ndarray, camera_directions: np.ndarray) -> np.ndarray | None:
    """Return the world-to-camera rotation that best aligns unit directions."""
    if len(world_directions) < 2 or len(camera_directions) != len(world_directions):
        return None
    covariance = np.asarray(world_directions, dtype=np.float64).T @ np.asarray(camera_directions, dtype=np.float64)
    try:
        u, _singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    if not np.isfinite(rotation).all() or np.linalg.det(rotation) < 0.9:
        return None
    return rotation


def _pose_from_fixed_center_matches(
    *,
    frame_width: int,
    frame_height: int,
    keypoints: list[Any],
    matches: list[Any],
    landmark_points: np.ndarray,
    camera_center: np.ndarray,
    fov_degrees: float,
    random_seed: int = 0,
    iteration_limit: int = 320,
) -> dict[str, Any] | None:
    """Estimate only rotation around a search-prior center from fresh matches.

    The supplied center is never accepted as localization evidence. It is used
    solely to turn fresh 2D/3D descriptor correspondences into a stable search
    orientation. The caller must still run the ordinary unconstrained guided
    PnP refinement before a pose can be published.
    """
    if len(matches) < 6 or not 30.0 <= float(fov_degrees) <= 120.0:
        return None
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    ranked = sorted(matches, key=lambda item: item.distance)
    # Keep the RANSAC pool descriptor-strong. A true view normally contributes
    # only a small minority of the raw ORB correspondences, so sampling the
    # full noisy tail needlessly lowers the chance of a clean three-match set.
    ransac_matches = ranked[: min(36, len(ranked))]
    object_points = np.asarray([landmark_points[item.trainIdx] for item in ransac_matches], dtype=np.float64)
    image_points = np.asarray([keypoints[item.queryIdx].pt for item in ransac_matches], dtype=np.float64)
    world_vectors = object_points - center.reshape(1, 3)
    world_norms = np.linalg.norm(world_vectors, axis=1)
    valid = world_norms > 0.10
    if int(np.count_nonzero(valid)) < 6:
        return None
    ransac_matches = [item for item, keep in zip(ransac_matches, valid) if bool(keep)]
    object_points = object_points[valid]
    image_points = image_points[valid]
    world_directions = world_vectors[valid] / world_norms[valid, None]

    focal = 0.5 * frame_width / math.tan(math.radians(float(fov_degrees)) / 2.0)
    camera_matrix = np.asarray(
        [[focal, 0.0, frame_width / 2.0], [0.0, focal, frame_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera_directions = np.column_stack(
        [
            (image_points[:, 0] - frame_width / 2.0) / focal,
            (image_points[:, 1] - frame_height / 2.0) / focal,
            np.ones(len(image_points), dtype=np.float64),
        ]
    )
    camera_directions /= np.linalg.norm(camera_directions, axis=1, keepdims=True)

    rng = np.random.default_rng(random_seed)
    # RoomPlan-center probes intentionally tolerate a little more angular
    # uncertainty than the ordinary fixed-center prior. Their only purpose is
    # to seed the later descriptor-gated PnP solve, which still uses the
    # strict six-pixel reprojection gate before a candidate can matter.
    angular_cosine = math.cos(math.radians(6.0 if iteration_limit < 120 else 3.5))
    best_indices: np.ndarray | None = None
    best_rotation: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    # Three-direction samples have a high success probability once descriptor
    # matching has reached this path. Hundreds of trials are sufficient and
    # avoid monopolising the single local worker for every scan-view seed.
    requested_iterations = max(40, int(iteration_limit))
    # Keep the normal search's historical minimum, while allowing the bounded
    # RoomPlan-center probes to request a genuinely smaller budget. The old
    # `max(120, ...)` floor silently turned their 72-iteration budget back into
    # 120 trials per center.
    minimum_iterations = 40 if requested_iterations < 120 else 120
    iterations = min(requested_iterations, max(minimum_iterations, len(ransac_matches) * 10))
    for _ in range(iterations):
        sample = rng.choice(len(ransac_matches), size=3, replace=False)
        rotation = _rotation_from_direction_pairs(world_directions[sample], camera_directions[sample])
        if rotation is None:
            continue
        predicted = (rotation @ world_directions.T).T
        cosine = np.sum(predicted * camera_directions, axis=1)
        indices = np.flatnonzero((cosine >= angular_cosine) & (predicted[:, 2] > 0.05))
        if len(indices) < 6:
            continue
        mean_cosine = float(np.mean(cosine[indices]))
        score = (int(len(indices)), mean_cosine)
        if best_score is None or score > best_score:
            best_score = score
            best_indices = indices
            best_rotation = rotation

    if best_indices is None or best_rotation is None:
        return None
    refined_rotation = _rotation_from_direction_pairs(world_directions[best_indices], camera_directions[best_indices])
    if refined_rotation is not None:
        best_rotation = refined_rotation
    predicted = (best_rotation @ world_directions.T).T
    cosine = np.sum(predicted * camera_directions, axis=1)
    best_indices = np.flatnonzero((cosine >= angular_cosine) & (predicted[:, 2] > 0.05))
    if len(best_indices) < 6:
        return None

    rvec, _ = cv2.Rodrigues(best_rotation)
    tvec = -(best_rotation @ center.reshape(3, 1))
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    # The angular gate is intentionally looser than final PnP. Retain the
    # aligned matches for diagnostics/search quality, then let guided PnP apply
    # its existing six-pixel acceptance threshold to fresh correspondences.
    inlier_image_points = image_points[best_indices]
    inlier_object_points = object_points[best_indices]
    return {
        "rvec": rvec,
        "tvec": tvec,
        "camera_matrix": camera_matrix,
        "intrinsics_source": "search-prior-fov",
        "fov_degrees": float(fov_degrees),
        "inlier_indices": best_indices,
        "inlier_count": int(len(best_indices)),
        "pool_size": len(ransac_matches),
        "mean_error": float(np.mean(residuals[best_indices])),
        "coverage_ratio": _coverage_ratio(inlier_image_points.astype(np.float32), frame_width, frame_height),
        "world_spread_m": float(np.linalg.norm(np.ptp(inlier_object_points, axis=0))),
        "positive_depth_ratio": 1.0,
        "fixed_center_rotation_search": True,
    }


_OBJECT_LABEL_GROUPS: dict[str, set[str]] = {
    "bed": {"bed"},
    "chair": {"chair"},
    "table": {"table", "desk", "dining table"},
    "storage": {"storage", "cabinet", "shelf", "bookcase", "wardrobe", "dresser", "nightstand"},
    "sofa": {"sofa", "couch"},
}


def _object_label_group(value: str) -> str | None:
    normalized = " ".join(str(value).strip().lower().replace("_", " ").split())
    for group, aliases in _OBJECT_LABEL_GROUPS.items():
        if normalized in aliases:
            return group
    return None


def _box_iou(first: list[float], second: list[float]) -> float:
    first_x1, first_y1, first_x2, first_y2 = (float(value) for value in first)
    second_x1, second_y1, second_x2, second_y2 = (float(value) for value in second)
    intersection = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1)) * max(
        0.0, min(first_y2, second_y2) - max(first_y1, second_y1)
    )
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-9 else 0.0


def _room_object_corners(room_object: Any) -> np.ndarray:
    """Return the eight native RoomPlan cuboid corners in world coordinates.

    RoomPlan object dimensions are expressed in the object's local axes.  The
    normalized scan also carries the full transform, so preserve its rotation
    instead of treating beds, desks, and storage as axis-aligned boxes.  The
    explicit normalized center remains the translation authority because older
    scans can contain transforms normalized through a different serialization
    path.
    """
    center = np.asarray(
        [room_object.center.x, room_object.center.y, room_object.center.z],
        dtype=np.float64,
    )
    dimensions = np.asarray(
        [room_object.dimensions.x, room_object.dimensions.y, room_object.dimensions.z],
        dtype=np.float64,
    )
    half = dimensions / 2.0
    local = np.asarray(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    rotation = np.eye(3, dtype=np.float64)
    transform = getattr(room_object, "transform", None)
    if transform is not None:
        try:
            raw_rotation = np.asarray(transform.values, dtype=np.float64)[:3, :3]
            # RoomPlan transforms should be rigid, but orthonormalize the
            # rotation so small serialization/float drift cannot stretch the
            # cuboid during projection.
            u, _singular_values, vt = np.linalg.svd(raw_rotation)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
        except (TypeError, ValueError, np.linalg.LinAlgError):
            rotation = np.eye(3, dtype=np.float64)
    return local @ rotation.T + center


def _project_room_object_bbox(
    room_object: Any,
    *,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[list[float], float] | None:
    corners = _room_object_corners(room_object)
    rotation, _ = cv2.Rodrigues(rvec)
    camera_space = (rotation @ corners.T + tvec.reshape(3, 1)).T
    positive = camera_space[:, 2] > 0.08
    positive_ratio = float(np.mean(positive))
    if int(np.count_nonzero(positive)) < 6:
        return None
    projected, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, None)
    pixels = projected.reshape(-1, 2)[positive]
    if not np.isfinite(pixels).all():
        return None
    x1 = max(0.0, min(float(frame_width), float(np.min(pixels[:, 0]))))
    y1 = max(0.0, min(float(frame_height), float(np.min(pixels[:, 1]))))
    x2 = max(0.0, min(float(frame_width), float(np.max(pixels[:, 0]))))
    y2 = max(0.0, min(float(frame_height), float(np.max(pixels[:, 1]))))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return [x1, y1, x2, y2], positive_ratio


def _semantic_cuboid_score(
    payload: CameraLocalizationRequest,
    *,
    frame_index: int,
    assignment: list[tuple[int, int]],
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    """Score a semantic pose by projecting rotated RoomPlan cuboids.

    Detector boxes remain coarse observations; the score is intentionally used
    as an initializer/verifier signal rather than pretending that a 2D box is a
    precise metric feature correspondence.
    """
    matches: list[dict[str, Any]] = []
    groups: set[str] = set()
    for object_index, detection_index in assignment:
        room_object = payload.room_objects[object_index]
        detection = payload.object_detections[detection_index]
        if detection.frame_index != frame_index:
            continue
        projected = _project_room_object_bbox(
            room_object,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if projected is None:
            continue
        projected_box, positive_depth_ratio = projected
        observed_box = [
            max(0.0, min(float(frame_width), float(detection.bbox[0]))),
            max(0.0, min(float(frame_height), float(detection.bbox[1]))),
            max(0.0, min(float(frame_width), float(detection.bbox[2]))),
            max(0.0, min(float(frame_height), float(detection.bbox[3]))),
        ]
        observed_area = max(0.0, observed_box[2] - observed_box[0]) * max(0.0, observed_box[3] - observed_box[1])
        projected_area = max(0.0, projected_box[2] - projected_box[0]) * max(0.0, projected_box[3] - projected_box[1])
        if observed_area <= 1.0 or projected_area <= 1.0:
            continue
        iou = _box_iou(projected_box, observed_box)
        observed_center = np.asarray(
            [(observed_box[0] + observed_box[2]) / 2.0, (observed_box[1] + observed_box[3]) / 2.0],
            dtype=np.float64,
        )
        projected_center = np.asarray(
            [(projected_box[0] + projected_box[2]) / 2.0, (projected_box[1] + projected_box[3]) / 2.0],
            dtype=np.float64,
        )
        normalized_center_distance = float(
            np.linalg.norm(
                (projected_center - observed_center)
                / np.asarray([max(1.0, frame_width), max(1.0, frame_height)], dtype=np.float64)
            )
        )
        center_score = max(0.0, 1.0 - normalized_center_distance / 0.30)
        area_ratio = min(observed_area, projected_area) / max(observed_area, projected_area)
        weight = max(0.10, float(detection.confidence) * float(room_object.confidence))
        component = weight * (iou + 0.20 * center_score + 0.10 * area_ratio)
        group = _object_label_group(room_object.label)
        if group is not None:
            groups.add(group)
        matches.append(
            {
                "object_id": room_object.id,
                "label": str(room_object.label),
                "detection_label": str(detection.label),
                "iou": float(iou),
                "center_score": float(center_score),
                "area_ratio": float(area_ratio),
                "positive_depth_ratio": float(positive_depth_ratio),
                "component": float(component),
            }
        )
    if not matches:
        return {
            "score": 0.0,
            "matched_object_count": 0,
            "semantic_group_count": 0,
            "mean_iou": 0.0,
            "minimum_iou": 0.0,
            "mean_center_score": 0.0,
            "mean_area_ratio": 0.0,
            "matches": [],
        }
    return {
        "score": round(float(sum(item["component"] for item in matches)), 6),
        "matched_object_count": len(matches),
        "semantic_group_count": len(groups),
        "mean_iou": round(float(np.mean([item["iou"] for item in matches])), 6),
        "minimum_iou": round(float(min(item["iou"] for item in matches)), 6),
        "mean_center_score": round(float(np.mean([item["center_score"] for item in matches])), 6),
        "mean_area_ratio": round(float(np.mean([item["area_ratio"] for item in matches])), 6),
        "matches": [
            {
                **item,
                "iou": round(float(item["iou"]), 6),
                "center_score": round(float(item["center_score"]), 6),
                "area_ratio": round(float(item["area_ratio"]), 6),
                "positive_depth_ratio": round(float(item["positive_depth_ratio"]), 6),
                "component": round(float(item["component"]), 6),
            }
            for item in matches
        ],
    }


def _semantic_cuboid_rank(score: dict[str, Any]) -> tuple[int, float, float, int, int, float, float]:
    """Rank semantic cuboid hypotheses by consistency before raw object count.

    A hypothesis with one nearly unrelated assignment can accumulate a larger
    summed score simply because it contains a third object.  Treat a usable
    minimum IoU as the first gate, then prefer the tighter overall projection
    before considering how many objects contributed.
    """
    minimum_iou = float(score.get("minimum_iou") or 0.0)
    matched_object_count = int(score.get("matched_object_count") or 0)
    semantic_group_count = int(score.get("semantic_group_count") or 0)
    raw_score = score.get("score")
    if raw_score is None:
        raw_score = score.get("cuboid_score")
    return (
        int(minimum_iou >= 0.20 and matched_object_count >= 2 and semantic_group_count >= 2),
        float(score.get("mean_iou") or 0.0),
        minimum_iou,
        semantic_group_count,
        matched_object_count,
        float(raw_score or 0.0),
        float(score.get("mean_center_score") or 0.0),
    )


def _semantic_pose_from_camera_to_world(
    camera_to_world: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    fov_degrees: float | None,
    seed: dict[str, Any],
) -> dict[str, Any]:
    world_to_cv = _world_to_cv(camera_to_world)
    rvec, _ = cv2.Rodrigues(world_to_cv[:3, :3])
    tvec = world_to_cv[:3, 3].reshape(3, 1)
    return {
        **seed,
        "rvec": rvec,
        "tvec": tvec,
        "camera_matrix": camera_matrix,
        "fov_degrees": fov_degrees,
    }


def _refine_semantic_cuboid_pose(
    payload: CameraLocalizationRequest,
    *,
    frame_index: int,
    frame_width: int,
    frame_height: int,
    seed: dict[str, Any],
) -> dict[str, Any]:
    assignment = list(seed.get("semantic_object_assignment") or [])
    if len(assignment) < 3:
        return seed
    if sum(payload.room_objects[object_index].transform is not None for object_index, _ in assignment) < 3:
        # Center-only legacy objects do not contain enough information to turn
        # detector rectangles into a rotated-cuboid objective. Keep their PnP
        # seed unchanged rather than optimizing against an invented alignment.
        return seed
    original_matrix = np.asarray(_camera_to_world(seed["rvec"], seed["tvec"]), dtype=np.float64)
    base_rotation = original_matrix[:3, :3].copy()
    base_center = original_matrix[:3, 3].copy()
    state = {
        "x": float(base_center[0]),
        "y": float(base_center[1]),
        "z": float(base_center[2]),
        "yaw": 0.0,
        "fov": float(seed["fov_degrees"]) if seed.get("fov_degrees") is not None else None,
    }

    def evaluated(candidate_state: dict[str, float | None]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        yaw_radians = math.radians(float(candidate_state["yaw"] or 0.0))
        c, s = math.cos(yaw_radians), math.sin(yaw_radians)
        yaw_rotation = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
        matrix = original_matrix.copy()
        matrix[:3, :3] = yaw_rotation @ base_rotation
        matrix[:3, 3] = [float(candidate_state["x"]), float(candidate_state["y"]), float(candidate_state["z"])]
        if float(np.linalg.norm(matrix[[0, 2], 3] - base_center[[0, 2]])) > 0.85:
            return None
        if abs(float(matrix[1, 3] - base_center[1])) > 0.35 or abs(float(candidate_state["yaw"] or 0.0)) > 18.0:
            return None
        scene_prior = _pose_scene_prior(matrix, payload)
        if not scene_prior.get("accepted"):
            return None
        candidate_fov = candidate_state["fov"]
        if payload.intrinsics is None and candidate_fov is not None:
            candidate_fov = float(candidate_fov)
            if not 30.0 <= candidate_fov <= 120.0:
                return None
            focal = 0.5 * frame_width / math.tan(math.radians(candidate_fov) / 2.0)
            camera_matrix = np.asarray(
                [[focal, 0.0, frame_width / 2.0], [0.0, focal, frame_height / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        else:
            camera_matrix = np.asarray(seed["camera_matrix"], dtype=np.float64)
        pose = _semantic_pose_from_camera_to_world(
            matrix,
            camera_matrix=camera_matrix,
            fov_degrees=None if candidate_fov is None else float(candidate_fov),
            seed=seed,
        )
        score = _semantic_cuboid_score(
            payload,
            frame_index=frame_index,
            assignment=assignment,
            rvec=pose["rvec"],
            tvec=pose["tvec"],
            camera_matrix=pose["camera_matrix"],
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return pose, score

    initial = evaluated(state)
    if initial is None:
        return seed
    best_pose, best_score = initial
    initial_score = dict(best_score)
    # Dependency-free bounded coordinate descent.  The first pass can move a
    # rough center-based semantic PnP seed substantially; later passes tighten
    # the placement without turning the verifier into a room-wide search.
    passes = (
        (0.30, 0.12, 8.0),
        (0.15, 0.06, 4.0),
        (0.075, 0.03, 2.0),
    )
    for horizontal_step, vertical_step, yaw_step in passes:
        for _sweep in range(3):
            sweep_improved = False
            # Translation and apparent object scale are coupled in perspective.
            # Probe the bounded X/Z neighborhood jointly before the one-axis
            # updates so a diagonal move can cross a shallow IoU valley.
            xz_options: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
            for x_direction in (-1.0, 0.0, 1.0):
                for z_direction in (-1.0, 0.0, 1.0):
                    if x_direction == 0.0 and z_direction == 0.0:
                        continue
                    trial = dict(state)
                    trial["x"] = float(state["x"] or 0.0) + x_direction * horizontal_step
                    trial["z"] = float(state["z"] or 0.0) + z_direction * horizontal_step
                    result = evaluated(trial)
                    if result is not None:
                        xz_options.append(
                            (
                                *result,
                                float(trial["x"] or 0.0),
                                float(trial["z"] or 0.0),
                            )
                        )
            if xz_options:
                candidate_pose, candidate_score, candidate_x, candidate_z = max(
                    xz_options,
                    key=lambda item: _semantic_cuboid_rank(item[1]),
                )
                best_key = _semantic_cuboid_rank(best_score)
                candidate_key = _semantic_cuboid_rank(candidate_score)
                if candidate_key > best_key:
                    state["x"] = candidate_x
                    state["z"] = candidate_z
                    best_pose = candidate_pose
                    best_score = candidate_score
                    sweep_improved = True
            for parameter, step in (
                ("x", horizontal_step),
                ("z", horizontal_step),
                ("y", vertical_step),
                ("yaw", yaw_step),
            ):
                current_value = float(state[parameter] or 0.0)
                options: list[tuple[dict[str, Any], dict[str, Any], float]] = []
                for direction in (-1.0, 1.0):
                    trial = dict(state)
                    trial[parameter] = current_value + direction * step
                    result = evaluated(trial)
                    if result is not None:
                        options.append((*result, float(trial[parameter] or 0.0)))
                if not options:
                    continue
                candidate_pose, candidate_score, candidate_value = max(
                    options,
                    key=lambda item: _semantic_cuboid_rank(item[1]),
                )
                best_key = _semantic_cuboid_rank(best_score)
                candidate_key = _semantic_cuboid_rank(candidate_score)
                if candidate_key > best_key:
                    state[parameter] = candidate_value
                    best_pose = candidate_pose
                    best_score = candidate_score
                    sweep_improved = True
            if not sweep_improved:
                break

    refined_matrix = np.asarray(_camera_to_world(best_pose["rvec"], best_pose["tvec"]), dtype=np.float64)
    best_pose["semantic_cuboid"] = {
        **best_score,
        "initial_score": float(initial_score["score"]),
        "initial_mean_iou": float(initial_score["mean_iou"]),
        "center_shift_m": round(float(np.linalg.norm(refined_matrix[:3, 3] - base_center)), 6),
        "yaw_shift_degrees": round(float(state["yaw"] or 0.0), 4),
        "fov_shift_degrees": (
            0.0
        ),
    }
    return best_pose


def _semantic_object_assignments(payload: CameraLocalizationRequest, frame_index: int) -> list[list[tuple[int, int]]]:
    """Return bounded map-object/detection assignments for one frame.

    YOLO-World may return the same furniture through two synonymous prompts
    (for example ``table`` and ``desk``). Collapse overlapping detections
    inside a semantic group before considering assignments, then enumerate only
    the small ambiguity introduced by repeated furniture of one category.
    """
    map_groups: dict[str, list[int]] = {}
    for object_index, room_object in enumerate(payload.room_objects):
        group = _object_label_group(room_object.label)
        if group is not None:
            map_groups.setdefault(group, []).append(object_index)

    detection_candidates: list[tuple[int, Any, str]] = []
    for detection_index, detection in enumerate(payload.object_detections):
        if detection.frame_index != frame_index or detection.confidence < 0.20:
            continue
        if len(detection.bbox) != 4 or not all(math.isfinite(float(value)) for value in detection.bbox):
            continue
        group = _object_label_group(detection.label)
        if group is not None:
            detection_candidates.append((detection_index, detection, group))

    grouped_detections: dict[str, list[tuple[int, Any]]] = {}
    for detection_index, detection, group in detection_candidates:
        grouped_detections.setdefault(group, []).append((detection_index, detection))
    for group, items in grouped_detections.items():
        ordered = sorted(items, key=lambda item: float(item[1].confidence), reverse=True)
        kept: list[tuple[int, Any]] = []
        for item in ordered:
            if all(_box_iou(item[1].bbox, previous[1].bbox) < 0.50 for previous in kept):
                kept.append(item)
            if len(kept) >= 4:
                break
        grouped_detections[group] = kept

    group_options: list[list[tuple[float, list[tuple[int, int]]]]] = []
    for group in sorted(set(map_groups) & set(grouped_detections)):
        object_indices = map_groups[group][:4]
        detection_items = grouped_detections[group]
        detection_indices = [item[0] for item in detection_items]
        detection_confidences = {item[0]: float(item[1].confidence) for item in detection_items}
        options: list[tuple[float, list[tuple[int, int]]]] = []
        for count in range(1, min(len(object_indices), len(detection_indices), 3) + 1):
            for selected_objects in combinations(object_indices, count):
                for selected_detections in permutations(detection_indices, count):
                    pairs = list(zip(selected_objects, selected_detections))
                    score = sum(detection_confidences[index] for index in selected_detections) + 0.05 * count
                    options.append((score, pairs))
        options.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        group_options.append(options[:24])

    assignments: list[tuple[float, list[tuple[int, int]]]] = [(0.0, [])]
    for options in group_options:
        expanded: list[tuple[float, list[tuple[int, int]]]] = []
        for base_score, base_pairs in assignments:
            for option_score, option_pairs in options:
                expanded.append((base_score + option_score, [*base_pairs, *option_pairs]))
        expanded.sort(key=lambda item: (len(item[1]), item[0]), reverse=True)
        assignments = expanded[:96]

    valid: list[tuple[float, list[tuple[int, int]]]] = []
    for score, pairs in assignments:
        if len(pairs) < 3 or len({
            _object_label_group(payload.room_objects[object_index].label)
            for object_index, _detection_index in pairs
        }) < 2:
            continue
        valid.append((score, pairs))
    valid.sort(key=lambda item: (len(item[1]), item[0]), reverse=True)
    # Semantic detections are only pose initializers. Keep the best bounded
    # assignments so ambiguous synonym prompts cannot multiply the expensive
    # PnP seed stage into a second localization search.
    return [pairs for _score, pairs in valid[:8]]


def _semantic_object_pose_seeds(
    payload: CameraLocalizationRequest,
    *,
    frame_index: int,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Estimate bounded pose seeds from recognized native RoomPlan objects."""
    assignments = _semantic_object_assignments(payload, frame_index)
    if not assignments:
        return []

    fov_hypotheses = _camera_matrix_candidates(
        payload,
        frame_width,
        frame_height,
        preferred_fov_degrees=(payload.search_prior.fov_degrees if payload.search_prior else None),
    )
    if payload.intrinsics is None:
        preferred = payload.search_prior.fov_degrees if payload.search_prior else None
        requested = float(payload.fov_degrees)
        fov_hypotheses = [
            item
            for item in fov_hypotheses
            if item[2] is not None
            and (
                abs(float(item[2]) - requested) <= 0.5
                or abs(float(item[2]) - 60.0) <= 0.5
                or abs(float(item[2]) - 74.0) <= 0.5
                or abs(float(item[2]) - 96.0) <= 0.5
                or (preferred is not None and abs(float(item[2]) - float(preferred)) <= 0.5)
            )
        ]

    seeds: list[dict[str, Any]] = []
    for assignment in assignments:
        for anchor_kind in ("center", "floor-contact"):
            object_points: list[list[float]] = []
            image_points: list[list[float]] = []
            confidences: list[float] = []
            labels: list[str] = []
            for object_index, detection_index in assignment:
                room_object = payload.room_objects[object_index]
                detection = payload.object_detections[detection_index]
                center = np.asarray(
                    [room_object.center.x, room_object.center.y, room_object.center.z],
                    dtype=np.float64,
                )
                dimensions_y = max(0.0, float(room_object.dimensions.y))
                x1, y1, x2, y2 = (float(value) for value in detection.bbox)
                if anchor_kind == "floor-contact":
                    center[1] -= dimensions_y / 2.0
                    image = [(x1 + x2) / 2.0, y2]
                else:
                    image = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                object_points.append(center.tolist())
                image_points.append(image)
                confidences.append(float(detection.confidence) * float(room_object.confidence))
                labels.append(str(room_object.label))

            if len(object_points) < 3:
                continue
            object_array = np.asarray(object_points, dtype=np.float64)
            image_array = np.asarray(image_points, dtype=np.float64)
            if not np.isfinite(object_array).all() or not np.isfinite(image_array).all():
                continue
            for camera_matrix, _intrinsics_source, fov_degrees in fov_hypotheses:
                if len(object_array) == 3:
                    # OpenCV's solvePnPRansac wrapper rejects three points even
                    # though the SQPnP solver supports the minimal case. Keep
                    # every returned solution as a seed and let the RoomPlan
                    # scene prior plus fresh ORB refinement reject ambiguity.
                    try:
                        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                            object_array,
                            image_array,
                            camera_matrix,
                            np.zeros((4, 1), dtype=np.float64),
                            flags=cv2.SOLVEPNP_SQPNP,
                        )
                    except cv2.error:
                        continue
                    if not ok:
                        continue
                    pose_solutions = list(zip(rvecs, tvecs))
                else:
                    try:
                        ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
                            object_array,
                            image_array,
                            camera_matrix,
                            np.zeros((4, 1), dtype=np.float64),
                            iterationsCount=260,
                            reprojectionError=36.0,
                            confidence=0.995,
                            flags=cv2.SOLVEPNP_EPNP,
                        )
                    except cv2.error:
                        continue
                    if not ok:
                        continue
                    pose_solutions = [(rvec, tvec)]

                for rvec, tvec in pose_solutions:
                    projected, _ = cv2.projectPoints(object_array, rvec, tvec, camera_matrix, None)
                    residuals = np.linalg.norm(projected.reshape(-1, 2) - image_array, axis=1)
                    inlier_indices = np.flatnonzero(residuals <= 36.0)
                    if len(inlier_indices) < 3:
                        continue
                    world_to_cv, _ = cv2.Rodrigues(rvec)
                    camera_space = (world_to_cv @ object_array.T + tvec.reshape(3, 1)).T
                    positive_depth_ratio = float(np.mean(camera_space[inlier_indices, 2] > 0.05))
                    # Three rough detector boxes produce a minimal SQPnP seed;
                    # one center can fall just behind the minimal solution
                    # while the remaining two still give a useful direction.
                    # This is still only an initializer and cannot bypass the
                    # fresh ORB/PnP acceptance gates below.
                    minimum_positive_depth = 2.0 / 3.0 if len(object_array) == 3 else 0.75
                    if positive_depth_ratio < minimum_positive_depth:
                        continue
                    pose_matrix = np.asarray(_camera_to_world(rvec, tvec), dtype=np.float64)
                    scene_prior = _pose_scene_prior(pose_matrix, payload)
                    if not scene_prior.get("accepted"):
                        continue
                    mean_error = float(np.mean(residuals[inlier_indices]))
                    seeds.append(
                        {
                            "rvec": rvec,
                            "tvec": tvec,
                            "camera_matrix": camera_matrix,
                            "intrinsics_source": "semantic-object-seed",
                            "fov_degrees": fov_degrees,
                            "inlier_indices": inlier_indices,
                            "inlier_count": int(len(inlier_indices)),
                            "pool_size": len(object_array),
                            "mean_error": mean_error,
                            "coverage_ratio": _coverage_ratio(image_array[inlier_indices].astype(np.float32), frame_width, frame_height),
                            "world_spread_m": float(np.linalg.norm(np.ptp(object_array[inlier_indices], axis=0))),
                            "positive_depth_ratio": positive_depth_ratio,
                            "semantic_object_seed": True,
                            "semantic_object_match_count": len(assignment),
                            "semantic_object_labels": sorted(set(labels)),
                            "semantic_object_anchor": anchor_kind,
                            "semantic_object_confidence": round(float(np.mean(confidences)), 6),
                            "semantic_object_assignment": list(assignment),
                        }
                    )

    seeds = [
        _refine_semantic_cuboid_pose(
            payload,
            frame_index=frame_index,
            frame_width=frame_width,
            frame_height=frame_height,
            seed=seed,
        )
        for seed in seeds
    ]
    seeds.sort(
        key=lambda item: (
            _semantic_cuboid_rank(item.get("semantic_cuboid") or {}),
            item["inlier_count"],
            -item["mean_error"],
            item["semantic_object_confidence"],
        ),
        reverse=True,
    )
    unique: list[dict[str, Any]] = []
    for seed in seeds:
        matrix = np.asarray(_camera_to_world(seed["rvec"], seed["tvec"]), dtype=np.float64)
        if all(float(np.linalg.norm(matrix[:3, 3] - np.asarray(_camera_to_world(existing["rvec"], existing["tvec"]), dtype=np.float64)[:3, 3])) >= 0.20 for existing in unique):
            unique.append(seed)
        if len(unique) >= 6:
            break
    return unique


def _poses_agree(first: np.ndarray, second: np.ndarray) -> bool:
    translation_delta = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_delta_degrees = math.degrees(math.acos(cosine))
    return translation_delta <= 0.35 and rotation_delta_degrees <= 10.0


def _segment_distance(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> float:
    dx, dz = bx - ax, bz - az
    length_sq = dx * dx + dz * dz
    if length_sq <= 1e-12:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length_sq))
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))


def _polygon_distance(px: float, pz: float, polygon: list[tuple[float, float]]) -> float:
    inside = False
    previous = polygon[-1]
    minimum = float("inf")
    for current in polygon:
        ax, az = previous
        bx, bz = current
        minimum = min(minimum, _segment_distance(px, pz, ax, az, bx, bz))
        if (az > pz) != (bz > pz):
            crossing_x = (bx - ax) * (pz - az) / (bz - az) + ax
            if px < crossing_x:
                inside = not inside
        previous = current
    return 0.0 if inside else minimum


def _room_perimeter_search_centers(
    payload: CameraLocalizationRequest,
    *,
    camera_y: float,
    spacing_m: float = 0.55,
    inset_m: float = 0.25,
    max_centers: int = 32,
) -> list[np.ndarray]:
    """Sample bounded, RoomPlan-derived centers for a moved fixed camera.

    Fixed household cameras are commonly placed near a wall.  When a fresh PnP
    seed has a physically plausible height but an ambiguous X/Z translation,
    sample the room perimeter at that *fresh* height at two useful inset depths
    toward the room centroid. Fixed laptop cameras are often 25-60 cm in from
    a wall on a desk, so a single shallow inset can miss the correct basin.
    These centers are search initializations
    only; the caller must still recover an unconstrained pose from fresh image
    correspondences before the result can be published.
    """
    if not payload.room_zones or max_centers <= 0:
        return []
    spacing = max(0.35, float(spacing_m))
    inset = max(0.0, min(float(inset_m), 0.45))
    raw: list[np.ndarray] = []
    for zone in payload.room_zones:
        polygon = np.asarray([(float(point.x), float(point.z)) for point in zone.polygon], dtype=np.float64)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        centroid = np.mean(polygon, axis=0)
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            edge = end - start
            length = float(np.linalg.norm(edge))
            if length <= 1e-6:
                continue
            steps = max(1, int(math.ceil(length / spacing)))
            for step in range(steps):
                # Mid-cell samples cover the whole perimeter without
                # over-weighting shared polygon vertices.
                t = (step + 0.5) / steps
                boundary = start + t * edge
                toward_center = centroid - boundary
                toward_norm = float(np.linalg.norm(toward_center))
                inset_values = (inset, max(inset, 0.55))
                for inset_value in inset_values:
                    xz = boundary if toward_norm <= 1e-6 else boundary + (toward_center / toward_norm) * min(inset_value, toward_norm * 0.75)
                    raw.append(np.asarray([xz[0], float(camera_y), xz[1]], dtype=np.float64))
        raw.append(np.asarray([centroid[0], float(camera_y), centroid[1]], dtype=np.float64))

    if not raw:
        return []
    # Deduplicate close samples across adjacent/overlapping RoomPlan zones.
    unique: list[np.ndarray] = []
    for center in raw:
        if all(float(np.linalg.norm(center[[0, 2]] - existing[[0, 2]])) >= 0.18 for existing in unique):
            unique.append(center)
    if len(unique) <= max_centers:
        return unique
    # Preserve coverage over the full perimeter rather than truncating one side
    # of the room when a complex polygon produces many samples.
    indices = np.linspace(0, len(unique) - 1, max_centers, dtype=np.int32)
    return [unique[int(index)] for index in indices]


def _room_search_camera_y_values(
    payload: CameraLocalizationRequest,
    seed_camera_y: float,
) -> list[float]:
    """Return bounded generic camera heights for a moved-camera search.

    A weak monocular PnP seed can preserve a plausible height while choosing a
    repeated-texture translation basin, and its recovered height can also be a
    low-height mirror solution. Keep the fresh seed as one hypothesis, then
    probe common fixed-camera heights measured from the RoomPlan floor. These
    are search initializers only; the final pose still has to be recovered from
    image correspondences and pass the normal scene checks.
    """
    values: list[float] = []
    floor_values = [float(zone.floor_y) for zone in payload.room_zones if math.isfinite(float(zone.floor_y))]
    if floor_values:
        floor_y = float(np.median(np.asarray(floor_values, dtype=np.float64)))
        seed_height = float(seed_camera_y) - floor_y
        if math.isfinite(seed_height) and 0.15 <= seed_height <= 3.50:
            values.append(float(seed_camera_y))
        values.extend(floor_y + height for height in (0.75, 1.05, 1.35, 1.65))
    elif math.isfinite(float(seed_camera_y)):
        values.append(float(seed_camera_y))
    result: list[float] = []
    for value in values:
        if not math.isfinite(value):
            continue
        if all(abs(value - existing) >= 0.12 for existing in result):
            result.append(value)
    return result


def _pose_scene_prior(matrix: np.ndarray, payload: CameraLocalizationRequest) -> dict[str, Any]:
    """Score a PnP hypothesis against physical facts already known by RoomPlan."""
    camera_up_alignment = float(matrix[1, 1])
    upright = camera_up_alignment >= math.cos(math.radians(55.0))
    if not payload.room_zones:
        return {
            "accepted": upright,
            "reason": "upright" if upright else "camera_not_upright",
            "camera_up_alignment": round(camera_up_alignment, 6),
        }

    x, y, z = (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))
    zone_candidates: list[tuple[float, float, str | None]] = []
    for zone in payload.room_zones:
        polygon = [(float(point.x), float(point.z)) for point in zone.polygon]
        if len(polygon) < 3:
            continue
        zone_candidates.append(
            (
                _polygon_distance(x, z, polygon),
                y - float(zone.floor_y),
                zone.id,
            )
        )
    if not zone_candidates:
        return {
            "accepted": upright,
            "reason": "upright" if upright else "camera_not_upright",
            "camera_up_alignment": round(camera_up_alignment, 6),
        }
    distance, height, zone_id = min(zone_candidates, key=lambda item: item[0])
    in_room = distance <= 0.75 and 0.15 <= height <= 3.50
    accepted = upright and in_room
    if not upright:
        reason = "camera_not_upright"
    elif not in_room:
        reason = "outside_roomplan_bounds"
    else:
        reason = "physically_plausible"
    return {
        "accepted": accepted,
        "reason": reason,
        "zone_id": zone_id,
        "camera_height_above_floor_m": round(height, 4),
        "distance_from_room_polygon_m": round(distance, 4),
        "camera_up_alignment": round(camera_up_alignment, 6),
    }


def _guided_person_calibration(payload: CameraLocalizationRequest) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Use a person standing on four known floor points as calibration markers."""
    if not payload.person_anchors:
        return None, {"status": "not_requested"}

    grouped: dict[tuple[float, float, float], list[int]] = {}
    for anchor in payload.person_anchors:
        if anchor.frame_index < len(payload.frames):
            point = (round(float(anchor.x), 4), round(float(anchor.y), 4), round(float(anchor.z), 4))
            grouped.setdefault(point, []).append(int(anchor.frame_index))
    if len(grouped) < 4:
        return None, {"status": "insufficient_targets", "target_count": len(grouped), "required_target_count": 4}

    targets = list(grouped.items())[:4]
    frame0 = payload.frames[targets[0][1][0]]
    camera_matrix, intrinsics_source = _camera_matrix(payload, frame0.width, frame0.height)
    object_points = np.asarray([point for point, _ in targets], dtype=np.float64)
    floor_points = object_points[:, [0, 2]]
    floor_spread = max((float(np.linalg.norm(a - b)) for a, b in combinations(floor_points, 2)), default=0.0)
    if floor_spread < 0.90:
        return None, {"status": "targets_too_close", "target_count": 4, "floor_spread_m": round(floor_spread, 4)}

    choices: list[list[dict[str, float]]] = []
    detection_count = 0
    for _point, frame_indices in targets:
        candidates: list[dict[str, float]] = []
        for frame_index in frame_indices:
            frame = payload.frames[frame_index]
            for detection in payload.object_detections:
                if detection.frame_index != frame_index or detection.label.strip().lower() != "person" or detection.confidence < 0.20:
                    continue
                x1, y1, x2, y2 = (float(value) for value in detection.bbox)
                if x2 <= x1 or y2 <= y1:
                    continue
                detection_count += 1
                candidates.append({
                    "x": ((x1 + x2) * 0.5) / float(frame.width),
                    "y": (y2 - max(1.0, (y2 - y1) * 0.015)) / float(frame.height),
                    "confidence": float(detection.confidence),
                })
        if not candidates:
            return None, {"status": "person_not_found", "resolved_target_count": len(choices), "person_detection_count": detection_count}
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        distinct: list[dict[str, float]] = []
        for candidate in candidates:
            if all(math.hypot(candidate["x"] - prior["x"], candidate["y"] - prior["y"]) > 0.08 for prior in distinct):
                distinct.append(candidate)
            if len(distinct) == 2:
                break
        choices.append(distinct)

    best: dict[str, Any] | None = None
    assignment_count = 0
    for assignment in product(*choices):
        assignment_count += 1
        image_points = np.asarray(
            [[item["x"] * frame0.width, item["y"] * frame0.height] for item in assignment],
            dtype=np.float64,
        )
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(object_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        if not ok:
            continue
        for rvec, tvec in zip(rvecs, tvecs):
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
            residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            matrix = np.asarray(_camera_to_world(rvec, tvec), dtype=np.float64)
            scene_prior = _pose_scene_prior(matrix, payload)
            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
            camera_points = (rotation @ object_points.T + np.asarray(tvec, dtype=np.float64).reshape(3, 1)).T
            positive_depth_ratio = float(np.mean(camera_points[:, 2] > 0.05))
            candidate = {
                "matrix": matrix,
                "scene_prior": scene_prior,
                "mean_error": float(np.mean(residuals)),
                "max_error": float(np.max(residuals)),
                "positive_depth_ratio": positive_depth_ratio,
                "person_confidence": float(np.mean([item["confidence"] for item in assignment])),
            }
            rank = (int(bool(scene_prior.get("accepted"))), positive_depth_ratio, -candidate["mean_error"], -candidate["max_error"], candidate["person_confidence"])
            if best is None or rank > best["rank"]:
                best = {**candidate, "rank": rank}

    if best is None:
        return None, {"status": "solve_failed", "assignment_count": assignment_count, "person_detection_count": detection_count}

    accepted = bool(best["scene_prior"].get("accepted")) and best["positive_depth_ratio"] >= 1.0 and best["mean_error"] <= 45.0 and best["max_error"] <= 95.0
    diagnostics = {
        "status": "accepted" if accepted else "rejected",
        "target_count": 4,
        "person_detection_count": detection_count,
        "assignment_count": assignment_count,
        "floor_spread_m": round(floor_spread, 4),
        "mean_reprojection_error_px": round(best["mean_error"], 4),
        "max_reprojection_error_px": round(best["max_error"], 4),
        "mean_person_confidence": round(best["person_confidence"], 4),
        "scene_prior": best["scene_prior"],
    }
    if not accepted:
        return None, diagnostics

    matrix = best["matrix"]
    confidence = max(0.55, min(0.97, 0.88 - best["mean_error"] / 220.0 + 0.10 * best["person_confidence"]))
    return {
        "status": "positioned",
        "coordinate_frame": "roomplan-local",
        "camera_to_world": [[round(float(value), 8) for value in row] for row in matrix],
        "confidence": round(confidence, 6),
        "inlier_count": 4,
        "match_count": 4,
        "reprojection_error_px": round(best["mean_error"], 6),
        "intrinsics_source": intrinsics_source,
        "intrinsics": [[round(float(value), 8) for value in row] for row in camera_matrix],
        "diagnostics": {
            "selected_camera_center": [round(float(matrix[index, 3]), 4) for index in range(3)],
            "selected_estimate_source": "guided-person-floor",
            "guided_person_calibration": diagnostics,
            "raw_frames_persisted": False,
        },
    }, diagnostics


def _localization_candidate(
    frame_index: int,
    view_id: str,
    matches: list[Any],
    pose: dict[str, Any],
    landmark_view_ids: list[str],
) -> dict[str, Any]:
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
    inlier_landmark_view_ids = sorted(
        {
            landmark_view_ids[matches[int(index)].trainIdx]
            for index in pose["inlier_indices"]
            if 0 <= int(index) < len(matches) and 0 <= matches[int(index)].trainIdx < len(landmark_view_ids)
        }
    )
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
            "inlier_landmark_view_ids": inlier_landmark_view_ids,
            "inlier_landmark_view_count": len(inlier_landmark_view_ids),
            "inlier_ratio": round(pool_ratio, 6),
            "global_inlier_ratio": round(global_ratio, 6),
            "selected_match_pool_size": pool_size,
            "selected_fov_degrees": pose["fov_degrees"],
            "image_coverage_ratio": round(coverage, 6),
            "world_spread_m": round(world_spread, 6),
            "positive_depth_ratio": round(float(pose["positive_depth_ratio"]), 6),
            "pose_guided_refinement": bool(pose.get("guided_match_count")),
            "pose_guided_match_count": int(pose.get("guided_match_count") or 0),
            "pose_guided_attempt_match_count": int(pose.get("guided_attempt_match_count") or 0),
            "pose_guided_attempt_view_count": int(pose.get("guided_attempt_view_count") or 0),
            "pose_guided_attempt_view_ids": list(pose.get("guided_attempt_view_ids") or []),
            "pose_guided_attempt_radius_px": float(pose.get("guided_attempt_radius_px") or 0.0),
            "search_prior_distance_m": pose.get("search_prior_distance_m"),
            "search_prior_aligned": bool(pose.get("search_prior_aligned")),
            "search_prior_view_match": bool(pose.get("search_prior_view_match")),
            "search_prior_center_attempted": bool(pose.get("search_prior_center_attempted")),
            "search_prior_center_guided": bool(pose.get("search_prior_center_guided")),
            "search_prior_recovery": bool(pose.get("search_prior_recovery")),
            "seed_center_rotation_recovery": bool(pose.get("seed_center_rotation_recovery")),
            "room_center_search_attempted": bool(pose.get("room_center_search_attempted")),
            "room_center_search_center_count": int(pose.get("room_center_search_center_count") or 0),
            "room_center_search_hypothesis_count": int(pose.get("room_center_search_hypothesis_count") or 0),
            "room_center_search_hypothesis_stats": list(pose.get("room_center_search_hypothesis_stats") or []),
            "room_center_search_recovery": bool(pose.get("room_center_search_recovery")),
            "fixed_center_rotation_search": bool(pose.get("fixed_center_rotation_search")),
            "raw_frames_persisted": False,
        },
        "_pose_matrix": np.asarray(_camera_to_world(rvec, tvec), dtype=np.float64),
    }


def _candidate_scan_views(candidate: dict[str, Any]) -> set[str]:
    diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
    # Consensus is agreement between independently solved RoomPlan viewpoints.
    # An `all-views` solve can contain landmarks originating from many scan
    # views, but those points still belong to one PnP hypothesis and must not
    # masquerade as several agreeing hypotheses.  Its direct cross-view support
    # is tracked separately by `inlier_landmark_view_count`.
    fallback = diagnostics.get("landmark_view_id")
    if isinstance(fallback, str) and fallback and fallback != "all-views":
        return {fallback}
    return set()


def _candidate_fov(candidate: dict[str, Any]) -> float | None:
    diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
    value = diagnostics.get("selected_fov_degrees")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _fov_hypotheses_agree(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_fov = _candidate_fov(first)
    second_fov = _candidate_fov(second)
    if first_fov is None or second_fov is None:
        return True
    return abs(first_fov - second_fov) <= 8.0


def _consensus_stats(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    pose_agreeing = [
        other
        for other in candidates
        if _poses_agree(candidate["_pose_matrix"], other["_pose_matrix"])
    ]
    agreeing = [other for other in pose_agreeing if _fov_hypotheses_agree(candidate, other)]
    scan_views: set[str] = set()
    for other in agreeing:
        scan_views.update(_candidate_scan_views(other))
    fovs = [value for value in (_candidate_fov(other) for other in agreeing) if value is not None]
    return {
        "frame_count": len({int(other["diagnostics"]["frame_index"]) for other in agreeing}),
        "scan_view_count": len(scan_views),
        "scan_views": sorted(scan_views),
        "pose_only_frame_count": len({int(other["diagnostics"]["frame_index"]) for other in pose_agreeing}),
        "fov_span_degrees": round(max(fovs) - min(fovs), 3) if fovs else 0.0,
    }


def _candidate_is_positioned(candidate: dict[str, Any]) -> bool:
    diagnostics = candidate["diagnostics"]
    if candidate.get("_scene_plausible") is False:
        return False
    scan_view_count = int(candidate.get("_consensus_scan_view_count") or 0)
    frame_count = int(candidate.get("_consensus_frame_count") or 0)
    direct_scan_view_count = int(diagnostics.get("inlier_landmark_view_count") or 0)

    geometric_quality = (
        candidate["inlier_count"] >= 6
        and diagnostics["global_inlier_ratio"] >= 0.05
        and candidate["reprojection_error_px"] <= 4.5
        and diagnostics["image_coverage_ratio"] >= 0.01
        and diagnostics["world_spread_m"] >= 0.50
    )
    if not geometric_quality:
        return False

    # Browser camera intrinsics are unknown.  Repeated frames from the same
    # fixed webcam are therefore not independent evidence: they can repeat the
    # same wrong PnP/FOV hypothesis.  Require the pose to be supported by more
    # than one RoomPlan scan viewpoint.  A single all-landmark solve needs a
    # stronger ten-inlier floor; clustered per-view solves can use two camera
    # frames or three independent scan views.
    if direct_scan_view_count >= 2 and candidate["inlier_count"] >= 10:
        return True
    if scan_view_count >= 3:
        return True
    return scan_view_count >= 2 and frame_count >= 2


def localize_camera(payload: CameraLocalizationRequest) -> dict[str, Any]:
    guided_person_result, guided_person_diagnostics = _guided_person_calibration(payload)
    if guided_person_result is not None:
        return guided_person_result

    landmark_points, landmark_descriptors, landmark_view_ids = _landmark_arrays(payload)
    landmark_responses = np.asarray(
        [max(0.0, float(landmark.response)) for landmark in payload.landmarks],
        dtype=np.float32,
    )
    learned_matcher = get_or_fit_matcher(
        landmark_points,
        landmark_descriptors,
        landmark_responses,
        landmark_view_ids,
    )
    learned_matcher_diagnostics = (
        {"status": "ready", **learned_matcher.diagnostics}
        if learned_matcher is not None
        else {"status": "unavailable", "reason": "insufficient_training_pairs_or_torch_unavailable"}
    )
    orb = cv2.ORB_create(nfeatures=2600, scaleFactor=1.2, nlevels=8, fastThreshold=7, edgeThreshold=19)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    candidates: list[dict[str, Any]] = []
    semantic_cuboid_candidates: list[dict[str, Any]] = []
    best_match_count = 0
    best_feature_count = 0
    best_min_distance: float | None = None
    person_masked_frame_count = 0
    person_masked_area_ratios: list[float] = []
    view_groups: dict[str, list[int]] = {}
    for index, view_id in enumerate(landmark_view_ids):
        view_groups.setdefault(view_id, []).append(index)
    grouped_landmarks = [
        (view_id, np.asarray(indices, dtype=np.int32))
        for view_id, indices in sorted(view_groups.items())
        if len(indices) >= 6
    ]
    all_landmark_indices = np.arange(len(landmark_points), dtype=np.int32)

    for frame_index, frame in enumerate(payload.frames):
        jpeg = _jpeg_bytes(frame.frame_base64)
        try:
            image = decode_jpeg(jpeg, frame.width, frame.height)
        except VisionInferenceError as exc:
            raise LocalizationInputError(str(exc)) from exc
        gray = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        dynamic_mask = None
        person_boxes = [
            detection.bbox
            for detection in payload.object_detections
            if detection.frame_index == frame_index
            and detection.label.strip().lower() == "person"
            and detection.confidence >= 0.20
        ]
        if person_boxes:
            dynamic_mask = np.full((frame.height, frame.width), 255, dtype=np.uint8)
            for box in person_boxes:
                x1, y1, x2, y2 = (float(value) for value in box)
                pad_x = max(12.0, (x2 - x1) * 0.08)
                pad_y = max(12.0, (y2 - y1) * 0.08)
                left = max(0, min(frame.width - 1, int(math.floor(x1 - pad_x))))
                top = max(0, min(frame.height - 1, int(math.floor(y1 - pad_y))))
                right = max(left + 1, min(frame.width, int(math.ceil(x2 + pad_x))))
                bottom = max(top + 1, min(frame.height, int(math.ceil(y2 + pad_y))))
                dynamic_mask[top:bottom, left:right] = 0
            masked_ratio = float(np.mean(dynamic_mask == 0))
            # If a detector ever covers nearly the whole image, preserve the
            # ordinary visual path instead of starving PnP of static features.
            if masked_ratio <= 0.68:
                person_masked_frame_count += 1
                person_masked_area_ratios.append(masked_ratio)
            else:
                dynamic_mask = None
        keypoints, descriptors = orb.detectAndCompute(gray, dynamic_mask)
        if descriptors is None or len(keypoints) < 6:
            continue
        best_feature_count = max(best_feature_count, len(keypoints))
        query_responses = np.asarray([max(0.0, float(keypoint.response)) for keypoint in keypoints], dtype=np.float32)

        # Use the real local detector's furniture boxes as a bounded pose seed
        # when the native RoomPlan map contains recognizable objects. The seed
        # itself is never publishable: it must expand into fresh ORB matches
        # and pass the same PnP, reprojection, room, and cross-view checks.
        semantic_seeds = _semantic_object_pose_seeds(
            payload,
            frame_index=frame_index,
            frame_width=frame.width,
            frame_height=frame.height,
        )
        for semantic_seed in semantic_seeds:
            semantic_matrix = np.asarray(_camera_to_world(semantic_seed["rvec"], semantic_seed["tvec"]), dtype=np.float64)
            semantic_cuboid = semantic_seed.get("semantic_cuboid") if isinstance(semantic_seed.get("semantic_cuboid"), dict) else {}
            semantic_cuboid_candidates.append(
                {
                    "kind": "semantic-cuboid",
                    "frame_index": frame_index,
                    "camera_center": [round(float(semantic_matrix[index, 3]), 4) for index in range(3)],
                    "selected_fov_degrees": semantic_seed.get("fov_degrees"),
                    "match_count": int(semantic_seed.get("semantic_object_match_count") or 0),
                    "labels": list(semantic_seed.get("semantic_object_labels") or []),
                    "anchor": semantic_seed.get("semantic_object_anchor"),
                    "confidence": float(semantic_seed.get("semantic_object_confidence") or 0.0),
                    "cuboid_score": float(semantic_cuboid.get("score") or 0.0),
                    "mean_iou": float(semantic_cuboid.get("mean_iou") or 0.0),
                    "minimum_iou": float(semantic_cuboid.get("minimum_iou") or 0.0),
                    "mean_center_score": float(semantic_cuboid.get("mean_center_score") or 0.0),
                    "mean_area_ratio": float(semantic_cuboid.get("mean_area_ratio") or 0.0),
                    "matched_object_count": int(semantic_cuboid.get("matched_object_count") or 0),
                    "semantic_group_count": int(semantic_cuboid.get("semantic_group_count") or 0),
                    "center_shift_m": float(semantic_cuboid.get("center_shift_m") or 0.0),
                    "yaw_shift_degrees": float(semantic_cuboid.get("yaw_shift_degrees") or 0.0),
                    "fov_shift_degrees": float(semantic_cuboid.get("fov_shift_degrees") or 0.0),
                }
            )
            semantic_matches = _pose_guided_matches(
                frame_width=frame.width,
                frame_height=frame.height,
                keypoints=keypoints,
                descriptors=descriptors,
                landmark_points=landmark_points,
                landmark_descriptors=landmark_descriptors,
                pose=semantic_seed,
                radius_px=96.0,
            )
            semantic_radius_px = 96.0
            if len(semantic_matches) < 12:
                wider_semantic_matches = _pose_guided_matches(
                    frame_width=frame.width,
                    frame_height=frame.height,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    landmark_points=landmark_points,
                    landmark_descriptors=landmark_descriptors,
                    pose=semantic_seed,
                    radius_px=160.0,
                )
                if len(wider_semantic_matches) > len(semantic_matches):
                    semantic_matches = wider_semantic_matches
                    semantic_radius_px = 160.0
            if len(semantic_matches) < 8:
                continue
            semantic_refined = _refine_pose_with_guided_matches(
                frame_width=frame.width,
                frame_height=frame.height,
                keypoints=keypoints,
                descriptors=descriptors,
                landmark_points=landmark_points,
                landmark_descriptors=landmark_descriptors,
                pose=semantic_seed,
                matches=semantic_matches,
            )
            if semantic_refined is None:
                continue
            semantic_pose, refined_matches = semantic_refined
            semantic_pose["semantic_object_guided_match_count"] = len(refined_matches)
            semantic_pose["semantic_object_guided_radius_px"] = semantic_radius_px
            semantic_candidate = _localization_candidate(
                frame_index,
                "all-views",
                refined_matches,
                semantic_pose,
                landmark_view_ids,
            )
            semantic_candidate["diagnostics"]["learned_matcher"] = learned_matcher_diagnostics
            semantic_candidate["diagnostics"]["semantic_object_seed"] = {
                "match_count": int(semantic_seed.get("semantic_object_match_count") or 0),
                "labels": list(semantic_seed.get("semantic_object_labels") or []),
                "anchor": semantic_seed.get("semantic_object_anchor"),
                "confidence": float(semantic_seed.get("semantic_object_confidence") or 0.0),
                "cuboid": semantic_cuboid,
                "guided_match_count": len(refined_matches),
                "guided_radius_px": semantic_radius_px,
            }
            semantic_matrix = semantic_candidate["_pose_matrix"]
            semantic_scene_prior = _pose_scene_prior(semantic_matrix, payload)
            semantic_candidate["_scene_plausible"] = bool(semantic_scene_prior["accepted"])
            semantic_candidate["diagnostics"]["scene_prior"] = semantic_scene_prior
            candidates.append(semantic_candidate)

        matched_groups: list[tuple[str, np.ndarray, list[Any]]] = []
        for view_id, landmark_indices in grouped_landmarks:
            matches = _descriptor_matches(
                descriptors,
                landmark_descriptors,
                landmark_indices,
                matcher,
                query_responses=query_responses,
                landmark_responses=landmark_responses,
                learned_matcher=learned_matcher,
            )
            best_match_count = max(best_match_count, len(matches))
            if matches:
                frame_min_distance = float(min(match.distance for match in matches))
                best_min_distance = frame_min_distance if best_min_distance is None else min(best_min_distance, frame_min_distance)
            if len(matches) >= 6:
                matched_groups.append((view_id, landmark_indices, matches))

        # Descriptor matching is cheap compared with the PnP/FOV sweep.  Rank
        # every RoomPlan scan view, but solve only the strongest four plus one
        # all-landmarks hypothesis.  A real fixed view should rank the scan
        # viewpoints that actually saw the same surfaces near the top, while
        # retaining several independent views for consensus.
        matched_groups.sort(
            key=lambda item: (
                len(item[2]),
                sum(match.distance <= 44 for match in item[2]),
                -float(np.median([match.distance for match in item[2]])),
            ),
            reverse=True,
        )
        solve_groups = matched_groups[:3]
        if payload.search_prior is not None and payload.search_prior.landmark_view_id is not None:
            prior_group = next(
                (item for item in matched_groups if item[0] == payload.search_prior.landmark_view_id),
                None,
            )
            if prior_group is not None and all(item[0] != prior_group[0] for item in solve_groups):
                if len(solve_groups) >= 4:
                    solve_groups[-1] = prior_group
                else:
                    solve_groups.append(prior_group)
        if len(grouped_landmarks) >= 2:
            all_matches = _descriptor_matches(
                descriptors,
                landmark_descriptors,
                all_landmark_indices,
                matcher,
                query_responses=query_responses,
                landmark_responses=landmark_responses,
                learned_matcher=learned_matcher,
            )
            best_match_count = max(best_match_count, len(all_matches))
            if all_matches:
                frame_min_distance = float(min(match.distance for match in all_matches))
                best_min_distance = frame_min_distance if best_min_distance is None else min(best_min_distance, frame_min_distance)
            if len(all_matches) >= 6:
                solve_groups.append(("all-views", all_landmark_indices, all_matches))

        for group_rank, (view_id, _landmark_indices, matches) in enumerate(solve_groups):
            preferred_fov = None
            if payload.search_prior is not None and payload.intrinsics is None:
                preferred_fov = payload.search_prior.fov_degrees
            pose = _pose_from_matches(
                payload=payload,
                frame_width=frame.width,
                frame_height=frame.height,
                keypoints=keypoints,
                matches=matches,
                landmark_points=landmark_points,
                preferred_fov_degrees=preferred_fov,
            )
            if pose is not None:
                seed_matrix = np.asarray(_camera_to_world(pose["rvec"], pose["tvec"]), dtype=np.float64)
                seed_prior = _pose_scene_prior(seed_matrix, payload)
                search_prior_distance_m: float | None = None
                search_prior_center: np.ndarray | None = None
                search_prior_view_match = False
                search_prior_compatible = False
                search_prior_aligned = False
                if payload.search_prior is not None:
                    search_prior_center = np.asarray(payload.search_prior.center, dtype=np.float64)
                    search_prior_distance_m = float(np.linalg.norm(seed_matrix[:3, 3] - search_prior_center))
                    search_prior_view_match = (
                        view_id == "all-views"
                        if payload.search_prior.landmark_view_id is None
                        else view_id == payload.search_prior.landmark_view_id
                    )
                    prior_fov = payload.search_prior.fov_degrees
                    fov_matches = (
                        prior_fov is None
                        or pose.get("fov_degrees") is None
                        or abs(float(pose["fov_degrees"]) - float(prior_fov)) <= 12.0
                    )
                    search_prior_compatible = search_prior_view_match and (prior_fov is None or fov_matches)
                    search_prior_aligned = search_prior_compatible and search_prior_distance_m <= 0.80
                pose["search_prior_distance_m"] = None if search_prior_distance_m is None else round(search_prior_distance_m, 4)
                pose["search_prior_view_match"] = search_prior_view_match
                pose["search_prior_aligned"] = search_prior_aligned
                guided_variants: list[tuple[str, dict[str, Any], list[Any], float]] = []
                # First keep the ordinary seed-guided path. A nearby prior can
                # widen its spatial descriptor search, but it still has to earn
                # fresh geometric support below.
                if seed_prior["accepted"] or search_prior_aligned:
                    guided_matches = _pose_guided_matches(
                        frame_width=frame.width,
                        frame_height=frame.height,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        landmark_points=landmark_points,
                        landmark_descriptors=landmark_descriptors,
                        pose=pose,
                    )
                    guided_radius_px = 36.0
                    if len(guided_matches) < 8:
                        wide_guided_matches = _pose_guided_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            landmark_points=landmark_points,
                            landmark_descriptors=landmark_descriptors,
                            pose=pose,
                            radius_px=72.0,
                        )
                        if len(wide_guided_matches) > len(guided_matches):
                            guided_matches = wide_guided_matches
                            guided_radius_px = 72.0
                    if search_prior_aligned and len(guided_matches) < 12:
                        prior_guided_matches = _pose_guided_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            landmark_points=landmark_points,
                            landmark_descriptors=landmark_descriptors,
                            pose=pose,
                            radius_px=128.0,
                        )
                        if len(prior_guided_matches) > len(guided_matches):
                            guided_matches = prior_guided_matches
                            guided_radius_px = 128.0
                    guided_variants.append(("seed", pose, guided_matches, guided_radius_px))

                # A moved camera often gives us a useful translation before
                # PnP has the rotation right: the candidate sits at a sane
                # height inside the RoomPlan polygon, but fails only the
                # upright check. Use that *fresh* center as a search
                # initialization and estimate rotation again from fresh
                # descriptor directions. This does not make the seed valid;
                # the unconstrained guided PnP below still has to recover at
                # least ten strong cross-view inliers before it can matter.
                seed_height_plausible = (
                    seed_prior.get("reason") in {"camera_not_upright", "outside_roomplan_bounds"}
                    and isinstance(seed_prior.get("camera_height_above_floor_m"), (int, float))
                    and 0.15 <= float(seed_prior["camera_height_above_floor_m"]) <= 3.50
                )
                seed_position_plausible = (
                    seed_height_plausible
                    and isinstance(seed_prior.get("distance_from_room_polygon_m"), (int, float))
                    and float(seed_prior["distance_from_room_polygon_m"]) <= 0.75
                )
                if (
                    seed_position_plausible
                    and payload.intrinsics is None
                    and pose.get("fov_degrees") is not None
                    # Re-estimating rotation around every plausible PnP center
                    # was the remaining timeout hotspot on moved cameras. The
                    # two strongest descriptor groups are enough to seed this
                    # recovery; the all-landmarks guided stage can still pull
                    # in support from every RoomPlan view afterwards.
                    and (group_rank < 2 or search_prior_view_match)
                ):
                    seed_center_pose = _pose_from_fixed_center_matches(
                        frame_width=frame.width,
                        frame_height=frame.height,
                        keypoints=keypoints,
                        matches=matches,
                        landmark_points=landmark_points,
                        camera_center=seed_matrix[:3, 3],
                        fov_degrees=float(pose["fov_degrees"]),
                        random_seed=frame_index * 193 + len(matches),
                    )
                    if seed_center_pose is not None:
                        seed_center_matches = _pose_guided_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            landmark_points=landmark_points,
                            landmark_descriptors=landmark_descriptors,
                            pose=seed_center_pose,
                            radius_px=72.0,
                        )
                        seed_center_radius_px = 72.0
                        if len(seed_center_matches) < 12:
                            wide_seed_center_matches = _pose_guided_matches(
                                frame_width=frame.width,
                                frame_height=frame.height,
                                keypoints=keypoints,
                                descriptors=descriptors,
                                landmark_points=landmark_points,
                                landmark_descriptors=landmark_descriptors,
                                pose=seed_center_pose,
                                radius_px=112.0,
                            )
                            if len(wide_seed_center_matches) > len(seed_center_matches):
                                seed_center_matches = wide_seed_center_matches
                                seed_center_radius_px = 112.0
                        if 6 <= len(seed_center_matches) < 12:
                            provisional_pose = _provisional_pose_from_guided_matches(
                                frame_width=frame.width,
                                frame_height=frame.height,
                                keypoints=keypoints,
                                landmark_points=landmark_points,
                                pose=seed_center_pose,
                                matches=seed_center_matches,
                            )
                            if provisional_pose is not None:
                                expanded_seed_center_matches = _pose_guided_matches(
                                    frame_width=frame.width,
                                    frame_height=frame.height,
                                    keypoints=keypoints,
                                    descriptors=descriptors,
                                    landmark_points=landmark_points,
                                    landmark_descriptors=landmark_descriptors,
                                    pose=provisional_pose,
                                    radius_px=72.0,
                                )
                                if len(expanded_seed_center_matches) > len(seed_center_matches):
                                    seed_center_pose = provisional_pose
                                    seed_center_matches = expanded_seed_center_matches
                                    seed_center_radius_px = 72.0
                        guided_variants.append(
                            ("seed-center-rotation", seed_center_pose, seed_center_matches, seed_center_radius_px)
                        )

                # A fresh PnP seed can recover the correct *height* while
                # landing in the wrong repeated-texture basin in X/Z. Search
                # a bounded set of centers derived from the RoomPlan
                # perimeter at that fresh height. Keep this independent of
                # the historical prior: a rejected recurring cluster is only
                # a search hint and must never suppress a fresh room-derived
                # search for a moved camera.
                # Each sampled center must first recover a fresh orientation,
                # then earn ordinary cross-view guided correspondences. The
                # sampled center itself is never publishable evidence.
                if (
                    (
                        seed_prior["accepted"]
                        or seed_height_plausible
                        or seed_prior.get("reason") in {"camera_not_upright", "outside_roomplan_bounds"}
                    )
                    and payload.room_zones
                    and payload.intrinsics is None
                    and pose.get("fov_degrees") is not None
                    # One fresh frame is enough to probe RoomPlan-derived
                    # perimeter centers. The remaining fixed frames still
                    # provide independent consensus for the final pose, while
                    # avoiding the same bounded search twice per request.
                    and frame_index == len(payload.frames) - 1
                    and (group_rank == 0 or view_id == "all-views")
                ):
                    pose["room_center_search_attempted"] = True
                    room_search_centers: list[tuple[np.ndarray, float]] = []
                    room_search_fovs: list[float] = []
                    for fov in (float(payload.fov_degrees), 74.0, float(pose["fov_degrees"])):
                        if 30.0 <= fov <= 120.0 and all(abs(fov - existing) >= 0.5 for existing in room_search_fovs):
                            room_search_fovs.append(fov)
                    if payload.search_prior is not None and payload.search_prior.fov_degrees is not None:
                        prior_fov = float(payload.search_prior.fov_degrees)
                        if all(abs(prior_fov - existing) >= 0.5 for existing in room_search_fovs):
                            room_search_fovs.append(prior_fov)
                    room_search_fovs = room_search_fovs[:3]
                    room_search_camera_y_values = _room_search_camera_y_values(payload, float(seed_matrix[1, 3]))
                    if len(room_search_camera_y_values) > 3:
                        room_search_camera_y_values = room_search_camera_y_values[:3]
                    for room_fov in room_search_fovs:
                        for camera_y in room_search_camera_y_values:
                            room_search_centers.extend(
                                (
                                    center,
                                    room_fov,
                                )
                                for center in _room_perimeter_search_centers(
                                    payload,
                                    camera_y=camera_y,
                                    spacing_m=0.60,
                                    max_centers=10,
                                )
                            )
                    pose["room_center_search_center_count"] = len(room_search_centers)
                    room_search_hypotheses: list[tuple[tuple[int, int, int, float], dict[str, Any], list[Any], float]] = []
                    for center_index, (room_center, room_fov) in enumerate(room_search_centers):
                        room_pose = _pose_from_fixed_center_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            matches=matches,
                            landmark_points=landmark_points,
                            camera_center=room_center,
                            fov_degrees=room_fov,
                            random_seed=frame_index * 997 + group_rank * 101 + center_index,
                            iteration_limit=40,
                        )
                        if room_pose is None:
                            continue
                        room_matches = _pose_guided_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            landmark_points=landmark_points,
                            landmark_descriptors=landmark_descriptors,
                            pose=room_pose,
                            radius_px=96.0,
                        )
                        room_radius_px = 96.0
                        if len(room_matches) < 10:
                            wider_room_matches = _pose_guided_matches(
                                frame_width=frame.width,
                                frame_height=frame.height,
                                keypoints=keypoints,
                                descriptors=descriptors,
                                landmark_points=landmark_points,
                                landmark_descriptors=landmark_descriptors,
                                pose=room_pose,
                                radius_px=144.0,
                            )
                            if len(wider_room_matches) > len(room_matches):
                                room_matches = wider_room_matches
                                room_radius_px = 144.0
                        room_views = {
                            landmark_view_ids[item.trainIdx]
                            for item in room_matches
                            if 0 <= item.trainIdx < len(landmark_view_ids)
                        }
                        if len(room_matches) < 8 or len(room_views) < 2:
                            continue
                        room_pose["room_search_center"] = room_center.tolist()
                        room_pose["room_search_fov_degrees"] = room_fov
                        room_pose["room_center_search_recovery"] = True
                        room_pose["room_center_search_attempted"] = True
                        room_pose["room_center_search_center_count"] = len(room_search_centers)
                        score = (
                            len(room_matches),
                            len(room_views),
                            int(room_pose["inlier_count"]),
                            -float(room_pose["mean_error"]),
                        )
                        room_search_hypotheses.append((score, room_pose, room_matches, room_radius_px))
                    pose["room_center_search_hypothesis_count"] = len(room_search_hypotheses)
                    room_search_hypotheses.sort(key=lambda item: item[0], reverse=True)
                    pose["room_center_search_hypothesis_stats"] = [
                        {
                            "center": [round(float(value), 4) for value in room_pose["room_search_center"]],
                            "fov_degrees": float(room_pose["room_search_fov_degrees"]),
                            "guided_match_count": len(room_matches),
                            "guided_view_count": len(
                                {
                                    landmark_view_ids[item.trainIdx]
                                    for item in room_matches
                                    if 0 <= item.trainIdx < len(landmark_view_ids)
                                }
                            ),
                            "rotation_inlier_count": int(room_pose["inlier_count"]),
                            "rotation_mean_error_px": round(float(room_pose["mean_error"]), 4),
                            "radius_px": float(room_radius_px),
                        }
                        for _score, room_pose, room_matches, room_radius_px in room_search_hypotheses
                    ]
                    # Coarse angular scoring is only an initializer and can
                    # rank a repeated texture basin above the true wall
                    # position. The fixed-center probes are already bounded;
                    # carry every surviving hypothesis into the normal PnP
                    # refinement so the final geometric checks choose the
                    # pose rather than the coarse score.
                    for _score, room_pose, room_matches, room_radius_px in room_search_hypotheses:
                        room_pose["room_center_search_hypothesis_count"] = len(room_search_hypotheses)
                        room_pose["room_center_search_hypothesis_stats"] = pose["room_center_search_hypothesis_stats"]
                        guided_variants.append(("room-center-search", room_pose, room_matches, room_radius_px))

                # A recurring fixed-camera center is more useful as a search
                # initialization than as evidence. Prefer estimating a fresh
                # rotation directly around that center from this frame's
                # descriptor correspondences. If that rotation RANSAC cannot
                # find enough support, retain the older seed-rotation fallback.
                # Either way, the center is temporary: only the unconstrained
                # guided PnP result below can become a localization candidate.
                if search_prior_compatible and search_prior_center is not None:
                    prior_search_pose = None
                    prior_fov = payload.search_prior.fov_degrees if payload.search_prior is not None else None
                    if prior_fov is not None and payload.intrinsics is None:
                        prior_search_pose = _pose_from_fixed_center_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            matches=matches,
                            landmark_points=landmark_points,
                            camera_center=search_prior_center,
                            fov_degrees=float(prior_fov),
                            random_seed=frame_index * 97 + len(matches),
                        )
                    if prior_search_pose is None:
                        prior_search_pose = _pose_with_camera_center(pose, search_prior_center)
                    if prior_fov is not None and payload.intrinsics is None:
                        prior_fov = float(prior_fov)
                        prior_focal = 0.5 * frame.width / math.tan(math.radians(prior_fov) / 2.0)
                        prior_search_pose["camera_matrix"] = np.asarray(
                            [
                                [prior_focal, 0.0, frame.width / 2.0],
                                [0.0, prior_focal, frame.height / 2.0],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float64,
                        )
                        prior_search_pose["fov_degrees"] = prior_fov
                    prior_matches = _pose_guided_matches(
                        frame_width=frame.width,
                        frame_height=frame.height,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        landmark_points=landmark_points,
                        landmark_descriptors=landmark_descriptors,
                        pose=prior_search_pose,
                        radius_px=96.0,
                    )
                    prior_radius_px = 96.0
                    if len(prior_matches) < 12:
                        very_wide_prior_matches = _pose_guided_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            landmark_points=landmark_points,
                            landmark_descriptors=landmark_descriptors,
                            pose=prior_search_pose,
                            radius_px=160.0,
                        )
                        if len(very_wide_prior_matches) > len(prior_matches):
                            prior_matches = very_wide_prior_matches
                            prior_radius_px = 160.0
                    # Six or seven geometrically guided matches are useful
                    # search evidence but intentionally remain below the real
                    # refinement floor. Use them once to improve projection,
                    # then rematch all landmarks. Only an expanded set that
                    # later passes the normal >=8 guided refinement can matter.
                    if 6 <= len(prior_matches) < 12:
                        provisional_pose = _provisional_pose_from_guided_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            landmark_points=landmark_points,
                            pose=prior_search_pose,
                            matches=prior_matches,
                        )
                        if provisional_pose is not None:
                            expanded_prior_matches = _pose_guided_matches(
                                frame_width=frame.width,
                                frame_height=frame.height,
                                keypoints=keypoints,
                                descriptors=descriptors,
                                landmark_points=landmark_points,
                                landmark_descriptors=landmark_descriptors,
                                pose=provisional_pose,
                                radius_px=72.0,
                            )
                            expanded_radius_px = 72.0
                            if len(expanded_prior_matches) < 12:
                                wider_expanded_matches = _pose_guided_matches(
                                    frame_width=frame.width,
                                    frame_height=frame.height,
                                    keypoints=keypoints,
                                    descriptors=descriptors,
                                    landmark_points=landmark_points,
                                    landmark_descriptors=landmark_descriptors,
                                    pose=provisional_pose,
                                    radius_px=112.0,
                                )
                                if len(wider_expanded_matches) > len(expanded_prior_matches):
                                    expanded_prior_matches = wider_expanded_matches
                                    expanded_radius_px = 112.0
                            if len(expanded_prior_matches) > len(prior_matches):
                                prior_search_pose = provisional_pose
                                prior_matches = expanded_prior_matches
                                prior_radius_px = expanded_radius_px
                    guided_variants.append(("prior-center", prior_search_pose, prior_matches, prior_radius_px))

                if guided_variants:
                    def variant_views(variant_matches: list[Any]) -> list[str]:
                        return sorted(
                            {
                                landmark_view_ids[item.trainIdx]
                                for item in variant_matches
                                if 0 <= item.trainIdx < len(landmark_view_ids)
                            }
                        )

                    attempt_kind, _attempt_pose, attempt_matches, attempt_radius = max(
                        guided_variants,
                        key=lambda item: (len(item[2]), len(variant_views(item[2]))),
                    )
                    attempt_views = variant_views(attempt_matches)
                    pose["guided_attempt_match_count"] = len(attempt_matches)
                    pose["guided_attempt_view_count"] = len(attempt_views)
                    pose["guided_attempt_view_ids"] = attempt_views
                    pose["guided_attempt_radius_px"] = attempt_radius
                    pose["search_prior_center_attempted"] = any(item[0] == "prior-center" for item in guided_variants)

                    best_guided: tuple[tuple[int, int, float], dict[str, Any], list[Any], bool, bool, str, float] | None = None
                    for variant_kind, guided_seed_pose, variant_matches, variant_radius in guided_variants:
                        guided = _refine_pose_with_guided_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            landmark_points=landmark_points,
                            landmark_descriptors=landmark_descriptors,
                            pose=guided_seed_pose,
                            matches=variant_matches,
                        )
                        if variant_kind == "room-center-search":
                            room_search_center = guided_seed_pose.get("room_search_center")
                            room_search_fov = guided_seed_pose.get("room_search_fov_degrees")
                            for hypothesis in pose.get("room_center_search_hypothesis_stats", []):
                                if (
                                    hypothesis.get("center")
                                    == [round(float(value), 4) for value in (room_search_center or [])]
                                    and (
                                        room_search_fov is None
                                        or abs(float(hypothesis.get("fov_degrees", 0.0)) - float(room_search_fov)) < 0.5
                                    )
                                ):
                                    hypothesis["guided_refinement_succeeded"] = guided is not None
                                    if guided is not None:
                                        refined_probe_pose, refined_probe_matches = guided
                                        refined_probe_matrix = np.asarray(
                                            _camera_to_world(
                                                refined_probe_pose["rvec"],
                                                refined_probe_pose["tvec"],
                                            ),
                                            dtype=np.float64,
                                        )
                                        hypothesis["refinement_inlier_count"] = int(
                                            refined_probe_pose["inlier_count"]
                                        )
                                        hypothesis["refinement_mean_error_px"] = round(
                                            float(refined_probe_pose["mean_error"]), 4
                                        )
                                        hypothesis["refinement_guided_view_count"] = len(
                                            {
                                                landmark_view_ids[item.trainIdx]
                                                for item in refined_probe_matches
                                                if 0 <= item.trainIdx < len(landmark_view_ids)
                                            }
                                        )
                                        hypothesis["refinement_center_distance_m"] = round(
                                            float(
                                                np.linalg.norm(
                                                    refined_probe_matrix[:3, 3]
                                                    - np.asarray(room_search_center, dtype=np.float64)
                                                )
                                            ),
                                            4,
                                        )
                                    break
                        if guided is None:
                            continue
                        guided_pose, refined_matches = guided
                        guided_matrix = np.asarray(
                            _camera_to_world(guided_pose["rvec"], guided_pose["tvec"]),
                            dtype=np.float64,
                        )
                        guided_prior = _pose_scene_prior(guided_matrix, payload)
                        guided_views = {
                            landmark_view_ids[refined_matches[int(index)].trainIdx]
                            for index in guided_pose["inlier_indices"]
                            if 0 <= int(index) < len(refined_matches)
                            and 0 <= refined_matches[int(index)].trainIdx < len(landmark_view_ids)
                        }
                        seed_to_guided_translation = float(np.linalg.norm(guided_matrix[:3, 3] - seed_matrix[:3, 3]))
                        seed_to_guided_rotation = seed_matrix[:3, :3].T @ guided_matrix[:3, :3]
                        seed_to_guided_cosine = float(
                            np.clip((np.trace(seed_to_guided_rotation) - 1.0) / 2.0, -1.0, 1.0)
                        )
                        seed_to_guided_rotation_degrees = math.degrees(math.acos(seed_to_guided_cosine))
                        seed_consistent = (
                            seed_prior["accepted"]
                            and seed_to_guided_translation <= 0.80
                            and seed_to_guided_rotation_degrees <= 20.0
                        )
                        prior_recovery = (
                            search_prior_center is not None
                            and (variant_kind == "prior-center" or (not seed_prior["accepted"] and search_prior_aligned))
                            and guided_pose["inlier_count"] >= max(10, pose["inlier_count"] + 4)
                            and float(np.linalg.norm(guided_matrix[:3, 3] - search_prior_center)) <= 0.65
                        )
                        seed_center_recovery = (
                            variant_kind == "seed-center-rotation"
                            and seed_position_plausible
                            and guided_pose["inlier_count"] >= max(10, pose["inlier_count"] + 4)
                            and float(np.linalg.norm(guided_matrix[:3, 3] - seed_matrix[:3, 3])) <= 0.65
                        )
                        room_search_center = guided_seed_pose.get("room_search_center")
                        room_center_recovery = (
                            variant_kind == "room-center-search"
                            and isinstance(room_search_center, list)
                            and len(room_search_center) == 3
                            and guided_pose["inlier_count"] >= max(10, pose["inlier_count"] + 4)
                            and float(
                                np.linalg.norm(
                                    guided_matrix[:3, 3] - np.asarray(room_search_center, dtype=np.float64)
                                )
                            ) <= 0.70
                        )
                        if not (
                            guided_prior["accepted"]
                            and len(guided_views) >= 2
                            and guided_pose["inlier_count"] >= pose["inlier_count"] + 2
                            and (seed_consistent or prior_recovery or seed_center_recovery or room_center_recovery)
                        ):
                            continue
                        score = (
                            int(guided_pose["inlier_count"]),
                            len(guided_views),
                            -float(guided_pose["mean_error"]),
                        )
                        if best_guided is None or score > best_guided[0]:
                            best_guided = (
                                score,
                                guided_pose,
                                refined_matches,
                                prior_recovery,
                                seed_center_recovery,
                                variant_kind,
                                variant_radius,
                            )

                    if best_guided is not None:
                        _, guided_pose, guided_matches, prior_recovery, seed_center_recovery, variant_kind, variant_radius = best_guided
                        guided_views = variant_views(guided_matches)
                        guided_pose["guided_attempt_match_count"] = len(guided_matches)
                        guided_pose["guided_attempt_view_count"] = len(guided_views)
                        guided_pose["guided_attempt_view_ids"] = guided_views
                        guided_pose["guided_attempt_radius_px"] = variant_radius
                        guided_pose["search_prior_center_attempted"] = True
                        guided_pose["search_prior_center_guided"] = variant_kind == "prior-center"
                        guided_pose["search_prior_recovery"] = prior_recovery
                        guided_pose["seed_center_rotation_recovery"] = seed_center_recovery
                        guided_pose["room_center_search_attempted"] = bool(
                            guided_pose.get("room_center_search_attempted")
                            or variant_kind == "room-center-search"
                        )
                        guided_pose["room_center_search_recovery"] = room_center_recovery
                        if guided_pose.get("room_center_search_attempted"):
                            guided_pose["room_center_search_center_count"] = int(
                                guided_pose.get("room_center_search_center_count")
                                or pose.get("room_center_search_center_count")
                                or 0
                            )
                            guided_pose["room_center_search_hypothesis_count"] = int(
                                guided_pose.get("room_center_search_hypothesis_count")
                                or pose.get("room_center_search_hypothesis_count")
                                or 0
                            )
                        pose = guided_pose
                        matches = guided_matches
                candidate = _localization_candidate(
                    frame_index,
                    view_id,
                    matches,
                    pose,
                    landmark_view_ids,
                )
                candidate["diagnostics"]["learned_matcher"] = learned_matcher_diagnostics
                scene_prior = _pose_scene_prior(candidate["_pose_matrix"], payload)
                candidate["_scene_plausible"] = bool(scene_prior["accepted"])
                candidate["diagnostics"]["scene_prior"] = scene_prior
                candidates.append(candidate)

    if not candidates:
        fallback_matrix, _ = _camera_matrix(payload, payload.frames[0].width, payload.frames[0].height)
        semantic_cuboid_candidates.sort(
            key=_semantic_cuboid_rank,
            reverse=True,
        )
        selected_semantic = semantic_cuboid_candidates[0] if semantic_cuboid_candidates else None
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
                "reason": (
                    "semantic_geometry_unverified"
                    if selected_semantic is not None
                    else ("pnp_no_geometric_consensus" if best_match_count >= 6 else "insufficient_feature_matches")
                ),
                "best_feature_count": best_feature_count,
                "best_match_count": best_match_count,
                "best_min_descriptor_distance": best_min_distance,
                "learned_matcher": learned_matcher_diagnostics,
                "semantic_cuboid_candidates": semantic_cuboid_candidates[:12],
                "selected_camera_center": selected_semantic.get("camera_center") if selected_semantic is not None else None,
                "selected_estimate_source": "semantic-cuboid" if selected_semantic is not None else None,
                "dynamic_person_mask": {
                    "masked_frame_count": person_masked_frame_count,
                    "max_masked_area_ratio": round(max(person_masked_area_ratios), 4) if person_masked_area_ratios else 0.0,
                },
                "guided_person_calibration": guided_person_diagnostics,
                "raw_frames_persisted": False,
            },
        }
    for candidate in candidates:
        consensus = _consensus_stats(candidate, candidates)
        candidate["_consensus_frame_count"] = int(consensus["frame_count"])
        candidate["_consensus_scan_view_count"] = int(consensus["scan_view_count"])
        candidate["_consensus_scan_views"] = list(consensus["scan_views"])
        candidate["_pose_only_frame_count"] = int(consensus["pose_only_frame_count"])
        candidate["_consensus_fov_span_degrees"] = float(consensus["fov_span_degrees"])
    candidate_summaries = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            int(item.get("_scene_plausible") is not False),
            item["_consensus_scan_view_count"],
            item["_consensus_frame_count"],
            item["inlier_count"],
            -item["reprojection_error_px"],
        ),
        reverse=True,
    )[:12]:
        matrix = candidate["_pose_matrix"]
        scene_prior = candidate["diagnostics"].get("scene_prior") or {}
        candidate_summaries.append(
            {
                "frame_index": int(candidate["diagnostics"]["frame_index"]),
                "landmark_view_id": candidate["diagnostics"].get("landmark_view_id"),
                "camera_center": [round(float(matrix[index, 3]), 4) for index in range(3)],
                "inlier_count": int(candidate["inlier_count"]),
                "match_count": int(candidate["match_count"]),
                "reprojection_error_px": round(float(candidate["reprojection_error_px"]), 4),
                "selected_fov_degrees": candidate["diagnostics"].get("selected_fov_degrees"),
                "pose_guided_refinement": bool(candidate["diagnostics"].get("pose_guided_refinement")),
                "pose_guided_match_count": int(candidate["diagnostics"].get("pose_guided_match_count") or 0),
                "pose_guided_attempt_match_count": int(candidate["diagnostics"].get("pose_guided_attempt_match_count") or 0),
                "pose_guided_attempt_view_count": int(candidate["diagnostics"].get("pose_guided_attempt_view_count") or 0),
                "pose_guided_attempt_view_ids": list(candidate["diagnostics"].get("pose_guided_attempt_view_ids") or []),
                "pose_guided_attempt_radius_px": float(candidate["diagnostics"].get("pose_guided_attempt_radius_px") or 0.0),
                "search_prior_distance_m": candidate["diagnostics"].get("search_prior_distance_m"),
                "search_prior_aligned": bool(candidate["diagnostics"].get("search_prior_aligned")),
                "search_prior_center_attempted": bool(candidate["diagnostics"].get("search_prior_center_attempted")),
                "search_prior_center_guided": bool(candidate["diagnostics"].get("search_prior_center_guided")),
                "search_prior_recovery": bool(candidate["diagnostics"].get("search_prior_recovery")),
                "seed_center_rotation_recovery": bool(candidate["diagnostics"].get("seed_center_rotation_recovery")),
                "room_center_search_attempted": bool(candidate["diagnostics"].get("room_center_search_attempted")),
                "room_center_search_center_count": int(candidate["diagnostics"].get("room_center_search_center_count") or 0),
                "room_center_search_hypothesis_count": int(candidate["diagnostics"].get("room_center_search_hypothesis_count") or 0),
                "room_center_search_hypothesis_stats": list(candidate["diagnostics"].get("room_center_search_hypothesis_stats") or []),
                "room_center_search_recovery": bool(candidate["diagnostics"].get("room_center_search_recovery")),
                "scene_plausible": bool(candidate.get("_scene_plausible") is not False),
                "scene_reason": scene_prior.get("reason"),
                "consensus_frame_count": int(candidate["_consensus_frame_count"]),
                "consensus_scan_view_count": int(candidate["_consensus_scan_view_count"]),
                "pose_only_consensus_frame_count": int(candidate["_pose_only_frame_count"]),
            }
        )
    best = max(
        candidates,
        key=lambda item: (
            int(_candidate_is_positioned(item)),
            int(item.get("_scene_plausible") is not False),
            item["_consensus_scan_view_count"],
            item["_consensus_frame_count"],
            item["inlier_count"],
            int(item["diagnostics"].get("inlier_landmark_view_count") or 0),
            item["confidence"],
            -item["reprojection_error_px"],
        ),
    )
    positioned = _candidate_is_positioned(best)
    diagnostics = best["diagnostics"]
    consensus_frame_count = int(best.pop("_consensus_frame_count"))
    consensus_scan_view_count = int(best.pop("_consensus_scan_view_count"))
    consensus_scan_views = list(best.pop("_consensus_scan_views"))
    pose_only_frame_count = int(best.pop("_pose_only_frame_count"))
    consensus_fov_span_degrees = float(best.pop("_consensus_fov_span_degrees"))
    pose_matrix = best.pop("_pose_matrix")
    diagnostics["pose_candidate_frame_count"] = len({int(item["diagnostics"]["frame_index"]) for item in candidates})
    diagnostics["consensus_frame_count"] = consensus_frame_count
    diagnostics["consensus_scan_view_count"] = consensus_scan_view_count
    diagnostics["consensus_scan_views"] = consensus_scan_views
    diagnostics["pose_only_consensus_frame_count"] = pose_only_frame_count
    diagnostics["consensus_fov_span_degrees"] = consensus_fov_span_degrees
    diagnostics["candidate_summaries"] = candidate_summaries
    geometric_selected_center = [round(float(pose_matrix[index, 3]), 4) for index in range(3)]
    semantic_cuboid_candidates.sort(
        key=_semantic_cuboid_rank,
        reverse=True,
    )
    diagnostics["semantic_cuboid_candidates"] = semantic_cuboid_candidates[:12]
    selected_semantic = semantic_cuboid_candidates[0] if semantic_cuboid_candidates else None
    semantic_is_strong = bool(
        selected_semantic is not None
        and int(selected_semantic.get("matched_object_count") or 0) >= 2
        and int(selected_semantic.get("semantic_group_count") or 0) >= 2
        and float(selected_semantic.get("minimum_iou") or 0.0) >= 0.30
        and float(selected_semantic.get("mean_iou") or 0.0) >= 0.55
    )
    diagnostics["geometric_selected_camera_center"] = geometric_selected_center
    if not positioned and semantic_is_strong:
        diagnostics["selected_camera_center"] = selected_semantic["camera_center"]
        diagnostics["selected_estimate_source"] = "semantic-cuboid"
    else:
        diagnostics["selected_camera_center"] = geometric_selected_center
        diagnostics["selected_estimate_source"] = "visual-pnp"
    diagnostics["dynamic_person_mask"] = {
        "masked_frame_count": person_masked_frame_count,
        "max_masked_area_ratio": round(max(person_masked_area_ratios), 4) if person_masked_area_ratios else 0.0,
    }
    diagnostics["guided_person_calibration"] = guided_person_diagnostics
    if positioned:
        best["status"] = "positioned"
        best["camera_to_world"] = [[round(float(value), 8) for value in row] for row in pose_matrix]
        evidence_bonus = 0.60 + 0.05 * min(4, consensus_scan_view_count) + 0.025 * min(4, consensus_frame_count)
        best["confidence"] = round(max(float(best["confidence"]), min(0.95, evidence_bonus)), 6)
    for candidate in candidates:
        candidate.pop("_pose_matrix", None)
        candidate.pop("_scene_plausible", None)
        candidate.pop("_consensus_frame_count", None)
        candidate.pop("_consensus_scan_view_count", None)
        candidate.pop("_consensus_scan_views", None)
        candidate.pop("_pose_only_frame_count", None)
        candidate.pop("_consensus_fov_span_degrees", None)
    return best
