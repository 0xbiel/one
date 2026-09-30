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
import os
import threading
from typing import Any, Callable

import cv2
import numpy as np

from .contracts import CameraLocalizationRequest, VisualLandmarkBuildRequest
from .learned_matcher import LearnedMatcher, get_or_fit_matcher
from .low_light import prepare_feature_gray
from .real_vision import VisionInferenceError, decode_jpeg


_HAMMING_POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
_SOLVER_THREAD_STATE = threading.local()


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
    # Keep more independent candidates per iPhone viewpoint. The backend still
    # voxel-deduplicates and caps the merged index, so this adds coverage rather
    # than accepting duplicate matches as confidence.
    orb = cv2.ORB_create(nfeatures=2800, scaleFactor=1.2, nlevels=8, fastThreshold=5, edgeThreshold=17)
    # SIFT is retained as an optional second descriptor for the same metric
    # landmark. ORB is still the compact baseline and remains required for
    # backwards compatibility; SIFT is materially more tolerant of the
    # iPhone-to-browser scale and illumination change in this registration
    # problem. It is CPU feature extraction only; matching/learned scoring and
    # finalist pose polish still use the configured accelerator where present.
    sift = cv2.SIFT_create(nfeatures=2200, contrastThreshold=0.012) if hasattr(cv2, "SIFT_create") else None
    candidates: list[tuple[np.ndarray, bytes, float]] = []
    sift_candidates: list[tuple[np.ndarray, bytes, float]] = []
    frame_feature_counts: list[int] = []
    sift_feature_counts: list[int] = []
    depth_feature_counts: list[int] = []
    sift_depth_feature_counts: list[int] = []
    low_light_frames: list[dict[str, Any]] = []
    feature_frames: list[dict[str, Any]] = []
    for frame in payload.frames:
        jpeg = _jpeg_bytes(frame.frame_base64)
        try:
            image = decode_jpeg(jpeg, frame.width, frame.height)
        except VisionInferenceError as exc:
            raise LocalizationInputError(str(exc)) from exc
        gray, light_diagnostics = prepare_feature_gray(image)
        low_light_frames.append({"frame_index": len(low_light_frames), **light_diagnostics})
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        sift_keypoints: list[Any] = []
        sift_descriptors: np.ndarray | None = None
        if sift is not None:
            sift_keypoints, sift_descriptors = sift.detectAndCompute(gray, None)
        sift_feature_counts.append(len(sift_keypoints))
        if descriptors is None or not keypoints:
            frame_feature_counts.append(0)
            depth_feature_counts.append(0)
            sift_depth_feature_counts.append(0)
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
        sift_accepted = 0
        if frame.depth_base64 is not None and frame.depth_width is not None and frame.depth_height is not None:
            depth = _depth_array(frame.depth_base64, frame.depth_width, frame.depth_height)
            for keypoint, descriptor in zip(keypoints, descriptors):
                depth_m = _depth_at(depth, keypoint.pt[0], keypoint.pt[1], frame.width, frame.height)
                if depth_m is None:
                    continue
                point = _world_point(keypoint.pt[0], keypoint.pt[1], depth_m, intrinsics, camera_to_world)
                candidates.append((point, bytes(descriptor.tolist()), float(keypoint.response)))
                accepted += 1
            if sift_descriptors is not None:
                for keypoint, descriptor in zip(sift_keypoints, sift_descriptors):
                    depth_m = _depth_at(depth, keypoint.pt[0], keypoint.pt[1], frame.width, frame.height)
                    if depth_m is None:
                        continue
                    point = _world_point(keypoint.pt[0], keypoint.pt[1], depth_m, intrinsics, camera_to_world)
                    sift_candidates.append(
                        (
                            point,
                            bytes(np.asarray(descriptor, dtype=np.float32).reshape(-1).tobytes()),
                            float(keypoint.response),
                        )
                    )
                    sift_accepted += 1
        depth_feature_counts.append(accepted)
        sift_depth_feature_counts.append(sift_accepted)

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
    selected = sorted(voxels.values(), key=lambda item: item[2], reverse=True)[:8_000]
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
                "sift_feature_counts": sift_feature_counts,
                "depth_feature_counts": depth_feature_counts,
                "sift_depth_feature_counts": sift_depth_feature_counts,
                "triangulated_feature_count": len(triangulated),
                "low_light": {
                    "enabled": True,
                    "preprocessed_frame_count": int(sum(bool(item.get("low_light")) for item in low_light_frames)),
                    "frames": low_light_frames,
                },
                "raw_frames_persisted": False,
            },
        }
    # Attach the nearest SIFT observation to each retained ORB landmark. Depth
    # quantization and RoomPlan-to-camera alignment can move the same physical
    # surface point across adjacent 3 cm voxels, so search neighbouring voxels
    # inside a bounded 8 cm metric gate instead of requiring an exact voxel
    # collision. The descriptor remains optional so old artifacts and
    # triangulated ORB-only landmarks stay valid.
    sift_by_voxel: dict[tuple[int, int, int], tuple[np.ndarray, bytes, float]] = {}
    for point, descriptor, response in sift_candidates:
        key = tuple(int(round(float(component) / 0.03)) for component in point)
        current = sift_by_voxel.get(key)
        if current is None or response > current[2]:
            sift_by_voxel[key] = (point, descriptor, response)

    serialized_landmarks: list[dict[str, Any]] = []
    attached_sift_count = 0
    for point, descriptor, response in selected:
        landmark: dict[str, Any] = {
            "point": [round(float(value), 6) for value in point],
            "descriptor_base64": base64.b64encode(descriptor).decode("ascii"),
            "response": round(response, 6),
        }
        voxel = tuple(int(round(float(component) / 0.03)) for component in point)
        nearest: tuple[np.ndarray, bytes, float] | None = None
        nearest_distance = 0.08
        for dx, dy, dz in product((-1, 0, 1), repeat=3):
            candidate = sift_by_voxel.get((voxel[0] + dx, voxel[1] + dy, voxel[2] + dz))
            if candidate is None:
                continue
            distance = float(np.linalg.norm(candidate[0] - point))
            if distance <= nearest_distance and (nearest is None or distance < nearest_distance or candidate[2] > nearest[2]):
                nearest = candidate
                nearest_distance = distance
        if nearest is not None:
            landmark["sift_descriptor_base64"] = base64.b64encode(nearest[1]).decode("ascii")
            attached_sift_count += 1
        serialized_landmarks.append(landmark)

    return {
        "status": "ready",
        "schema_version": "roomplan-visual-landmarks.v1",
        "detector": "opencv-orb+sift" if attached_sift_count > 0 else "opencv-orb",
        "landmarks": serialized_landmarks,
        "diagnostics": {
            "landmark_count": len(selected),
            "source_frame_count": len(payload.frames),
            "frame_feature_counts": frame_feature_counts,
            "sift_feature_counts": sift_feature_counts,
            "depth_feature_counts": depth_feature_counts,
            "sift_depth_feature_counts": sift_depth_feature_counts,
            "sift_landmark_count": len(sift_candidates),
            "sift_attached_count": attached_sift_count,
            "triangulated_feature_count": len(triangulated),
            "low_light": {
                "enabled": True,
                "preprocessed_frame_count": int(sum(bool(item.get("low_light")) for item in low_light_frames)),
                "frames": low_light_frames,
            },
            "raw_frames_persisted": False,
        },
    }


def _landmark_feature_arrays(
    payload: CameraLocalizationRequest,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    descriptors: list[np.ndarray] = []
    view_ids: list[str] = []
    sift_descriptors: list[np.ndarray] = []
    sift_available: list[bool] = []
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
        raw_sift = landmark.sift_descriptor_base64
        if raw_sift:
            try:
                sift = np.frombuffer(base64.b64decode(raw_sift, validate=True), dtype=np.float32)
            except (binascii.Error, ValueError) as exc:
                raise LocalizationInputError("landmark SIFT descriptor is invalid base64") from exc
            if sift.shape != (128,) or not np.isfinite(sift).all():
                raise LocalizationInputError("landmark SIFT descriptors must contain 128 finite float32 values")
            sift_descriptors.append(sift)
            sift_available.append(True)
        else:
            sift_descriptors.append(np.zeros(128, dtype=np.float32))
            sift_available.append(False)
    if not descriptors:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 32), dtype=np.uint8),
            view_ids,
            np.empty((0, 128), dtype=np.float32),
            np.empty((0,), dtype=bool),
        )
    return (
        np.asarray(points, dtype=np.float32),
        np.vstack(descriptors).astype(np.uint8),
        view_ids,
        np.vstack(sift_descriptors).astype(np.float32),
        np.asarray(sift_available, dtype=bool),
    )


def _landmark_arrays(payload: CameraLocalizationRequest) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Backwards-compatible ORB-only view of the landmark index."""

    points, descriptors, view_ids, _sift_descriptors, _sift_available = _landmark_feature_arrays(payload)
    return points, descriptors, view_ids


def _descriptor_matches(
    query_descriptors: np.ndarray,
    landmark_descriptors: np.ndarray,
    landmark_indices: np.ndarray,
    matcher: Any,
    *,
    query_responses: np.ndarray | None = None,
    landmark_responses: np.ndarray | None = None,
    learned_matcher: LearnedMatcher | None = None,
    ratio_threshold: float = 0.84,
    distance_limit: float | None = 72.0,
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
            (distance_limit is None or first.distance <= distance_limit)
            and first.distance < ratio_threshold * second.distance
            and reverse_best.get(first.trainIdx) == first.queryIdx
        )
        if baseline_accept:
            # A ratio-pass + reverse-unique ORB match is already strong direct
            # evidence. Keep the learned matcher as an ambiguity rescue path
            # rather than letting a map-specific score replace a geometrically
            # useful direct match before PnP has established a pose. Live room
            # testing showed that replacing these matches can collapse a
            # plausible multi-view basin into a single-view upside-down pose.
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


def _merge_descriptor_matches(
    orb_matches: list[Any],
    sift_matches: list[Any],
    *,
    sift_query_offset: int,
) -> list[Any]:
    """Merge ORB/SIFT matches with family-normalized uniqueness guards."""

    ranked: list[tuple[float, int, Any]] = []
    for match in orb_matches:
        ranked.append((min(127.0, float(match.distance) / 64.0), 0, match))
    for match in sift_matches:
        ranked.append(
            (
                # SIFT's L2 scale is not comparable to Hamming distance. The
                # divisor only makes the shared PnP pool ranking bounded; the
                # ratio test was already applied in _descriptor_matches.
                min(127.0, float(match.distance) / 4.0),
                1,
                cv2.DMatch(
                    _queryIdx=int(match.queryIdx) + int(sift_query_offset),
                    _trainIdx=int(match.trainIdx),
                    _distance=min(127.0, float(match.distance) / 4.0),
                ),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: list[Any] = []
    used_queries: set[int] = set()
    used_landmarks: set[int] = set()
    for _normalized_distance, _family, match in ranked:
        query_index = int(match.queryIdx)
        train_index = int(match.trainIdx)
        if query_index in used_queries or train_index in used_landmarks:
            continue
        used_queries.add(query_index)
        used_landmarks.add(train_index)
        selected.append(match)
    return selected


def _temporal_consensus_matches(
    observations: dict[int, list[tuple[int, float, float, float]]],
    *,
    minimum_frame_support: int = 2,
    maximum_jitter_px: float = 12.0,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Collapse repeated fixed-camera matches into stable burst correspondences.

    The calibration burst is captured while the camera is stationary. A real
    RoomPlan landmark should therefore keep landing at nearly the same image
    coordinate across frames, while many repeated-texture descriptor matches
    wander between unrelated pixels. Keep only landmarks observed in multiple
    frames with bounded pixel jitter and average their 2D location before PnP.
    """
    keypoints: list[Any] = []
    matches: list[Any] = []
    support_counts: list[int] = []
    jitters: list[float] = []
    supported_frames: set[int] = set()

    for landmark_index, raw_observations in observations.items():
        # A detector should produce one winning correspondence per landmark in
        # a frame, but retain the strongest one explicitly if a caller ever
        # supplies duplicates.
        by_frame: dict[int, tuple[float, float, float]] = {}
        for frame_index, x, y, distance in raw_observations:
            current = by_frame.get(int(frame_index))
            candidate = (float(x), float(y), float(distance))
            if current is None or candidate[2] < current[2]:
                by_frame[int(frame_index)] = candidate
        if len(by_frame) < minimum_frame_support:
            continue

        frame_items = sorted(by_frame.items())
        xs = np.asarray([item[1][0] for item in frame_items], dtype=np.float64)
        ys = np.asarray([item[1][1] for item in frame_items], dtype=np.float64)
        center_x = float(np.median(xs))
        center_y = float(np.median(ys))
        radial = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
        jitter = float(np.max(radial)) if radial.size else 0.0
        if jitter > maximum_jitter_px:
            continue

        distances = np.asarray([item[1][2] for item in frame_items], dtype=np.float64)
        query_index = len(keypoints)
        keypoints.append(cv2.KeyPoint(center_x, center_y, 1.0))
        matches.append(
            cv2.DMatch(
                _queryIdx=query_index,
                _trainIdx=int(landmark_index),
                _distance=float(np.median(distances)),
            )
        )
        support_counts.append(len(frame_items))
        jitters.append(jitter)
        supported_frames.update(frame_index for frame_index, _value in frame_items)

    return keypoints, matches, {
        "stable_landmark_count": len(matches),
        "support_frame_count": len(supported_frames),
        "minimum_frame_support": minimum_frame_support,
        "median_landmark_frame_support": round(float(np.median(support_counts)), 3) if support_counts else 0.0,
        "maximum_pixel_jitter": round(max(jitters), 3) if jitters else None,
        "median_pixel_jitter": round(float(np.median(jitters)), 3) if jitters else None,
    }


def _camera_matrix(payload: CameraLocalizationRequest, width: int, height: int) -> tuple[np.ndarray, str]:
    if payload.intrinsics is not None:
        matrix = _matrix(payload.intrinsics.values, (3, 3))
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise LocalizationInputError("camera intrinsics must contain positive focal lengths")
        return matrix, "provided"
    if payload.fov_degrees is None:
        raise LocalizationInputError("camera FOV is unknown; focal length must be solved from localization evidence")
    focal = 0.5 * width / math.tan(math.radians(float(payload.fov_degrees)) / 2.0)
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

    # No single camera FOV is assumed.  When the caller has no calibrated
    # intrinsics, search a bounded focal-length range and let fresh geometric
    # evidence choose the camera model.  An explicitly supplied FOV remains a
    # search hint, not a hard-coded default.
    requested = float(payload.fov_degrees) if payload.fov_degrees is not None else None
    values = [36.0, 42.0, 48.0, 54.0, 60.0, 66.0, 74.0, 84.0, 96.0, 108.0]
    if requested is not None:
        values.append(requested)
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
        source = "estimated-fov-sweep" if payload.fov_degrees is None else "estimated-fov"
        result.append((matrix, source, fov))
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


def _ransac_pnp_hypothesis(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    iterations: int = 1_200,
    reprojection_error: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Return the strongest of several minimal/non-minimal PnP hypotheses.

    AP3P is useful for a minimal hypothesis, but a single PnP flag can be
    brittle for nearly planar indoor landmarks or when the match pool contains
    repeated texture.  MAGSAC++ and DSAC both motivate evaluating multiple
    hypotheses with a robust consensus score instead of trusting one hard
    sample. OpenCV's PnP RANSAC API does not expose MAGSAC on this platform, so
    we keep the robust estimator CPU-side and use the accelerator only for the
    accepted finalist polish below.

    The solver choice is selected using only the current correspondence pool;
    no scene-specific threshold or learned pose is introduced here.
    """
    if len(object_points) < 4 or len(object_points) != len(image_points):
        return None
    flags: list[tuple[int, str]] = [
        (cv2.SOLVEPNP_AP3P, "ap3p"),
        (cv2.SOLVEPNP_EPNP, "epnp"),
    ]
    sqpnp = getattr(cv2, "SOLVEPNP_SQPNP", None)
    if sqpnp is not None:
        flags.append((int(sqpnp), "sqpnp"))

    best: tuple[tuple[float, ...], np.ndarray, np.ndarray, str] | None = None
    for flag, label in flags:
        try:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
                iterationsCount=iterations,
                reprojectionError=reprojection_error,
                confidence=0.999,
                flags=flag,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        try:
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
        except cv2.error:
            continue
        residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        inlier_mask = residuals <= reprojection_error
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < 6:
            continue
        inlier_residuals = residuals[inlier_mask]
        # Soft consensus is a bounded tie-breaker. It rewards a hypothesis
        # with more uniformly small residuals without replacing the strict
        # final reprojection and multi-view gates.
        soft_consensus = float(np.sum(1.0 / (1.0 + (residuals / 4.0) ** 2)))
        score = (
            float(inlier_count),
            soft_consensus,
            -float(np.median(inlier_residuals)),
            -float(np.mean(inlier_residuals)),
        )
        if best is None or score > best[0]:
            best = (score, np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64), label)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _gpu_pose_refine(
    *,
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    residual_threshold: float,
) -> dict[str, Any] | None:
    """Polish a PnP hypothesis with a bounded differentiable GPU solve.

    OpenCV's RANSAC/PnP implementation has no usable MPS backend on macOS,
    so it remains the robust hypothesis generator.  The final nonlinear pose
    polish is differentiable PyTorch and runs on MPS/CUDA when available.  It
    is deliberately conservative: the GPU result is accepted only when it
    retains at least as many positive-depth inliers and improves the measured
    reprojection error.  This makes the accelerator a real part of the solve,
    without allowing an optimizer to manufacture confidence from a weaker
    hypothesis.
    """

    diagnostics: dict[str, Any] = {"status": "unavailable", "device": None, "accepted": False}

    def finish(result: dict[str, Any] | None) -> dict[str, Any] | None:
        attempts = getattr(_SOLVER_THREAD_STATE, "gpu_pose_refinement_attempts", None)
        if isinstance(attempts, list) and len(attempts) < 8:
            attempts.append(dict(diagnostics))
        return result

    object_array = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_array = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if len(object_array) > 32:
        diagnostics["reason"] = "bounded-to-small-finalist-pool"
        return finish(None)
    remaining = int(getattr(_SOLVER_THREAD_STATE, "gpu_pose_refinement_budget", 1))
    if remaining <= 0:
        diagnostics["reason"] = "per-request-gpu-budget-exhausted"
        return finish(None)
    _SOLVER_THREAD_STATE.gpu_pose_refinement_budget = remaining - 1
    try:
        import torch
        from .gpu_runtime import GPU_COMPUTE_LOCK
    except (ImportError, OSError):
        diagnostics["reason"] = "torch-unavailable"
        return finish(None)

    requested = (os.getenv("ONE_GEOMETRY_SOLVER_DEVICE") or os.getenv("ONE_GEOMETRY_DEVICE") or "auto").strip().lower()
    mps_available = bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)())
    cuda_available = bool(torch.cuda.is_available())
    if requested in {"mps", "metal"} and mps_available:
        device = torch.device("mps")
        device_label = "mps"
    elif requested == "cuda" and cuda_available:
        device = torch.device("cuda")
        device_label = "cuda"
    elif requested in {"auto", ""} and mps_available:
        device = torch.device("mps")
        device_label = "mps"
    elif requested in {"auto", ""} and cuda_available:
        device = torch.device("cuda")
        device_label = "cuda"
    else:
        diagnostics.update({"device": "cpu", "reason": "no-supported-gpu"})
        return finish(None)

    if len(object_array) < 6 or len(object_array) != len(image_array):
        diagnostics.update({"device": device_label, "reason": "insufficient-correspondences"})
        return finish(None)
    initial_rvec = np.asarray(rvec, dtype=np.float32).reshape(3)
    initial_tvec = np.asarray(tvec, dtype=np.float32).reshape(3)
    if not np.isfinite(initial_rvec).all() or not np.isfinite(initial_tvec).all():
        diagnostics.update({"device": device_label, "reason": "non-finite-seed"})
        return finish(None)

    camera = np.asarray(camera_matrix, dtype=np.float32)
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        diagnostics.update({"device": device_label, "reason": "invalid-camera-matrix"})
        return finish(None)

    try:
        with GPU_COMPUTE_LOCK:
            return finish(_gpu_pose_refine_locked(
                torch=torch,
                device=device,
                device_label=device_label,
                diagnostics=diagnostics,
                object_array=object_array,
                image_array=image_array,
                initial_rvec=initial_rvec,
                initial_tvec=initial_tvec,
                camera=camera,
                object_points=object_points,
                image_points=image_points,
                camera_matrix=camera_matrix,
                rvec=rvec,
                tvec=tvec,
                residual_threshold=residual_threshold,
            ))
    except (RuntimeError, ValueError, cv2.error) as exc:
        diagnostics.update({"device": device_label, "reason": f"gpu-refinement-failed:{type(exc).__name__}"})
        return finish({"accepted": False, "diagnostics": diagnostics})


def _gpu_pose_refine_locked(
    *,
    torch: Any,
    device: Any,
    device_label: str,
    diagnostics: dict[str, Any],
    object_array: np.ndarray,
    image_array: np.ndarray,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    camera: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    residual_threshold: float,
) -> dict[str, Any] | None:
    """Implementation called while the process-wide GPU lock is held."""

    try:
        baseline_rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        baseline_camera_points = (
            baseline_rotation.astype(np.float32) @ object_array.T + initial_tvec.reshape(3, 1)
        ).T
        baseline_projected = camera @ baseline_camera_points.T
        baseline_projected = (baseline_projected[:2] / np.maximum(baseline_projected[2:3], 1e-5)).T
        baseline_residuals = np.linalg.norm(baseline_projected - image_array, axis=1)
        baseline_inliers = np.flatnonzero(
            (baseline_residuals <= float(residual_threshold)) & (baseline_camera_points[:, 2] > 0.05)
        )
        if len(baseline_inliers) < 6:
            diagnostics.update({"device": device_label, "reason": "seed-has-too-few-inliers"})
            return {"accepted": False, "diagnostics": diagnostics}

        points = torch.from_numpy(object_array).to(device)
        observations = torch.from_numpy(image_array).to(device)
        camera_tensor = torch.from_numpy(camera).to(device)
        initial = torch.from_numpy(np.concatenate((initial_rvec, initial_tvec))).to(device)
        parameters = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([parameters], lr=0.012)

        def project(parameters_tensor: Any) -> tuple[Any, Any]:
            rotation_vector = parameters_tensor[:3]
            translation = parameters_tensor[3:]
            theta2 = torch.sum(rotation_vector * rotation_vector)
            theta = torch.sqrt(torch.clamp(theta2, min=1e-12))
            skew = torch.stack(
                (
                    torch.stack((torch.zeros((), device=device), -rotation_vector[2], rotation_vector[1])),
                    torch.stack((rotation_vector[2], torch.zeros((), device=device), -rotation_vector[0])),
                    torch.stack((-rotation_vector[1], rotation_vector[0], torch.zeros((), device=device))),
                )
            )
            identity = torch.eye(3, device=device)
            coefficient_a = torch.where(
                theta2 < 1e-8,
                1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
                torch.sin(theta) / theta,
            )
            coefficient_b = torch.where(
                theta2 < 1e-8,
                0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
                (1.0 - torch.cos(theta)) / theta2,
            )
            rotation = identity + coefficient_a * skew + coefficient_b * (skew @ skew)
            camera_points = points @ rotation.T + translation
            projected = camera_points @ camera_tensor.T
            projected = projected[:, :2] / torch.clamp(projected[:, 2:3], min=1e-4)
            return projected, camera_points

        for _step in range(48):
            optimizer.zero_grad(set_to_none=True)
            projected_tensor, camera_points_tensor = project(parameters)
            residual = torch.sqrt(torch.sum((projected_tensor - observations) ** 2, dim=1) + 1e-6)
            delta = torch.as_tensor(4.0, device=device)
            robust_loss = torch.where(
                residual <= delta,
                0.5 * residual * residual,
                delta * (residual - 0.5 * delta),
            ).mean()
            # Keep the nonlinear polish in the local basin found by robust
            # PnP; no global search or learned scene-specific displacement is
            # allowed here.
            trust_loss = 0.0005 * torch.sum((parameters - initial) ** 2)
            loss = robust_loss + trust_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            projected_tensor, camera_points_tensor = project(parameters)
            projected_numpy = projected_tensor.detach().cpu().numpy()
            camera_points_numpy = camera_points_tensor.detach().cpu().numpy()
            rotation_numpy = camera_points_tensor.new_zeros((3, 3))
            # Re-evaluate the Rodrigues matrix through the public parameter
            # representation so the CPU bridge receives a proper rotation.
            rotation_vector_numpy = parameters[:3].detach().cpu().numpy().reshape(3, 1)
            rotation_numpy, _ = cv2.Rodrigues(rotation_vector_numpy)
            translation_numpy = parameters[3:].detach().cpu().numpy().reshape(3, 1)
        if not np.isfinite(projected_numpy).all() or not np.isfinite(camera_points_numpy).all():
            diagnostics.update({"device": device_label, "reason": "non-finite-gpu-result"})
            return {"accepted": False, "diagnostics": diagnostics}

        residuals = np.linalg.norm(projected_numpy - image_array, axis=1)
        inlier_indices = np.flatnonzero(
            (residuals <= float(residual_threshold)) & (camera_points_numpy[:, 2] > 0.05)
        )
        mean_error = float(np.mean(residuals[inlier_indices])) if len(inlier_indices) else float("inf")
        baseline_mean_error = float(np.mean(baseline_residuals[baseline_inliers]))
        # A larger thresholded inlier set is not automatically better: a
        # shallow optimizer can buy one extra marginal point by pulling the
        # pose away from the tight CPU basin. Require both support growth and
        # a bounded error change, or a meaningful error reduction with equal
        # support. This keeps GPU polish conservative and reproducible.
        error_not_materially_worse = (
            math.isfinite(mean_error)
            and mean_error <= baseline_mean_error + max(0.15, baseline_mean_error * 0.10)
        )
        accepted = bool(
            len(inlier_indices) >= len(baseline_inliers)
            and len(inlier_indices) >= 6
            and error_not_materially_worse
            and (
                len(inlier_indices) > len(baseline_inliers)
                or mean_error + 0.03 < baseline_mean_error
            )
        )
        diagnostics.update(
            {
                "status": "accepted" if accepted else "rejected",
                "device": device_label,
                "accepted": accepted,
                "iterations": 48,
                "baseline_inlier_count": int(len(baseline_inliers)),
                "gpu_inlier_count": int(len(inlier_indices)),
                "baseline_mean_error_px": round(baseline_mean_error, 5),
                "gpu_mean_error_px": round(mean_error, 5) if math.isfinite(mean_error) else None,
            }
        )
        if not accepted:
            return {"accepted": False, "diagnostics": diagnostics}
        return {
            "accepted": True,
            "rvec": rotation_vector_numpy,
            "tvec": translation_numpy,
            "inlier_indices": inlier_indices,
            "mean_error": mean_error,
            "camera_points": camera_points_numpy,
            "diagnostics": diagnostics,
        }
    except (RuntimeError, ValueError, cv2.error) as exc:
        # The caller adds the device label to this bounded diagnostic.  Do not
        # let a transient unsupported MPS operation fail the whole localization
        # job; CPU PnP remains the safe fallback.
        diagnostics.update({"device": device_label, "reason": f"gpu-refinement-failed:{type(exc).__name__}"})
        return {"accepted": False, "diagnostics": diagnostics}


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
            hypothesis = _ransac_pnp_hypothesis(
                object_points,
                image_points,
                camera_matrix,
                iterations=1_200,
                reprojection_error=8.0,
            )
            if hypothesis is None:
                continue
            rvec, tvec, pnp_solver = hypothesis

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
                "pnp_solver": pnp_solver,
            }
            # ORB/PnP can rank a large pool on the CPU, but the final local
            # nonlinear polish should still run on the accelerator. Include
            # the closest rejected correspondences in a small bounded pool so
            # the robust GPU loss can recover a missed inlier; passing only
            # the CPU inliers would make GPU polish capable of reducing error
            # but incapable of increasing support. The 32-point cap and the
            # 32px residual cutoff keep ambiguous matches from becoming a
            # second global search or a learned scene-specific shortcut.
            near_indices = np.flatnonzero(residuals <= 32.0)
            if len(near_indices) > 32:
                near_indices = near_indices[np.argsort(residuals[near_indices])[:32]]
            gpu_source_indices = np.asarray(
                sorted(set(int(index) for index in near_indices).union(int(index) for index in inlier_indices)),
                dtype=np.int64,
            )
            if len(gpu_source_indices) > 32:
                gpu_source_indices = gpu_source_indices[np.argsort(residuals[gpu_source_indices])[:32]]
            gpu_refinement = _gpu_pose_refine(
                object_points=all_object_points[gpu_source_indices],
                image_points=all_image_points[gpu_source_indices],
                camera_matrix=camera_matrix,
                rvec=rvec,
                tvec=tvec,
                residual_threshold=8.0,
            ) if len(gpu_source_indices) >= 6 else None
            candidate["gpu_pose_refinement"] = (
                gpu_refinement["diagnostics"]
                if gpu_refinement is not None
                else {"status": "not-accepted-or-unavailable"}
            )
            if gpu_refinement is not None and bool(gpu_refinement.get("accepted")):
                rvec = gpu_refinement["rvec"]
                tvec = gpu_refinement["tvec"]
                inlier_indices = gpu_source_indices[gpu_refinement["inlier_indices"]]
                inlier_object_points = all_object_points[inlier_indices]
                inlier_image_points = all_image_points[inlier_indices]
                candidate.update(
                    {
                        "rvec": rvec,
                        "tvec": tvec,
                        "inlier_indices": inlier_indices,
                        "inlier_count": int(len(inlier_indices)),
                        "mean_error": float(gpu_refinement["mean_error"]),
                        "coverage_ratio": _coverage_ratio(inlier_image_points, frame_width, frame_height),
                        "world_spread_m": float(np.linalg.norm(np.ptp(inlier_object_points, axis=0))),
                    }
                )
            # A low reprojection error alone is not enough to choose the FOV
            # seed. Repeated indoor textures can produce a tight mirrored or
            # upside-down PnP solution which is impossible in the RoomPlan
            # world frame. Prefer hypotheses that already satisfy the same
            # physical prior used by the final acceptance gate, while keeping
            # geometric evidence as the tiebreaker. This only changes which
            # seed gets the later fresh guided/consensus checks; it does not
            # make a pose publishable by itself.
            scene_prior = _pose_scene_prior(
                np.asarray(_camera_to_world(rvec, tvec), dtype=np.float64),
                payload,
            )
            candidate["seed_scene_prior"] = scene_prior
            # Once RANSAC has found the same number of inliers, prefer the
            # camera model that actually reprojects them more tightly.  The
            # previous coverage-first ordering could select a much wider or
            # narrower FOV simply because its six points happened to span a
            # larger image area.
            score = (
                int(bool(scene_prior.get("accepted"))),
                candidate["inlier_count"],
                -candidate["mean_error"],
                candidate["coverage_ratio"],
            )
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
    learned_matcher: LearnedMatcher | None = None,
    query_responses: np.ndarray | None = None,
    landmark_responses: np.ndarray | None = None,
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

    learned_scores: dict[tuple[int, int], float] = {}
    if learned_matcher is not None and by_query:
        query_indices: list[int] = []
        landmark_indices: list[int] = []
        for query_index, options in by_query.items():
            if len(options) < 2:
                continue
            for _descriptor_distance, _pixel_distance, landmark_index in options:
                query_indices.append(query_index)
                landmark_indices.append(landmark_index)
        if query_indices:
            query_array = np.asarray(query_indices, dtype=np.int32)
            landmark_array = np.asarray(landmark_indices, dtype=np.int32)
            scores = learned_matcher.score_pairs(
                descriptors[query_array],
                landmark_descriptors[landmark_array],
                query_responses[query_array] if query_responses is not None else None,
                landmark_responses[landmark_array] if landmark_responses is not None else None,
            )
            learned_scores = {
                (int(query_index), int(landmark_index)): float(score)
                for query_index, landmark_index, score in zip(query_array, landmark_array, scores)
            }

    result: list[Any] = []
    for query_index, options in by_query.items():
        options.sort(key=lambda item: (item[0], item[1]))
        best_distance, best_pixel_distance, landmark_index = options[0]
        learned_score = learned_scores.get((query_index, landmark_index))
        if learned_matcher is not None and len(options) > 1:
            baseline_distance = best_distance
            baseline_score = learned_score
            learned_candidates = [
                (
                    learned_scores.get((query_index, option[2]), -1.0),
                    option,
                )
                for option in options
                if option[0] <= baseline_distance + 12.0
            ]
            learned_candidates.sort(key=lambda item: (item[0], -item[1][0], -item[1][1]), reverse=True)
            if learned_candidates:
                candidate_score, candidate = learned_candidates[0]
                if (
                    candidate_score >= learned_matcher.threshold
                    and (
                        baseline_score is None
                        or baseline_score < learned_matcher.threshold
                        or candidate_score >= baseline_score + 0.08
                    )
                ):
                    best_distance, best_pixel_distance, landmark_index = candidate
                    learned_score = candidate_score
        if len(options) > 1:
            # RoomPlan can observe the same physical surface point from two
            # nearby scan viewpoints. Those observations become separate
            # landmark records with nearly identical 3D coordinates and ORB
            # descriptors. Treating that pair as descriptor ambiguity throws
            # away exactly the cross-view support needed by a fixed webcam.
            # Only compare the winning landmark with a genuinely different
            # 3D point; the final PnP/reprojection/scene gates still decide
            # whether the retained correspondence is geometrically valid.
            best_world = np.asarray(landmark_points[landmark_index], dtype=np.float64)
            distinct_competitor = next(
                (
                    option
                    for option in options
                    if option[2] != landmark_index
                    and float(
                        np.linalg.norm(
                            np.asarray(landmark_points[option[2]], dtype=np.float64) - best_world
                        )
                    ) >= 0.08
                ),
                None,
            )
            if (
                distinct_competitor is not None
                and best_distance >= 0.86 * distinct_competitor[0]
                and best_distance > 32.0
            ):
                competitor_score = learned_scores.get((query_index, distinct_competitor[2]))
                learned_override = bool(
                    learned_matcher is not None
                    and learned_score is not None
                    and learned_score >= learned_matcher.threshold
                    and best_distance <= distinct_competitor[0] + 12.0
                    and (
                        competitor_score is None
                        or learned_score >= competitor_score + 0.08
                    )
                )
                if not learned_override:
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
    minimum_inliers: int = 8,
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
    minimum_inliers = max(6, int(minimum_inliers))
    if len(matches) < minimum_inliers:
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
        if len(inlier_indices) < minimum_inliers:
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
        if len(inlier_indices) < minimum_inliers:
            return None

        inlier_object_points = object_points[inlier_indices]
        inlier_image_points = image_points[inlier_indices]
        world_to_cv, _ = cv2.Rodrigues(rvec)
        camera_space = (world_to_cv @ inlier_object_points.T + tvec.reshape(3, 1)).T
        positive_depth_ratio = float(np.mean(camera_space[:, 2] > 0.05))
        if positive_depth_ratio < 0.9:
            return None
        refined = {
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
        gpu_refinement = (
            _gpu_pose_refine(
                object_points=object_points,
                image_points=image_points,
                camera_matrix=camera_matrix,
                rvec=rvec,
                tvec=tvec,
                residual_threshold=6.0,
            )
            if len(object_points) <= 32 and len(inlier_indices) >= 8
            else None
        )
        refined["gpu_pose_refinement"] = (
            gpu_refinement["diagnostics"]
            if gpu_refinement is not None
            else {"status": "not-accepted-or-unavailable"}
        )
        if gpu_refinement is not None and bool(gpu_refinement.get("accepted")):
            rvec = gpu_refinement["rvec"]
            tvec = gpu_refinement["tvec"]
            inlier_indices = gpu_refinement["inlier_indices"]
            inlier_object_points = object_points[inlier_indices]
            inlier_image_points = image_points[inlier_indices]
            refined.update(
                {
                    "rvec": rvec,
                    "tvec": tvec,
                    "inlier_indices": inlier_indices,
                    "inlier_count": int(len(inlier_indices)),
                    "mean_error": float(gpu_refinement["mean_error"]),
                    "coverage_ratio": _coverage_ratio(inlier_image_points, frame_width, frame_height),
                    "world_spread_m": float(np.linalg.norm(np.ptp(inlier_object_points, axis=0))),
                    "positive_depth_ratio": float(
                        np.mean(gpu_refinement["camera_points"][inlier_indices, 2] > 0.05)
                    ),
                }
            )
        return refined

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
    """Use a small guided match pool only to improve the next projection.

    This intentionally has a lower floor than the real guided refinement, but
    its result is never returned as a localization candidate. It exists only
    to reproject all landmarks more accurately so a subsequent fresh matching
    pass can reach the ordinary eight-match refinement threshold. Small pools
    are especially sensitive to one repeated-texture mismatch, so evaluate the
    full pool and each leave-one-out variant while still requiring six fresh
    correspondences to agree with the recovered provisional pose.
    """
    if len(matches) < 6:
        return None
    object_points = np.asarray([landmark_points[item.trainIdx] for item in matches], dtype=np.float32)
    image_points = np.asarray([keypoints[item.queryIdx].pt for item in matches], dtype=np.float32)
    camera_matrix = np.asarray(pose["camera_matrix"], dtype=np.float64)
    subset_indices = [np.arange(len(matches), dtype=np.int32)]
    if len(matches) >= 7:
        subset_indices.extend(
            np.asarray([index for index in range(len(matches)) if index != omitted], dtype=np.int32)
            for omitted in range(len(matches))
        )

    provisional_candidates: list[dict[str, Any]] = []
    for subset in subset_indices:
        if len(subset) < 6:
            continue
        solve_attempts = (
            (True, cv2.SOLVEPNP_ITERATIVE),
            # Semantic/object seeds can have the right camera center basin but
            # a poor rotation.  A provisional recovery is search-only, so give
            # the same correspondences one seed-independent EPNP attempt before
            # abandoning them.  The recovered pose still has to explain at
            # least six members of the original pool and can never be returned
            # without the ordinary strict guided refinement afterwards.
            (False, cv2.SOLVEPNP_EPNP),
        )
        for use_extrinsic_guess, flags in solve_attempts:
            rvec = np.asarray(pose["rvec"], dtype=np.float64).copy()
            tvec = np.asarray(pose["tvec"], dtype=np.float64).copy()
            try:
                ok, rvec, tvec, ransac_inliers = cv2.solvePnPRansac(
                    object_points[subset],
                    image_points[subset],
                    camera_matrix,
                    np.zeros((4, 1), dtype=np.float64),
                    rvec=rvec,
                    tvec=tvec,
                    useExtrinsicGuess=use_extrinsic_guess,
                    iterationsCount=400,
                    reprojectionError=8.0,
                    confidence=0.995,
                    flags=flags,
                )
            except cv2.error:
                continue
            if not ok:
                continue

            fit_indices = (
                subset[np.asarray(ransac_inliers).reshape(-1)]
                if ransac_inliers is not None else subset
            )

            # Score every original correspondence, including the omitted one.
            # A leave-one-out solve is useful only if at least six observations
            # from the actual pool agree with it; omitting a point never creates
            # extra evidence by itself.
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
            residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            inlier_indices = np.flatnonzero(residuals <= 8.0)
            if len(inlier_indices) < 6:
                continue
            # RANSAC's input pool can contain rejected matches. Record the
            # actual subset used for the final fit, not the size of that pool.
            refinement_indices = inlier_indices.copy()
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points[inlier_indices],
                    image_points[inlier_indices],
                    camera_matrix,
                    np.zeros((4, 1), dtype=np.float64),
                    rvec,
                    tvec,
                )
                fit_indices = refinement_indices
                projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
                residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
                inlier_indices = np.flatnonzero(residuals <= 8.0)
            except cv2.error:
                pass
            if len(inlier_indices) < 6:
                continue

            inlier_object_points = object_points[inlier_indices]
            inlier_image_points = image_points[inlier_indices]
            world_to_cv, _ = cv2.Rodrigues(rvec)
            camera_space = (world_to_cv @ inlier_object_points.T + tvec.reshape(3, 1)).T
            positive_depth_ratio = float(np.mean(camera_space[:, 2] > 0.05))
            if positive_depth_ratio < 0.9:
                continue
            provisional_candidates.append(
                {
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
                    "search_prior_provisional_refinement": True,
                    "provisional_subset_size": int(len(fit_indices)),
                    "provisional_hypothesis_pool_size": int(len(subset)),
                    "provisional_seed_independent": not use_extrinsic_guess,
                }
            )

    if not provisional_candidates:
        return None
    return max(
        provisional_candidates,
        key=lambda candidate: (
            int(candidate["inlier_count"]),
            -float(candidate["mean_error"]),
            float(candidate["coverage_ratio"]),
            int(candidate["provisional_subset_size"]),
        ),
    )


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
    "chair": {"chair", "armchair", "stool"},
    "table": {"table", "desk", "dining table"},
    "storage": {"storage", "cabinet", "shelf", "bookcase", "wardrobe", "dresser", "nightstand"},
    "sofa": {"sofa", "couch"},
}

# Chairs and people move frequently enough that their current image position is
# not evidence of where the fixed camera sits in the RoomPlan scan. Reflective
# openings are unreliable for a different reason: the pixels visible in glass
# can change completely between the iPhone RoomPlan pass and the fixed camera
# even though the room itself did not move. Keep all of these detections for
# masking/diagnostics, but never let them seed a camera pose.
_MOVABLE_SEMANTIC_GROUPS = {"chair"}
_REFLECTIVE_FEATURE_LABELS = {"window", "mirror", "glass", "glass door", "sliding glass door"}
_DYNAMIC_FEATURE_LABELS = {
    "person",
    "people",
    "human",
    "chair",
    "armchair",
    "stool",
    *_REFLECTIVE_FEATURE_LABELS,
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
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[float], float] | None:
    corners = _room_object_corners(room_object)
    rotation, _ = cv2.Rodrigues(rvec)
    camera_space = (rotation @ corners.T + tvec.reshape(3, 1)).T
    positive = camera_space[:, 2] > 0.08
    positive_ratio = float(np.mean(positive))
    if diagnostics is not None:
        depths = camera_space[:, 2]
        diagnostics.update(
            {
                "positive_depth_count": int(np.count_nonzero(positive)),
                "positive_depth_ratio": round(positive_ratio, 6),
                "camera_depths_m": [round(float(value), 6) for value in depths],
                "minimum_camera_depth_m": round(float(np.min(depths)), 6),
                "maximum_camera_depth_m": round(float(np.max(depths)), 6),
            }
        )
    # Large furniture close to a fixed camera can legitimately cross the
    # camera near plane (the live shelf/storage case is a common example).
    # Four front-facing cuboid corners still define a bounded visible face;
    # fewer than four does not provide enough geometry for a useful box.
    if int(np.count_nonzero(positive)) < 4:
        if diagnostics is not None:
            diagnostics["rejection_reason"] = "insufficient-positive-depth-corners"
        return None
    projected, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, None)
    pixels = projected.reshape(-1, 2)[positive]
    if not np.isfinite(pixels).all():
        if diagnostics is not None:
            diagnostics["rejection_reason"] = "non-finite-projection"
        return None
    x1 = max(0.0, min(float(frame_width), float(np.min(pixels[:, 0]))))
    y1 = max(0.0, min(float(frame_height), float(np.min(pixels[:, 1]))))
    x2 = max(0.0, min(float(frame_width), float(np.max(pixels[:, 0]))))
    y2 = max(0.0, min(float(frame_height), float(np.max(pixels[:, 1]))))
    if diagnostics is not None:
        diagnostics["projected_bbox"] = [round(value, 4) for value in (x1, y1, x2, y2)]
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        if diagnostics is not None:
            diagnostics["rejection_reason"] = "projected-bbox-too-small"
        return None
    if diagnostics is not None:
        diagnostics["rejection_reason"] = None
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
    attempts: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    groups: set[str] = set()
    contradictory_groups: set[str] = set()
    for object_index, detection_index in assignment:
        room_object = payload.room_objects[object_index]
        detection = payload.object_detections[detection_index]
        observed_box = [
            max(0.0, min(float(frame_width), float(detection.bbox[0]))),
            max(0.0, min(float(frame_height), float(detection.bbox[1]))),
            max(0.0, min(float(frame_width), float(detection.bbox[2]))),
            max(0.0, min(float(frame_height), float(detection.bbox[3]))),
        ]
        observed_area = max(0.0, observed_box[2] - observed_box[0]) * max(0.0, observed_box[3] - observed_box[1])
        frame_area = max(1.0, float(frame_width * frame_height))
        observed_area_ratio = observed_area / frame_area
        attempt: dict[str, Any] = {
            "object_index": int(object_index),
            "object_id": room_object.id,
            "label": str(room_object.label),
            "object_center": [
                round(float(room_object.center.x), 6),
                round(float(room_object.center.y), 6),
                round(float(room_object.center.z), 6),
            ],
            "object_dimensions": [
                round(float(room_object.dimensions.x), 6),
                round(float(room_object.dimensions.y), 6),
                round(float(room_object.dimensions.z), 6),
            ],
            "detection_index": int(detection_index),
            "detection_label": str(detection.label),
            "detection_confidence": round(float(detection.confidence), 6),
            "detection_bbox": [round(float(value), 4) for value in detection.bbox],
            "observed_area_px2": round(float(observed_area), 4),
            "observed_area_ratio": round(float(observed_area_ratio), 6),
        }
        if detection.frame_index != frame_index:
            attempt["rejection_reason"] = "wrong-frame"
            attempts.append(attempt)
            continue
        projection_diagnostics: dict[str, Any] = {}
        projected = _project_room_object_bbox(
            room_object,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            frame_width=frame_width,
            frame_height=frame_height,
            diagnostics=projection_diagnostics,
        )
        attempt.update(projection_diagnostics)
        if projected is None:
            # A strong detector box is direct evidence that its assigned room
            # object is visible in this frame. If every cuboid corner is behind
            # the camera, the pose contradicts that observation. Keep this
            # deliberately narrower than the generic projection rejection so
            # near-plane furniture with 1-3 visible corners is not treated as
            # equally definitive negative evidence.
            if (
                projection_diagnostics.get("rejection_reason") == "insufficient-positive-depth-corners"
                and int(projection_diagnostics.get("positive_depth_count") or 0) == 0
                and float(detection.confidence) >= 0.50
                and observed_area_ratio >= 0.005
            ):
                group = _object_label_group(room_object.label)
                contradiction = {
                    "object_id": room_object.id,
                    "label": str(room_object.label),
                    "detection_label": str(detection.label),
                    "detection_confidence": round(float(detection.confidence), 6),
                    "observed_area_ratio": round(float(observed_area_ratio), 6),
                    "positive_depth_count": 0,
                    "reason": "visible-detection-fully-behind-camera",
                }
                contradictions.append(contradiction)
                attempt["contradiction"] = True
                attempt["contradiction_reason"] = contradiction["reason"]
                if group is not None:
                    contradictory_groups.add(group)
            attempts.append(attempt)
            continue
        projected_box, positive_depth_ratio = projected
        projected_area = max(0.0, projected_box[2] - projected_box[0]) * max(0.0, projected_box[3] - projected_box[1])
        attempt["projected_area_px2"] = round(float(projected_area), 4)
        if observed_area <= 1.0 or projected_area <= 1.0:
            attempt["rejection_reason"] = "degenerate-area"
            attempts.append(attempt)
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
        attempt.update(
            {
                "rejection_reason": None,
                "iou": round(float(iou), 6),
                "center_score": round(float(center_score), 6),
                "area_ratio": round(float(area_ratio), 6),
                "component": round(float(component), 6),
            }
        )
        attempts.append(attempt)
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
            "contains_movable_semantic_group": False,
            "supported_object_count": 0,
            "supported_group_count": 0,
            "supported_mean_iou": 0.0,
            "supported_minimum_iou": 0.0,
            "contradictory_object_count": len(contradictions),
            "contradictory_group_count": len(contradictory_groups),
            "mean_iou": 0.0,
            "minimum_iou": 0.0,
            "mean_center_score": 0.0,
            "mean_area_ratio": 0.0,
            "matches": [],
            "contradictions": contradictions,
            "attempts": attempts,
        }
    supported_matches = [
        item
        for item in matches
        if float(item["iou"]) >= 0.30 and float(item["positive_depth_ratio"]) >= 0.50
    ]
    supported_groups = {
        group
        for item in supported_matches
        if (group := _object_label_group(str(item["label"]))) is not None
    }
    contains_movable_semantic_group = any(
        _object_label_group(str(item.get("label") or "")) in _MOVABLE_SEMANTIC_GROUPS
        for item in matches
    )
    return {
        "score": round(float(sum(item["component"] for item in matches)), 6),
        "matched_object_count": len(matches),
        "semantic_group_count": len(groups),
        "contains_movable_semantic_group": contains_movable_semantic_group,
        "supported_object_count": len(supported_matches),
        "supported_group_count": len(supported_groups),
        "supported_mean_iou": (
            round(float(np.mean([item["iou"] for item in supported_matches])), 6)
            if supported_matches
            else 0.0
        ),
        "supported_minimum_iou": (
            round(float(min(item["iou"] for item in supported_matches)), 6)
            if supported_matches
            else 0.0
        ),
        "contradictory_object_count": len(contradictions),
        "contradictory_group_count": len(contradictory_groups),
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
        "contradictions": contradictions,
        "attempts": attempts,
    }


def _semantic_cuboid_rank(score: dict[str, Any]) -> tuple[int, int, float, float, int, int, float, float]:
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
    contradiction_free = int(int(score.get("contradictory_object_count") or 0) == 0)
    return (
        contradiction_free,
        int(minimum_iou >= 0.20 and matched_object_count >= 2 and semantic_group_count >= 2),
        float(score.get("mean_iou") or 0.0),
        minimum_iou,
        semantic_group_count,
        matched_object_count,
        float(raw_score or 0.0),
        float(score.get("mean_center_score") or 0.0),
    )


def _semantic_cuboid_has_robust_support(score: dict[str, Any]) -> bool:
    """Accept a semantic basin as a search hint when two groups agree.

    Open-vocabulary detection can add one plausible-sounding but geometrically
    unrelated furniture box.  That outlier must not erase two independent
    RoomPlan objects that project consistently.  This signal remains a
    search/diagnostic hint; it never activates a camera registration by itself.
    """
    return (
        int(score.get("contradictory_object_count") or 0) == 0
        and not bool(score.get("contains_movable_semantic_group"))
        and int(score.get("supported_object_count") or 0) >= 2
        and int(score.get("supported_group_count") or 0) >= 2
        and float(score.get("supported_minimum_iou") or 0.0) >= 0.30
        and float(score.get("supported_mean_iou") or 0.0) >= 0.55
    )


def _semantic_supported_labels(score: dict[str, Any]) -> set[str]:
    matches = score.get("matches")
    if not isinstance(matches, list):
        return set()
    return {
        str(item.get("label") or "").strip().lower()
        for item in matches
        if isinstance(item, dict)
        and str(item.get("label") or "").strip()
        and float(item.get("iou") or 0.0) >= 0.30
        and float(item.get("positive_depth_ratio") or 0.0) >= 0.50
    }


def _semantic_candidate_matrix(candidate: dict[str, Any]) -> np.ndarray | None:
    value = candidate.get("candidate_camera_to_world")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix


def _rotation_delta_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _semantic_cuboid_consensus(
    candidates: list[dict[str, Any]],
    *,
    burst_frame_count: int,
    dynamic_mask_max_ratio: float = 0.0,
) -> dict[str, Any] | None:
    """Return a strict multi-frame RoomPlan-object pose consensus.

    A single cuboid fit is intentionally only a search hint.  A fixed camera
    becomes independently localizable from semantics when several different
    burst frames recover the same *full* pose from at least two RoomPlan object
    groups, with a stable focal-length hypothesis and fresh pose-gated ORB
    support.  This keeps semantic localization separate from the weaker
    one-frame initializer path and avoids lowering any visual-PnP threshold.
    """
    if burst_frame_count < 3:
        return None
    prepared: list[tuple[dict[str, Any], np.ndarray, int, float, set[str]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not _semantic_cuboid_has_robust_support(candidate):
            continue
        matrix = _semantic_candidate_matrix(candidate)
        frame_index = candidate.get("frame_index")
        fov_degrees = candidate.get("selected_fov_degrees")
        if matrix is None or not isinstance(frame_index, int) or not isinstance(fov_degrees, (int, float)):
            continue
        labels = _semantic_supported_labels(candidate)
        if len(labels) < 2:
            continue
        prepared.append((candidate, matrix, frame_index, float(fov_degrees), labels))
    if not prepared:
        return None

    occlusion_relaxed = burst_frame_count >= 4 and dynamic_mask_max_ratio >= 0.45
    minimum_frames = 3 if occlusion_relaxed else (4 if burst_frame_count >= 4 else 3)
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for anchor_candidate, anchor_matrix, _anchor_frame, anchor_fov, _anchor_labels in prepared:
        by_frame: dict[int, tuple[float, float, dict[str, Any], np.ndarray, float, set[str]]] = {}
        for candidate, matrix, frame_index, fov_degrees, labels in prepared:
            if abs(fov_degrees - anchor_fov) > 4.0:
                continue
            translation_delta = float(np.linalg.norm(matrix[:3, 3] - anchor_matrix[:3, 3]))
            rotation_delta = _rotation_delta_degrees(anchor_matrix, matrix)
            if translation_delta > 0.25 or rotation_delta > 5.5:
                continue
            current = by_frame.get(frame_index)
            rank = (translation_delta + 0.03 * rotation_delta, -float(candidate.get("supported_mean_iou") or 0.0))
            if current is None or rank < (current[0], current[1]):
                by_frame[frame_index] = (
                    rank[0],
                    rank[1],
                    candidate,
                    matrix,
                    fov_degrees,
                    labels,
                )
        if len(by_frame) < minimum_frames:
            continue

        members = list(by_frame.values())
        common_labels = set.intersection(*(item[5] for item in members))
        if len(common_labels) < 2:
            continue
        centers = np.asarray([item[3][:3, 3] for item in members], dtype=np.float64)
        median_center = np.median(centers, axis=0)
        center_residuals = np.linalg.norm(centers - median_center.reshape(1, 3), axis=1)
        mean_center_residual = float(np.mean(center_residuals))
        max_center_residual = float(np.max(center_residuals))
        if mean_center_residual > 0.12 or max_center_residual > 0.22:
            continue

        # Choose an observed rotation medoid rather than averaging rotations;
        # only the translation is robustly median-combined. This preserves an
        # actual physically validated camera orientation from the burst.
        medoid_index = min(
            range(len(members)),
            key=lambda index: sum(
                float(np.linalg.norm(members[index][3][:3, 3] - other[3][:3, 3]))
                + 0.03 * _rotation_delta_degrees(members[index][3], other[3])
                for other in members
            ),
        )
        medoid_matrix = members[medoid_index][3]
        rotation_residuals = [_rotation_delta_degrees(medoid_matrix, item[3]) for item in members]
        mean_rotation_residual = float(np.mean(rotation_residuals))
        max_rotation_residual = float(max(rotation_residuals))
        if mean_rotation_residual > 3.5 or max_rotation_residual > 5.5:
            continue

        visual_support_frames = sum(
            int(item[2].get("guided_match_count") or 0) >= 6
            for item in members
        )
        if visual_support_frames < 2:
            continue
        if occlusion_relaxed and len(members) == 3 and visual_support_frames < 3:
            continue
        fovs = sorted(item[4] for item in members)
        fov_span = float(fovs[-1] - fovs[0])
        if fov_span > 4.0:
            continue
        selected_fov = float(fovs[len(fovs) // 2])
        consensus_matrix = medoid_matrix.copy()
        consensus_matrix[:3, 3] = median_center
        mean_supported_iou = float(
            np.mean([float(item[2].get("supported_mean_iou") or 0.0) for item in members])
        )
        support_observation_count = sum(int(item[2].get("supported_object_count") or 0) for item in members)
        matched_observation_count = sum(int(item[2].get("matched_object_count") or 0) for item in members)
        confidence = min(
            0.92,
            0.65
            + 0.025 * len(members)
            + 0.015 * visual_support_frames
            + 0.08 * min(1.0, mean_supported_iou),
        )
        result = {
            "camera_to_world": consensus_matrix,
            "camera_center": [round(float(value), 4) for value in median_center],
            "fov_degrees": selected_fov,
            "frame_count": len(members),
            "frame_indices": sorted(int(item[2]["frame_index"]) for item in members),
            "visual_support_frame_count": visual_support_frames,
            "support_observation_count": support_observation_count,
            "matched_observation_count": matched_observation_count,
            "common_labels": sorted(common_labels),
            "mean_center_residual_m": round(mean_center_residual, 4),
            "max_center_residual_m": round(max_center_residual, 4),
            "mean_rotation_residual_degrees": round(mean_rotation_residual, 3),
            "max_rotation_residual_degrees": round(max_rotation_residual, 3),
            "fov_span_degrees": round(fov_span, 3),
            "mean_supported_iou": round(mean_supported_iou, 6),
            "confidence": round(confidence, 6),
            "occlusion_relaxed_frame_requirement": bool(occlusion_relaxed and len(members) == 3),
            "members": [
                {
                    "frame_index": int(item[2]["frame_index"]),
                    "camera_center": list(item[2].get("camera_center") or []),
                    "selected_fov_degrees": item[2].get("selected_fov_degrees"),
                    "supported_object_count": int(item[2].get("supported_object_count") or 0),
                    "supported_group_count": int(item[2].get("supported_group_count") or 0),
                    "supported_mean_iou": float(item[2].get("supported_mean_iou") or 0.0),
                    "supported_minimum_iou": float(item[2].get("supported_minimum_iou") or 0.0),
                    "guided_match_count": int(item[2].get("guided_match_count") or 0),
                }
                for item in sorted(members, key=lambda item: int(item[2]["frame_index"]))
            ],
        }
        score = (
            len(members),
            visual_support_frames,
            len(common_labels),
            mean_supported_iou,
            -mean_center_residual,
            -mean_rotation_residual,
        )
        if best is None or score > best[0]:
            best = (score, result)
    return None if best is None else best[1]


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
    # Two independent RoomPlan cuboids are enough for a coarse initializer when
    # each contributes its centre and floor-contact anchor.  This is still only
    # a search hint; fresh visual correspondences remain mandatory below.
    if len(assignment) < 2:
        return seed
    if sum(payload.room_objects[object_index].transform is not None for object_index, _ in assignment) < 2:
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


def _semantic_object_assignments(
    payload: CameraLocalizationRequest,
    frame_index: int,
    *,
    allow_movable_seed: bool = False,
) -> list[list[tuple[int, int]]]:
    """Return bounded map-object/detection assignments for one frame.

    YOLO-World may return the same furniture through two synonymous prompts
    (for example ``table`` and ``desk``). Collapse overlapping detections
    inside a semantic group before considering assignments, then enumerate only
    the small ambiguity introduced by repeated furniture of one category.
    """
    map_groups: dict[str, list[int]] = {}
    for object_index, room_object in enumerate(payload.room_objects):
        group = _object_label_group(room_object.label)
        if group is not None and (allow_movable_seed or group not in _MOVABLE_SEMANTIC_GROUPS):
            map_groups.setdefault(group, []).append(object_index)

    detection_candidates: list[tuple[int, Any, str]] = []
    for detection_index, detection in enumerate(payload.object_detections):
        if detection.frame_index != frame_index or detection.confidence < 0.20:
            continue
        if len(detection.bbox) != 4 or not all(math.isfinite(float(value)) for value in detection.bbox):
            continue
        group = _object_label_group(detection.label)
        if group is not None and (allow_movable_seed or group not in _MOVABLE_SEMANTIC_GROUPS):
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

    # The bounded product above intentionally retains the densest full-group
    # assignments. Also enumerate the small group subsets explicitly so a
    # weak third detector cannot hide a strong two-object initializer.
    assignment_candidates = list(assignments)
    for subset_size in range(2, min(len(group_options), 3)):
        for group_indexes in combinations(range(len(group_options)), subset_size):
            subset_assignments: list[tuple[float, list[tuple[int, int]]]] = [(0.0, [])]
            for group_index in group_indexes:
                expanded_subset: list[tuple[float, list[tuple[int, int]]]] = []
                for base_score, base_pairs in subset_assignments:
                    for option_score, option_pairs in group_options[group_index]:
                        expanded_subset.append(
                            (base_score + option_score, [*base_pairs, *option_pairs])
                        )
                expanded_subset.sort(key=lambda item: (len(item[1]), item[0]), reverse=True)
                subset_assignments = expanded_subset[:24]
            assignment_candidates.extend(subset_assignments)

    valid: list[tuple[float, list[tuple[int, int]]]] = []
    for score, pairs in assignment_candidates:
        # Two different semantic groups are a useful coarse initializer even
        # when only one mapped object exists in each group (a common living
        # room has exactly one sofa and one table). The pose-seed stage expands
        # each pair into centre + floor-contact correspondences, so it still
        # has four 3D/2D observations for SQPnP/EPNP. No semantic pose can be
        # published without independent visual PnP evidence.
        if len(pairs) < 2 or len({
            _object_label_group(payload.room_objects[object_index].label)
            for object_index, _detection_index in pairs
        }) < 2:
            continue
        valid.append((score, pairs))
    valid.sort(key=lambda item: (len(item[1]), item[0]), reverse=True)
    # Keep one representative for each semantic-group subset before filling
    # the remaining budget by score. A weak desk/table box should not force a
    # good stationary-chair + sofa pair to share a six-point PnP seed with it.
    # Every returned assignment is still only an initializer; fresh visual
    # correspondences and the normal multi-view gate remain mandatory.
    representatives: dict[frozenset[str], tuple[float, list[tuple[int, int]]]] = {}
    for score, pairs in valid:
        groups = frozenset(
            _object_label_group(payload.room_objects[object_index].label)
            for object_index, _detection_index in pairs
        )
        if groups and groups not in representatives:
            representatives[groups] = (score, pairs)
    selected: list[list[tuple[int, int]]] = [pairs for _score, pairs in representatives.values()]
    selected_signatures = {tuple(pairs) for pairs in selected}
    for _score, pairs in valid:
        signature = tuple(pairs)
        if signature in selected_signatures:
            continue
        selected.append(pairs)
        selected_signatures.add(signature)
        if len(selected) >= 8:
            break
    return selected[:8]


def _stable_movable_object_detection(payload: CameraLocalizationRequest) -> bool:
    """Allow movable furniture only as a stable burst search initializer.

    A chair is never enough to publish a pose. It can help enter the right
    PnP basin only when the detector sees the same movable group in most of the
    fixed-camera burst with bounded image motion. This makes the exception
    useful for a static chair in the live view while preserving the existing
    multi-view semantic acceptance gate.
    """
    frame_count = len(payload.frames)
    if frame_count < 3:
        return False
    minimum_frames = max(3, int(math.ceil(frame_count * 0.60)))
    for group in _MOVABLE_SEMANTIC_GROUPS:
        observations_by_frame: dict[int, list[tuple[float, float, float, float, float]]] = {}
        for detection in payload.object_detections:
            # Temporal agreement can recover a partially occluded static chair
            # whose per-frame detector confidence is modest. Keep a separate
            # strong-confidence requirement below so one weak, repeated false
            # positive cannot unlock a movable semantic initializer.
            if _object_label_group(detection.label) != group or detection.confidence < 0.15:
                continue
            if len(detection.bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in detection.bbox)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                continue
            width = max(1.0, float(payload.frames[detection.frame_index].width)) if 0 <= detection.frame_index < frame_count else 1.0
            height = max(1.0, float(payload.frames[detection.frame_index].height)) if 0 <= detection.frame_index < frame_count else 1.0
            observations_by_frame.setdefault(int(detection.frame_index), []).append(
                (
                    ((x1 + x2) / 2.0) / width,
                    ((y1 + y2) / 2.0) / height,
                    max(0.0, x2 - x1) / width,
                    max(0.0, y2 - y1) / height,
                    float(detection.confidence),
                )
            )

        # A frame can contain more than one chair. Selecting the largest box
        # per frame is not temporal tracking: a partial chair at the edge of
        # one frame can displace the actual stationary chair and make the
        # otherwise stable burst look inconsistent. Treat each observation as
        # a bounded cluster proposal, choose at most one nearby observation per
        # frame, and accept the best cluster that spans most of the burst.
        prototypes = [observation for items in observations_by_frame.values() for observation in items]
        best_cluster: tuple[int, float, float] | None = None
        for prototype in prototypes:
            selected: list[tuple[float, float, float, float, float]] = []
            for items in observations_by_frame.values():
                nearby = [
                    item
                    for item in items
                    if float(np.linalg.norm(np.asarray(item[:2]) - np.asarray(prototype[:2]))) <= 0.08
                    and float(np.linalg.norm(np.asarray(item[2:4]) - np.asarray(prototype[2:4]))) <= 0.16
                ]
                if nearby:
                    selected.append(
                        min(
                            nearby,
                            key=lambda item: (
                                float(np.linalg.norm(np.asarray(item[:2]) - np.asarray(prototype[:2])))
                                + float(np.linalg.norm(np.asarray(item[2:4]) - np.asarray(prototype[2:4]))),
                                -item[4],
                            ),
                        )
                    )
            if len(selected) < minimum_frames:
                continue
            values = np.asarray([item[:4] for item in selected], dtype=np.float64)
            center_spread = float(
                np.max(np.linalg.norm(values[:, :2] - np.median(values[:, :2], axis=0), axis=1))
            )
            size_spread = float(
                np.max(np.linalg.norm(values[:, 2:] - np.median(values[:, 2:], axis=0), axis=1))
            )
            if center_spread > 0.08 or size_spread > 0.16:
                continue
            strong_observation_count = sum(item[4] >= 0.35 for item in selected)
            if strong_observation_count < max(2, int(math.ceil(minimum_frames * 0.50))):
                continue
            cluster = (len(selected), float(np.mean(values[:, 0])), float(np.mean(values[:, 1])))
            if best_cluster is None or cluster[0] > best_cluster[0]:
                best_cluster = cluster
        if best_cluster is not None:
            return True
    return False


def _semantic_object_pose_seeds(
    payload: CameraLocalizationRequest,
    *,
    frame_index: int,
    frame_width: int,
    frame_height: int,
    allow_movable_seed: bool = False,
) -> list[dict[str, Any]]:
    """Estimate bounded pose seeds from recognized native RoomPlan objects."""
    assignments = _semantic_object_assignments(
        payload,
        frame_index,
        allow_movable_seed=allow_movable_seed,
    )
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
        requested = float(payload.fov_degrees) if payload.fov_degrees is not None else None
        fov_hypotheses = [
            item
            for item in fov_hypotheses
            if item[2] is not None
            and (
                (requested is not None and abs(float(item[2]) - requested) <= 0.5)
                or abs(float(item[2]) - 54.0) <= 0.5
                or abs(float(item[2]) - 74.0) <= 0.5
                or abs(float(item[2]) - 96.0) <= 0.5
                or (preferred is not None and abs(float(item[2]) - float(preferred)) <= 0.5)
            )
        ]

    seeds: list[dict[str, Any]] = []
    for assignment in assignments:
        for anchor_kind in ("center", "floor-contact", "center+floor-contact"):
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
                center_image = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                floor_center = center.copy()
                floor_center[1] -= dimensions_y / 2.0
                floor_image = [(x1 + x2) / 2.0, y2]
                if anchor_kind == "floor-contact":
                    object_points.append(floor_center.tolist())
                    image_points.append(floor_image)
                elif anchor_kind == "center+floor-contact":
                    object_points.extend((center.tolist(), floor_center.tolist()))
                    image_points.extend((center_image, floor_image))
                else:
                    object_points.append(center.tolist())
                    image_points.append(center_image)
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
                    # Three points can have several exactly fitting poses.
                    # SQPnP may return only an upside-down branch even when a
                    # physically valid root exists. AP3P enumerates those roots
                    # so the unchanged scene gates can choose plausible seeds.
                    try:
                        ok, rvecs, tvecs = cv2.solveP3P(
                            object_array,
                            image_array,
                            camera_matrix,
                            np.zeros((4, 1), dtype=np.float64),
                            flags=cv2.SOLVEPNP_AP3P,
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
                            # Center and floor-contact anchors from the same
                            # detector box are correlated, not independent
                            # object observations. Do not double their support.
                            "semantic_object_inlier_count": len({
                                int(index) // (2 if anchor_kind == "center+floor-contact" else 1)
                                for index in inlier_indices
                            }),
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
    preferred_fov = payload.search_prior.fov_degrees if payload.search_prior else payload.fov_degrees
    if preferred_fov is None:
        preferred_fov = payload.fov_degrees
    seeds.sort(
        key=lambda item: (
            _semantic_cuboid_rank(item.get("semantic_cuboid") or {}),
            item["semantic_object_inlier_count"],
            # Minimal solutions can all fit to floating-point precision at
            # different FOVs. Sub-millipixel noise is not calibration evidence;
            # use the supplied hint only to break these equally fitting cases.
            -round(float(item["mean_error"]), 3),
            -abs(float(item["fov_degrees"]) - float(preferred_fov))
            if preferred_fov is not None and item["fov_degrees"] is not None else 0.0,
            item["semantic_object_anchor"] == "center",
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


def _room_bounded_landmark_indices(
    payload: CameraLocalizationRequest,
    landmark_points: np.ndarray,
    *,
    margin_m: float = 0.75,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prefer landmarks belonging to the scanned room, with a safe fallback.

    LiDAR/RGB scan frames can see through doors and windows, so depth-backed
    landmarks may legitimately extend well beyond the RoomPlan floor polygon.
    Those points are poor fixed-camera localization evidence because the final
    camera itself must lie in the scanned room. Keep a generous wall/opening
    margin, and disable the filter automatically when it would remove most of
    a small or unusually shaped scan.
    """
    total = int(len(landmark_points))
    all_indices = np.arange(total, dtype=np.int32)
    if total == 0 or not payload.room_zones:
        return all_indices, {
            "applied": False,
            "reason": "room_bounds_unavailable",
            "total_landmark_count": total,
            "eligible_landmark_count": total,
            "retained_ratio": 1.0 if total else 0.0,
            "margin_m": margin_m,
        }

    zones: list[tuple[float, list[tuple[float, float]]]] = []
    for zone in payload.room_zones:
        polygon = [(float(point.x), float(point.z)) for point in zone.polygon]
        if len(polygon) >= 3:
            zones.append((float(zone.floor_y), polygon))
    if not zones:
        return all_indices, {
            "applied": False,
            "reason": "room_bounds_invalid",
            "total_landmark_count": total,
            "eligible_landmark_count": total,
            "retained_ratio": 1.0 if total else 0.0,
            "margin_m": margin_m,
        }

    eligible: list[int] = []
    for index, point in enumerate(landmark_points):
        x, y, z = (float(value) for value in point)
        if any(
            floor_y - 0.50 <= y <= floor_y + 4.00
            and _polygon_distance(x, z, polygon) <= margin_m
            for floor_y, polygon in zones
        ):
            eligible.append(index)

    minimum_safe_count = max(40, int(math.ceil(total * 0.35)))
    apply_filter = len(eligible) >= minimum_safe_count
    retained = len(eligible) if apply_filter else total
    return (
        np.asarray(eligible, dtype=np.int32) if apply_filter else all_indices,
        {
            "applied": apply_filter,
            "reason": "room_bounds" if apply_filter else "fallback_preserved_full_landmark_set",
            "total_landmark_count": total,
            "eligible_landmark_count": len(eligible),
            "filtered_landmark_count": total - len(eligible) if apply_filter else 0,
            "retained_ratio": round(retained / max(1, total), 6),
            "margin_m": margin_m,
        },
    )


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


def _support_surface_search_centers(
    payload: CameraLocalizationRequest,
    *,
    max_centers: int = 24,
) -> list[np.ndarray]:
    """Sample generic fixed-camera centers on RoomPlan support surfaces.

    A fixed household camera is often placed on a table or desk rather than
    against a wall.  The object is only a bounded search prior: every sampled
    center still has to recover its orientation from fresh image matches and
    pass the ordinary multi-view geometric gates.  Do not use detector boxes
    or a scene-specific coordinate here; this is derived only from the native
    RoomPlan cuboid and therefore generalizes to a new room.
    """
    if max_centers <= 0:
        return []
    raw: list[np.ndarray] = []
    support_heights = (0.08, 0.24, 0.42)
    # Cover the center and the four cardinal placements without exploding the
    # fixed-center PnP budget.  The offsets are fractions of the actual
    # cuboid footprint, so a small side table and a large desk are treated
    # proportionally.
    surface_offsets = (
        (0.0, 0.0),
        (-0.38, 0.0),
        (0.38, 0.0),
        (0.0, -0.38),
        (0.0, 0.38),
    )
    for room_object in payload.room_objects:
        if _object_label_group(room_object.label) != "table":
            continue
        if float(room_object.confidence) < 0.45:
            continue
        dimensions = np.asarray(
            [float(room_object.dimensions.x), float(room_object.dimensions.y), float(room_object.dimensions.z)],
            dtype=np.float64,
        )
        center = np.asarray(
            [float(room_object.center.x), float(room_object.center.y), float(room_object.center.z)],
            dtype=np.float64,
        )
        if not np.isfinite(dimensions).all() or not np.isfinite(center).all():
            continue
        if dimensions[0] <= 0.25 or dimensions[1] <= 0.15 or dimensions[2] <= 0.25:
            continue
        rotation = np.eye(3, dtype=np.float64)
        transform = getattr(room_object, "transform", None)
        if transform is not None:
            try:
                raw_rotation = np.asarray(transform.values, dtype=np.float64)[:3, :3]
                u, _singular_values, vt = np.linalg.svd(raw_rotation)
                rotation = u @ vt
                if np.linalg.det(rotation) < 0.0:
                    u[:, -1] *= -1.0
                    rotation = u @ vt
            except (TypeError, ValueError, np.linalg.LinAlgError):
                rotation = np.eye(3, dtype=np.float64)
        half_x = dimensions[0] * 0.5
        half_z = dimensions[2] * 0.5
        surface_y = center[1] + dimensions[1] * 0.5
        for height in support_heights:
            for offset_x, offset_z in surface_offsets:
                local = np.asarray([offset_x * half_x, 0.0, offset_z * half_z], dtype=np.float64)
                position = center + rotation @ local
                position[1] = surface_y + height
                raw.append(position)

    unique: list[np.ndarray] = []
    for center in raw:
        if all(float(np.linalg.norm(center - existing)) >= 0.12 for existing in unique):
            unique.append(center)
    if len(unique) <= max_centers:
        return unique
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
    """Use a person standing on known floor points as calibration markers."""
    if not payload.person_anchors:
        return None, {"status": "not_requested"}

    grouped: dict[tuple[float, float, float], list[int]] = {}
    for anchor in payload.person_anchors:
        if anchor.frame_index < len(payload.frames):
            point = (round(float(anchor.x), 4), round(float(anchor.y), 4), round(float(anchor.z), 4))
            grouped.setdefault(point, []).append(int(anchor.frame_index))
    if len(grouped) < 4:
        return None, {"status": "insufficient_targets", "target_count": len(grouped), "required_target_count": 4}

    targets = list(grouped.items())[:8]
    frame0 = payload.frames[targets[0][1][0]]
    object_points = np.asarray([point for point, _ in targets], dtype=np.float64)
    floor_points = object_points[:, [0, 2]]
    floor_spread = max((float(np.linalg.norm(a - b)) for a, b in combinations(floor_points, 2)), default=0.0)
    if floor_spread < 0.90:
        return None, {"status": "targets_too_close", "target_count": len(targets), "floor_spread_m": round(floor_spread, 4)}
    floor_hull_area = float(cv2.contourArea(cv2.convexHull(floor_points.astype(np.float32)).reshape(-1, 1, 2)))
    if floor_hull_area < 0.20:
        return None, {
            "status": "targets_poorly_conditioned",
            "target_count": len(targets),
            "floor_spread_m": round(floor_spread, 4),
            "floor_hull_area_m2": round(floor_hull_area, 4),
        }

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
        clusters: list[list[dict[str, float]]] = []
        for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
            matching = next((
                cluster for cluster in clusters
                if math.hypot(
                    candidate["x"] - float(np.median([item["x"] for item in cluster])),
                    candidate["y"] - float(np.median([item["y"] for item in cluster])),
                ) <= 0.065
            ), None)
            if matching is None:
                clusters.append([candidate])
            else:
                matching.append(candidate)
        representatives = [{
            "x": float(np.median([item["x"] for item in cluster])),
            "y": float(np.median([item["y"] for item in cluster])),
            "confidence": float(np.mean([item["confidence"] for item in cluster])),
            "support": float(len(cluster)),
        } for cluster in clusters]
        representatives.sort(key=lambda item: (item["support"], item["confidence"]), reverse=True)
        choices.append(representatives[:2])

    best: dict[str, Any] | None = None
    assignment_count = 0
    tested_fovs: set[float] = set()

    def evaluate_camera_matrix(camera_matrix: np.ndarray, intrinsics_source: str, fov_degrees: float | None) -> None:
        nonlocal best, assignment_count
        if fov_degrees is not None:
            tested_fovs.add(round(float(fov_degrees), 3))
        for assignment in product(*choices):
            assignment_count += 1
            image_points = np.asarray(
                [[item["x"] * frame0.width, item["y"] * frame0.height] for item in assignment],
                dtype=np.float64,
            )
            try:
                ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    camera_matrix,
                    None,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error:
                continue
            if not ok:
                continue
            for raw_rvec, raw_tvec in zip(rvecs, tvecs):
                rvec = np.asarray(raw_rvec, dtype=np.float64).copy()
                tvec = np.asarray(raw_tvec, dtype=np.float64).copy()
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, None, rvec, tvec)
                except cv2.error:
                    pass
                projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
                residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
                matrix = np.asarray(_camera_to_world(rvec, tvec), dtype=np.float64)
                scene_prior = _pose_scene_prior(matrix, payload)
                rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
                camera_points = (rotation @ object_points.T + np.asarray(tvec, dtype=np.float64).reshape(3, 1)).T
                positive_depth_ratio = float(np.mean(camera_points[:, 2] > 0.05))
                candidate = {
                    "matrix": matrix,
                    "camera_matrix": camera_matrix,
                    "intrinsics_source": intrinsics_source,
                    "fov_degrees": fov_degrees,
                    "scene_prior": scene_prior,
                    "mean_error": float(np.mean(residuals)),
                    "max_error": float(np.max(residuals)),
                    "residuals": [float(value) for value in residuals],
                    "image_points": image_points,
                    "positive_depth_ratio": positive_depth_ratio,
                    "person_confidence": float(np.mean([item["confidence"] for item in assignment])),
                    "person_support": float(np.mean([item["support"] for item in assignment])),
                }
                rank = (
                    int(bool(scene_prior.get("accepted"))),
                    positive_depth_ratio,
                    -candidate["mean_error"],
                    -candidate["max_error"],
                    candidate["person_support"],
                    candidate["person_confidence"],
                )
                if best is None or rank > best["rank"]:
                    best = {**candidate, "rank": rank}

    for camera_matrix, intrinsics_source, fov_degrees in _camera_matrix_candidates(payload, frame0.width, frame0.height):
        evaluate_camera_matrix(camera_matrix, intrinsics_source, fov_degrees)

    if payload.intrinsics is None and best is not None and best.get("fov_degrees") is not None:
        coarse_fov = float(best["fov_degrees"])
        for offset in (-6.0, -3.0, -1.5, -0.75, 0.75, 1.5, 3.0, 6.0):
            fov = round(coarse_fov + offset, 3)
            if fov < 30.0 or fov > 120.0 or fov in tested_fovs:
                continue
            focal = 0.5 * frame0.width / math.tan(math.radians(fov) / 2.0)
            camera_matrix = np.asarray(
                [[focal, 0.0, frame0.width / 2.0], [0.0, focal, frame0.height / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            evaluate_camera_matrix(camera_matrix, "estimated-fov-sweep", fov)

    if best is None:
        return None, {"status": "solve_failed", "assignment_count": assignment_count, "person_detection_count": detection_count}

    accepted = bool(best["scene_prior"].get("accepted")) and best["positive_depth_ratio"] >= 1.0 and best["mean_error"] <= 35.0 and best["max_error"] <= 80.0
    diagnostics = {
        "status": "accepted" if accepted else "rejected",
        "target_count": len(targets),
        "person_detection_count": detection_count,
        "assignment_count": assignment_count,
        "floor_spread_m": round(floor_spread, 4),
        "floor_hull_area_m2": round(floor_hull_area, 4),
        "mean_reprojection_error_px": round(best["mean_error"], 4),
        "max_reprojection_error_px": round(best["max_error"], 4),
        "per_target_reprojection_error_px": [round(value, 4) for value in best["residuals"]],
        "mean_person_confidence": round(best["person_confidence"], 4),
        "mean_person_frame_support": round(best["person_support"], 4),
        "selected_fov_degrees": round(float(best["fov_degrees"]), 3) if best.get("fov_degrees") is not None else None,
        "tested_fov_degrees": sorted(tested_fovs),
        "scene_prior": best["scene_prior"],
    }
    if not accepted:
        return None, diagnostics

    matrix = best["matrix"]
    confidence = max(
        0.55,
        min(0.97, 0.84 - best["mean_error"] / 220.0 + 0.08 * best["person_confidence"] + 0.02 * min(4, len(targets) - 4)),
    )
    return {
        "status": "positioned",
        "coordinate_frame": "roomplan-local",
        "camera_to_world": [[round(float(value), 8) for value in row] for row in matrix],
        "confidence": round(confidence, 6),
        "inlier_count": len(targets),
        "match_count": len(targets),
        "reprojection_error_px": round(best["mean_error"], 6),
        "intrinsics_source": best["intrinsics_source"],
        "intrinsics": [[round(float(value), 8) for value in row] for row in best["camera_matrix"]],
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
            "pnp_solver": pose.get("pnp_solver"),
            "gpu_pose_refinement": dict(pose.get("gpu_pose_refinement") or {}),
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
            "support_surface_search_attempted": bool(pose.get("support_surface_search_attempted")),
            "support_surface_search_center_count": int(pose.get("support_surface_search_center_count") or 0),
            "support_surface_search_hypothesis_count": int(pose.get("support_surface_search_hypothesis_count") or 0),
            "support_surface_search_hypothesis_stats": list(pose.get("support_surface_search_hypothesis_stats") or []),
            "support_surface_search_recovery": bool(pose.get("support_surface_search_recovery")),
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


def _person_feature_mask(
    payload: CameraLocalizationRequest,
    *,
    frame_index: int,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, float | None, bool]:
    """Mask transient or visually unstable regions before static-room matching.

    The legacy helper name is retained for callers/tests. People use the lower
    confidence floor because missing even a partial bystander can corrupt ORB.
    Large windows/mirrors use a slightly higher floor so an uncertain
    open-vocabulary box cannot erase most of the image. Chairs require a normal
    semantic confidence before being excluded.
    Returns ``(mask, masked_ratio, fully_occluded)``.
    """
    def mask_confidence_floor(label: str) -> float:
        if label in {"person", "people", "human"}:
            return 0.05
        if label in _REFLECTIVE_FEATURE_LABELS:
            # Open-vocabulary detectors often call framed artwork or dark
            # furniture a mirror/window.  Do not erase that static texture
            # from PnP unless the reflective classification is reasonably
            # strong; true glass still remains masked at the higher floor.
            return 0.35
        return 0.20

    dynamic_boxes = [
        detection.bbox
        for detection in payload.object_detections
        if detection.frame_index == frame_index
        and detection.label.strip().lower() in _DYNAMIC_FEATURE_LABELS
        and detection.confidence >= mask_confidence_floor(detection.label.strip().lower())
    ]
    if not dynamic_boxes:
        return None, None, False

    mask = np.full((height, width), 255, dtype=np.uint8)
    for box in dynamic_boxes:
        x1, y1, x2, y2 = (float(value) for value in box)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            continue
        pad_x = max(12.0, (x2 - x1) * 0.08)
        pad_y = max(12.0, (y2 - y1) * 0.08)
        left = max(0, min(width - 1, int(math.floor(x1 - pad_x))))
        top = max(0, min(height - 1, int(math.floor(y1 - pad_y))))
        right = max(left + 1, min(width, int(math.ceil(x2 + pad_x))))
        bottom = max(top + 1, min(height, int(math.ceil(y2 + pad_y))))
        mask[top:bottom, left:right] = 0

    masked_ratio = float(np.mean(mask == 0))
    if masked_ratio <= 0.0:
        return None, None, False
    # Do not discard a frame just because a person occupies most of it. A
    # close bystander can leave only a narrow but still useful strip of static
    # room, and ORB/PnP can decide whether that strip contains enough evidence.
    # Only call the frame fully occluded when the mask leaves literally no
    # usable pixels at all.
    if not np.any(mask):
        return None, masked_ratio, True
    return mask, masked_ratio, False


def localize_camera(
    payload: CameraLocalizationRequest,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    # Keep the expensive differentiable polish bounded per request.  The
    # detector and CPU PnP remain fully concurrent; only a couple of finalists
    # may enter the serialized accelerator stage.
    _SOLVER_THREAD_STATE.gpu_pose_refinement_budget = 2
    _SOLVER_THREAD_STATE.gpu_pose_refinement_attempts = []

    def report(progress: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(max(1, min(100, int(progress))), stage)

    def low_light_summary() -> dict[str, Any]:
        return {
            "enabled": True,
            "preprocessed_frame_count": int(sum(bool(item.get("low_light")) for item in low_light_frames)),
            "frames": list(low_light_frames),
        }

    report(1, "Preparing fixed-camera reference frames")
    guided_person_result, guided_person_diagnostics = _guided_person_calibration(payload)
    if guided_person_result is not None:
        report(100, "Camera pose solved")
        return guided_person_result

    (
        landmark_points,
        landmark_descriptors,
        landmark_view_ids,
        landmark_sift_descriptors,
        landmark_sift_available,
    ) = _landmark_feature_arrays(payload)
    landmark_indices, landmark_scene_filter = _room_bounded_landmark_indices(payload, landmark_points)
    if len(landmark_indices) != len(landmark_points):
        landmark_points = landmark_points[landmark_indices]
        landmark_descriptors = landmark_descriptors[landmark_indices]
        landmark_view_ids = [landmark_view_ids[int(index)] for index in landmark_indices]
        landmark_sift_descriptors = landmark_sift_descriptors[landmark_indices]
        landmark_sift_available = landmark_sift_available[landmark_indices]
        response_source = [payload.landmarks[int(index)] for index in landmark_indices]
    else:
        response_source = list(payload.landmarks)
    landmark_responses = np.asarray(
        [max(0.0, float(landmark.response)) for landmark in response_source],
        dtype=np.float32,
    )
    descriptor_family_diagnostics = {
        "orb_landmark_count": int(len(landmark_descriptors)),
        "sift_landmark_count": int(np.count_nonzero(landmark_sift_available)),
        "sift_matching_enabled": bool(np.count_nonzero(landmark_sift_available) >= 6),
        "legacy_visual_index": bool(np.count_nonzero(landmark_sift_available) == 0),
        "orb_pnp_hypothesis_backend": "opencv-cpu-ransac-ensemble",
        "pose_polish_backend": "torch-mps-or-cuda-when-available",
    }
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
    report(
        8,
        (
            f"Loaded {len(landmark_points)} room-bounded RoomPlan landmarks"
            if landmark_scene_filter["applied"]
            else "Loaded RoomPlan visual landmarks"
        ),
    )
    orb = cv2.ORB_create(nfeatures=3400, scaleFactor=1.2, nlevels=8, fastThreshold=5, edgeThreshold=17)
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.012) if hasattr(cv2, "SIFT_create") else None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    sift_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) if sift is not None else None
    candidates: list[dict[str, Any]] = []
    semantic_cuboid_candidates: list[dict[str, Any]] = []
    allow_movable_semantic_seed = _stable_movable_object_detection(payload)
    best_match_count = 0
    best_feature_count = 0
    best_min_distance: float | None = None
    person_masked_frame_count = 0
    person_masked_area_ratios: list[float] = []
    person_occluded_frame_count = 0
    low_light_frames: list[dict[str, Any]] = []
    room_center_search_used = False
    support_surface_search_used = False
    support_surface_search_summary: dict[str, Any] = {
        "room_object_labels": [str(room_object.label).strip().lower() for room_object in payload.room_objects],
        "support_surface_object_count": 0,
        "center_count": 0,
        "attempted": False,
        "hypothesis_count": 0,
    }
    if payload.room_objects:
        support_surface_search_summary["support_surface_object_count"] = sum(
            _object_label_group(room_object.label) == "table" and float(room_object.confidence) >= 0.45
            for room_object in payload.room_objects
        )
    view_groups: dict[str, list[int]] = {}
    for index, view_id in enumerate(landmark_view_ids):
        view_groups.setdefault(view_id, []).append(index)
    grouped_landmarks = [
        (view_id, np.asarray(indices, dtype=np.int32))
        for view_id, indices in sorted(view_groups.items())
        if len(indices) >= 6
    ]
    all_landmark_indices = np.arange(len(landmark_points), dtype=np.int32)
    burst_match_observations: dict[str, dict[int, list[tuple[int, float, float, float]]]] = {}

    def record_burst_matches(view_id: str, frame_index: int, keypoints: list[Any], matches: list[Any]) -> None:
        view_observations = burst_match_observations.setdefault(view_id, {})
        for match in matches:
            if not (0 <= int(match.queryIdx) < len(keypoints)):
                continue
            x, y = keypoints[int(match.queryIdx)].pt
            view_observations.setdefault(int(match.trainIdx), []).append(
                (frame_index, float(x), float(y), float(match.distance))
            )

    frame_count = max(1, len(payload.frames))
    for frame_index, frame in enumerate(payload.frames):
        frame_start_progress = 10 + int(68 * frame_index / frame_count)
        frame_end_progress = 10 + int(68 * (frame_index + 1) / frame_count)
        report(frame_start_progress, f"Extracting features from reference frame {frame_index + 1} of {frame_count}")
        jpeg = _jpeg_bytes(frame.frame_base64)
        try:
            image = decode_jpeg(jpeg, frame.width, frame.height)
        except VisionInferenceError as exc:
            raise LocalizationInputError(str(exc)) from exc
        gray, light_diagnostics = prepare_feature_gray(image)
        low_light_frames.append({"frame_index": frame_index, **light_diagnostics})
        dynamic_mask, masked_ratio, person_occluded = _person_feature_mask(
            payload,
            frame_index=frame_index,
            width=frame.width,
            height=frame.height,
        )
        if masked_ratio is not None:
            person_masked_area_ratios.append(masked_ratio)
        if person_occluded:
            person_occluded_frame_count += 1
            continue
        if dynamic_mask is not None and masked_ratio is not None:
            person_masked_frame_count += 1
        keypoints, descriptors = orb.detectAndCompute(gray, dynamic_mask)
        sift_keypoints: list[Any] = []
        sift_query_descriptors: np.ndarray | None = None
        if sift is not None:
            sift_keypoints, sift_query_descriptors = sift.detectAndCompute(gray, dynamic_mask)
        if (descriptors is None or len(keypoints) < 6) and (
            sift_query_descriptors is None or len(sift_keypoints) < 6
        ):
            report(frame_end_progress, f"Reference frame {frame_index + 1} had too few stable features")
            continue
        best_feature_count = max(best_feature_count, len(keypoints) + len(sift_keypoints))
        # Initial PnP may use both descriptor families. Guided refinement keeps
        # using the ORB set because its spatially gated matcher and learned
        # scorer operate on the legacy ORB index; SIFT correspondences still
        # supply additional independent 2D-3D hypotheses.
        pose_keypoints = [*keypoints, *sift_keypoints]
        query_responses = np.asarray([max(0.0, float(keypoint.response)) for keypoint in keypoints], dtype=np.float32)
        report(
            min(frame_end_progress - 1, frame_start_progress + max(1, (frame_end_progress - frame_start_progress) // 3)),
            f"Matching reference frame {frame_index + 1} to RoomPlan landmarks",
        )

        # Use the real local detector's furniture boxes as a bounded pose seed
        # when the native RoomPlan map contains recognizable objects. The seed
        # itself is never publishable: it must expand into fresh ORB matches
        # and pass the same PnP, reprojection, room, and cross-view checks.
        semantic_seeds = _semantic_object_pose_seeds(
            payload,
            frame_index=frame_index,
            frame_width=frame.width,
            frame_height=frame.height,
            allow_movable_seed=allow_movable_semantic_seed,
        )
        for semantic_seed in semantic_seeds:
            semantic_matrix = np.asarray(_camera_to_world(semantic_seed["rvec"], semantic_seed["tvec"]), dtype=np.float64)
            semantic_cuboid = semantic_seed.get("semantic_cuboid") if isinstance(semantic_seed.get("semantic_cuboid"), dict) else {}
            semantic_candidate_diagnostics = {
                "kind": "semantic-cuboid",
                "frame_index": frame_index,
                "camera_center": [round(float(semantic_matrix[index, 3]), 4) for index in range(3)],
                # Diagnostic only. This pose is never returned as the active
                # camera registration unless it later earns ordinary geometric
                # support and passes the normal positioned gates. Keeping the
                # matrix here lets review tooling render the rejected semantic
                # hypothesis without confusing it with a saved pose.
                "candidate_camera_to_world": [
                    [round(float(value), 8) for value in row]
                    for row in semantic_matrix
                ],
                "selected_fov_degrees": semantic_seed.get("fov_degrees"),
                "match_count": int(semantic_seed.get("semantic_object_match_count") or 0),
                "labels": list(semantic_seed.get("semantic_object_labels") or []),
                "anchor": semantic_seed.get("semantic_object_anchor"),
                "confidence": float(semantic_seed.get("semantic_object_confidence") or 0.0),
                "cuboid_score": float(semantic_cuboid.get("score") or 0.0),
                "mean_iou": float(semantic_cuboid.get("mean_iou") or 0.0),
                "minimum_iou": float(semantic_cuboid.get("minimum_iou") or 0.0),
                "supported_object_count": int(semantic_cuboid.get("supported_object_count") or 0),
                "supported_group_count": int(semantic_cuboid.get("supported_group_count") or 0),
                "supported_mean_iou": float(semantic_cuboid.get("supported_mean_iou") or 0.0),
                "supported_minimum_iou": float(semantic_cuboid.get("supported_minimum_iou") or 0.0),
                "contradictory_object_count": int(semantic_cuboid.get("contradictory_object_count") or 0),
                "contradictory_group_count": int(semantic_cuboid.get("contradictory_group_count") or 0),
                "mean_center_score": float(semantic_cuboid.get("mean_center_score") or 0.0),
                "mean_area_ratio": float(semantic_cuboid.get("mean_area_ratio") or 0.0),
                "matched_object_count": int(semantic_cuboid.get("matched_object_count") or 0),
                "semantic_group_count": int(semantic_cuboid.get("semantic_group_count") or 0),
                "contains_movable_semantic_group": bool(semantic_cuboid.get("contains_movable_semantic_group")),
                "matches": list(semantic_cuboid.get("matches") or []),
                "contradictions": list(semantic_cuboid.get("contradictions") or []),
                "attempts": list(semantic_cuboid.get("attempts") or []),
                "center_shift_m": float(semantic_cuboid.get("center_shift_m") or 0.0),
                "yaw_shift_degrees": float(semantic_cuboid.get("yaw_shift_degrees") or 0.0),
                "fov_shift_degrees": float(semantic_cuboid.get("fov_shift_degrees") or 0.0),
                "guided_match_count": 0,
                "guided_radius_px": 0.0,
                "provisional_expansion": False,
                "provisional_expanded_match_count": 0,
                "guided_refinement_succeeded": False,
            }
            semantic_cuboid_candidates.append(semantic_candidate_diagnostics)
            semantic_matches = _pose_guided_matches(
                frame_width=frame.width,
                frame_height=frame.height,
                keypoints=keypoints,
                descriptors=descriptors,
                landmark_points=landmark_points,
                landmark_descriptors=landmark_descriptors,
                pose=semantic_seed,
                radius_px=96.0,
                learned_matcher=learned_matcher,
                query_responses=query_responses,
                landmark_responses=landmark_responses,
            )
            semantic_match_pools: list[tuple[float, list[Any]]] = [(96.0, semantic_matches)]
            if len(semantic_matches) >= 6:
                tighter_semantic_matches = _pose_guided_matches(
                    frame_width=frame.width,
                    frame_height=frame.height,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    landmark_points=landmark_points,
                    landmark_descriptors=landmark_descriptors,
                    pose=semantic_seed,
                    radius_px=64.0,
                    learned_matcher=learned_matcher,
                    query_responses=query_responses,
                    landmark_responses=landmark_responses,
                )
                if len(tighter_semantic_matches) >= 6:
                    semantic_match_pools.insert(0, (64.0, tighter_semantic_matches))
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
                    learned_matcher=learned_matcher,
                    query_responses=query_responses,
                    landmark_responses=landmark_responses,
                )
                if len(wider_semantic_matches) > len(semantic_matches):
                    semantic_match_pools.append((160.0, wider_semantic_matches))

            # A wider spatial gate can add useful cross-view support, but raw
            # match count is not a quality metric. In repeated room textures it
            # can instead add just enough wrong correspondences to make PnP
            # fail. Try each bounded pool independently and keep only a pool
            # that earns the unchanged strict guided-refinement gates.
            unique_semantic_match_pools: list[tuple[float, list[Any]]] = []
            seen_pool_signatures: set[tuple[tuple[int, int], ...]] = set()
            for radius_px, pool_matches in semantic_match_pools:
                signature = tuple(sorted((int(item.queryIdx), int(item.trainIdx)) for item in pool_matches))
                if signature in seen_pool_signatures:
                    continue
                seen_pool_signatures.add(signature)
                unique_semantic_match_pools.append((radius_px, pool_matches))
            semantic_candidate_diagnostics["guided_match_pools"] = [
                {"radius_px": radius_px, "match_count": len(pool_matches)}
                for radius_px, pool_matches in unique_semantic_match_pools
            ]
            diagnostic_best_radius, diagnostic_best_matches = max(
                unique_semantic_match_pools,
                key=lambda item: (len(item[1]), -item[0]),
            )
            semantic_candidate_diagnostics["guided_match_count"] = len(diagnostic_best_matches)
            semantic_candidate_diagnostics["guided_radius_px"] = diagnostic_best_radius

            best_semantic_refined: tuple[tuple[int, float, float, float], tuple[dict[str, Any], list[Any]], float] | None = None
            for semantic_radius_px, pool_matches in unique_semantic_match_pools:
                if len(pool_matches) < 6:
                    continue
                semantic_refined = None
                if len(pool_matches) >= 8:
                    semantic_refined = _refine_pose_with_guided_matches(
                        frame_width=frame.width,
                        frame_height=frame.height,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        landmark_points=landmark_points,
                        landmark_descriptors=landmark_descriptors,
                        pose=semantic_seed,
                        matches=pool_matches,
                        # Semantic boxes are search hints, not evidence. A
                        # six-inlier exploratory result is useful only if it
                        # later agrees across fixed-camera frames and multiple
                        # RoomPlan scan views; normal visual candidates retain
                        # the stricter eight-inlier refinement floor.
                        minimum_inliers=6,
                    )

                # If the strict pass cannot make eight clean inliers, a small
                # pool can still contain a useful six-point core plus one or
                # more repeated-texture outliers. Recover only a provisional
                # pose, collect fresh correspondences, then run the same strict
                # refiner again. The provisional pose is never publishable.
                if semantic_refined is None and len(pool_matches) <= 32:
                    # The wider semantic gate can contain a useful core plus
                    # repeated-texture outliers. Use only the best bounded
                    # correspondences to form the provisional pose; then
                    # collect fresh matches and send the full pool through the
                    # unchanged strict refiner. This keeps the initializer
                    # exploratory while preserving the publication gates.
                    provisional_matches = pool_matches[:16]
                    provisional_semantic_pose = _provisional_pose_from_guided_matches(
                        frame_width=frame.width,
                        frame_height=frame.height,
                        keypoints=keypoints,
                        landmark_points=landmark_points,
                        pose=semantic_seed,
                        matches=provisional_matches,
                    )
                    if provisional_semantic_pose is not None:
                        expansion_radius_px = max(96.0, min(160.0, semantic_radius_px * 1.5))
                        expanded_semantic_matches = _pose_guided_matches(
                            frame_width=frame.width,
                            frame_height=frame.height,
                            keypoints=keypoints,
                            descriptors=descriptors,
                            landmark_points=landmark_points,
                            landmark_descriptors=landmark_descriptors,
                            pose=provisional_semantic_pose,
                            radius_px=expansion_radius_px,
                            learned_matcher=learned_matcher,
                            query_responses=query_responses,
                            landmark_responses=landmark_responses,
                        )
                        semantic_candidate_diagnostics["provisional_expansion"] = True
                        semantic_candidate_diagnostics["provisional_expanded_match_count"] = max(
                            int(semantic_candidate_diagnostics["provisional_expanded_match_count"]),
                            len(expanded_semantic_matches),
                        )
                        if len(expanded_semantic_matches) >= 8:
                            semantic_refined = _refine_pose_with_guided_matches(
                                frame_width=frame.width,
                                frame_height=frame.height,
                                keypoints=keypoints,
                                descriptors=descriptors,
                                landmark_points=landmark_points,
                                landmark_descriptors=landmark_descriptors,
                                pose=provisional_semantic_pose,
                                matches=expanded_semantic_matches,
                                minimum_inliers=6,
                            )
                            if semantic_refined is not None:
                                pool_matches = expanded_semantic_matches
                                semantic_radius_px = expansion_radius_px

                if semantic_refined is None:
                    continue
                refined_pose, _refined_matches = semantic_refined
                score = (
                    int(refined_pose["inlier_count"]),
                    -float(refined_pose["mean_error"]),
                    float(refined_pose["coverage_ratio"]),
                    -float(semantic_radius_px),
                )
                if best_semantic_refined is None or score > best_semantic_refined[0]:
                    best_semantic_refined = (score, semantic_refined, semantic_radius_px)

            if best_semantic_refined is None:
                continue
            _semantic_score, semantic_refined, semantic_radius_px = best_semantic_refined
            semantic_candidate_diagnostics["guided_refinement_succeeded"] = True
            semantic_pose, refined_matches = semantic_refined
            semantic_candidate_diagnostics["guided_match_count"] = len(refined_matches)
            semantic_candidate_diagnostics["guided_radius_px"] = semantic_radius_px
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
            semantic_candidate["diagnostics"]["landmark_scene_filter"] = landmark_scene_filter
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
            orb_matches = (
                _descriptor_matches(
                    descriptors,
                    landmark_descriptors,
                    landmark_indices,
                    matcher,
                    query_responses=query_responses,
                    landmark_responses=landmark_responses,
                    learned_matcher=learned_matcher,
                )
                if descriptors is not None and len(keypoints) >= 2
                else []
            )
            sift_indices = landmark_indices[landmark_sift_available[landmark_indices]]
            sift_matches = (
                _descriptor_matches(
                    sift_query_descriptors,
                    landmark_sift_descriptors,
                    sift_indices,
                    sift_matcher,
                    ratio_threshold=0.78,
                    distance_limit=None,
                )
                if (
                    sift_query_descriptors is not None
                    and sift_matcher is not None
                    and len(sift_keypoints) >= 2
                    and len(sift_indices) >= 2
                )
                else []
            )
            matches = _merge_descriptor_matches(
                orb_matches,
                sift_matches,
                sift_query_offset=len(keypoints),
            )
            best_match_count = max(best_match_count, len(matches))
            if matches:
                frame_min_distance = float(min(match.distance for match in matches))
                best_min_distance = frame_min_distance if best_min_distance is None else min(best_min_distance, frame_min_distance)
            record_burst_matches(view_id, frame_index, pose_keypoints, matches)
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
            orb_all_matches = (
                _descriptor_matches(
                    descriptors,
                    landmark_descriptors,
                    all_landmark_indices,
                    matcher,
                    query_responses=query_responses,
                    landmark_responses=landmark_responses,
                    learned_matcher=learned_matcher,
                )
                if descriptors is not None and len(keypoints) >= 2
                else []
            )
            sift_all_indices = all_landmark_indices[landmark_sift_available]
            sift_all_matches = (
                _descriptor_matches(
                    sift_query_descriptors,
                    landmark_sift_descriptors,
                    sift_all_indices,
                    sift_matcher,
                    ratio_threshold=0.78,
                    distance_limit=None,
                )
                if (
                    sift_query_descriptors is not None
                    and sift_matcher is not None
                    and len(sift_keypoints) >= 2
                    and len(sift_all_indices) >= 2
                )
                else []
            )
            all_matches = _merge_descriptor_matches(
                orb_all_matches,
                sift_all_matches,
                sift_query_offset=len(keypoints),
            )
            best_match_count = max(best_match_count, len(all_matches))
            if all_matches:
                frame_min_distance = float(min(match.distance for match in all_matches))
                best_min_distance = frame_min_distance if best_min_distance is None else min(best_min_distance, frame_min_distance)
            record_burst_matches("all-views", frame_index, pose_keypoints, all_matches)
            if len(all_matches) >= 6:
                solve_groups.append(("all-views", all_landmark_indices, all_matches))

        solve_group_count = max(1, len(solve_groups))
        for group_rank, (view_id, _landmark_indices, matches) in enumerate(solve_groups):
            solve_span = max(1, frame_end_progress - frame_start_progress)
            solve_progress = frame_start_progress + max(
                1,
                int(solve_span * (0.45 + 0.45 * group_rank / solve_group_count)),
            )
            report(
                min(frame_end_progress - 1, solve_progress),
                f"Solving pose hypothesis {group_rank + 1} of {len(solve_groups)} for reference frame {frame_index + 1}",
            )
            preferred_fov = None
            if payload.search_prior is not None and payload.intrinsics is None:
                preferred_fov = payload.search_prior.fov_degrees
            pose = _pose_from_matches(
                payload=payload,
                frame_width=frame.width,
                frame_height=frame.height,
                keypoints=pose_keypoints,
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
                        learned_matcher=learned_matcher,
                        query_responses=query_responses,
                        landmark_responses=landmark_responses,
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
                            learned_matcher=learned_matcher,
                            query_responses=query_responses,
                            landmark_responses=landmark_responses,
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
                            learned_matcher=learned_matcher,
                            query_responses=query_responses,
                            landmark_responses=landmark_responses,
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
                        keypoints=pose_keypoints,
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
                            learned_matcher=learned_matcher,
                            query_responses=query_responses,
                            landmark_responses=landmark_responses,
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
                                learned_matcher=learned_matcher,
                                query_responses=query_responses,
                                landmark_responses=landmark_responses,
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
                                    learned_matcher=learned_matcher,
                                    query_responses=query_responses,
                                    landmark_responses=landmark_responses,
                                )
                                if len(expanded_seed_center_matches) > len(seed_center_matches):
                                    seed_center_pose = provisional_pose
                                    seed_center_matches = expanded_seed_center_matches
                                    seed_center_radius_px = 72.0
                        guided_variants.append(
                            ("seed-center-rotation", seed_center_pose, seed_center_matches, seed_center_radius_px)
                        )

                # A fixed camera is often sitting on a table or desk rather
                # than near the room perimeter. Use RoomPlan support surfaces
                # as bounded center hypotheses before the broader perimeter
                # sweep. These centers are search initializers only; the
                # ordinary unconstrained guided PnP and multi-view checks
                # below remain mandatory.
                if (
                    (
                        seed_prior["accepted"]
                        or seed_height_plausible
                        or seed_prior.get("reason") in {"camera_not_upright", "outside_roomplan_bounds"}
                    )
                    and payload.room_objects
                    and payload.intrinsics is None
                    and pose.get("fov_degrees") is not None
                    and not support_surface_search_used
                    and (group_rank == 0 or view_id == "all-views")
                ):
                    support_centers = _support_surface_search_centers(payload, max_centers=15)
                    if support_centers:
                        support_surface_search_used = True
                        support_surface_search_summary["attempted"] = True
                        support_surface_search_summary["center_count"] = len(support_centers)
                        pose["support_surface_search_attempted"] = True
                        support_search_fovs: list[float] = []
                        for support_fov in (float(pose["fov_degrees"]), 74.0):
                            if 30.0 <= support_fov <= 120.0 and all(
                                abs(support_fov - existing) >= 0.5 for existing in support_search_fovs
                            ):
                                support_search_fovs.append(support_fov)
                        support_search_centers = [
                            (center, support_fov)
                            for support_fov in support_search_fovs
                            for center in support_centers
                        ]
                        pose["support_surface_search_center_count"] = len(support_search_centers)
                        support_search_hypotheses: list[
                            tuple[tuple[int, int, int, float], dict[str, Any], list[Any], float]
                        ] = []
                        for center_index, (support_center, support_fov) in enumerate(support_search_centers):
                            support_pose = _pose_from_fixed_center_matches(
                                frame_width=frame.width,
                                frame_height=frame.height,
                                keypoints=pose_keypoints,
                                matches=matches,
                                landmark_points=landmark_points,
                                camera_center=support_center,
                                fov_degrees=support_fov,
                                random_seed=frame_index * 1301 + group_rank * 107 + center_index,
                                iteration_limit=40,
                            )
                            if support_pose is None:
                                continue
                            for support_radius_px in (72.0, 112.0, 144.0):
                                support_matches = _pose_guided_matches(
                                    frame_width=frame.width,
                                    frame_height=frame.height,
                                    keypoints=keypoints,
                                    descriptors=descriptors,
                                    landmark_points=landmark_points,
                                    landmark_descriptors=landmark_descriptors,
                                    pose=support_pose,
                                    radius_px=support_radius_px,
                                    learned_matcher=learned_matcher,
                                    query_responses=query_responses,
                                    landmark_responses=landmark_responses,
                                )
                                if len(support_matches) < 8:
                                    continue
                                support_views = {
                                    landmark_view_ids[item.trainIdx]
                                    for item in support_matches
                                    if 0 <= item.trainIdx < len(landmark_view_ids)
                                }
                                support_pose_variant = dict(support_pose)
                                support_pose_variant["support_surface_search_center"] = support_center.tolist()
                                support_pose_variant["support_surface_search_fov_degrees"] = support_fov
                                support_pose_variant["support_surface_search_recovery"] = True
                                support_pose_variant["support_surface_search_attempted"] = True
                                support_pose_variant["support_surface_search_center_count"] = len(support_search_centers)
                                score = (
                                    len(support_matches),
                                    len(support_views),
                                    int(support_pose_variant["inlier_count"]),
                                    -float(support_pose_variant["mean_error"]),
                                )
                                support_search_hypotheses.append(
                                    (score, support_pose_variant, support_matches, support_radius_px)
                                )
                        pose["support_surface_search_hypothesis_count"] = len(support_search_hypotheses)
                        support_surface_search_summary["hypothesis_count"] = max(
                            int(support_surface_search_summary["hypothesis_count"]),
                            len(support_search_hypotheses),
                        )
                        support_search_hypotheses.sort(key=lambda item: item[0], reverse=True)
                        pose["support_surface_search_hypothesis_stats"] = [
                            {
                                "center": [round(float(value), 4) for value in support_pose["support_surface_search_center"]],
                                "fov_degrees": float(support_pose["support_surface_search_fov_degrees"]),
                                "guided_match_count": len(support_matches),
                                "guided_view_count": len(
                                    {
                                        landmark_view_ids[item.trainIdx]
                                        for item in support_matches
                                        if 0 <= item.trainIdx < len(landmark_view_ids)
                                    }
                                ),
                                "rotation_inlier_count": int(support_pose["inlier_count"]),
                                "rotation_mean_error_px": round(float(support_pose["mean_error"]), 4),
                                "radius_px": float(support_radius_px),
                            }
                            for _score, support_pose, support_matches, support_radius_px in support_search_hypotheses[:12]
                        ]
                        for _score, support_pose, _support_matches, _support_radius_px in support_search_hypotheses:
                            support_pose["support_surface_search_hypothesis_count"] = len(support_search_hypotheses)
                            support_pose["support_surface_search_hypothesis_stats"] = pose[
                                "support_surface_search_hypothesis_stats"
                            ]
                        for _score, support_pose, support_matches, support_radius_px in support_search_hypotheses:
                            guided_variants.append(
                                ("support-surface-search", support_pose, support_matches, support_radius_px)
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
                    # One fresh qualifying pose is enough to probe RoomPlan-
                    # derived perimeter centers. Do not tie that opportunity
                    # to the final burst frame: a weak/blurred last frame can
                    # fail initial PnP and would otherwise disable recovery for
                    # the entire request.
                    and not room_center_search_used
                    and (group_rank == 0 or view_id == "all-views")
                ):
                    room_center_search_used = True
                    pose["room_center_search_attempted"] = True
                    room_search_centers: list[tuple[np.ndarray, float]] = []
                    room_search_fovs: list[float] = []
                    # Keep this bounded, but cover both ordinary webcam and
                    # genuinely wide-angle camera models. The main solver
                    # already treats FOV as unknown; room-center recovery must
                    # not silently collapse that search to the seed FOV plus
                    # one mid-range value.
                    fov_hints = [float(pose["fov_degrees"]), 74.0, 96.0]
                    if payload.fov_degrees is not None:
                        fov_hints.insert(0, float(payload.fov_degrees))
                    for fov in fov_hints:
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
                            keypoints=pose_keypoints,
                            matches=matches,
                            landmark_points=landmark_points,
                            camera_center=room_center,
                            fov_degrees=room_fov,
                            random_seed=frame_index * 997 + group_rank * 101 + center_index,
                            iteration_limit=40,
                        )
                        if room_pose is None:
                            continue
                        room_match_pools: list[tuple[float, list[Any]]] = []
                        seen_room_pool_signatures: set[tuple[tuple[int, int], ...]] = set()
                        for room_radius_px in (72.0, 96.0, 144.0):
                            room_matches = _pose_guided_matches(
                                frame_width=frame.width,
                                frame_height=frame.height,
                                keypoints=keypoints,
                                descriptors=descriptors,
                                landmark_points=landmark_points,
                                landmark_descriptors=landmark_descriptors,
                                pose=room_pose,
                                radius_px=room_radius_px,
                                learned_matcher=learned_matcher,
                                query_responses=query_responses,
                                landmark_responses=landmark_responses,
                            )
                            signature = tuple(
                                sorted((int(item.queryIdx), int(item.trainIdx)) for item in room_matches)
                            )
                            if signature in seen_room_pool_signatures:
                                continue
                            seen_room_pool_signatures.add(signature)
                            room_match_pools.append((room_radius_px, room_matches))
                        for room_radius_px, room_matches in room_match_pools:
                            room_views = {
                                landmark_view_ids[item.trainIdx]
                                for item in room_matches
                                if 0 <= item.trainIdx < len(landmark_view_ids)
                            }
                            if len(room_matches) < 8 or len(room_views) < 2:
                                continue
                            room_variant_pose = dict(room_pose)
                            room_variant_pose["room_search_center"] = room_center.tolist()
                            room_variant_pose["room_search_fov_degrees"] = room_fov
                            room_variant_pose["room_center_search_recovery"] = True
                            room_variant_pose["room_center_search_attempted"] = True
                            room_variant_pose["room_center_search_center_count"] = len(room_search_centers)
                            score = (
                                len(room_matches),
                                len(room_views),
                                int(room_variant_pose["inlier_count"]),
                                -float(room_variant_pose["mean_error"]),
                            )
                            room_search_hypotheses.append(
                                (score, room_variant_pose, room_matches, room_radius_px)
                            )
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
                            keypoints=pose_keypoints,
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
                        learned_matcher=learned_matcher,
                        query_responses=query_responses,
                        landmark_responses=landmark_responses,
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
                            learned_matcher=learned_matcher,
                            query_responses=query_responses,
                            landmark_responses=landmark_responses,
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
                                learned_matcher=learned_matcher,
                                query_responses=query_responses,
                                landmark_responses=landmark_responses,
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
                                    learned_matcher=learned_matcher,
                                    query_responses=query_responses,
                                    landmark_responses=landmark_responses,
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

                    best_guided: tuple[
                        tuple[int, int, float],
                        dict[str, Any],
                        list[Any],
                        bool,
                        bool,
                        bool,
                        str,
                        float,
                    ] | None = None
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
                        room_provisional_expansion = False
                        room_provisional_inlier_count = 0
                        room_provisional_expanded_match_count = 0
                        if (
                            variant_kind == "room-center-search"
                            and guided is None
                            and 6 <= len(variant_matches) <= 16
                        ):
                            # The fixed-center rotation search is deliberately
                            # coarse, so its 8-11 cross-view matches can be too
                            # far from a local PnP minimum even when they point
                            # at the correct room basin. Use the existing
                            # provisional solver only to improve reprojection,
                            # gather a fresh match pool, then require the same
                            # strict guided refinement as every publishable
                            # candidate. The provisional pose itself can never
                            # become localization evidence.
                            provisional_room_pose = _provisional_pose_from_guided_matches(
                                frame_width=frame.width,
                                frame_height=frame.height,
                                keypoints=keypoints,
                                landmark_points=landmark_points,
                                pose=guided_seed_pose,
                                matches=variant_matches,
                            )
                            if provisional_room_pose is not None:
                                room_provisional_expansion = True
                                room_provisional_inlier_count = int(
                                    provisional_room_pose.get("inlier_count") or 0
                                )
                                expanded_room_matches = _pose_guided_matches(
                                    frame_width=frame.width,
                                    frame_height=frame.height,
                                    keypoints=keypoints,
                                    descriptors=descriptors,
                                    landmark_points=landmark_points,
                                    landmark_descriptors=landmark_descriptors,
                                    pose=provisional_room_pose,
                                    radius_px=96.0,
                                    learned_matcher=learned_matcher,
                                    query_responses=query_responses,
                                    landmark_responses=landmark_responses,
                                )
                                expanded_room_radius = 96.0
                                if len(expanded_room_matches) < 12:
                                    wider_room_matches = _pose_guided_matches(
                                        frame_width=frame.width,
                                        frame_height=frame.height,
                                        keypoints=keypoints,
                                        descriptors=descriptors,
                                        landmark_points=landmark_points,
                                        landmark_descriptors=landmark_descriptors,
                                        pose=provisional_room_pose,
                                        radius_px=144.0,
                                        learned_matcher=learned_matcher,
                                        query_responses=query_responses,
                                        landmark_responses=landmark_responses,
                                    )
                                    if len(wider_room_matches) > len(expanded_room_matches):
                                        expanded_room_matches = wider_room_matches
                                        expanded_room_radius = 144.0
                                room_provisional_expanded_match_count = len(expanded_room_matches)
                                if len(expanded_room_matches) >= 8:
                                    guided = _refine_pose_with_guided_matches(
                                        frame_width=frame.width,
                                        frame_height=frame.height,
                                        keypoints=keypoints,
                                        descriptors=descriptors,
                                        landmark_points=landmark_points,
                                        landmark_descriptors=landmark_descriptors,
                                        pose=provisional_room_pose,
                                        matches=expanded_room_matches,
                                    )
                                    if guided is not None:
                                        guided_seed_pose = provisional_room_pose
                                        variant_matches = expanded_room_matches
                                        variant_radius = expanded_room_radius
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
                                    and abs(float(hypothesis.get("radius_px", 0.0)) - float(variant_radius)) < 0.5
                                ):
                                    hypothesis["provisional_expansion"] = room_provisional_expansion
                                    hypothesis["provisional_inlier_count"] = room_provisional_inlier_count
                                    hypothesis["provisional_expanded_match_count"] = room_provisional_expanded_match_count
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
                        support_surface_search_center = guided_seed_pose.get("support_surface_search_center")
                        support_surface_recovery = (
                            variant_kind == "support-surface-search"
                            and isinstance(support_surface_search_center, list)
                            and len(support_surface_search_center) == 3
                            and guided_pose["inlier_count"] >= max(10, pose["inlier_count"] + 4)
                            and float(
                                np.linalg.norm(
                                    guided_matrix[:3, 3]
                                    - np.asarray(support_surface_search_center, dtype=np.float64)
                                )
                            ) <= 0.70
                        )
                        if not (
                            guided_prior["accepted"]
                            and len(guided_views) >= 2
                            and guided_pose["inlier_count"] >= pose["inlier_count"] + 2
                            and (
                                seed_consistent
                                or prior_recovery
                                or seed_center_recovery
                                or room_center_recovery
                                or support_surface_recovery
                            )
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
                                support_surface_recovery,
                                variant_kind,
                                variant_radius,
                            )

                    if best_guided is not None:
                        (
                            _,
                            guided_pose,
                            guided_matches,
                            prior_recovery,
                            seed_center_recovery,
                            support_surface_recovery,
                            variant_kind,
                            variant_radius,
                        ) = best_guided
                        guided_views = variant_views(guided_matches)
                        guided_pose["guided_attempt_match_count"] = len(guided_matches)
                        guided_pose["guided_attempt_view_count"] = len(guided_views)
                        guided_pose["guided_attempt_view_ids"] = guided_views
                        guided_pose["guided_attempt_radius_px"] = variant_radius
                        guided_pose["search_prior_center_attempted"] = True
                        guided_pose["search_prior_center_guided"] = variant_kind == "prior-center"
                        guided_pose["search_prior_recovery"] = prior_recovery
                        guided_pose["seed_center_rotation_recovery"] = seed_center_recovery
                        guided_pose["support_surface_search_recovery"] = support_surface_recovery
                        guided_pose["room_center_search_attempted"] = bool(
                            guided_pose.get("room_center_search_attempted")
                            or variant_kind == "room-center-search"
                        )
                        guided_pose["room_center_search_recovery"] = variant_kind == "room-center-search"
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
                candidate["diagnostics"]["landmark_scene_filter"] = landmark_scene_filter
                scene_prior = _pose_scene_prior(candidate["_pose_matrix"], payload)
                candidate["_scene_plausible"] = bool(scene_prior["accepted"])
                candidate["diagnostics"]["scene_prior"] = scene_prior
                candidates.append(candidate)

        report(frame_end_progress, f"Finished reference frame {frame_index + 1} of {frame_count}")

    # A fixed-camera calibration burst gives us one extra source of evidence:
    # the same landmark should stay at the same image coordinate across several
    # frames. Aggregate those repeated observations before one additional PnP
    # solve. This can recover independent RoomPlan-view support that is too
    # sparse to reach six correspondences in any single frame, without
    # weakening the final multi-view acceptance gates.
    frame_shapes = {(frame.width, frame.height) for frame in payload.frames}
    if len(payload.frames) >= 2 and len(frame_shapes) == 1 and burst_match_observations:
        report(80, "Combining stable matches across the fixed-camera burst")
        frame_width, frame_height = next(iter(frame_shapes))
        temporal_groups: list[tuple[str, list[Any], list[Any], dict[str, Any]]] = []
        temporal_minimum_support = max(2, min(4, math.ceil(len(payload.frames) * 0.5)))
        temporal_maximum_jitter_px = 6.0 if len(payload.frames) >= 3 else 8.0
        for view_id, observations in burst_match_observations.items():
            temporal_keypoints, temporal_matches, temporal_diagnostics = _temporal_consensus_matches(
                observations,
                minimum_frame_support=temporal_minimum_support,
                maximum_jitter_px=temporal_maximum_jitter_px,
            )
            if len(temporal_matches) >= 6:
                temporal_groups.append((view_id, temporal_keypoints, temporal_matches, temporal_diagnostics))

        per_view_groups = sorted(
            (item for item in temporal_groups if item[0] != "all-views"),
            key=lambda item: (
                len(item[2]),
                int(item[3]["support_frame_count"]),
                -float(item[3]["median_pixel_jitter"] or 0.0),
            ),
            reverse=True,
        )[:3]
        all_views_group = next((item for item in temporal_groups if item[0] == "all-views"), None)
        solve_temporal_groups = [*per_view_groups, *([all_views_group] if all_views_group is not None else [])]
        for view_id, temporal_keypoints, temporal_matches, temporal_diagnostics in solve_temporal_groups:
            preferred_fov = None
            if payload.search_prior is not None and payload.intrinsics is None:
                preferred_fov = payload.search_prior.fov_degrees
            temporal_pose = _pose_from_matches(
                payload=payload,
                frame_width=frame_width,
                frame_height=frame_height,
                keypoints=temporal_keypoints,
                matches=temporal_matches,
                landmark_points=landmark_points,
                preferred_fov_degrees=preferred_fov,
            )
            if temporal_pose is None:
                continue
            temporal_candidate = _localization_candidate(
                -1,
                view_id,
                temporal_matches,
                temporal_pose,
                landmark_view_ids,
            )
            temporal_candidate["diagnostics"]["learned_matcher"] = learned_matcher_diagnostics
            temporal_candidate["diagnostics"]["landmark_scene_filter"] = landmark_scene_filter
            temporal_candidate["diagnostics"]["temporal_burst_consensus"] = temporal_diagnostics
            temporal_scene_prior = _pose_scene_prior(temporal_candidate["_pose_matrix"], payload)
            temporal_candidate["_scene_plausible"] = bool(temporal_scene_prior["accepted"])
            temporal_candidate["diagnostics"]["scene_prior"] = temporal_scene_prior
            candidates.append(temporal_candidate)

    report(82, "Checking pose agreement across reference frames")
    semantic_consensus = _semantic_cuboid_consensus(
        semantic_cuboid_candidates,
        burst_frame_count=len(payload.frames),
        dynamic_mask_max_ratio=max(person_masked_area_ratios) if person_masked_area_ratios else 0.0,
    )
    semantic_consensus_diagnostics: dict[str, Any] | None = None
    semantic_consensus_scene_prior: dict[str, Any] | None = None
    semantic_consensus_intrinsics: np.ndarray | None = None
    semantic_consensus_intrinsics_source: str | None = None
    if semantic_consensus is not None:
        consensus_matrix = np.asarray(semantic_consensus["camera_to_world"], dtype=np.float64)
        semantic_consensus_scene_prior = _pose_scene_prior(consensus_matrix, payload)
        semantic_consensus_diagnostics = {
            **{key: value for key, value in semantic_consensus.items() if key != "camera_to_world"},
            "camera_to_world": [
                [round(float(value), 8) for value in row]
                for row in consensus_matrix
            ],
            "scene_prior": semantic_consensus_scene_prior,
        }
        frame = payload.frames[0]
        if payload.intrinsics is not None:
            semantic_consensus_intrinsics, semantic_consensus_intrinsics_source = _camera_matrix(
                payload,
                frame.width,
                frame.height,
            )
        else:
            consensus_fov = float(semantic_consensus["fov_degrees"])
            focal = 0.5 * frame.width / math.tan(math.radians(consensus_fov) / 2.0)
            semantic_consensus_intrinsics = np.asarray(
                [
                    [focal, 0.0, frame.width / 2.0],
                    [0.0, focal, frame.height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            semantic_consensus_intrinsics_source = (
                "estimated-fov" if payload.fov_degrees is not None else "estimated-fov-sweep"
            )
    semantic_consensus_accepted = bool(
        semantic_consensus is not None
        and semantic_consensus_scene_prior is not None
        and semantic_consensus_scene_prior.get("accepted")
        and semantic_consensus_intrinsics is not None
        and semantic_consensus_intrinsics_source is not None
    )

    if not candidates:
        report(94, "Evaluating localization confidence")
        fallback_matrix: np.ndarray | None = None
        if payload.intrinsics is not None or payload.fov_degrees is not None:
            fallback_matrix, _ = _camera_matrix(payload, payload.frames[0].width, payload.frames[0].height)
        semantic_cuboid_candidates.sort(
            key=_semantic_cuboid_rank,
            reverse=True,
        )
        selected_semantic = semantic_cuboid_candidates[0] if semantic_cuboid_candidates else None
        selected_semantic_fov = selected_semantic.get("selected_fov_degrees") if selected_semantic is not None else None
        if semantic_consensus_accepted:
            assert semantic_consensus is not None
            assert semantic_consensus_diagnostics is not None
            assert semantic_consensus_intrinsics is not None
            assert semantic_consensus_intrinsics_source is not None
            consensus_matrix = np.asarray(semantic_consensus["camera_to_world"], dtype=np.float64)
            result = {
                "status": "positioned",
                "coordinate_frame": "roomplan-local",
                "camera_to_world": [
                    [round(float(value), 8) for value in row]
                    for row in consensus_matrix
                ],
                "confidence": float(semantic_consensus["confidence"]),
                "inlier_count": 0,
                "match_count": best_match_count,
                "reprojection_error_px": None,
                "intrinsics_source": semantic_consensus_intrinsics_source,
                "intrinsics": [
                    [round(float(value), 8) for value in row]
                    for row in semantic_consensus_intrinsics
                ],
                "diagnostics": {
                    "reason": "semantic_cuboid_consensus",
                    "best_feature_count": best_feature_count,
                    "best_match_count": best_match_count,
                    "best_min_descriptor_distance": best_min_distance,
                    "learned_matcher": learned_matcher_diagnostics,
                    "descriptor_families": descriptor_family_diagnostics,
                    "landmark_scene_filter": landmark_scene_filter,
                    "semantic_cuboid_candidates": semantic_cuboid_candidates[:12],
                    "semantic_cuboid_consensus": semantic_consensus_diagnostics,
                    "selected_camera_center": semantic_consensus["camera_center"],
                    "selected_candidate_camera_to_world": semantic_consensus_diagnostics["camera_to_world"],
                    "selected_fov_degrees": semantic_consensus["fov_degrees"],
                    "selected_estimate_confidence": semantic_consensus["confidence"],
                    "selected_estimate_source": "semantic-cuboid-consensus",
                    "gpu_pose_refinement_attempts": list(
                        getattr(_SOLVER_THREAD_STATE, "gpu_pose_refinement_attempts", [])
                    ),
                    "dynamic_person_mask": {
                        "masked_frame_count": person_masked_frame_count,
                        "occluded_frame_count": person_occluded_frame_count,
                        "max_masked_area_ratio": round(max(person_masked_area_ratios), 4) if person_masked_area_ratios else 0.0,
                    },
                    "low_light": low_light_summary(),
                    "guided_person_calibration": guided_person_diagnostics,
                    "support_surface_search": dict(support_surface_search_summary),
                    "raw_frames_persisted": False,
                },
            }
            report(100, "Camera pose solved")
            return result
        result = {
            "status": "needs_rescan",
            "coordinate_frame": "roomplan-local",
            "camera_to_world": None,
            "confidence": 0.0,
            "inlier_count": 0,
            "match_count": best_match_count,
            "reprojection_error_px": None,
            "intrinsics_source": (
                "provided"
                if payload.intrinsics is not None
                else ("estimated-fov" if payload.fov_degrees is not None else "estimated-fov-sweep")
            ),
            "intrinsics": (
                [[round(float(value), 8) for value in row] for row in fallback_matrix]
                if fallback_matrix is not None
                else None
            ),
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
                "descriptor_families": descriptor_family_diagnostics,
                "landmark_scene_filter": landmark_scene_filter,
                "semantic_cuboid_candidates": semantic_cuboid_candidates[:12],
                "semantic_cuboid_consensus": semantic_consensus_diagnostics,
                "selected_camera_center": selected_semantic.get("camera_center") if selected_semantic is not None else None,
                "selected_candidate_camera_to_world": selected_semantic.get("candidate_camera_to_world") if selected_semantic is not None else None,
                "selected_fov_degrees": selected_semantic_fov,
                "selected_estimate_confidence": selected_semantic.get("confidence") if selected_semantic is not None else None,
                "selected_estimate_source": "semantic-cuboid" if selected_semantic is not None else None,
                "gpu_pose_refinement_attempts": list(
                    getattr(_SOLVER_THREAD_STATE, "gpu_pose_refinement_attempts", [])
                ),
                "dynamic_person_mask": {
                    "masked_frame_count": person_masked_frame_count,
                    "occluded_frame_count": person_occluded_frame_count,
                    "max_masked_area_ratio": round(max(person_masked_area_ratios), 4) if person_masked_area_ratios else 0.0,
                },
                "low_light": low_light_summary(),
                "guided_person_calibration": guided_person_diagnostics,
                "support_surface_search": dict(support_surface_search_summary),
                "raw_frames_persisted": False,
            },
        }
        report(100, "Localization finished without a confident pose")
        return result
    report(90, "Comparing pose candidates across RoomPlan viewpoints")
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
                "gpu_pose_refinement": dict(candidate["diagnostics"].get("gpu_pose_refinement") or {}),
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
                "support_surface_search_attempted": bool(candidate["diagnostics"].get("support_surface_search_attempted")),
                "support_surface_search_center_count": int(candidate["diagnostics"].get("support_surface_search_center_count") or 0),
                "support_surface_search_hypothesis_count": int(candidate["diagnostics"].get("support_surface_search_hypothesis_count") or 0),
                "support_surface_search_hypothesis_stats": list(candidate["diagnostics"].get("support_surface_search_hypothesis_stats") or []),
                "support_surface_search_recovery": bool(candidate["diagnostics"].get("support_surface_search_recovery")),
                "support_surface_search": dict(candidate["diagnostics"].get("support_surface_search") or {}),
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
    report(96, "Validating the best pose against the scanned room")
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
    diagnostics["semantic_cuboid_consensus"] = semantic_consensus_diagnostics
    selected_semantic = semantic_cuboid_candidates[0] if semantic_cuboid_candidates else None
    semantic_is_strong = bool(
        selected_semantic is not None
        and _semantic_cuboid_has_robust_support(selected_semantic)
    )
    diagnostics["geometric_selected_camera_center"] = geometric_selected_center
    semantic_consensus_selected = bool(not positioned and semantic_consensus_accepted)
    if semantic_consensus_selected:
        assert semantic_consensus is not None
        assert semantic_consensus_diagnostics is not None
        assert semantic_consensus_intrinsics is not None
        assert semantic_consensus_intrinsics_source is not None
        pose_matrix = np.asarray(semantic_consensus["camera_to_world"], dtype=np.float64)
        positioned = True
        best["inlier_count"] = 0
        best["reprojection_error_px"] = None
        best["intrinsics_source"] = semantic_consensus_intrinsics_source
        best["intrinsics"] = [
            [round(float(value), 8) for value in row]
            for row in semantic_consensus_intrinsics
        ]
        best["confidence"] = float(semantic_consensus["confidence"])
        diagnostics["selected_camera_center"] = semantic_consensus["camera_center"]
        diagnostics["selected_candidate_camera_to_world"] = semantic_consensus_diagnostics["camera_to_world"]
        diagnostics["selected_fov_degrees"] = semantic_consensus["fov_degrees"]
        diagnostics["selected_estimate_confidence"] = semantic_consensus["confidence"]
        diagnostics["selected_estimate_source"] = "semantic-cuboid-consensus"
    elif not positioned and semantic_is_strong:
        diagnostics["selected_camera_center"] = selected_semantic["camera_center"]
        diagnostics["selected_candidate_camera_to_world"] = selected_semantic.get("candidate_camera_to_world")
        diagnostics["selected_fov_degrees"] = selected_semantic.get("selected_fov_degrees")
        diagnostics["selected_estimate_confidence"] = selected_semantic.get("confidence")
        diagnostics["selected_estimate_source"] = "semantic-cuboid"
    else:
        diagnostics["selected_camera_center"] = geometric_selected_center
        diagnostics["selected_candidate_camera_to_world"] = None
        diagnostics["selected_estimate_confidence"] = best.get("confidence")
        diagnostics["selected_estimate_source"] = "visual-pnp"
    dynamic_mask_diagnostics = {
        "masked_frame_count": person_masked_frame_count,
        "occluded_frame_count": person_occluded_frame_count,
        "max_masked_area_ratio": round(max(person_masked_area_ratios), 4) if person_masked_area_ratios else 0.0,
        "masked_labels": sorted(_DYNAMIC_FEATURE_LABELS),
    }
    diagnostics["dynamic_object_mask"] = dynamic_mask_diagnostics
    diagnostics["dynamic_person_mask"] = dynamic_mask_diagnostics
    diagnostics["low_light"] = low_light_summary()
    diagnostics["guided_person_calibration"] = guided_person_diagnostics
    diagnostics["support_surface_search"] = dict(support_surface_search_summary)
    diagnostics["descriptor_families"] = descriptor_family_diagnostics
    diagnostics["gpu_pose_refinement_attempts"] = list(
        getattr(_SOLVER_THREAD_STATE, "gpu_pose_refinement_attempts", [])
    )
    if positioned:
        best["status"] = "positioned"
        best["camera_to_world"] = [[round(float(value), 8) for value in row] for row in pose_matrix]
        if semantic_consensus_selected:
            best["confidence"] = round(float(best["confidence"]), 6)
        else:
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
    report(100, "Camera pose solved" if positioned else "Localization finished without a confident pose")
    return best
