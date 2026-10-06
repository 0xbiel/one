"""Conservative fall signals from learned GPU human-pose sequences.

The production path intentionally does not infer a fall from a person box.
Each frame must carry a 17-joint COCO pose from the local pose model. The API
keeps only a bounded derived-pose window and the geometry worker scores it with
an evaluated ST-GCN plus causal-TCN model. A small posture/persistence gate
remains as a conservative review debounce around that learned score.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


POSE_MODEL_VERSION = "stgcn-causal-tcn-fall-v1"


@dataclass(frozen=True)
class FallPoseWindow:
    """A bounded derived-pose window and the last learned score for its track."""

    frames: tuple[dict[str, Any], ...]
    should_score: bool
    latest_probability: float | None
    latest_threshold: float | None
    latest_model_version: str | None


@dataclass
class _PoseWindowState:
    last_at: datetime
    frames: deque[dict[str, Any]]
    observations: int = 0
    latest_probability: float | None = None
    latest_threshold: float | None = None
    latest_model_version: str | None = None


class PoseSequenceWindowStore:
    """Keep short-lived COCO pose windows keyed by camera and stable track."""

    def __init__(self, *, window_size: int = 15, score_stride: int = 3, stale_after_seconds: float = 12.0):
        self.window_size = max(8, min(int(window_size), 64))
        self.score_stride = max(1, min(int(score_stride), self.window_size))
        self.stale_after = timedelta(seconds=max(4.0, float(stale_after_seconds)))
        self._states: dict[tuple[str, int], _PoseWindowState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _frame(item: dict[str, Any]) -> dict[str, Any] | None:
        raw_keypoints = item.get("pose_keypoints")
        bbox = item.get("bbox")
        if not isinstance(raw_keypoints, (list, tuple)) or len(raw_keypoints) != 17:
            return None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        points: list[list[float]] = []
        try:
            for point in raw_keypoints:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    return None
                values = [float(point[0]), float(point[1]), float(point[2]) if len(point) >= 3 else 1.0]
                if not all(math.isfinite(value) for value in values):
                    return None
                points.append([values[0], values[1], min(max(values[2], 0.0), 1.0)])
            box = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
            return None
        return {"keypoints": points, "bbox": box}

    def append(
        self,
        camera_id: str,
        item: dict[str, Any],
        *,
        observed_at: datetime,
    ) -> FallPoseWindow | None:
        track_id = item.get("track_id")
        frame = self._frame(item)
        if not isinstance(track_id, int) or track_id < 1 or frame is None:
            return None
        timestamp = _utc(observed_at)
        key = (camera_id, track_id)
        with self._lock:
            for stale_key, state in list(self._states.items()):
                if timestamp - state.last_at > self.stale_after:
                    self._states.pop(stale_key, None)
            state = self._states.get(key)
            if state is None or timestamp - state.last_at > self.stale_after:
                state = _PoseWindowState(timestamp, deque(maxlen=self.window_size))
                self._states[key] = state
            state.frames.append(frame)
            state.observations += 1
            state.last_at = timestamp
            has_window = len(state.frames) >= self.window_size
            should_score = has_window and (
                state.observations == self.window_size
                or (state.observations - self.window_size) % self.score_stride == 0
            )
            return FallPoseWindow(
                frames=tuple(state.frames) if should_score else (),
                should_score=should_score,
                latest_probability=state.latest_probability,
                latest_threshold=state.latest_threshold,
                latest_model_version=state.latest_model_version,
            )

    def record_score(
        self,
        camera_id: str,
        track_id: int,
        *,
        probability: float,
        threshold: float | None,
        model_version: str | None,
    ) -> None:
        if not math.isfinite(float(probability)):
            return
        with self._lock:
            state = self._states.get((camera_id, track_id))
            if state is None:
                return
            state.latest_probability = min(max(float(probability), 0.0), 1.0)
            state.latest_threshold = float(threshold) if isinstance(threshold, (int, float)) and math.isfinite(float(threshold)) else None
            state.latest_model_version = str(model_version)[:120] if model_version else None


@dataclass(frozen=True)
class FallSignal:
    track_id: int
    observed_at: datetime
    confidence: float
    explanation: str
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class _PoseFeatures:
    posture: str
    center_y: float
    aspect: float
    axis_degrees: float
    visible_keypoints: int
    quality: float


@dataclass
class _TrackState:
    last_at: datetime
    last_center_y: float
    last_posture: str
    upright_center_y: float | None
    sequence: deque[_PoseFeatures] = field(default_factory=lambda: deque(maxlen=12))
    candidate_started_at: datetime | None = None
    candidate_hits: int = 0
    episode_open: bool = False
    refractory_until: datetime | None = None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _point(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
        confidence = float(value[2]) if len(value) >= 3 else 1.0
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, confidence)):
        return None
    return x, y, max(0.0, min(1.0, confidence))


def _midpoint(points: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _pose_features(
    item: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    minimum_keypoint_confidence: float,
    minimum_visible_keypoints: int,
) -> _PoseFeatures | None:
    raw = item.get("pose_keypoints")
    if not isinstance(raw, (list, tuple)) or len(raw) != 17 or frame_width <= 0 or frame_height <= 0:
        return None
    points = [_point(value) for value in raw]
    if any(point is None for point in points):
        return None
    typed_points = [point for point in points if point is not None]
    visible = [point for point in typed_points if point[2] >= minimum_keypoint_confidence]
    if len(visible) < minimum_visible_keypoints:
        return None

    normalized = [(x / frame_width, y / frame_height, confidence) for x, y, confidence in visible]
    min_x, max_x = min(point[0] for point in normalized), max(point[0] for point in normalized)
    min_y, max_y = min(point[1] for point in normalized), max(point[1] for point in normalized)
    horizontal_extent = max_x - min_x
    vertical_extent = max_y - min_y
    if horizontal_extent <= 1e-4 or vertical_extent <= 1e-4:
        return None

    shoulder_points = [
        typed_points[index]
        for index in (5, 6)
        if typed_points[index][2] >= minimum_keypoint_confidence
    ]
    hip_points = [
        typed_points[index]
        for index in (11, 12)
        if typed_points[index][2] >= minimum_keypoint_confidence
    ]
    shoulders = _midpoint(shoulder_points)
    hips = _midpoint(hip_points)
    axis_degrees = 45.0
    if shoulders is not None and hips is not None:
        dx, dy = shoulders[0] - hips[0], shoulders[1] - hips[1]
        axis_degrees = math.degrees(math.atan2(abs(dy), abs(dx)))

    aspect = vertical_extent / horizontal_extent
    if axis_degrees >= 52.0 and aspect >= 1.15:
        posture = "upright"
    elif axis_degrees <= 38.0 or aspect <= 0.95:
        posture = "low"
    else:
        posture = "transition"

    torso = [point for point in (shoulders, hips) if point is not None]
    center = _midpoint([(point[0], point[1], 1.0) for point in torso]) if torso else None
    center_y = center[1] / frame_height if center is not None else (min_y + max_y) / 2.0
    quality = sum(point[2] for point in visible) / len(visible)
    return _PoseFeatures(
        posture=posture,
        center_y=max(0.0, min(1.0, center_y)),
        aspect=aspect,
        axis_degrees=max(0.0, min(90.0, axis_degrees)),
        visible_keypoints=len(visible),
        quality=max(0.0, min(1.0, quality)),
    )


class PoseSequenceFallDetector:
    """Track pose transitions and emit at most one signal per episode."""

    model_version = POSE_MODEL_VERSION

    def __init__(
        self,
        *,
        min_hits: int = 3,
        minimum_confirmation_seconds: float = 0.75,
        confirmation_window_seconds: float = 4.0,
        stale_after_seconds: float = 12.0,
        minimum_person_confidence: float = 0.30,
        minimum_keypoint_confidence: float = 0.35,
        minimum_visible_keypoints: int = 6,
        require_learned_model: bool = False,
        learned_threshold: float = 0.65,
        refractory_seconds: float = 15.0,
    ):
        self.min_hits = max(2, min_hits)
        self.minimum_confirmation = timedelta(seconds=max(0.25, float(minimum_confirmation_seconds)))
        self.minimum_person_confidence = min(max(float(minimum_person_confidence), 0.20), 0.80)
        self.minimum_keypoint_confidence = min(max(float(minimum_keypoint_confidence), 0.10), 0.80)
        self.minimum_visible_keypoints = max(5, min(int(minimum_visible_keypoints), 17))
        self.require_learned_model = bool(require_learned_model)
        self.learned_threshold = min(max(float(learned_threshold), 0.50), 0.99)
        self.confirmation_window = timedelta(seconds=max(1.0, confirmation_window_seconds))
        self.stale_after = timedelta(seconds=max(4.0, stale_after_seconds))
        self.refractory = timedelta(seconds=max(0.0, float(refractory_seconds)))
        self._tracks: dict[tuple[str, int], _TrackState] = {}
        self._lock = threading.RLock()

    def update(
        self,
        camera_id: str,
        item: dict[str, Any],
        *,
        frame_width: int,
        frame_height: int,
        observed_at: datetime,
    ) -> FallSignal | None:
        """Update one pose track and return only a newly confirmed signal."""

        if str(item.get("label") or "").strip().lower() != "person":
            return None
        track_id = item.get("track_id")
        confidence = item.get("confidence")
        if (
            not isinstance(track_id, int)
            or track_id < 1
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or float(confidence) < self.minimum_person_confidence
        ):
            return None
        features = _pose_features(
            item,
            frame_width=frame_width,
            frame_height=frame_height,
            minimum_keypoint_confidence=self.minimum_keypoint_confidence,
            minimum_visible_keypoints=self.minimum_visible_keypoints,
        )
        # No rectangle-only fallback: a fall event needs actual joint evidence.
        if features is None:
            return None
        learned_probability = item.get("fall_sequence_probability")
        if isinstance(learned_probability, (int, float)) and math.isfinite(float(learned_probability)):
            learned_probability = min(max(float(learned_probability), 0.0), 1.0)
        else:
            learned_probability = None
        learned_threshold = item.get("fall_sequence_threshold")
        if not isinstance(learned_threshold, (int, float)) or not math.isfinite(float(learned_threshold)):
            learned_threshold = self.learned_threshold
        learned_threshold = min(max(float(learned_threshold), 0.50), 0.99)
        learned_positive = (
            not self.require_learned_model
            or (learned_probability is not None and learned_probability >= learned_threshold)
        )
        timestamp = _utc(observed_at)
        key = (camera_id, track_id)

        with self._lock:
            for stale_key, state in list(self._tracks.items()):
                if timestamp - state.last_at > self.stale_after:
                    self._tracks.pop(stale_key, None)

            state = self._tracks.get(key)
            if state is None or timestamp - state.last_at > self.stale_after:
                state = _TrackState(
                    last_at=timestamp,
                    last_center_y=features.center_y,
                    last_posture=features.posture,
                    upright_center_y=features.center_y if features.posture == "upright" else None,
                )
                state.sequence.append(features)
                self._tracks[key] = state
                return None
            if timestamp <= state.last_at:
                # Replayed or out-of-order observations must never advance a
                # safety candidate. This also prevents a burst of duplicate
                # frames with one timestamp from satisfying persistence.
                return None

            state.sequence.append(features)
            in_refractory = state.refractory_until is not None and timestamp < state.refractory_until
            if features.posture == "upright":
                if state.upright_center_y is None:
                    state.upright_center_y = features.center_y
                else:
                    state.upright_center_y = state.upright_center_y * 0.7 + features.center_y * 0.3
                state.candidate_started_at = None
                state.candidate_hits = 0
                state.episode_open = False

            if in_refractory:
                # A brief pose reclassification after a confirmed event must
                # not create a second alert for the same physical episode.
                # Keep the track alive and continue observing it, but require
                # the refractory interval to expire before a new candidate can
                # accumulate.
                state.candidate_started_at = None
                state.candidate_hits = 0

            reference_y = state.upright_center_y if state.upright_center_y is not None else state.last_center_y
            vertical_drop = features.center_y - reference_y
            within_window = (
                state.candidate_started_at is not None
                and timestamp - state.candidate_started_at <= self.confirmation_window
            )
            if self.require_learned_model and not learned_positive:
                # Keep observing the posture so an upright baseline is not
                # lost while the temporal model is warming up. Do not allow a
                # low/unknown model score to accumulate a safety event.
                state.candidate_started_at = None
                state.candidate_hits = 0
            fresh_learned_evidence = (
                not self.require_learned_model
                or item.get("fall_sequence_score_fresh") is True
            )
            if (
                features.posture == "low"
                and vertical_drop >= 0.05
                and features.quality >= 0.50
                and state.upright_center_y is not None
                and not state.episode_open
                and not in_refractory
                and learned_positive
                and fresh_learned_evidence
            ):
                if not within_window:
                    state.candidate_started_at = timestamp
                    state.candidate_hits = 1
                else:
                    state.candidate_hits += 1
                candidate_duration = (
                    timestamp - state.candidate_started_at
                    if state.candidate_started_at is not None
                    else timedelta(0)
                )
                if state.candidate_hits >= self.min_hits and candidate_duration >= self.minimum_confirmation:
                    state.episode_open = True
                    recent = list(state.sequence)[-state.candidate_hits:]
                    average_quality = sum(item.quality for item in recent) / max(1, len(recent))
                    shape_score = min(1.0, max(0.0, (1.15 - features.aspect) / 0.45))
                    axis_score = min(1.0, max(0.0, (52.0 - features.axis_degrees) / 52.0))
                    drop_score = min(1.0, max(0.0, (vertical_drop - 0.05) / 0.20))
                    signal_confidence = min(
                        0.92,
                        0.48
                        + 0.10 * shape_score
                        + 0.10 * axis_score
                        + 0.10 * drop_score
                        + 0.08 * average_quality
                        + 0.06 * min(1.0, state.candidate_hits / self.min_hits),
                    )
                    if learned_probability is not None:
                        signal_confidence = min(
                            0.96,
                            signal_confidence * (0.82 + 0.18 * learned_probability),
                        )
                    state.refractory_until = timestamp + self.refractory
                    signal = FallSignal(
                        track_id=track_id,
                        observed_at=timestamp,
                        confidence=round(signal_confidence, 3),
                        explanation=(
                            "A possible fall pattern was observed in a tracked human-pose sequence: "
                            "the body changed from an upright to a low horizontal posture and remained there "
                            "across multiple frames. Please check in with them. This is a safety signal for "
                            "human review, not a diagnosis."
                        ),
                        metrics={
                            "pose_model_version": str(item.get("pose_model_version") or POSE_MODEL_VERSION),
                            "pose_visible_keypoints": features.visible_keypoints,
                            "pose_quality": round(average_quality, 3),
                            "body_aspect_ratio": round(features.aspect, 3),
                            "body_axis_degrees": round(features.axis_degrees, 3),
                            "vertical_drop": round(vertical_drop, 3),
                            "confirmation_frames": state.candidate_hits,
                            "confirmation_seconds": round(candidate_duration.total_seconds(), 3),
                            "detection_confidence": round(float(confidence), 3),
                            "learned_fall_probability": round(float(learned_probability), 3) if learned_probability is not None else "test-rule-only",
                            "learned_fall_threshold": round(learned_threshold, 3),
                            "fall_sequence_model_version": str(item.get("fall_sequence_model_version") or POSE_MODEL_VERSION),
                        },
                    )
                    state.last_at = timestamp
                    state.last_center_y = features.center_y
                    state.last_posture = features.posture
                    return signal
            elif features.posture not in {"low", "transition"}:
                state.candidate_started_at = None
                state.candidate_hits = 0

            state.last_at = timestamp
            state.last_center_y = features.center_y
            state.last_posture = features.posture
            return None

    def debug_status(
        self,
        camera_id: str,
        item: dict[str, Any],
        *,
        frame_width: int,
        frame_height: int,
        observed_at: datetime,
    ) -> dict[str, Any]:
        """Expose the current gate state without changing detector state."""
        track_id = item.get("track_id")
        if not isinstance(track_id, int):
            return {"status": "not_tracking", "reason": "track_unavailable"}
        features = _pose_features(
            item,
            frame_width=frame_width,
            frame_height=frame_height,
            minimum_keypoint_confidence=self.minimum_keypoint_confidence,
            minimum_visible_keypoints=self.minimum_visible_keypoints,
        )
        if features is None:
            return {"status": "blocked", "reason": "insufficient_pose_keypoints"}
        timestamp = _utc(observed_at)
        with self._lock:
            state = self._tracks.get((camera_id, track_id))
            if state is None:
                return {"status": "tracking", "reason": "building_track_history", "posture": features.posture,
                        "visible_keypoints": features.visible_keypoints, "pose_quality": round(features.quality, 3),
                        "confirmation_hits": 0, "required_hits": self.min_hits}
            reference_y = state.upright_center_y if state.upright_center_y is not None else state.last_center_y
            drop = features.center_y - reference_y
            learned = item.get("fall_sequence_probability")
            threshold = item.get("fall_sequence_threshold", self.learned_threshold)
            learned_positive = isinstance(learned, (int, float)) and float(learned) >= float(threshold)
            if state.episode_open:
                reason = "confirmed_review_signal"
            elif features.posture != "low":
                reason = "posture_not_low"
            elif drop < 0.05:
                reason = "upright_to_low_drop_not_met"
            elif self.require_learned_model and not learned_positive:
                reason = "learned_score_below_threshold_or_unavailable"
            elif self.require_learned_model and item.get("fall_sequence_score_fresh") is not True:
                reason = "waiting_for_fresh_learned_score"
            elif state.candidate_hits < self.min_hits:
                reason = "awaiting_repeated_confirmation"
            else:
                reason = "awaiting_minimum_confirmation_time"
            elapsed = (timestamp - state.candidate_started_at).total_seconds() if state.candidate_started_at else 0.0
            return {
                "status": "confirmed" if state.episode_open else "candidate" if state.candidate_hits else "observing",
                "reason": reason,
                "posture": features.posture,
                "visible_keypoints": features.visible_keypoints,
                "pose_quality": round(features.quality, 3),
                "vertical_drop": round(drop, 3),
                "confirmation_hits": state.candidate_hits,
                "required_hits": self.min_hits,
                "confirmation_seconds": round(max(0.0, elapsed), 3),
                "required_seconds": round(self.minimum_confirmation.total_seconds(), 3),
            }


# Keep the original import name for callers and tests while making the strict
# pose-sequence implementation the only production behavior.
FallDetectionTracker = PoseSequenceFallDetector
