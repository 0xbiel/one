"""Bounded vision stabilization and RoomPlan-aware projection primitives.

Production detection is delegated to the private local YOLO-World geometry
worker. The deterministic and OWLv2 adapters remain available only as explicit
test/integration seams; production wiring does not silently fall back to them.
"""
from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Frame:
    camera_id: str
    data: bytes
    width: int
    height: int
    captured_at: datetime


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    frame_at: datetime
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class Detector(Protocol):
    model_version: str

    def detect(self, frame: Frame, candidate_labels: Sequence[str]) -> list[Detection]: ...


class DeterministicDemoDetector:
    """No-download detector used for demos and tests.

    It returns one repeatable centre-ish detection for the first allowlisted label.
    Frame bytes only choose a small deterministic horizontal offset; they are never
    persisted by this component.
    """

    model_version = "demo-deterministic-v1"

    def detect(self, frame: Frame, candidate_labels: Sequence[str]) -> list[Detection]:
        if not candidate_labels or not frame.data:
            return []
        offset = int(hashlib.sha256(frame.data).hexdigest()[:2], 16) / 2550 - 0.05
        cx = min(max(frame.width * (0.5 + offset), frame.width * 0.2), frame.width * 0.8)
        cy = frame.height * 0.5
        w, h = frame.width * 0.2, frame.height * 0.2
        return [Detection(candidate_labels[0], 0.85, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), frame.captured_at)]


class OWLv2Detector:
    """OWLv2-compatible interface; model execution is intentionally injected."""

    model_version = "owlv2-unconfigured"

    def __init__(self, infer=None):
        self._infer = infer

    def detect(self, frame: Frame, candidate_labels: Sequence[str]) -> list[Detection]:
        if self._infer is None:
            raise RuntimeError("OWLv2 is not configured; use DeterministicDemoDetector for offline demos")
        return list(self._infer(frame, candidate_labels))


class LocalServiceDetector:
    """Production detector backed by the host-side real YOLO-World service."""

    model_version = "local-yolo-world-unavailable"

    def __init__(self, service: object):
        self._service = service

    def detect(self, frame: Frame, candidate_labels: Sequence[str]) -> list[Detection]:
        detect = getattr(self._service, "detect", None)
        if not callable(detect):
            raise RuntimeError("real local vision service is not configured")
        result = detect(
            frame_base64=base64.b64encode(frame.data).decode("ascii"),
            width=frame.width,
            height=frame.height,
            candidate_labels=list(candidate_labels),
        )
        if not isinstance(result, dict) or result.get("status") != "ready":
            raise RuntimeError("real local vision service is unavailable")
        version = result.get("model_version")
        if isinstance(version, str) and version:
            self.model_version = version[:120]
        raw_detections = result.get("detections")
        if not isinstance(raw_detections, list):
            raise RuntimeError("real local vision service returned an invalid response")
        detections: list[Detection] = []
        for item in raw_detections:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            confidence = item.get("confidence")
            bbox = item.get("bbox")
            if (
                not isinstance(label, str)
                or not isinstance(confidence, (int, float))
                or not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                continue
            values = tuple(float(value) for value in bbox)
            if not all(math.isfinite(value) for value in values):
                continue
            detections.append(Detection(label, float(confidence), values, frame.captured_at))
        return detections


@dataclass
class _Track:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    hits: int
    first_at: datetime
    last_at: datetime


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


class TemporalStabilityTracker:
    def __init__(self, min_hits: int = 3, window_seconds: float = 4.0, iou_threshold: float = 0.2):
        self.min_hits, self.window_seconds, self.iou_threshold = min_hits, window_seconds, iou_threshold
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        stable: list[Detection] = []
        used_track_ids: set[int] = set()
        for detection in detections:
            matches = [
                track
                for track in self._tracks
                if track.track_id not in used_track_ids
                and track.label == detection.label
                and _iou(track.bbox, detection.bbox) >= self.iou_threshold
                and (detection.frame_at - track.last_at).total_seconds() <= self.window_seconds
            ]
            match = max(matches, key=lambda track: _iou(track.bbox, detection.bbox), default=None)
            if match:
                match.bbox, match.last_at, match.hits = detection.bbox, detection.frame_at, match.hits + 1
            else:
                match = _Track(self._next_track_id, detection.label, detection.bbox, 1, detection.frame_at, detection.frame_at)
                self._next_track_id += 1
                self._tracks.append(match)
            used_track_ids.add(match.track_id)
            if match.hits >= self.min_hits:
                stable.append(Detection(detection.label, detection.confidence, detection.bbox, detection.frame_at, match.track_id))
        self._tracks = [track for track in self._tracks if (datetime.now(timezone.utc) - track.last_at).total_seconds() <= self.window_seconds]
        return stable


@dataclass(frozen=True)
class Calibration:
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: tuple[tuple[float, float, float, float], ...]
    accuracy_m: float = 0.5
    floor_y: float | None = None


@dataclass(frozen=True)
class Projection:
    world_xyz: tuple[float, float, float] | None
    uncertainty_m: float
    zone: str
    quality: str


def _zone(x: float, y: float, width: int, height: int) -> str:
    col = "left" if x < width / 3 else "right" if x > width * 2 / 3 else "centre"
    row = "top" if y < height / 3 else "bottom" if y > height * 2 / 3 else "middle"
    return f"{row}-{col}"


def project_detection(detection: Detection, frame: Frame, calibration: Calibration | None, depth_m: float | None = None, depth_uncertainty_m: float = 0.75) -> Projection:
    x, y = detection.center
    if calibration is None or calibration.fx <= 0 or calibration.fy <= 0:
        return Projection(None, max(2.0, (calibration.accuracy_m if calibration else 0.5) * 4), _zone(x, y, frame.width, frame.height), "zone-fallback")
    matrix = calibration.camera_to_world
    if depth_m is not None and depth_m > 0:
        # Stored RoomPlan extrinsics use ARKit camera coordinates: +X right,
        # +Y up and -Z forward. Image y increases downward.
        cam = (
            (x - calibration.cx) * depth_m / calibration.fx,
            -(y - calibration.cy) * depth_m / calibration.fy,
            -depth_m,
            1.0,
        )
        world = tuple(sum(matrix[row][col] * cam[col] for col in range(4)) for row in range(3))
        pixel_error_m = depth_m * max(1 / calibration.fx, 1 / calibration.fy) * 8
        uncertainty = math.sqrt(calibration.accuracy_m**2 + depth_uncertainty_m**2 + pixel_error_m**2)
        return Projection((world[0], world[1], world[2]), uncertainty, _zone(x, y, frame.width, frame.height), "calibrated-depth")

    if calibration.floor_y is None:
        return Projection(None, max(2.0, calibration.accuracy_m * 4), _zone(x, y, frame.width, frame.height), "zone-fallback")

    # Monocular fixed cameras do not provide per-pixel depth. Approximate an
    # object's ground contact by intersecting the ray through the bottom centre
    # of its bounding box with the RoomPlan floor plane.
    x1, _, x2, y2 = detection.bbox
    px = (x1 + x2) / 2.0
    py = y2
    local_direction = (
        (px - calibration.cx) / calibration.fx,
        -(py - calibration.cy) / calibration.fy,
        -1.0,
    )
    origin = (matrix[0][3], matrix[1][3], matrix[2][3])
    direction = tuple(
        sum(matrix[row][col] * local_direction[col] for col in range(3))
        for row in range(3)
    )
    if abs(direction[1]) < 1e-6:
        return Projection(None, max(2.0, calibration.accuracy_m * 4), _zone(x, y, frame.width, frame.height), "zone-fallback")
    distance = (calibration.floor_y - origin[1]) / direction[1]
    if distance <= 0 or distance > 30:
        return Projection(None, max(2.0, calibration.accuracy_m * 4), _zone(x, y, frame.width, frame.height), "zone-fallback")
    world = tuple(origin[index] + direction[index] * distance for index in range(3))
    uncertainty = max(0.45, calibration.accuracy_m + distance * 0.06)
    return Projection(world, uncertainty, _zone(x, y, frame.width, frame.height), "calibrated-floor-ray")


class CameraVisionPipeline:
    """Bounded frame-to-detection pipeline; one detector call per submitted frame."""

    def __init__(self, detector: Detector, tracker: TemporalStabilityTracker | None = None):
        self.detector = detector
        self._seed_tracker = tracker
        self._trackers: dict[str, TemporalStabilityTracker] = {}

    def ingest(self, frame: Frame, candidate_labels: Sequence[str], calibration: Calibration | None = None, depth_m: float | None = None) -> list[dict]:
        detections = self.detector.detect(frame, candidate_labels)
        tracker = self._trackers.get(frame.camera_id)
        if tracker is None:
            if self._seed_tracker is not None and not self._trackers:
                tracker = self._seed_tracker
            else:
                tracker = TemporalStabilityTracker()
            self._trackers[frame.camera_id] = tracker
        stable = tracker.update(detections)
        return [{"label": item.label, "confidence": item.confidence, "bbox": item.bbox, "track_id": item.track_id, "projection": project_detection(item, frame, calibration, depth_m).__dict__} for item in stable]
