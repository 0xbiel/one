"""Local GPU human-pose inference for temporal safety signals.

The object detector still supplies the household-object contract.  This
runtime is a separate COCO keypoint model so a fall decision is based on a
sequence of body joints rather than a rectangle changing aspect ratio.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .gpu_runtime import GPU_COMPUTE_LOCK
from .runtime import RuntimeInferenceError, RuntimeUnavailable


class PoseEngine:
    """Bounded Ultralytics pose runtime sharing the configured accelerator."""

    keypoint_count = 17

    def __init__(self, settings: Any, runtime: Any) -> None:
        self.ready = False
        self.reason: str | None = None
        self.model_path: Path | None = getattr(settings, "pose_model_path", None)
        self.device_label = getattr(runtime, "device_label", None)
        self.detector_confidence = float(getattr(settings, "pose_detector_confidence", 0.25))
        self.keypoint_confidence = float(getattr(settings, "pose_keypoint_confidence", 0.35))
        self.model_version = "yolo11n-pose-unavailable"
        self._model: Any = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.model_path:
            self.reason = "ONE_POSE_MODEL_PATH must point to an explicit pose checkpoint"
            return
        if not self.model_path.is_file():
            self.reason = "configured pose checkpoint does not exist"
            return
        if self.device_label not in {"mps", "cuda", "cpu"}:
            self.reason = "the object runtime did not select a usable accelerator for pose inference"
            return
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.model_path), verbose=False)
            model.to(self.device_label)
        except Exception as exc:  # pragma: no cover - depends on local checkpoint/runtime
            self.reason = str(exc) or "the configured pose model could not be loaded"
            return
        self._model = model
        self.model_version = f"{self.model_path.stem}-coco-keypoints-v1"[:120]
        self.ready = True

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "unavailable",
            "model_version": self.model_version,
            "model_configured": self.model_path is not None,
            "loaded": self.ready,
            "device": self.device_label,
            "keypoint_count": self.keypoint_count,
            "detector_confidence": self.detector_confidence,
            "keypoint_confidence": self.keypoint_confidence,
            "reason": self.reason,
            "raw_frames_persisted": False,
        }

    @staticmethod
    def _values(value: Any) -> list:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().tolist()
        return np.asarray(value).tolist()

    def extract(self, image: np.ndarray | bytes, width: int, height: int) -> list[dict[str, Any]]:
        if not self.ready or self._model is None:
            raise RuntimeUnavailable(self.reason or "pose runtime is unavailable")
        if isinstance(image, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeInferenceError("pose image is not a decodable JPEG")
        if image.ndim != 3 or image.shape[1] != width or image.shape[0] != height:
            raise RuntimeInferenceError("pose image dimensions do not match the request")
        try:
            with GPU_COMPUTE_LOCK:
                with self._lock:
                    results = self._model.predict(
                        source=image,
                        device=self.device_label,
                        imgsz=640,
                        conf=self.detector_confidence,
                        iou=0.7,
                        max_det=20,
                        verbose=False,
                    )
            if not results:
                return []
            result = results[0]
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None or keypoints is None:
                return []
            raw_boxes = self._values(getattr(boxes, "xyxy", []))
            raw_scores = self._values(getattr(boxes, "conf", []))
            raw_keypoints = self._values(getattr(keypoints, "data", []))
            output: list[dict[str, Any]] = []
            for raw_box, raw_score, raw_points in zip(raw_boxes, raw_scores, raw_keypoints):
                if len(raw_box) != 4 or len(raw_points) != self.keypoint_count:
                    continue
                score = float(raw_score)
                if score < self.detector_confidence:
                    continue
                points: list[list[float]] = []
                visible_scores: list[float] = []
                for point in raw_points:
                    if len(point) < 2:
                        points = []
                        break
                    x, y = float(point[0]), float(point[1])
                    confidence = float(point[2]) if len(point) >= 3 else 1.0
                    if not np.isfinite([x, y, confidence]).all():
                        points = []
                        break
                    confidence = max(0.0, min(1.0, confidence))
                    points.append([
                        round(min(max(x, 0.0), float(width)), 3),
                        round(min(max(y, 0.0), float(height)), 3),
                        round(confidence, 5),
                    ])
                    if confidence >= self.keypoint_confidence:
                        visible_scores.append(confidence)
                if len(points) != self.keypoint_count or len(visible_scores) < 5:
                    continue
                x1, y1, x2, y2 = (float(value) for value in raw_box)
                if x2 <= x1 or y2 <= y1:
                    continue
                output.append({
                    "bbox": [
                        round(min(max(x1, 0.0), float(width)), 3),
                        round(min(max(y1, 0.0), float(height)), 3),
                        round(min(max(x2, 0.0), float(width)), 3),
                        round(min(max(y2, 0.0), float(height)), 3),
                    ],
                    "confidence": round(score, 6),
                    "keypoints": points,
                    "keypoint_confidence": round(sum(visible_scores) / len(visible_scores), 5),
                })
            return output[:20]
        except (RuntimeUnavailable, RuntimeInferenceError):
            raise
        except Exception as exc:  # pragma: no cover - depends on accelerator/runtime
            raise RuntimeInferenceError("the configured pose model failed during inference") from exc


def attach_pose_detections(
    detections: list[dict[str, Any]],
    poses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach poses to person boxes with a one-to-one global assignment."""

    def iou(left: list[float], right: list[float]) -> float:
        lx1, ly1, lx2, ly2 = left
        rx1, ry1, rx2, ry2 = right
        ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    candidates: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        if str(detection.get("label") or "").strip().lower() != "person":
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        for pose_index, pose in enumerate(poses):
            pose_bbox = pose.get("bbox")
            if not isinstance(pose_bbox, list) or len(pose_bbox) != 4:
                continue
            overlap = iou(bbox, pose_bbox)
            if overlap >= 0.15:
                candidates.append((overlap, detection_index, pose_index))

    used_detections: set[int] = set()
    used_poses: set[int] = set()
    for _overlap, detection_index, pose_index in sorted(candidates, reverse=True):
        if detection_index in used_detections or pose_index in used_poses:
            continue
        detection = detections[detection_index]
        pose = poses[pose_index]
        detection["pose_keypoints"] = pose["keypoints"]
        detection["pose_confidence"] = pose["keypoint_confidence"]
        used_detections.add(detection_index)
        used_poses.add(pose_index)
    return detections
