"""Real, bounded YOLO-World object detection for transient camera frames."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class VisionInferenceError(RuntimeError):
    pass


def decode_jpeg(jpeg_bytes: bytes, expected_width: int, expected_height: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise VisionInferenceError("the submitted frame is not a decodable JPEG")
    height, width = image.shape[:2]
    if width != expected_width or height != expected_height:
        raise VisionInferenceError("decoded JPEG dimensions do not match the request")
    return image


def _names(result: Any) -> dict[int, str]:
    raw = getattr(result, "names", {})
    if isinstance(raw, dict):
        return {int(key): str(value).strip().lower() for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {index: str(value).strip().lower() for index, value in enumerate(raw)}
    return {}


def detections_from_result(result: Any, *, width: int, height: int, minimum_confidence: float) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    confidences = getattr(boxes, "conf", None)
    classes = getattr(boxes, "cls", None)
    if xyxy is None or confidences is None or classes is None:
        return []
    coordinates = xyxy.detach().cpu().tolist() if hasattr(xyxy, "detach") else list(xyxy)
    scores = confidences.detach().cpu().tolist() if hasattr(confidences, "detach") else list(confidences)
    class_ids = classes.detach().cpu().tolist() if hasattr(classes, "detach") else list(classes)
    names = _names(result)
    detections: list[dict[str, Any]] = []
    for raw_box, raw_score, raw_class in zip(coordinates, scores, class_ids):
        if len(raw_box) != 4:
            continue
        score = float(raw_score)
        if score < minimum_confidence:
            continue
        label = names.get(int(raw_class), "")
        if not label:
            continue
        x1, y1, x2, y2 = (float(value) for value in raw_box)
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "label": label,
                "confidence": round(score, 6),
                "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            }
        )
    return detections[:100]

