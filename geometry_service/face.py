"""Local face detection and embedding extraction.

The worker returns derived vectors to the API over the private local network;
it never writes submitted frames or aligned crops. Identity matching and
consent enforcement remain API responsibilities.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:  # Optional in API-only development environments.
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

from .runtime import RuntimeInferenceError, RuntimeUnavailable


class FaceEngine:
    model_version = "yunet-sface-minifas-unavailable"

    def __init__(self, settings: Any) -> None:
        self.ready = False
        self.reason: str | None = None
        self.detector_model_path: Path | None = getattr(settings, "face_detector_model_path", None)
        self.recognizer_model_path: Path | None = getattr(settings, "face_recognizer_model_path", None)
        self.liveness_model_path: Path | None = getattr(settings, "face_liveness_model_path", None)
        self.confidence = float(getattr(settings, "face_detector_confidence", 0.90))
        self.liveness_threshold = float(getattr(settings, "face_liveness_threshold", 0.0))
        self.liveness_min_real_probability = float(getattr(settings, "face_liveness_min_real_probability", 0.65))
        self.liveness_min_saturation = float(getattr(settings, "face_liveness_min_saturation", 0.04))
        self._detector: Any = None
        self._recognizer: Any = None
        self._liveness: Any = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if cv2 is None or np is None:
            self.reason = "OpenCV is not installed in the geometry worker environment"
            return
        if not self.detector_model_path or not self.recognizer_model_path or not self.liveness_model_path:
            self.reason = (
                "ONE_FACE_DETECTOR_MODEL_PATH, ONE_FACE_RECOGNIZER_MODEL_PATH, and "
                "ONE_FACE_LIVENESS_MODEL_PATH must point to explicit files"
            )
            return
        if (
            not self.detector_model_path.is_file()
            or not self.recognizer_model_path.is_file()
            or not self.liveness_model_path.is_file()
        ):
            self.reason = "configured face detector, recognizer, or liveness model does not exist"
            return
        try:
            detector_factory = getattr(getattr(cv2, "FaceDetectorYN", None), "create", None)
            recognizer_factory = getattr(getattr(cv2, "FaceRecognizerSF", None), "create", None)
            if detector_factory is None or recognizer_factory is None:
                raise RuntimeUnavailable("OpenCV FaceDetectorYN and FaceRecognizerSF are required")
            self._detector = detector_factory(
                str(self.detector_model_path),
                "",
                (320, 320),
                self.confidence,
                0.3,
                5000,
            )
            self._recognizer = recognizer_factory(str(self.recognizer_model_path), "")
            self._liveness = cv2.dnn.readNetFromONNX(str(self.liveness_model_path))
        except Exception as exc:  # pragma: no cover - depends on optional model files
            self.reason = str(exc) or "the configured face or liveness models could not be loaded"
            return
        self.ready = True
        self.model_version = "yunet-sface-minifas-v2se-2021dec-v1"

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "unavailable",
            "model_version": self.model_version,
            "detector_configured": self.detector_model_path is not None,
            "recognizer_configured": self.recognizer_model_path is not None,
            "liveness": {
                "status": "ready" if self._liveness is not None and self.ready else "unavailable",
                "model_version": "minifas-v2se-celab-spoof-v1",
                "model_configured": self.liveness_model_path is not None,
                "threshold": self.liveness_threshold,
                "min_real_probability": self.liveness_min_real_probability,
                "min_saturation": self.liveness_min_saturation,
            },
            "reason": self.reason,
            "raw_frames_persisted": False,
        }

    @staticmethod
    def _face_row(row: Any) -> dict[str, Any] | None:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size < 15:
            return None
        x, y, width, height = (float(value) for value in values[:4])
        confidence = float(values[14])
        if width <= 0 or height <= 0 or confidence <= 0:
            return None
        return {
            "bbox": [round(x, 3), round(y, 3), round(x + width, 3), round(y + height, 3)],
            "confidence": round(confidence, 6),
            "landmarks": [round(float(value), 3) for value in values[4:14]],
            "yaw": round(FaceEngine._landmark_yaw(values[4:14]), 6),
        }

    @staticmethod
    def _landmark_yaw(values: Any) -> float:
        """Return a normalized, camera-relative yaw proxy from YuNet landmarks."""

        points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 3:
            return 0.0
        left_eye, right_eye, nose = points[0], points[1], points[2]
        eye_mid = (left_eye + right_eye) / 2.0
        eye_span = float(np.linalg.norm(left_eye - right_eye))
        if not np.isfinite(eye_span) or eye_span <= 1e-4:
            return 0.0
        # The sign is camera-relative; liveness only needs a centered view and
        # two opposite side views, so mirrored front cameras are harmless.
        return float(np.clip((float(nose[0]) - float(eye_mid[0])) / eye_span, -2.0, 2.0))

    @staticmethod
    def _square_face_crop(image: Any, bbox: list[float], expansion: float = 2.7) -> Any:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            raise RuntimeInferenceError("face crop has invalid dimensions")
        size = max(width, height) * expansion
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        left = int(round(center_x - size / 2.0))
        top = int(round(center_y - size / 2.0))
        right = left + max(1, int(round(size)))
        bottom = top + max(1, int(round(size)))
        crop_left, crop_top = max(0, left), max(0, top)
        crop_right, crop_bottom = min(image.shape[1], right), min(image.shape[0], bottom)
        crop = image[crop_top:crop_bottom, crop_left:crop_right]
        if crop.size == 0:
            raise RuntimeInferenceError("face crop is outside the image")
        padded = cv2.copyMakeBorder(
            crop,
            max(0, -top),
            max(0, bottom - image.shape[0]),
            max(0, -left),
            max(0, right - image.shape[1]),
            cv2.BORDER_REFLECT_101,
        )
        return cv2.resize(padded, (128, 128), interpolation=cv2.INTER_AREA)

    def _liveness_result(self, image: Any, bbox: list[float]) -> dict[str, Any]:
        if self._liveness is None:
            raise RuntimeUnavailable("face liveness model is unavailable")
        crop_bgr = self._square_face_crop(image, bbox)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = crop_rgb.transpose(2, 0, 1).astype(np.float32)[None, ...] / 255.0
        self._liveness.setInput(tensor)
        raw = np.asarray(self._liveness.forward(), dtype=np.float32).reshape(-1)
        if raw.size < 2 or not np.isfinite(raw[:2]).all():
            raise RuntimeInferenceError("face liveness model returned an invalid score")
        logits = raw[:2]
        logits = logits - float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        real_probability = float(probabilities[0])
        logit_difference = float(raw[0] - raw[1])
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        # A color-diversity floor is a deliberately narrow secondary PAD cue:
        # it rejects the grayscale/blurred print proxy that MiniFAS alone
        # accepted in local testing, while remaining explicit and measurable.
        is_live = (
            logit_difference >= self.liveness_threshold
            and real_probability >= self.liveness_min_real_probability
            and mean_saturation >= self.liveness_min_saturation
        )
        return {
            "status": "live" if is_live else "spoof",
            "score": round(real_probability, 6),
            "logit_difference": round(logit_difference, 6),
            "mean_saturation": round(mean_saturation, 6),
            "min_saturation": round(self.liveness_min_saturation, 6),
            "model_version": "minifas-v2se-celab-spoof-v1",
        }

    def extract(self, jpeg_bytes: bytes, width: int, height: int) -> list[dict[str, Any]]:
        if cv2 is None or np is None or not self.ready or self._detector is None or self._recognizer is None:
            raise RuntimeUnavailable(self.reason or "face runtime is unavailable")
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise RuntimeInferenceError("the submitted frame is not a decodable JPEG")
        if image.shape[1] != width or image.shape[0] != height:
            raise RuntimeInferenceError("decoded JPEG dimensions do not match the request")
        with self._lock:
            self._detector.setInputSize((width, height))
            _ignored, faces = self._detector.detect(image)
            if faces is None:
                return []
            results: list[dict[str, Any]] = []
            for raw_face in np.asarray(faces):
                face = self._face_row(raw_face)
                if face is None:
                    continue
                aligned = self._recognizer.alignCrop(image, raw_face)
                feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(feature))
                if not np.isfinite(norm) or norm <= 0:
                    continue
                feature = feature / norm
                face["embedding"] = [round(float(value), 8) for value in feature]
                face["liveness"] = self._liveness_result(image, face["bbox"])
                results.append(face)
            return results[:8]


def active_liveness_challenge(yaws: list[float]) -> dict[str, Any]:
    """Evaluate the ordered center/side/opposite-side enrollment challenge."""

    finite: list[float] = []
    for value in yaws:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite.append(numeric)
    baseline = finite[0] if finite else 0.0
    center_seen = bool(finite) and abs(baseline) <= 0.18
    relative = [value - baseline for value in finite[1:]]
    first_side_index = next((index for index, value in enumerate(relative) if abs(value) >= 0.18), None)
    first_side = relative[first_side_index] if first_side_index is not None else None
    opposite_side_index = None
    if first_side_index is not None and first_side is not None:
        opposite_side_index = next(
            (
                index
                for index in range(first_side_index + 1, len(relative))
                if abs(relative[index]) >= 0.18 and relative[index] * first_side < 0
            ),
            None,
        )
    positive_side_seen = any(value >= 0.18 for value in relative)
    negative_side_seen = any(value <= -0.18 for value in relative)
    ordered_motion = first_side_index is not None and opposite_side_index is not None
    span = max(finite) - min(finite) if finite else 0.0
    failure_reasons: list[str] = []
    if len(finite) < 3:
        failure_reasons.append("not_enough_samples")
    if not center_seen:
        failure_reasons.append("first_sample_not_centered")
    if not ordered_motion:
        failure_reasons.append("ordered_side_to_opposite_motion_missing")
    if span < 0.32:
        failure_reasons.append("yaw_span_too_small")
    passed = len(finite) >= 3 and center_seen and ordered_motion and span >= 0.32
    return {
        "status": "passed" if passed else "failed",
        "required": ["center", "turn_one_side", "turn_opposite_side"],
        "center_seen": center_seen,
        "positive_side_seen": positive_side_seen,
        "negative_side_seen": negative_side_seen,
        "ordered_motion": ordered_motion,
        "sample_count": len(finite),
        "center_reference_yaw": round(baseline, 4),
        "yaw_span": round(span, 4),
        "failure_reasons": failure_reasons,
    }
