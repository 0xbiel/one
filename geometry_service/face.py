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
    model_version = "yunet-sface-unavailable"

    def __init__(self, settings: Any) -> None:
        self.ready = False
        self.reason: str | None = None
        self.detector_model_path: Path | None = getattr(settings, "face_detector_model_path", None)
        self.recognizer_model_path: Path | None = getattr(settings, "face_recognizer_model_path", None)
        self.confidence = float(getattr(settings, "face_detector_confidence", 0.90))
        self._detector: Any = None
        self._recognizer: Any = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if cv2 is None or np is None:
            self.reason = "OpenCV is not installed in the geometry worker environment"
            return
        if not self.detector_model_path or not self.recognizer_model_path:
            self.reason = "ONE_FACE_DETECTOR_MODEL_PATH and ONE_FACE_RECOGNIZER_MODEL_PATH must point to explicit files"
            return
        if not self.detector_model_path.is_file() or not self.recognizer_model_path.is_file():
            self.reason = "configured face detector or recognizer model does not exist"
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
        except Exception as exc:  # pragma: no cover - depends on optional model files
            self.reason = str(exc) or "the configured face models could not be loaded"
            return
        self.ready = True
        self.model_version = "yunet-sface-2021dec-v1"

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "unavailable",
            "model_version": self.model_version,
            "detector_configured": self.detector_model_path is not None,
            "recognizer_configured": self.recognizer_model_path is not None,
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
                results.append(face)
            return results[:8]
