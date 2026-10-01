"""Environment-backed configuration for the geometry service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class ServiceSettings:
    """Configuration with safe bounds for a single in-memory request."""

    mode: str
    checkpoint_path: Path | None
    config_path: Path | None
    device_preference: str
    allow_cpu: bool
    max_frames: int
    max_frame_bytes: int
    max_total_frame_bytes: int
    max_request_bytes: int
    minimum_confidence: float
    positioning_workers: int
    face_detector_model_path: Path | None = None
    face_recognizer_model_path: Path | None = None
    face_detector_confidence: float = 0.90
    face_liveness_model_path: Path | None = None
    face_liveness_threshold: float = 0.0
    face_liveness_min_real_probability: float = 0.65
    face_liveness_min_saturation: float = 0.04
    pose_model_path: Path | None = None
    pose_detector_confidence: float = 0.25
    pose_keypoint_confidence: float = 0.35
    fall_sequence_model_path: Path | None = None
    fall_sequence_threshold: float = 0.65
    fall_sequence_window: int = 15

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        mode = os.getenv("ONE_GEOMETRY_MODE", "model").strip().lower()
        if mode == "production":
            mode = "model"

        # ONE_* names are the public configuration contract. The real model
        # path is explicit so an absent checkpoint cannot become a fixture.
        checkpoint = os.getenv("ONE_GEOMETRY_MODEL_PATH", "").strip()
        config = os.getenv("ONE_GEOMETRY_MODEL_CONFIG", "").strip()
        if not config:
            config = str(Path(__file__).with_name("model_config.yolo-world.json"))
        # The browser normally submits 16 frames. Keep the public service
        # contract aligned with the backend's hard ceiling of 20 so an
        # environment override cannot turn this into an unbounded batch.
        max_frames = min(max(_env_int("GEOMETRY_MAX_FRAMES", 20), 16), 20)
        max_frame_bytes = min(
            max(_env_int("GEOMETRY_MAX_FRAME_BYTES", 3_000_000), 64_000),
            3_000_000,
        )
        max_total_frame_bytes = min(
            max(_env_int("GEOMETRY_MAX_TOTAL_FRAME_BYTES", 18_000_000), max_frame_bytes),
            18_000_000,
        )
        max_request_bytes = min(
            max(_env_int("GEOMETRY_MAX_REQUEST_BYTES", 26_000_000), 256_000),
            32_000_000,
        )
        minimum_confidence = min(
            max(_env_float("GEOMETRY_MIN_CONFIDENCE", 0.60), 0.0),
            1.0,
        )
        positioning_workers = min(
            max(_env_int("ONE_POSITIONING_WORKERS", 3), 1),
            8,
        )
        face_detector = os.getenv("ONE_FACE_DETECTOR_MODEL_PATH", "").strip()
        face_recognizer = os.getenv("ONE_FACE_RECOGNIZER_MODEL_PATH", "").strip()
        face_detector_confidence = min(
            max(_env_float("ONE_FACE_DETECTOR_CONFIDENCE", 0.90), 0.50),
            0.99,
        )
        face_liveness = os.getenv("ONE_FACE_LIVENESS_MODEL_PATH", "").strip()
        face_liveness_threshold = min(
            max(_env_float("ONE_FACE_LIVENESS_THRESHOLD", 0.0), -10.0),
            10.0,
        )
        face_liveness_min_real_probability = min(
            max(_env_float("ONE_FACE_LIVENESS_MIN_REAL_PROBABILITY", 0.65), 0.50),
            0.99,
        )
        face_liveness_min_saturation = min(
            max(_env_float("ONE_FACE_LIVENESS_MIN_SATURATION", 0.04), 0.0),
            1.0,
        )
        pose_model = os.getenv("ONE_POSE_MODEL_PATH", "").strip()
        pose_detector_confidence = min(
            max(_env_float("ONE_POSE_DETECTOR_CONFIDENCE", 0.25), 0.05),
            0.95,
        )
        pose_keypoint_confidence = min(
            max(_env_float("ONE_POSE_KEYPOINT_CONFIDENCE", 0.35), 0.05),
            0.95,
        )
        fall_sequence_model = os.getenv("ONE_FALL_SEQUENCE_MODEL_PATH", "").strip()
        fall_sequence_threshold = min(
            max(_env_float("ONE_FALL_SEQUENCE_THRESHOLD", 0.65), 0.50),
            0.99,
        )
        fall_sequence_window = min(
            max(_env_int("ONE_FALL_SEQUENCE_WINDOW", 15), 8),
            64,
        )

        return cls(
            mode=mode,
            checkpoint_path=Path(checkpoint).expanduser() if checkpoint else None,
            config_path=Path(config).expanduser() if config else None,
            device_preference=os.getenv("ONE_GEOMETRY_DEVICE", os.getenv("GEOMETRY_SERVICE_DEVICE", "auto")).strip().lower(),
            allow_cpu=_env_bool(
                "ONE_GEOMETRY_ALLOW_CPU",
                _env_bool("GEOMETRY_SERVICE_ALLOW_CPU", False),
            ),
            max_frames=max_frames,
            max_frame_bytes=max_frame_bytes,
            max_total_frame_bytes=max_total_frame_bytes,
            max_request_bytes=max_request_bytes,
            minimum_confidence=minimum_confidence,
            positioning_workers=positioning_workers,
            face_detector_model_path=Path(face_detector).expanduser() if face_detector else None,
            face_recognizer_model_path=Path(face_recognizer).expanduser() if face_recognizer else None,
            face_detector_confidence=face_detector_confidence,
            face_liveness_model_path=Path(face_liveness).expanduser() if face_liveness else None,
            face_liveness_threshold=face_liveness_threshold,
            face_liveness_min_real_probability=face_liveness_min_real_probability,
            face_liveness_min_saturation=face_liveness_min_saturation,
            pose_model_path=Path(pose_model).expanduser() if pose_model else None,
            pose_detector_confidence=pose_detector_confidence,
            pose_keypoint_confidence=pose_keypoint_confidence,
            fall_sequence_model_path=Path(fall_sequence_model).expanduser() if fall_sequence_model else None,
            fall_sequence_threshold=fall_sequence_threshold,
            fall_sequence_window=fall_sequence_window,
        )
