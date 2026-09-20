"""Lazy real-model runtime loading with truthful readiness state."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import ServiceSettings
from .gpu_runtime import GPU_COMPUTE_LOCK
from .low_light import enhance_low_light_image


class RuntimeUnavailable(RuntimeError):
    """The configured production runtime cannot serve inference."""


class RuntimeInferenceError(RuntimeError):
    """The loaded model failed while producing an output."""


class RuntimeNeedsRescan(RuntimeError):
    """The model ran, but the sweep did not contain enough stable room structure."""


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeUnavailable(f"model config {field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeUnavailable(f"model config {field_name} must be finite")
    return result


def _validate_model_config(raw: Any, settings: ServiceSettings) -> tuple[dict[str, Any], float]:
    if not isinstance(raw, dict):
        raise RuntimeUnavailable("model config must contain a JSON object")
    if raw.get("backend") != "ultralytics-yolo-world":
        raise RuntimeUnavailable("model config backend must be ultralytics-yolo-world")
    model_version = raw.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise RuntimeUnavailable("model config must declare model_version")
    if raw.get("output_contract") != "camera-room-2d/v1":
        raise RuntimeUnavailable("model config output_contract must be camera-room-2d/v1")

    input_config = raw.get("input")
    if not isinstance(input_config, dict):
        raise RuntimeUnavailable("model config must declare input width and height")
    try:
        input_width = int(input_config["width"])
        input_height = int(input_config["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeUnavailable("model config input.width and input.height are required") from exc
    if not (16 <= input_width <= 2_048 and 16 <= input_height <= 2_048):
        raise RuntimeUnavailable("model config input dimensions must be between 16 and 2048")

    model_name = raw.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise RuntimeUnavailable("model config must declare model_name")

    prompts = raw.get("prompts")
    if (
        not isinstance(prompts, list)
        or not 3 <= len(prompts) <= 128
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise RuntimeUnavailable("model config prompts must contain 3 to 128 non-empty strings")

    normalization = raw.get("normalization", {})
    if not isinstance(normalization, dict):
        raise RuntimeUnavailable("model config normalization must be an object")
    for key in ("mean", "std"):
        values = normalization.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise RuntimeUnavailable(f"model config normalization.{key} must contain three values")
        for index, value in enumerate(values):
            numeric = _number(value, f"normalization.{key}[{index}]")
            if key == "std" and numeric <= 0:
                raise RuntimeUnavailable("model config normalization.std values must be positive")

    configured_threshold = raw.get("minimum_confidence", settings.minimum_confidence)
    threshold = _number(configured_threshold, "minimum_confidence")
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeUnavailable("model config minimum_confidence must be between 0 and 1")
    detection_threshold = _number(raw.get("detection_confidence", 0.20), "detection_confidence")
    if not 0.0 <= detection_threshold <= 1.0:
        raise RuntimeUnavailable("model config detection_confidence must be between 0 and 1")
    return raw, threshold


def _select_device(torch: Any, settings: ServiceSettings) -> tuple[Any, str]:
    preference = settings.device_preference or "auto"
    if preference not in {"auto", "mps", "cuda", "cpu"}:
        raise RuntimeUnavailable("ONE_GEOMETRY_DEVICE must be auto, mps, cuda, or cpu")

    mps_available = False
    try:
        mps_available = bool(torch.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        mps_available = False
    cuda_available = bool(torch.cuda.is_available())

    if preference in {"auto", "mps"} and mps_available:
        return torch.device("mps"), "mps"
    if preference in {"auto", "cuda"} and cuda_available:
        return torch.device("cuda"), "cuda"
    if preference == "mps":
        raise RuntimeUnavailable("PyTorch MPS is not available on this host")
    if preference == "cuda":
        raise RuntimeUnavailable("PyTorch CUDA is not available on this host")
    if preference == "cpu" and not settings.allow_cpu:
        raise RuntimeUnavailable("CPU inference is disabled; set ONE_GEOMETRY_ALLOW_CPU=1 explicitly")
    if preference == "auto" and not settings.allow_cpu:
        raise RuntimeUnavailable(
            "no supported GPU accelerator is available; PyTorch MPS/CUDA is required in model mode"
        )
    return torch.device("cpu"), "cpu"


def _to_python(value: Any) -> Any:
    """Convert common TorchScript return values without retaining tensors."""

    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _to_python(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


class RoomLayoutRuntime:
    """One process-local model runtime; request frames never leave memory."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.ready = False
        self.mode = settings.mode
        self.device_label: str | None = None
        self.torch_version: str | None = None
        self.model_version = "unavailable"
        self.model_config: dict[str, Any] = {}
        self.minimum_confidence = settings.minimum_confidence
        self.reason: str | None = None
        self._torch: Any = None
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.mode != "model":
            self.reason = "only real model mode is supported; set ONE_GEOMETRY_MODE=model"
            return
        if self.settings.checkpoint_path is None or self.settings.config_path is None:
            self.reason = (
                "ONE_GEOMETRY_MODEL_PATH and ONE_GEOMETRY_MODEL_CONFIG must point to explicit files"
            )
            return
        if not self.settings.checkpoint_path.is_file():
            self.reason = "configured model checkpoint does not exist"
            return
        if not self.settings.config_path.is_file():
            self.reason = "configured model config does not exist"
            return

        try:
            import torch
            from ultralytics import YOLOWorld
        except ImportError:
            self.reason = "PyTorch and Ultralytics are required for the real geometry model"
            return
        try:
            import numpy  # noqa: F401
            import PIL  # noqa: F401
        except ImportError:
            self.reason = "Pillow and NumPy are not installed"
            return
        self._torch = torch
        self.torch_version = str(getattr(torch, "__version__", "unknown"))

        try:
            device, device_label = _select_device(torch, self.settings)
            raw_config = json.loads(self.settings.config_path.read_text(encoding="utf-8"))
            model_config, threshold = _validate_model_config(raw_config, self.settings)
            model = YOLOWorld(str(self.settings.checkpoint_path), verbose=False)
            model.set_classes(model_config["prompts"])
            model.to(device)
            # Ultralytics' cached CLIP wrapper keeps its own ``device`` field.
            # ``nn.Module.to()`` moves the CLIP weights but does not update that
            # field, so subsequent ``set_classes()`` calls would tokenize on CPU
            # and then feed CPU tokens into MPS weights.
            clip_model = getattr(model.model, "clip_model", None)
            if clip_model is not None and hasattr(clip_model, "device"):
                clip_model.device = device
        except Exception as exc:  # pragma: no cover - depends on host/runtime/checkpoint
            self.reason = str(exc) or "the configured real geometry model could not be loaded"
            return

        self._model = model
        self.model_config = model_config
        self.minimum_confidence = threshold
        self.model_version = str(model_config["model_version"]).strip()[:120]
        self.device_label = device_label
        self.ready = True

    def health(self) -> dict[str, Any]:
        return {
            "service": "geometry-service",
            "status": "ready" if self.ready else "unavailable",
            "mode": self.mode,
            "runtime": {
                "framework": "pytorch-ultralytics" if self._torch is not None else None,
                "torch_version": self.torch_version,
                "device": self.device_label,
                "requested_device": self.settings.device_preference or "auto",
                "accelerator_required": self.mode == "model",
            },
            "model": {
                "checkpoint_configured": self.settings.checkpoint_path is not None,
                "config_configured": self.settings.config_path is not None,
                "loaded": self.ready,
                "model_version": self.model_version,
                "output_contract": "camera-room-2d/v1",
            },
            "limits": {
                "max_frames": self.settings.max_frames,
                "max_frame_bytes": self.settings.max_frame_bytes,
                "max_total_frame_bytes": self.settings.max_total_frame_bytes,
            },
            "reason": self.reason,
            "raw_frames_persisted": False,
        }

    def predict(self, batch: Any) -> Any:
        if not self.ready or self._model is None or self._torch is None:
            raise RuntimeUnavailable(self.reason or "geometry model is unavailable")
        try:
            from .real_layout import RealLayoutNeedsRescan, infer_real_room_layout

            with GPU_COMPUTE_LOCK:
                with self._model_lock:
                    return infer_real_room_layout(
                        self._model,
                        batch,
                        self.model_config,
                        self.device_label or "cpu",
                    )
        except Exception as exc:  # pragma: no cover - depends on an external checkpoint
            if isinstance(exc, RealLayoutNeedsRescan):
                raise RuntimeNeedsRescan(str(exc)) from exc
            if isinstance(exc, RuntimeInferenceError):
                raise
            raise RuntimeInferenceError("the configured real geometry model failed during inference") from exc

    def detect_jpeg(
        self,
        jpeg_bytes: bytes,
        width: int,
        height: int,
        candidate_labels: list[str],
        *,
        minimum_confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run the same local YOLO-World checkpoint for bounded object detection."""
        if not self.ready or self._model is None:
            raise RuntimeUnavailable(self.reason or "vision model is unavailable")
        try:
            from .real_vision import decode_jpeg, detections_from_result

            image = decode_jpeg(jpeg_bytes, width, height)
            # Give the local detector a conservative shadow-lifted view when
            # the camera is genuinely underexposed. Normal frames are passed
            # through unchanged, and localization still records the decision
            # separately in its own feature diagnostics.
            image, _light_diagnostics = enhance_low_light_image(image)
            configured_threshold = float(self.model_config.get("detection_confidence", 0.20))
            threshold = configured_threshold if minimum_confidence is None else max(0.01, min(configured_threshold, float(minimum_confidence)))
            with GPU_COMPUTE_LOCK:
                with self._model_lock:
                    self._model.set_classes(candidate_labels)
                    results = self._model.predict(
                        source=image,
                        device=self.device_label or "cpu",
                        imgsz=(int(self.model_config["input"]["height"]), int(self.model_config["input"]["width"])),
                        conf=threshold,
                        max_det=100,
                        verbose=False,
                    )
                    self._model.set_classes(self.model_config["prompts"])
            if not results:
                return []
            return detections_from_result(results[0], width=width, height=height, minimum_confidence=threshold)
        except Exception as exc:  # pragma: no cover - depends on accelerator/model runtime
            try:
                if self._model is not None:
                    self._model.set_classes(self.model_config.get("prompts", []))
            except Exception:
                pass
            if isinstance(exc, RuntimeUnavailable):
                raise
            raise RuntimeInferenceError("the configured real vision model failed during inference") from exc
