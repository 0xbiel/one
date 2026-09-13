"""Lazy PyTorch/MPS/CUDA runtime loading with truthful readiness state."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import ServiceSettings


class RuntimeUnavailable(RuntimeError):
    """The configured production runtime cannot serve inference."""


class RuntimeInferenceError(RuntimeError):
    """The loaded model failed while producing an output."""


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
        raise RuntimeUnavailable("CPU inference is disabled; set GEOMETRY_SERVICE_ALLOW_CPU=1 explicitly")
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
        self._load()

    def _load(self) -> None:
        if self.mode == "mock":
            self.ready = True
            self.device_label = "mock"
            self.model_version = "mock-room-layout-v1"
            return
        if self.mode != "model":
            self.reason = "ONE_GEOMETRY_MODE must be model or mock"
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
        except ImportError:
            self.reason = "PyTorch is not installed"
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
            model = torch.jit.load(str(self.settings.checkpoint_path), map_location=device)
            model.eval()
        except Exception as exc:  # pragma: no cover - depends on host/runtime/checkpoint
            self.reason = str(exc) or "the configured PyTorch model could not be loaded"
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
                "framework": "pytorch" if self._torch is not None else ("fixture" if self.mode == "mock" else None),
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
            with self._torch.inference_mode():
                output = self._model(batch)
            return _to_python(output)
        except Exception as exc:  # pragma: no cover - depends on an external checkpoint
            raise RuntimeInferenceError("the configured geometry model failed during inference") from exc
