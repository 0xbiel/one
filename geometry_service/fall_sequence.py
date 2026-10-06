"""Learned pose-sequence fall scoring for the local geometry worker.

The deployed classifier is intentionally small enough for a phone-facing local
worker, but it is a real learned temporal model: two spatial graph-convolution
blocks operate on the COCO-17 skeleton and a causal temporal-convolution head
scores a fixed sequence.  The API supplies only derived pose coordinates and
does not send raw camera frames to this endpoint.

The artifact format is explicit and self-describing.  A missing, malformed, or
unevaluated artifact leaves this engine unavailable; callers must not silently
fall back to a bounding-box rule.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - API-only environments
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .gpu_runtime import GPU_COMPUTE_LOCK
from .runtime import RuntimeInferenceError, RuntimeUnavailable


COCO_KEYPOINTS = 17
COCO_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 5), (0, 6),
)
FALL_SEQUENCE_FORMAT = "one-fall-sequence-model-v1"
FALL_SEQUENCE_MODEL_VERSION = "stgcn-causal-tcn-fall-v1"


def normalized_skeleton_adjacency() -> "torch.Tensor":
    """Return the symmetric degree-normalized COCO-17 graph with self loops."""

    if torch is None:
        raise RuntimeUnavailable("PyTorch is required for the pose-sequence model")
    adjacency = torch.eye(COCO_KEYPOINTS, dtype=torch.float32)
    for left, right in COCO_EDGES:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=1)
    inverse_sqrt = degree.clamp_min(1.0).pow(-0.5)
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


if nn is not None:

    class SpatialGraphBlock(nn.Module):
        """Graph aggregation followed by a local temporal convolution."""

        def __init__(self, in_channels: int, out_channels: int, adjacency: "torch.Tensor") -> None:
            super().__init__()
            self.register_buffer("adjacency", adjacency)
            self.temporal = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.08),
            )
            self.residual = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            # values: [batch, channels, time, joints]
            graph_values = torch.einsum("vw,bctw->bctv", self.adjacency, values)
            result = self.temporal(torch.cat((values, graph_values), dim=1))
            return F.relu(result + self.residual(values), inplace=True)


    class CausalTemporalBlock(nn.Module):
        """A residual TCN block with left-only padding."""

        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.dilation = dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, bias=False)
            self.norm1 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, bias=False)
            self.norm2 = nn.BatchNorm1d(channels)
            self.dropout = nn.Dropout(p=0.12)

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            padding = 2 * self.dilation
            result = self.conv1(F.pad(values, (padding, 0)))
            result = F.relu(self.norm1(result), inplace=True)
            result = self.dropout(result)
            result = self.conv2(F.pad(result, (padding, 0)))
            result = self.norm2(result)
            return F.relu(result + values, inplace=True)


    class SkeletonSTGCNTCN(nn.Module):
        """Compact ST-GCN plus causal-TCN classifier for COCO-17 sequences."""

        def __init__(self, input_channels: int = 3, hidden_channels: int = 32, class_count: int = 2) -> None:
            super().__init__()
            adjacency = normalized_skeleton_adjacency()
            self.graph1 = SpatialGraphBlock(input_channels, hidden_channels, adjacency)
            self.graph2 = SpatialGraphBlock(hidden_channels, hidden_channels, adjacency)
            self.temporal = nn.Sequential(
                CausalTemporalBlock(hidden_channels, dilation=1),
                CausalTemporalBlock(hidden_channels, dilation=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.10),
                nn.Linear(hidden_channels, class_count),
            )

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            # Input is [batch, time, joints, channels].
            result = values.permute(0, 3, 1, 2).contiguous()
            result = self.graph2(self.graph1(result))
            result = result.mean(dim=-1)  # [batch, channels, time]
            result = self.temporal(result)
            pooled = torch.cat((result.mean(dim=-1), result[:, :, -1]), dim=1)
            return self.classifier(pooled)


else:  # pragma: no cover - import guard for API-only environments

    SkeletonSTGCNTCN = None  # type: ignore[assignment,misc]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def sequence_features(
    frames: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> "np.ndarray":
    """Convert raw per-frame COCO joints into bbox-normalized model features."""

    if np is None:
        raise RuntimeUnavailable("NumPy is required for the pose-sequence model")
    if width <= 0 or height <= 0:
        raise RuntimeInferenceError("pose-sequence frame dimensions must be positive")
    rows: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeInferenceError(f"pose-sequence frame {index} is not an object")
        raw_points = frame.get("keypoints", frame.get("pose_keypoints"))
        if not isinstance(raw_points, (list, tuple)) or len(raw_points) != COCO_KEYPOINTS:
            raise RuntimeInferenceError(f"pose-sequence frame {index} must contain 17 keypoints")
        raw_bbox = frame.get("bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise RuntimeInferenceError(f"pose-sequence frame {index} must contain a bounding box")
        x1, y1, x2, y2 = (_finite_float(value) for value in raw_bbox)
        box_width = max(abs(x2 - x1), 1.0)
        box_height = max(abs(y2 - y1), 1.0)
        points = np.zeros((COCO_KEYPOINTS, 3), dtype=np.float32)
        for joint_index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                raise RuntimeInferenceError(f"pose-sequence frame {index} joint {joint_index} is invalid")
            x = _finite_float(raw_point[0], x1 + box_width / 2.0)
            y = _finite_float(raw_point[1], y1 + box_height / 2.0)
            confidence = min(max(_finite_float(raw_point[2], 1.0) if len(raw_point) >= 3 else 1.0, 0.0), 1.0)
            points[joint_index] = (
                min(max((x - x1) / box_width, 0.0), 1.0),
                min(max((y - y1) / box_height, 0.0), 1.0),
                confidence,
            )
        rows.append(points)
    if not rows:
        raise RuntimeInferenceError("pose-sequence request contains no frames")
    result = np.stack(rows, axis=0)
    return result


class FallSequenceEngine:
    """Load and score one evaluated pose-sequence artifact."""

    def __init__(self, settings: Any, runtime: Any) -> None:
        self.ready = False
        self.reason: str | None = None
        self.model_version = "unavailable"
        self.model_path: Path | None = getattr(settings, "fall_sequence_model_path", None)
        self.window_size = int(getattr(settings, "fall_sequence_window", 15))
        self.threshold = float(getattr(settings, "fall_sequence_threshold", 0.65))
        self.device_label = getattr(runtime, "device_label", None)
        self._device: Any = None
        self._torch: Any = None
        self._model: Any = None
        self._mean: Any = None
        self._std: Any = None
        self._load(settings, runtime)

    def _load(self, settings: Any, runtime: Any) -> None:
        if torch is None or np is None or SkeletonSTGCNTCN is None:
            self.reason = "PyTorch and NumPy are required for the learned fall model"
            return
        if self.model_path is None:
            self.reason = "ONE_FALL_SEQUENCE_MODEL_PATH must point to an evaluated artifact"
            return
        if not self.model_path.is_file():
            self.reason = "configured fall sequence model artifact does not exist"
            return
        if self.window_size < 8 or self.window_size > 64:
            self.reason = "fall sequence window must be between 8 and 64 frames"
            return
        device_label = str(self.device_label or "")
        allow_cpu = bool(getattr(settings, "allow_cpu", False))
        if device_label not in {"mps", "cuda", "cpu"}:
            self.reason = "the geometry runtime has no usable device for the fall sequence model"
            return
        if device_label == "cpu" and not allow_cpu:
            self.reason = "CPU fall-sequence inference is disabled; an MPS/CUDA worker is required"
            return
        try:
            try:
                # The artifact contains tensors plus primitive metadata only.
                # Prefer restricted loading so a model file cannot execute an
                # arbitrary pickle payload during worker startup.
                artifact = torch.load(self.model_path, map_location="cpu", weights_only=True)
            except TypeError:  # PyTorch versions before weights_only
                artifact = torch.load(self.model_path, map_location="cpu")
            if not isinstance(artifact, dict) or artifact.get("format") != FALL_SEQUENCE_FORMAT:
                raise RuntimeUnavailable("fall sequence artifact format is not supported")
            model_config = artifact.get("model_config")
            if not isinstance(model_config, dict):
                raise RuntimeUnavailable("fall sequence artifact is missing model_config")
            input_channels = int(model_config.get("input_channels", 3))
            hidden_channels = int(model_config.get("hidden_channels", 32))
            if input_channels != 3 or hidden_channels < 8 or hidden_channels > 256:
                raise RuntimeUnavailable("fall sequence artifact model_config is invalid")
            artifact_window = int(artifact.get("window_size", self.window_size))
            if artifact_window != self.window_size:
                raise RuntimeUnavailable("configured fall sequence window does not match the artifact")
            classes = artifact.get("classes")
            if classes != ["non_fall", "fall"]:
                raise RuntimeUnavailable("fall sequence artifact classes must be [non_fall, fall]")
            mean = np.asarray(artifact.get("feature_mean"), dtype=np.float32)
            std = np.asarray(artifact.get("feature_std"), dtype=np.float32)
            if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise RuntimeUnavailable("fall sequence artifact normalization is invalid")
            artifact_threshold = float(artifact.get("threshold", self.threshold))
            if not math.isfinite(artifact_threshold) or not 0.5 <= artifact_threshold <= 0.99:
                raise RuntimeUnavailable("fall sequence artifact threshold is invalid")
            # Production uses the stricter of the artifact-selected threshold
            # and the deployment threshold. This prevents an artifact trained
            # on a small or permissive validation set from silently lowering
            # the configured false-positive guardrail.
            threshold = max(artifact_threshold, self.threshold)
            model = SkeletonSTGCNTCN(input_channels, hidden_channels, class_count=2)
            state_dict = artifact.get("state_dict")
            if not isinstance(state_dict, dict):
                raise RuntimeUnavailable("fall sequence artifact is missing state_dict")
            model.load_state_dict(state_dict, strict=True)
            device = torch.device(device_label)
            model.to(device)
            model.eval()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            self.reason = str(exc)[:240] or "fall sequence artifact could not be loaded"
            return
        self._torch = torch
        self._device = device
        self._model = model
        self._mean = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        self._std = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        self.window_size = artifact_window
        self.threshold = threshold
        self.model_version = str(artifact.get("model_version") or FALL_SEQUENCE_MODEL_VERSION)[:120]
        self.ready = True

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "unavailable",
            "model_version": self.model_version,
            "artifact": str(self.model_path) if self.model_path is not None else None,
            "device": self.device_label,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "architecture": "stgcn-plus-causal-tcn",
            "classes": ["non_fall", "fall"],
            "reason": self.reason,
            "raw_frames_persisted": False,
        }

    def score(self, *, frames: Sequence[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
        if not self.ready or self._model is None or self._torch is None or self._mean is None or self._std is None:
            raise RuntimeUnavailable(self.reason or "fall sequence model is unavailable")
        if len(frames) != self.window_size:
            raise RuntimeInferenceError(f"fall sequence model requires exactly {self.window_size} frames")
        try:
            values = sequence_features(frames, width=width, height=height)
            tensor = self._torch.as_tensor(values, dtype=self._torch.float32, device=self._device).unsqueeze(0)
            tensor = (tensor - self._mean) / self._std
            with GPU_COMPUTE_LOCK:
                with self._torch.inference_mode():
                    logits = self._model(tensor)
                    probability = float(self._torch.softmax(logits, dim=-1)[0, 1].detach().cpu().item())
            if not math.isfinite(probability):
                raise RuntimeInferenceError("fall sequence model returned a non-finite probability")
            return {
                "status": "ready",
                "probability": round(min(max(probability, 0.0), 1.0), 6),
                "threshold": round(self.threshold, 6),
                "model_version": self.model_version,
                "diagnostics": {
                    "device": self.device_label,
                    "window_size": self.window_size,
                    "architecture": "stgcn-plus-causal-tcn",
                    "raw_frames_persisted": False,
                },
            }
        except RuntimeUnavailable:
            raise
        except RuntimeInferenceError:
            raise
        except Exception as exc:  # pragma: no cover - accelerator-specific failures
            raise RuntimeInferenceError("learned fall sequence inference failed") from exc
