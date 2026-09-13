"""Optional image preprocessing for the explicit TorchScript model contract."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from .frames import FrameInputError, FrameSample


class DependencyUnavailable(RuntimeError):
    """A production image dependency is not installed."""


def _rotate_for_orientation(image: Any, orientation: str) -> Any:
    normalized = orientation.strip().lower().replace("_", "-")
    if normalized in {"portrait", "portrait-up", "portrait-upside-down"}:
        if normalized == "portrait-upside-down":
            return image.rotate(180, expand=True)
        return image
    if normalized in {"landscape-left", "left"}:
        return image.rotate(90, expand=True)
    if normalized in {"landscape-right", "right"}:
        return image.rotate(270, expand=True)
    return image


def build_torch_batch(
    samples: list[FrameSample],
    model_config: dict[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Decode JPEGs and build a Torch tensor without retaining source images."""

    try:
        import numpy as np
        import torch
        from PIL import Image, ImageFile, ImageOps
    except ImportError as exc:  # pragma: no cover - depends on host installation
        raise DependencyUnavailable(
            "Pillow, NumPy, and PyTorch are required for model mode"
        ) from exc

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    input_config = model_config.get("input")
    if not isinstance(input_config, dict):
        raise DependencyUnavailable("model config must contain an input object")
    try:
        input_width = int(input_config["width"])
        input_height = int(input_config["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DependencyUnavailable("model config input.width and input.height are required") from exc
    if not (16 <= input_width <= 2_048 and 16 <= input_height <= 2_048):
        raise DependencyUnavailable("model input dimensions must be between 16 and 2048 pixels")

    normalization = model_config.get("normalization", {})
    mean = normalization.get("mean", [0.485, 0.456, 0.406])
    std = normalization.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
        raise DependencyUnavailable("model normalization mean/std must contain three values")
    try:
        mean_tensor = torch.tensor([float(value) for value in mean], dtype=torch.float32).view(3, 1, 1)
        std_tensor = torch.tensor([float(value) for value in std], dtype=torch.float32).view(3, 1, 1)
    except (TypeError, ValueError) as exc:
        raise DependencyUnavailable("model normalization mean/std must be numeric") from exc
    if bool(torch.any(std_tensor <= 0)):
        raise DependencyUnavailable("model normalization std values must be positive")

    tensors: list[Any] = []
    try:
        for sample in samples:
            try:
                with Image.open(BytesIO(sample.jpeg_bytes)) as source:
                    source.load()
                    image = ImageOps.exif_transpose(source).convert("RGB")
            except Exception as exc:
                raise FrameInputError("invalid_jpeg", "a frame could not be decoded as JPEG") from exc

            image = _rotate_for_orientation(image, sample.orientation)
            image = image.resize((input_width, input_height), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
            tensor = (tensor - mean_tensor) / std_tensor
            tensors.append(tensor)
            image.close()
    finally:
        # The only retained values are model tensors, not source JPEGs/images.
        image = None

    if not tensors:
        raise FrameInputError("no_frames", "at least one decodable frame is required")

    batch = torch.stack(tensors, dim=0).to(device)
    return batch, {
        "valid_frame_count": len(tensors),
        "input_width": input_width,
        "input_height": input_height,
        "preprocessing": "rgb-resize-normalize",
    }
