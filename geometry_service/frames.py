"""In-memory JPEG validation and frame-bound enforcement."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime

from .config import ServiceSettings
from .contracts import RoomLayoutRequest


class FrameInputError(ValueError):
    """A client-supplied frame cannot be safely processed."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class FrameSample:
    """Decoded request data held only for the duration of inference."""

    jpeg_bytes: bytes
    width: int
    height: int
    orientation: str
    captured_at: datetime | None


def _decode_jpeg(value: str, settings: ServiceSettings) -> bytes:
    encoded = value.strip()
    if encoded.startswith("data:image/jpeg;base64,"):
        encoded = encoded.partition(",")[2]

    # Reject oversized input before base64 decoding allocates a second buffer.
    max_encoded = (settings.max_frame_bytes * 4) // 3 + 32
    if len(encoded) > max_encoded:
        raise FrameInputError(
            "frame_too_large",
            f"each JPEG must be at most {settings.max_frame_bytes} decoded bytes",
            status_code=413,
        )

    compact = "".join(encoded.split())
    if not compact:
        raise FrameInputError("frame_empty", "frame_base64 is empty")
    if len(compact) > max_encoded:
        raise FrameInputError(
            "frame_too_large",
            f"each JPEG must be at most {settings.max_frame_bytes} decoded bytes",
            status_code=413,
        )

    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameInputError("invalid_base64", "frame_base64 is not valid base64") from exc

    if len(decoded) > settings.max_frame_bytes:
        raise FrameInputError(
            "frame_too_large",
            f"each JPEG must be at most {settings.max_frame_bytes} decoded bytes",
            status_code=413,
        )
    if len(decoded) < 4 or decoded[:2] != b"\xff\xd8" or decoded[-2:] != b"\xff\xd9":
        raise FrameInputError("invalid_jpeg", "each frame must be a complete JPEG")
    return decoded


def decode_request_frames(payload: RoomLayoutRequest, settings: ServiceSettings) -> list[FrameSample]:
    """Decode and bound all samples without writing or logging their contents."""

    if len(payload.frames) > settings.max_frames:
        raise FrameInputError(
            "too_many_frames",
            f"at most {settings.max_frames} frames may be submitted",
            status_code=413,
        )

    samples: list[FrameSample] = []
    total_bytes = 0
    try:
        for frame in payload.frames:
            if payload.resolution and (
                frame.width != payload.resolution.width or frame.height != payload.resolution.height
            ):
                raise FrameInputError(
                    "resolution_mismatch",
                    "each frame must match the request resolution",
                )
            jpeg_bytes = _decode_jpeg(frame.frame_base64, settings)
            total_bytes += len(jpeg_bytes)
            if total_bytes > settings.max_total_frame_bytes:
                raise FrameInputError(
                    "frames_too_large",
                    f"all JPEGs together must be at most {settings.max_total_frame_bytes} bytes",
                    status_code=413,
                )
            samples.append(
                FrameSample(
                    jpeg_bytes=jpeg_bytes,
                    width=frame.width,
                    height=frame.height,
                    orientation=payload.orientation,
                    captured_at=frame.captured_at,
                )
            )
    except Exception:
        # Do not leave decoded request bytes referenced after a rejected batch.
        samples.clear()
        raise
    return samples
