"""HTTP contracts for the local camera-room geometry service.

The request intentionally contains only short-lived base64 JPEGs.  The service
never writes those bytes to disk or includes them in a response.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_FRAME_BASE64_CHARS = 4_000_000
MAX_FRAMES = 20

MapSource = Literal["camera-cv-2d"]
MapDimension = Literal["2d"]
LayoutStatus = Literal["ready", "needs_rescan", "unavailable", "failed"]


class RoomLayoutResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0, le=7_680)
    height: int = Field(gt=0, le=4_320)


class RoomLayoutOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal["2d"] = "2d"
    coordinate_frame: Literal["camera-relative-image"] = "camera-relative-image"
    require_gpu: bool = True


class RoomLayoutFrame(BaseModel):
    """One bounded JPEG sample from the fixed camera sweep."""

    model_config = ConfigDict(extra="forbid")

    frame_base64: str = Field(
        min_length=1,
        max_length=MAX_FRAME_BASE64_CHARS,
        description="Base64-encoded JPEG. It is processed in memory only.",
    )
    width: int = Field(gt=0, le=7_680)
    height: int = Field(gt=0, le=4_320)
    captured_at: datetime | None = None

    @field_validator("frame_base64", mode="before")
    @classmethod
    def trim_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class RoomLayoutRequest(BaseModel):
    """Input contract shared with the backend room-layout adapter."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["room-layout-request.v1"] = "room-layout-request.v1"
    camera_id: str | None = Field(default=None, min_length=1, max_length=120)
    resolution: RoomLayoutResolution | None = None
    output: RoomLayoutOutput | None = None
    frames: list[RoomLayoutFrame] = Field(
        min_length=3,
        max_length=MAX_FRAMES,
        description="A short, bounded sweep; raw samples are never persisted.",
    )
    orientation: str = Field(min_length=1, max_length=32)
    room_label: str = Field(min_length=1, max_length=120)

    @field_validator("orientation", "room_label", mode="before")
    @classmethod
    def trim_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class Point2D(BaseModel):
    """Normalized camera-room coordinates, not meters."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class Point3D(BaseModel):
    """Camera-relative pose coordinates; units are explicitly non-metric."""

    x: float
    y: float
    z: float


class RotationDegrees(BaseModel):
    yaw: float
    pitch: float
    roll: float


class PolygonRoom(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    points: list[Point2D] = Field(min_length=3, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)


class WallSegment(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    start: Point2D
    end: Point2D
    confidence: float = Field(ge=0.0, le=1.0)


class CameraPose(BaseModel):
    coordinate_frame: Literal["camera-relative"] = "camera-relative"
    position: Point3D
    rotation_degrees: RotationDegrees
    confidence: float = Field(ge=0.0, le=1.0)


class RoomLayoutMetrics(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reprojection_error_px: float = Field(ge=0.0, le=100_000)
    homography_inlier_ratio: float = Field(ge=0.0, le=1.0)


class RoomLayoutGeometry(BaseModel):
    coordinate_frame: Literal["camera-relative-image"] = "camera-relative-image"
    polygons: list[PolygonRoom] = Field(min_length=1, max_length=100)
    walls: list[WallSegment] = Field(default_factory=list, max_length=256)
    camera_pose: dict[str, Any] = Field(min_length=1)
    intrinsics: dict[str, Any] = Field(default_factory=dict)
    metrics: RoomLayoutMetrics


class RoomLayoutResponse(BaseModel):
    """Backend-compatible result with explicit 2D provenance."""

    status: LayoutStatus
    schema_version: Literal["room-layout-response.v1"] = "room-layout-response.v1"
    source: MapSource = "camera-cv-2d"
    dimension: MapDimension = "2d"
    geometry: RoomLayoutGeometry | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metric_scale_known: Literal[False] = False
    model_version: str
    reason: str | None = Field(default=None, max_length=240)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
