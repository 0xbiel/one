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


class Size2D(BaseModel):
    """Normalized width and height for an image-space fixture."""

    x: float = Field(gt=0.0, le=1.0)
    y: float = Field(gt=0.0, le=1.0)


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


class RoomFurniture(BaseModel):
    """A recognizable, non-metric fixture detected in the camera sweep."""

    id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    center: Point2D
    size: Size2D
    rotation_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)
    confidence: float = Field(ge=0.0, le=1.0)


class RoomOpening(BaseModel):
    """A door or window represented as a normalized image-space segment."""

    id: str = Field(min_length=1, max_length=120)
    kind: Literal["door", "window"]
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
    furniture: list[RoomFurniture] = Field(default_factory=list, max_length=100)
    openings: list[RoomOpening] = Field(default_factory=list, max_length=100)
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


class VisionFrameRequest(BaseModel):
    """One transient frame for real local YOLO-World detection."""

    model_config = ConfigDict(extra="forbid")

    frame_base64: str = Field(min_length=1, max_length=MAX_FRAME_BASE64_CHARS)
    width: int = Field(gt=0, le=7_680)
    height: int = Field(gt=0, le=4_320)
    candidate_labels: list[str] = Field(min_length=1, max_length=32)

    @field_validator("candidate_labels")
    @classmethod
    def validate_labels(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip().lower() for item in value]
        if any(not item or len(item) > 80 for item in cleaned):
            raise ValueError("candidate labels must be non-empty strings up to 80 characters")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("candidate labels must be unique")
        return cleaned


class VisionDetection(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)


class VisionFrameResponse(BaseModel):
    status: Literal["ready", "unavailable", "failed"]
    model_version: str
    detections: list[VisionDetection] = Field(default_factory=list, max_length=100)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class Matrix3x3(BaseModel):
    values: list[list[float]]

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("matrix must be exactly 3x3")
        return value


class Matrix4x4(BaseModel):
    values: list[list[float]]

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("matrix must be exactly 4x4")
        return value


class VisualLandmarkFrame(BaseModel):
    """RGB + LiDAR depth sample captured inside the RoomPlan coordinate frame."""

    model_config = ConfigDict(extra="forbid")

    frame_base64: str = Field(min_length=1, max_length=MAX_FRAME_BASE64_CHARS)
    width: int = Field(gt=0, le=7_680)
    height: int = Field(gt=0, le=4_320)
    depth_base64: str = Field(min_length=1, max_length=2_000_000)
    depth_width: int = Field(gt=0, le=2_048)
    depth_height: int = Field(gt=0, le=2_048)
    intrinsics: Matrix3x3
    camera_to_world: Matrix4x4
    captured_at: datetime | None = None


class VisualLandmarkBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["roomplan-visual-landmarks.v1"] = "roomplan-visual-landmarks.v1"
    map_id: str = Field(min_length=1, max_length=120)
    frames: list[VisualLandmarkFrame] = Field(min_length=2, max_length=12)


class VisualLandmark(BaseModel):
    point: list[float] = Field(min_length=3, max_length=3)
    descriptor_base64: str = Field(min_length=1, max_length=256)
    response: float = 0.0


class VisualLandmarkBuildResponse(BaseModel):
    status: Literal["ready", "needs_rescan", "unavailable", "failed"]
    schema_version: Literal["roomplan-visual-landmarks.v1"] = "roomplan-visual-landmarks.v1"
    detector: Literal["opencv-orb"] = "opencv-orb"
    landmarks: list[VisualLandmark] = Field(default_factory=list, max_length=5_000)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class CameraLocalizationFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_base64: str = Field(min_length=1, max_length=MAX_FRAME_BASE64_CHARS)
    width: int = Field(gt=0, le=7_680)
    height: int = Field(gt=0, le=4_320)


class CameraLocalizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["roomplan-camera-localization.v1"] = "roomplan-camera-localization.v1"
    landmarks: list[VisualLandmark] = Field(min_length=6, max_length=5_000)
    frames: list[CameraLocalizationFrame] = Field(min_length=1, max_length=8)
    intrinsics: Matrix3x3 | None = None
    fov_degrees: float = Field(default=60.0, ge=30.0, le=120.0)


class CameraLocalizationResponse(BaseModel):
    status: Literal["positioned", "needs_rescan", "unavailable", "failed"]
    coordinate_frame: Literal["roomplan-local"] = "roomplan-local"
    camera_to_world: list[list[float]] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    inlier_count: int = Field(default=0, ge=0)
    match_count: int = Field(default=0, ge=0)
    reprojection_error_px: float | None = Field(default=None, ge=0.0)
    intrinsics_source: Literal["provided", "estimated-fov"]
    intrinsics: list[list[float]] | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
