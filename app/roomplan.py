"""Strict native RoomPlan upload contracts.

RoomPlan is the only producer allowed to create a metric 3D map.  The public
API keeps this contract deliberately small and explicit so arbitrary map JSON
cannot opt into the 3D renderer.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ROOMPLAN_SCHEMA_VERSION = "roomplan-normalized.v1"

_CONFIDENCE_VALUES = {"high": 0.95, "medium": 0.75, "low": 0.45}


class RoomPlanPoint3D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float

    @field_validator("x", "y", "z")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RoomPlan coordinates must be finite")
        return value


class RoomPlanDimensions3D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(gt=0, le=100)
    y: float = Field(gt=0, le=100)
    z: float = Field(gt=0, le=100)

    @field_validator("x", "y", "z")
    @classmethod
    def finite_dimension(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RoomPlan dimensions must be finite")
        return value


class RoomPlanElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    category: str = Field(min_length=1, max_length=80)
    confidence: Literal["high", "medium", "low"]
    center: RoomPlanPoint3D
    dimensions: RoomPlanDimensions3D
    transform: list[list[float]] = Field(min_length=4, max_length=4)
    vertices: list[RoomPlanPoint3D] = Field(default_factory=list, max_length=256)
    attributes: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("transform")
    @classmethod
    def four_by_four_finite_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("RoomPlan transforms must be 4x4 matrices")
        if any(not math.isfinite(item) for row in value for item in row):
            raise ValueError("RoomPlan transforms must contain finite values")
        return value

    @field_validator("attributes")
    @classmethod
    def valid_attributes(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 80 for item in value):
            raise ValueError("RoomPlan attributes must be non-empty short strings")
        return value


class RoomPlanSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    center: RoomPlanPoint3D
    story: int = Field(ge=0, le=100)


class RoomPlanNormalizedScan(BaseModel):
    """Canonical payload produced by the native iOS RoomPlan normalizer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["roomplan-normalized.v1"]
    producer: Literal["native-ios"]
    framework: Literal["RoomPlan"]
    units: Literal["m"]
    up_axis: Literal["Y"]
    coordinate_frame: Literal["roomplan-local"]
    geometry_type: Literal["3d"]
    captured_at: datetime | None = None
    room_id: str | None = Field(default=None, max_length=120)
    walls: list[RoomPlanElement] = Field(default_factory=list, max_length=1_000)
    floors: list[RoomPlanElement] = Field(default_factory=list, max_length=1_000)
    openings: list[RoomPlanElement] = Field(default_factory=list, max_length=1_000)
    doors: list[RoomPlanElement] = Field(default_factory=list, max_length=1_000)
    windows: list[RoomPlanElement] = Field(default_factory=list, max_length=1_000)
    objects: list[RoomPlanElement] = Field(default_factory=list, max_length=1_000)
    sections: list[RoomPlanSection] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def require_geometry(self) -> "RoomPlanNormalizedScan":
        if not any((self.walls, self.floors, self.openings, self.doors, self.windows, self.objects)):
            raise ValueError("RoomPlan scan must contain at least one 3D surface or object")
        return self


class RoomPlanScanMetadata(BaseModel):
    """Provenance asserted by a native RoomPlan/LiDAR producer."""

    model_config = ConfigDict(extra="forbid")

    provenance: Literal["native-roomplan"]
    device_model: str = Field(min_length=1, max_length=120)
    lidar: Literal[True]
    roomplan_version: str = Field(min_length=1, max_length=40)
    units: Literal["m", "meter", "meters"]
    up_axis: Literal["Y", "y"]
    geometry_type: Literal["3d"]


class RoomPlanMapIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str | None = None
    normalized_scan: RoomPlanNormalizedScan
    scan_metadata: RoomPlanScanMetadata

    @model_validator(mode="after")
    def metadata_matches_scan(self) -> "RoomPlanMapIn":
        if self.scan_metadata.units not in {self.normalized_scan.units, "meter", "meters"}:
            raise ValueError("RoomPlan metadata units must match the normalized scan")
        if self.scan_metadata.up_axis.upper() != self.normalized_scan.up_axis:
            raise ValueError("RoomPlan metadata up_axis must match the normalized scan")
        return self


def roomplan_geometry(scan: RoomPlanNormalizedScan) -> dict[str, Any]:
    """Expose the native scan as the frontend's actual 3D geometry shape.

    The values below are projections of RoomPlan elements, not synthesized
    room primitives. Surface vertices are retained when the native producer
    supplied them; object bounds and wall endpoints remain tied to the
    captured element's center, dimensions, and vertices.
    """

    collections = (
        ("wall", scan.walls),
        ("floor", scan.floors),
        ("opening", scan.openings),
        ("door", scan.doors),
        ("window", scan.windows),
        ("object", scan.objects),
    )
    surfaces: list[dict[str, Any]] = []
    walls: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for kind, elements in collections:
        for element in elements:
            confidence = _CONFIDENCE_VALUES[element.confidence]
            vertices = [point.model_dump(mode="json") for point in element.vertices]
            if len(vertices) >= 3:
                surfaces.append(
                    {
                        "id": element.id,
                        "kind": kind,
                        "vertices": vertices,
                        "faces": [list(range(len(vertices)))],
                        "confidence": confidence,
                    }
                )
            if kind == "wall" and len(vertices) >= 2:
                walls.append(
                    {
                        "id": element.id,
                        "start": vertices[0],
                        "end": vertices[1],
                        "confidence": confidence,
                    }
                )
            if kind == "object":
                objects.append(
                    {
                        "id": element.id,
                        "label": element.category,
                        "position": element.center.model_dump(mode="json"),
                        "dimensions": element.dimensions.model_dump(mode="json"),
                        "confidence": confidence,
                    }
                )
    return {
        "coordinate_space": scan.coordinate_frame,
        "polygons": [],
        "walls": walls,
        "surfaces": surfaces,
        "objects": objects,
        "roomplan_schema_version": scan.schema_version,
    }


def roomplan_usdz_metadata(
    *, sha256: str, byte_count: int, content_type: str, download_path: str
) -> dict[str, Any]:
    """Return the public, non-storage-key metadata for a USDZ attachment."""

    return {
        "available": True,
        "sha256": sha256,
        "bytes": byte_count,
        "content_type": content_type,
        "download_path": download_path,
    }
