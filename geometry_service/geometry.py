"""Validation and normalization for the camera-room-2d model output."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from .contracts import RoomLayoutGeometry, RoomLayoutResponse


class ModelOutputError(ValueError):
    """The configured model did not produce the documented geometry contract."""


class ConfidenceBelowThreshold(ModelOutputError):
    """The model saw a sweep but it was not reliable enough to persist."""

    def __init__(self, confidence: float, threshold: float) -> None:
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(f"geometry confidence {confidence:.3f} is below {threshold:.3f}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelOutputError(f"model output field {field_name} must be an object")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelOutputError(f"model output field {field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ModelOutputError(f"model output field {field_name} must be finite")
    return number


def _confidence(value: Any, field_name: str) -> float:
    number = _finite_number(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ModelOutputError(f"model output field {field_name} must be between 0 and 1")
    return number


def _identifier(value: Any, fallback: str, field_name: str) -> str:
    if value is None:
        return fallback
    if not isinstance(value, str) or not value.strip():
        raise ModelOutputError(f"model output field {field_name} must be a non-empty string")
    identifier = value.strip()
    if len(identifier) > 120:
        raise ModelOutputError(f"model output field {field_name} is too long")
    return identifier


def _normalized_point(value: Any, field_name: str) -> dict[str, float]:
    if isinstance(value, Mapping):
        x_value = value.get("x")
        y_value = value.get("y")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        x_value, y_value = value
    else:
        raise ModelOutputError(f"model output field {field_name} must be an [x, y] point")
    x = _finite_number(x_value, f"{field_name}.x")
    y = _finite_number(y_value, f"{field_name}.y")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ModelOutputError(f"model output field {field_name} must use normalized 0..1 coordinates")
    return {"x": round(x, 6), "y": round(y, 6)}


def _normalized_size(value: Any, field_name: str) -> dict[str, float]:
    if isinstance(value, Mapping):
        x_value = value.get("x", value.get("width"))
        y_value = value.get("y", value.get("height"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        x_value, y_value = value
    else:
        raise ModelOutputError(f"model output field {field_name} must be a [width, height] size")
    x = _finite_number(x_value, f"{field_name}.x")
    y = _finite_number(y_value, f"{field_name}.y")
    if not 0.0 < x <= 1.0 or not 0.0 < y <= 1.0:
        raise ModelOutputError(f"model output field {field_name} must use normalized 0..1 dimensions")
    return {"x": round(x, 6), "y": round(y, 6)}


def _rotation_degrees(value: Any, field_name: str) -> float:
    if value is None:
        return 0.0
    number = _finite_number(value, field_name)
    if not -180.0 <= number <= 180.0:
        raise ModelOutputError(f"model output field {field_name} must be between -180 and 180 degrees")
    return round(number, 4)


def _safe_diagnostics(raw: Any, base: dict[str, Any]) -> dict[str, Any]:
    """Keep diagnostics bounded and exclude arbitrary model payloads."""

    result = dict(base)
    if not isinstance(raw, Mapping):
        return result
    allowed = {
        "motion_stability",
        "reprojection_error_px",
        "homography_inlier_ratio",
        "visible_room_fraction",
        "inference_ms",
        "model_backend",
        "model_version",
        "structure_method",
        "detected_item_count",
        "furniture_count",
        "opening_count",
        "frame_count",
        "warnings",
    }
    for key in allowed:
        value = raw.get(key)
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            result[key] = value
        elif key == "warnings" and isinstance(value, list):
            result[key] = [item for item in value if isinstance(item, str)][:8]
    return result


def normalize_model_output(
    raw_output: Any,
    *,
    room_label: str,
    model_version: str,
    minimum_confidence: float,
    base_diagnostics: dict[str, Any],
) -> RoomLayoutResponse:
    """Validate the TorchScript output and create a persistable response."""

    if isinstance(raw_output, Mapping) and isinstance(raw_output.get("geometry"), Mapping):
        geometry_payload = _mapping(raw_output["geometry"], "geometry")
        top_level = raw_output
    else:
        geometry_payload = _mapping(raw_output, "output")
        top_level = raw_output if isinstance(raw_output, Mapping) else {}

    raw_polygons = geometry_payload.get("polygons")
    raw_walls = geometry_payload.get("walls")
    raw_pose = geometry_payload.get("camera_pose")
    raw_metrics = geometry_payload.get("metrics")
    if not isinstance(raw_polygons, list) or not raw_polygons:
        raise ModelOutputError("model output must contain at least one polygon room")
    if len(raw_polygons) > 100:
        raise ModelOutputError("model output contains too many rooms")
    if not isinstance(raw_walls, list) or not raw_walls:
        raise ModelOutputError("model output must contain at least one wall segment")
    if len(raw_walls) > 256:
        raise ModelOutputError("model output contains too many wall segments")
    raw_furniture = geometry_payload.get("furniture", [])
    if raw_furniture is None:
        raw_furniture = []
    if not isinstance(raw_furniture, list):
        raise ModelOutputError("model output field furniture must be a list")
    if len(raw_furniture) > 100:
        raise ModelOutputError("model output contains too many furniture items")
    raw_openings = geometry_payload.get("openings", [])
    if raw_openings is None:
        raw_openings = []
    if not isinstance(raw_openings, list):
        raise ModelOutputError("model output field openings must be a list")
    if len(raw_openings) > 100:
        raise ModelOutputError("model output contains too many openings")
    aliased_openings: list[dict[str, Any]] = []
    for alias, kind in (("doors", "door"), ("windows", "window")):
        values = geometry_payload.get(alias, [])
        if values is None:
            values = []
        if not isinstance(values, list):
            raise ModelOutputError(f"model output field {alias} must be a list")
        for value in values:
            item = dict(_mapping(value, alias))
            item.setdefault("kind", kind)
            aliased_openings.append(item)
    raw_openings = raw_openings + aliased_openings
    if len(raw_openings) > 100:
        raise ModelOutputError("model output contains too many openings")
    pose = _mapping(raw_pose, "camera_pose")
    metrics_payload = _mapping(raw_metrics, "metrics")

    polygons: list[dict[str, Any]] = []
    for index, raw_polygon in enumerate(raw_polygons):
        polygon_room = _mapping(raw_polygon, f"polygons[{index}]")
        polygon_value = polygon_room.get("points")
        if not isinstance(polygon_value, list) or len(polygon_value) < 3 or len(polygon_value) > 128:
            raise ModelOutputError(f"polygons[{index}].points must contain 3 to 128 points")
        polygon = [
            _normalized_point(point, f"polygons[{index}].points[{point_index}]")
            for point_index, point in enumerate(polygon_value)
        ]
        label = polygon_room.get("label") or room_label
        if not isinstance(label, str) or not label.strip():
            raise ModelOutputError(f"polygons[{index}].label must be a non-empty string")
        polygons.append(
            {
                "id": _identifier(polygon_room.get("id"), f"room-{index + 1}", f"polygons[{index}].id"),
                "label": label.strip()[:120],
                "points": polygon,
                "confidence": _confidence(polygon_room.get("confidence"), f"polygons[{index}].confidence"),
            }
        )

    walls: list[dict[str, Any]] = []
    for index, raw_wall in enumerate(raw_walls):
        wall = _mapping(raw_wall, f"walls[{index}]")
        walls.append(
            {
                "id": _identifier(wall.get("id"), f"wall-{index + 1}", f"walls[{index}].id"),
                "start": _normalized_point(wall.get("start"), f"walls[{index}].start"),
                "end": _normalized_point(wall.get("end"), f"walls[{index}].end"),
                "confidence": _confidence(wall.get("confidence"), f"walls[{index}].confidence"),
            }
        )

    furniture: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_furniture):
        item = _mapping(raw_item, f"furniture[{index}]")
        label = item.get("label") or item.get("name")
        if not isinstance(label, str) or not label.strip():
            raise ModelOutputError(f"furniture[{index}].label must be a non-empty string")
        furniture.append(
            {
                "id": _identifier(item.get("id"), f"furniture-{index + 1}", f"furniture[{index}].id"),
                "label": label.strip()[:120],
                "center": _normalized_point(item.get("center", item.get("position")), f"furniture[{index}].center"),
                "size": _normalized_size(item.get("size", item.get("dimensions")), f"furniture[{index}].size"),
                "rotation_degrees": _rotation_degrees(item.get("rotation_degrees", item.get("rotationDegrees")), f"furniture[{index}].rotation_degrees"),
                "confidence": _confidence(item.get("confidence"), f"furniture[{index}].confidence"),
            }
        )

    openings: list[dict[str, Any]] = []
    for index, raw_opening in enumerate(raw_openings):
        opening = _mapping(raw_opening, f"openings[{index}]")
        kind = opening.get("kind")
        if kind not in {"door", "window"}:
            raise ModelOutputError(f"openings[{index}].kind must be door or window")
        openings.append(
            {
                "id": _identifier(opening.get("id"), f"opening-{index + 1}", f"openings[{index}].id"),
                "kind": kind,
                "start": _normalized_point(opening.get("start"), f"openings[{index}].start"),
                "end": _normalized_point(opening.get("end"), f"openings[{index}].end"),
                "confidence": _confidence(opening.get("confidence"), f"openings[{index}].confidence"),
            }
        )

    raw_position = pose.get("position")
    raw_rotation = pose.get("rotation_degrees")
    position = _mapping(raw_position, "camera_pose.position")
    rotation = _mapping(raw_rotation, "camera_pose.rotation_degrees")
    camera_pose = {
        "coordinate_frame": "camera-relative",
        "position": {
            "x": _finite_number(position.get("x"), "camera_pose.position.x"),
            "y": _finite_number(position.get("y"), "camera_pose.position.y"),
            "z": _finite_number(position.get("z"), "camera_pose.position.z"),
        },
        "rotation_degrees": {
            "yaw": _finite_number(rotation.get("yaw"), "camera_pose.rotation_degrees.yaw"),
            "pitch": _finite_number(rotation.get("pitch"), "camera_pose.rotation_degrees.pitch"),
            "roll": _finite_number(rotation.get("roll"), "camera_pose.rotation_degrees.roll"),
        },
        "confidence": _confidence(pose.get("confidence"), "camera_pose.confidence"),
    }

    confidence = _confidence(metrics_payload.get("confidence"), "metrics.confidence")
    reprojection_error = _finite_number(
        metrics_payload.get("reprojection_error_px"),
        "metrics.reprojection_error_px",
    )
    if reprojection_error < 0:
        raise ModelOutputError("metrics.reprojection_error_px must be non-negative")
    homography_inlier_ratio = _confidence(
        metrics_payload.get("homography_inlier_ratio"),
        "metrics.homography_inlier_ratio",
    )
    if confidence < minimum_confidence:
        raise ConfidenceBelowThreshold(confidence, minimum_confidence)

    try:
        geometry = RoomLayoutGeometry.model_validate(
            {
                "coordinate_frame": "camera-relative-image",
                "polygons": polygons,
                "walls": walls,
                "furniture": furniture,
                "openings": openings,
                "camera_pose": camera_pose,
                "intrinsics": _mapping(geometry_payload.get("intrinsics", {}), "intrinsics"),
                "metrics": {
                    "confidence": confidence,
                    "reprojection_error_px": reprojection_error,
                    "homography_inlier_ratio": homography_inlier_ratio,
                },
            }
        )
    except ValidationError as exc:
        raise ModelOutputError("model output failed the camera-room-2d schema") from exc
    diagnostics = _safe_diagnostics(top_level.get("diagnostics"), base_diagnostics)
    diagnostics.update(
        {
            "coordinate_frame": "camera-relative-image",
            "metric_scale_known": False,
            "room_count": len(polygons),
            "wall_count": len(walls),
            "furniture_count": len(furniture),
            "opening_count": len(openings),
            "confidence": confidence,
            "reprojection_error_px": reprojection_error,
            "homography_inlier_ratio": homography_inlier_ratio,
        }
    )
    return RoomLayoutResponse(
        status="ready",
        geometry=geometry,
        confidence=round(confidence, 6),
        model_version=model_version,
        reason=None,
        diagnostics=diagnostics,
    )
