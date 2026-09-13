"""Dependency-free deterministic geometry fixture for local contract checks."""

from __future__ import annotations

import hashlib


def deterministic_room_layout(
    frame_bytes: list[bytes],
    room_label: str,
    orientation: str,
) -> dict:
    """Return stable camera-relative geometry without decoding or persisting JPEGs.

    This is deliberately opt-in via ``GEOMETRY_SERVICE_MODE=mock``.  It is a
    contract fixture, not a production vision fallback and never claims metric
    scale.
    """

    digest = hashlib.sha256()
    for frame in frame_bytes:
        digest.update(frame)
    digest.update(orientation.encode("utf-8"))
    seed = digest.digest()
    horizontal_jitter = (seed[0] / 255.0 - 0.5) * 0.025
    vertical_jitter = (seed[1] / 255.0 - 0.5) * 0.02
    left = max(0.05, min(0.18, 0.09 + horizontal_jitter))
    top = max(0.05, min(0.18, 0.11 + vertical_jitter))
    right = min(0.95, max(0.82, 0.91 - horizontal_jitter))
    bottom = min(0.95, max(0.78, 0.88 - vertical_jitter))
    confidence = 0.78 + (seed[2] / 255.0) * 0.08

    corners = [
        {"x": round(left, 5), "y": round(top, 5)},
        {"x": round(right, 5), "y": round(top, 5)},
        {"x": round(right, 5), "y": round(bottom, 5)},
        {"x": round(left, 5), "y": round(bottom, 5)},
    ]
    wall_points = list(corners) + [corners[0]]
    walls = [
        {
            "id": f"wall-{index + 1}",
            "start": wall_points[index],
            "end": wall_points[index + 1],
            "confidence": round(confidence, 5),
        }
        for index in range(4)
    ]
    return {
        "polygons": [
            {
                "id": "room-1",
                "label": room_label.strip(),
                "points": corners,
                "confidence": round(confidence, 5),
            }
        ],
        "walls": walls,
        "camera_pose": {
            "coordinate_frame": "camera-relative",
            "position": {"x": 0.5, "y": 0.5, "z": 0.0},
            "rotation_degrees": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "confidence": round(confidence, 5),
        },
        "intrinsics": {
            "source": "fixture",
            "coordinate_frame": "camera-relative-image",
        },
        "metrics": {
            "confidence": round(confidence, 5),
            "reprojection_error_px": 4.0,
            "homography_inlier_ratio": 0.86,
        },
        "confidence": round(confidence, 5),
        "diagnostics": {
            "mode": "deterministic-mock",
            "motion_stability": 0.9,
            "reprojection_error_px": 4.0,
        },
    }
