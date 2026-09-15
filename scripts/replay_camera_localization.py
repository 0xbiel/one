"""Replay saved fixed-camera frames against the local geometry worker.

This is an evaluation-only helper. It reads RoomPlan-derived numeric/artifact
data plus local JPEGs, sends them to the loopback geometry worker, and prints
numeric localization diagnostics. It never writes raw frames or turns the
human-marked floor reference into solver evidence.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request


_ROOMPLAN_CONFIDENCE = {"high": 0.95, "medium": 0.75, "low": 0.45}


def _parse_indices(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(index < 1 for index in result):
        raise argparse.ArgumentTypeError("indices must be positive frame numbers")
    return result


def _room_objects(room_map: dict) -> list[dict]:
    scan = room_map.get("normalized_scan") if isinstance(room_map.get("normalized_scan"), dict) else {}
    result: list[dict] = []
    for item in scan.get("objects", []):
        if not isinstance(item, dict):
            continue
        transform = item.get("transform")
        result.append(
            {
                "id": item["id"],
                "label": str(item["category"]).lower(),
                "center": item["center"],
                "dimensions": item["dimensions"],
                **({"transform": {"values": transform}} if isinstance(transform, list) else {}),
                "confidence": _ROOMPLAN_CONFIDENCE.get(str(item.get("confidence", "")).lower(), 1.0),
            }
        )
    return result


def _room_zones(room_map: dict) -> list[dict]:
    geometry = room_map.get("geometry") if isinstance(room_map.get("geometry"), dict) else {}
    result: list[dict] = []
    for zone in geometry.get("room_zones", []):
        if not isinstance(zone, dict):
            continue
        polygon = zone.get("polygon")
        floor_y = zone.get("floor_y")
        if not isinstance(polygon, list) or not isinstance(floor_y, (int, float)):
            continue
        result.append(
            {
                "id": zone.get("id"),
                "floor_y": float(floor_y),
                "polygon": [
                    {"x": float(point["x"]), "z": float(point["z"])}
                    for point in polygon
                    if isinstance(point, dict)
                    and isinstance(point.get("x"), (int, float))
                    and isinstance(point.get("z"), (int, float))
                ],
            }
        )
    return [zone for zone in result if len(zone["polygon"]) >= 3]


def _frames(directory: Path, indices: list[int], width: int, height: int) -> list[dict]:
    frames: list[dict] = []
    for index in indices:
        path = directory / f"frame_{index:03d}.jpg"
        payload = path.read_bytes()
        frames.append(
            {
                "frame_base64": base64.b64encode(payload).decode("ascii"),
                "width": width,
                "height": height,
            }
        )
    return frames


def _horizontal_error(center: object, ground_truth_x: float | None, ground_truth_z: float | None) -> float | None:
    if ground_truth_x is None or ground_truth_z is None:
        return None
    if not isinstance(center, list) or len(center) != 3:
        return None
    try:
        x, z = float(center[0]), float(center[2])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(z):
        return None
    return round(math.hypot(x - ground_truth_x, z - ground_truth_z), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", type=Path)
    parser.add_argument("--landmarks-json", type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--indices", type=_parse_indices, default=[1, 7, 13, 19])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov-degrees", type=float, default=60.0)
    parser.add_argument("--worker-url", default="http://127.0.0.1:8090/v1/camera-localization")
    parser.add_argument("--api-base")
    parser.add_argument("--email")
    parser.add_argument("--home-id")
    parser.add_argument("--camera-id")
    parser.add_argument("--ground-truth-x", type=float)
    parser.add_argument("--ground-truth-z", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frames = _frames(args.frames_dir, args.indices, args.width, args.height)

    def request_json(url: str, payload: dict, *, token: str | None = None) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(f"localization request returned HTTP {exc.code}: {detail}") from exc

    started = time.monotonic()
    if args.api_base:
        if not all((args.email, args.home_id, args.camera_id)):
            raise SystemExit("--api-base requires --email, --home-id, and --camera-id")
        api_base = args.api_base.rstrip("/")
        challenge = request_json(
            f"{api_base}/auth/email/request",
            {"email": args.email, "purpose": "login"},
        )
        code = challenge.get("dev_code")
        if not isinstance(code, str) or not code:
            raise SystemExit("API replay requires a development outbox sign-in code")
        session = request_json(
            f"{api_base}/auth/email/verify",
            {"email": args.email, "code": code},
        )
        token = session.get("access_token")
        if not isinstance(token, str) or not token:
            raise SystemExit("API sign-in did not return an access token")
        result = request_json(
            f"{api_base}/homes/{args.home_id}/cameras/{args.camera_id}/localize-roomplan",
            {"frames": frames, "fov_degrees": args.fov_degrees},
            token=token,
        )
    else:
        if args.map_json is None or args.landmarks_json is None:
            raise SystemExit("worker replay requires --map-json and --landmarks-json")
        room_map = json.loads(args.map_json.read_text())
        landmarks = json.loads(args.landmarks_json.read_text())
        result = request_json(
            args.worker_url,
            {
                "schema_version": "roomplan-camera-localization.v1",
                "landmarks": landmarks.get("landmarks", []),
                "frames": frames,
                "fov_degrees": args.fov_degrees,
                "room_zones": _room_zones(room_map),
                "room_objects": _room_objects(room_map),
            },
        )
    elapsed = time.monotonic() - started
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    semantic = diagnostics.get("semantic_cuboid_candidates") if isinstance(diagnostics.get("semantic_cuboid_candidates"), list) else []
    detection = diagnostics.get("semantic_object_detection") if isinstance(diagnostics.get("semantic_object_detection"), dict) else {}
    semantic_with_error = []
    for candidate in semantic[:6]:
        if not isinstance(candidate, dict):
            continue
        semantic_with_error.append(
            {
                **candidate,
                "horizontal_error_m": _horizontal_error(
                    candidate.get("camera_center"),
                    args.ground_truth_x,
                    args.ground_truth_z,
                ),
            }
        )
    selected_center = diagnostics.get("selected_camera_center")
    summary = {
        "status": result.get("status"),
        "elapsed_seconds": round(elapsed, 3),
        "frame_indices": args.indices,
        "selected_estimate_source": diagnostics.get("selected_estimate_source"),
        "selected_camera_center": selected_center,
        "horizontal_error_m": _horizontal_error(selected_center, args.ground_truth_x, args.ground_truth_z),
        "inlier_count": result.get("inlier_count"),
        "match_count": result.get("match_count"),
        "reprojection_error_px": result.get("reprojection_error_px"),
        "reason": diagnostics.get("reason"),
        "semantic_cuboid_candidates": semantic_with_error,
        "semantic_object_detection": {
            "status": detection.get("status"),
            "detected_count": detection.get("detected_count"),
            "detected_labels": detection.get("detected_labels"),
            "frames": detection.get("frames"),
        },
        "dynamic_person_mask": diagnostics.get("dynamic_person_mask"),
        "raw_frames_persisted": False,
    }
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
