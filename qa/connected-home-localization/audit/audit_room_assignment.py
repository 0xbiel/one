"""Synthetic contract/zone checks against pinned ONE source, not localization accuracy.

Usage: python audit_room_assignment.py [--source ../one] [--output result.json]
Executes actual source AST for the small geometry/selection helpers so OpenCV,
Torch, a live backend, model weights, and image solving are not needed. No app
source is modified. All poses supplied below are fabricated test estimates.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from pydantic import ValidationError


def source_functions(path, names, namespace):
    """Compile named original function nodes, preserving their source locations."""
    tree = ast.parse(path.read_text())
    selected = {n.name: n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name in names}
    assert set(selected) == set(names), (path, names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *[selected[n] for n in names]],
        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def assignment_from_result(result, payload, source):
    """Evaluation adapter: re-evaluate the actual final pose with ONE's helper.

    This does not infer a pose, use GT, or change the result. It intentionally
    ignores provisional/possibly stale diagnostics.selected_camera_center.
    """
    if result.get("status") != "positioned" or result.get("camera_to_world") is None:
        return {"assigned_zone_id": None, "reason": "solver_abstained"}
    matrix = np.asarray(result["camera_to_world"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return {"assigned_zone_id": None, "reason": "invalid_pose_matrix"}
    namespace = {"np": np, "math": math, "Any": Any}
    source_functions(Path(source) / "geometry_service/localization.py",
        ["_segment_distance", "_polygon_distance", "_pose_scene_prior"], namespace)
    prior = namespace["_pose_scene_prior"](matrix, payload)
    return {"assigned_zone_id": prior.get("zone_id") if prior["accepted"] else None,
            "final_pose_scene_prior": prior}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "one")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from geometry_service.contracts import CameraLocalizationRequest, CameraLocalizationResponse
    from app.roomplan import RoomPlanNormalizedScan, roomplan_geometry

    localization = source / "geometry_service/localization.py"
    backend = source / "app/main.py"
    namespace = {"np": np, "math": math, "Any": Any}
    source_functions(localization, ["_segment_distance", "_polygon_distance", "_pose_scene_prior"], namespace)
    source_functions(backend, ["roomplan_pose_scene_validation", "roomplan_map_for_camera"], namespace)
    namespace["roomplan_geometry_for_localization"] = lambda row: row

    def zone(name, x0, z0, x1, z1, floor_y=0):
        return {"id": name, "floor_y": floor_y, "polygon": [
            {"x": x, "z": z} for x, z in [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]]}

    # Three side-by-side rooms, each touching the same 2m-wide hallway.
    zones = [zone("room-a", 0, 0, 3, 4), zone("room-b", 3, 0, 6, 4),
             zone("room-c", 6, 0, 9, 4), zone("hallway", 0, -2, 9, 0)]
    raw_request = {"landmarks": [{"point": [i, 1, 1], "descriptor_base64":
        base64.b64encode(bytes(32)).decode()} for i in range(6)],
        "frames": [{"frame_base64": "synthetic-placeholder-not-decoded", "width": 640, "height": 480}],
        "room_zones": zones}
    payload = CameraLocalizationRequest.model_validate(raw_request)
    checks = [{"case": "room_id_omitted_validates", "passed": True}]
    try:
        CameraLocalizationRequest.model_validate({**raw_request, "room_id": "room-b"})
    except ValidationError as exc:
        checks.append({"case": "room_id_extra_is_forbidden", "passed": any(
            e["loc"] == ("room_id",) and e["type"] == "extra_forbidden" for e in exc.errors())})
    else:
        checks.append({"case": "room_id_extra_is_forbidden", "passed": False})
    checks.append({"case": "response_has_no_top_level_room_assignment", "passed":
                   "room_id" not in CameraLocalizationResponse.model_fields and
                   "zone_id" not in CameraLocalizationResponse.model_fields})

    def exercise(name, center, expected_id, expected_accept=True, requested_zones=None):
        selected_zones = zones if requested_zones is None else requested_zones
        request = CameraLocalizationRequest.model_validate({**raw_request, "room_zones": selected_zones})
        matrix = np.eye(4)
        matrix[:3, 3] = center
        worker = namespace["_pose_scene_prior"](matrix, request)
        accepted, api = namespace["roomplan_pose_scene_validation"](
            {"room_zones": selected_zones}, matrix.tolist())
        checks.append({"case": name, "synthetic_estimate_xyz": center,
            "worker": worker, "backend": api,
            "passed": worker.get("zone_id") == expected_id and worker["accepted"] == expected_accept
            and api.get("zone_id") == expected_id and accepted == expected_accept})

    for name, x, z in [("room-a", 1.5, 2), ("room-b", 4.5, 2), ("room-c", 7.5, 2), ("hallway", 4.5, -1)]:
        exercise(name + "_interior", [x, 1.6, z], name)
    exercise("shared_doorway_first_zone_wins", [1.5, 1.6, 0], "room-a")
    exercise("shared_doorway_reversed_order_changes_room", [1.5, 1.6, 0], "hallway", requested_zones=list(reversed(zones)))
    exercise("outside_0.5m_still_accepted", [9.5, 1.6, 2], "room-c")
    exercise("outside_0.8m_rejected", [9.8, 1.6, 2], "room-c", False)
    exercise("no_zones_no_assignment", [4.5, 1.6, 2], None, requested_zones=[])
    exercise("overlapping_stories_nearest_xz_ignores_height", [1.5, 5.6, 2], "lower", False,
             [zone("lower", 0, 0, 3, 4), zone("upper", 0, 0, 3, 4, floor_y=4)])

    def element(name, polygon, kind="floor"):
        xs = [p["x"] for p in polygon]; zs = [p["z"] for p in polygon]
        return {"id": name, "category": kind, "confidence": "high",
            "center": {"x": sum(xs)/len(xs), "y": 0, "z": sum(zs)/len(zs)},
            "dimensions": {"x": max(xs)-min(xs), "y": 0.1, "z": max(zs)-min(zs)},
            "transform": np.eye(4).tolist(), "vertices": [{**p, "y": 0} for p in polygon]}

    scan = {"schema_version": "roomplan-normalized.v1", "producer": "native-ios", "framework": "RoomPlan",
            "units": "m", "up_axis": "Y", "coordinate_frame": "roomplan-local", "geometry_type": "3d",
            "floors": [element(z["id"], z["polygon"]) for z in zones]}
    derived = roomplan_geometry(RoomPlanNormalizedScan.model_validate(scan))["room_zones"]
    checks.append({"case": "four_floor_surfaces_create_four_zones", "zone_ids": [z["id"] for z in derived],
                   "passed": [z["id"] for z in derived] == [z["id"] for z in zones]})
    wall_only = {**scan, "floors": [], "walls": [element("wall-proxy", z["polygon"], "wall") for z in zones]}
    derived = roomplan_geometry(RoomPlanNormalizedScan.model_validate(wall_only))["room_zones"]
    checks.append({"case": "wall_only_scan_collapses_into_one_convex_hull", "zone_ids": [z["id"] for z in derived],
                   "passed": len(derived) == 1})

    canonical = {"id": "full-home", "source": "roomplan-lidar-3d"}
    fragment = {"id": "room-b-fragment", "metadata": {"map_scope": "fragment", "canonical_map_id": "full-home"}}
    current = {"room_id": None}
    queries = []
    namespace.update(active_map_row=lambda _: canonical, map_source=lambda row: row["source"],
        roomplan_map_metadata=lambda row: row["metadata"],
        db=SimpleNamespace(one=lambda *a: current, many=lambda *a: queries.append(a) or [fragment]))
    selected = namespace["roomplan_map_for_camera"]("home", "camera")
    checks.append({"case": "unassigned_camera_uses_full_canonical_home", "passed": selected is canonical and not queries})
    current["room_id"] = "room-b"
    selected = namespace["roomplan_map_for_camera"]("home", "camera")
    checks.append({"case": "stored_camera_room_selects_fragment", "passed": selected is fragment})

    final_pose = np.eye(4); final_pose[:3, 3] = [4.5, 1.6, -1]
    result = {"status": "positioned", "camera_to_world": final_pose.tolist(),
              "diagnostics": {"scene_prior": {"zone_id": "room-a"}}}
    derived = assignment_from_result(result, payload, source)
    checks.append({"case": "adapter_scores_final_pose_not_stale_diagnostic", "result": derived,
                   "passed": derived["assigned_zone_id"] == "hallway"})
    derived = assignment_from_result({**result, "status": "needs_rescan"}, payload, source)
    checks.append({"case": "adapter_preserves_solver_abstention", "result": derived,
                   "passed": derived["assigned_zone_id"] is None})

    files = [localization, backend, source / "geometry_service/contracts.py", source / "app/roomplan.py"]
    report = {"scope": "Synthetic contract and actual-helper checks only; no image localization run, no accuracy claim",
        "source_commit_reported_by_parent": "9c05becdc3af59b15092075c1eb1dc90e2ab9e70",
        "execution": "Contracts imported normally; actual helper function AST executed unchanged without full module imports",
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "passed": all(c["passed"] for c in checks), "check_count": len(checks), "checks": checks}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
