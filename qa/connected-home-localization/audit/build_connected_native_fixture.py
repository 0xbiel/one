"""Create synthetic native-contract-shaped RoomPlan data from scan geometry only.

This is NOT a real iPhone/LiDAR capture. The discriminator constants are required
by ONE's strict schema; the separate provenance record documents that limitation.
No evaluator ground truth, query pose, query room label, or image is read.
"""
from __future__ import annotations
import os

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "pilot/fixtures/connected_home"
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("ONE_SOURCE_ROOT", str(ROOT.parents[1])))
from app.roomplan import RoomPlanNormalizedScan, roomplan_geometry


def transform(center):
    return [[1, 0, 0, center["x"]], [0, 1, 0, center["y"]],
            [0, 0, 1, center["z"]], [0, 0, 0, 1]]


def element(name, category, center, dimensions, vertices=None, matrix=None):
    return {"id": name, "category": category, "confidence": "high",
            "center": center, "dimensions": dimensions,
            "transform": matrix if matrix is not None else transform(center),
            "vertices": [] if vertices is None else vertices}


def wall_surface(box, category="wall"):
    """The source cuboid's vertical midplane, preserving its thickness in dimensions."""
    c, d = box["center"], box["dimensions"]
    axis = "z" if d["x"] < d["z"] else "x"
    vertices = []
    for horizontal, vertical in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        p = dict(c)
        p[axis] += horizontal * d[axis] / 2
        p["y"] += vertical * d["y"] / 2
        vertices.append(p)
    return element(box["id"], category, c, d, vertices)


def main():
    inputs = [FIXTURE / "scan.json", FIXTURE / "structural_boxes.json"]
    scan, boxes = [json.loads(p.read_text()) for p in inputs]
    floors, sections, walls, openings = [], [], [], []
    for zone in scan["room_zones"]:
        poly = zone["polygon"]
        xs, zs = [p["x"] for p in poly], [p["z"] for p in poly]
        center = {"x": sum(xs)/len(xs), "y": zone["floor_y"], "z": sum(zs)/len(zs)}
        floors.append(element("floor_" + zone["id"], "floor", center,
            {"x": max(xs)-min(xs), "y": 0.15, "z": max(zs)-min(zs)},
            [{"x": p["x"], "y": zone["floor_y"], "z": p["z"]} for p in poly]))
        sections.append({"id": zone["id"], "label": zone["id"].replace("_", " ").title(),
                         "center": center, "story": 0})
    for box in boxes:
        if box["id"] == "floor":
            continue  # The display floor is intentionally split into five zones.
        walls.append(wall_surface(box))
        if box["id"].startswith("lintel_"):
            height = box["center"]["y"] - box["dimensions"]["y"] / 2
            opening_box = {"id": box["id"].replace("lintel_", "opening_"),
                "center": {**box["center"], "y": height / 2},
                "dimensions": {**box["dimensions"], "y": height}}
            openings.append(wall_surface(opening_box, "opening"))
    objects = [element(o["id"], o["label"], o["center"], o["dimensions"],
                       matrix=o["transform"]["values"]) for o in scan["room_objects"]]
    # These generated semantic objects are axis-aligned. Check the entire 3D
    # doorway prism including source wall thickness, rather than only its center.
    clearance = []
    for opening in openings:
        obstructions = []
        for obj in objects:
            assert all(abs(obj["transform"][i][j] - int(i == j)) < 1e-6
                       for i in range(3) for j in range(3)), "Clearance check requires source AABBs"
            overlap = {axis: min(
                obj["center"][axis] + obj["dimensions"][axis]/2,
                opening["center"][axis] + opening["dimensions"][axis]/2) - max(
                obj["center"][axis] - obj["dimensions"][axis]/2,
                opening["center"][axis] - opening["dimensions"][axis]/2)
                for axis in "xyz"}
            if all(value > 1e-6 for value in overlap.values()):
                obstructions.append({"object_id": obj["id"], "overlap_m": overlap})
        clearance.append({"opening_id": opening["id"], "center": opening["center"],
                          "dimensions": opening["dimensions"], "object_count_checked": len(objects),
                          "obstructions": obstructions})
    assert not any(c["obstructions"] for c in clearance), clearance
    raw = {"schema_version": "roomplan-normalized.v1", "producer": "native-ios", "framework": "RoomPlan",
        "units": "m", "up_axis": "Y", "coordinate_frame": "roomplan-local", "geometry_type": "3d",
        "walls": walls, "floors": floors, "openings": openings, "doors": [], "windows": [],
        "objects": objects, "sections": sections}
    normalized = RoomPlanNormalizedScan.model_validate(raw)
    geometry = roomplan_geometry(normalized)
    assert len(geometry["room_zones"]) == 5
    assert len(geometry["objects"]) == 32
    assert len(geometry["walls"]) == 16
    for actual, expected in zip(geometry["room_zones"], scan["room_zones"]):
        assert actual["id"] == expected["id"]
        assert actual["floor_y"] == expected["floor_y"]
        assert actual["polygon"] == expected["polygon"]
    assert len(openings) == 4 and all(abs(o["dimensions"]["y"] - 2.2) < 1e-9 for o in openings)
    output = FIXTURE / "normalized_scan.json"
    output.write_text(normalized.model_dump_json(indent=2) + "\n")
    # Roundtrip the exact saved file through the actual backend model/normalizer.
    roundtrip = roomplan_geometry(RoomPlanNormalizedScan.model_validate_json(output.read_text()))
    assert roundtrip == geometry
    derived = FIXTURE / "normalized_geometry.json"
    derived.write_text(json.dumps(geometry, indent=2) + "\n")
    report = {"synthetic": True, "is_native_capture": False, "is_lidar_capture": False,
        "purpose": "Shape-compatible synthetic fixture for the actual ONE backend normalizer; no production upload",
        "schema_discriminator_caveat": "producer=native-ios and framework=RoomPlan are required literals, not evidence of native capture",
        "input_paths_read": [str(p.relative_to(FIXTURE)) for p in inputs],
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [output, derived]},
        "backend_source_sha256": hashlib.sha256((ROOT / "one/app/roomplan.py").read_bytes()).hexdigest(),
        "contract_validation": "passed", "saved_file_roundtrip": "passed",
        "normalizer_zone_count": len(geometry["room_zones"]),
        "normalizer_zone_ids": [z["id"] for z in geometry["room_zones"]],
        "normalizer_matches_input_polygons_and_floor_heights": True,
        "counts": {k: len(raw[k]) for k in ["floors", "walls", "objects", "openings", "doors", "windows", "sections"]},
        "doorway_clearance": {"passed": True, "method": "positive-volume AABB overlap, tolerance 1e-6m",
                              "object_opening_pairs_checked": len(objects) * len(openings), "openings": clearance},
        "modeling_notes": ["Four rooms plus one hallway in the supplied generated home",
            "The single display-floor cuboid is split into five native-shaped floor surface polygons",
            "Wall surfaces are source cuboid midplanes; source thickness remains in dimensions",
            "Four doorway voids under the source lintels are represented as openings, not invented door leaves",
            "Floor polygon surfaces remain at scan floor_y=0; dimensions.y is nominal slab thickness",
            "RoomPlanMapIn upload metadata is intentionally absent: truthful native/LiDAR provenance cannot be claimed",
            "No evaluator ground truth, query pose, query-room label, or image was read"]}
    (FIXTURE / "normalized_scan_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
