"""Independently audit final connected-home outputs; never imports summarizer.

No solver is run. Only original source helper AST, requests, results, and scoring
ground truth are read. Outputs under audit are validation artifacts only.
"""
from __future__ import annotations
import os
import ast
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "pilot"
SOURCE = Path(os.environ.get("ONE_SOURCE_ROOT", str(ROOT.parents[1])))
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
from geometry_service.contracts import CameraLocalizationRequest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
            print("MISMATCH: " + message, flush=True)
    manifest = json.loads((P / "manifest.json").read_text())
    truth = json.loads((P / "fixtures/connected_home/evaluator_groundtruth.json").read_text())
    queries = {r["id"]: r for r in truth["records"] if r["role"] == "query"}
    design = json.loads((P / "frozen_design.json").read_text())
    thresholds = design["thresholds"]
    stored = json.loads((P / "results_summary.json").read_text())
    stored_rows = {r["case"]: r for r in stored["rows"]}
    scan = json.loads((P / "fixtures/connected_home/scan.json").read_text())
    zones = [z["id"] for z in scan["room_zones"]]
    source_path = SOURCE / "geometry_service/localization.py"
    helpers = [n for n in ast.parse(source_path.read_text()).body if isinstance(n, ast.FunctionDef)
               and n.name in {"_segment_distance", "_polygon_distance", "_pose_scene_prior"}]
    ns = {"np": np, "math": math}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *helpers], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), ns)
    source_hashes = {p.name: sha(p) for p in sorted((SOURCE / "geometry_service").glob("*.py"))}
    fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    integrity = json.loads((ROOT / "source_integrity.json").read_text())
    for item in integrity["files"]:
        path = SOURCE / item["path"]
        check(path.exists(), "Missing source " + item["path"])
        if path.exists():
            raw = path.read_bytes()
            check(hashlib.sha256(raw).hexdigest() == item["sha256"], "Source SHA " + item["path"])
            check(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == item["git_blob"], "Git blob " + item["path"])
    rows, landmark_hashes = [], set()
    for item in manifest:
        case = item["case"]
        request_path = P / item["input"]
        check(sha(request_path) == item["sha256"], case + " request manifest hash")
        raw = json.loads(request_path.read_text())
        check(set(raw) <= {"schema_version", "landmarks", "frames", "room_objects", "room_zones", "intrinsics", "object_detections"}, case + " unexpected request fields")
        check(all(set(f) == {"frame_base64", "width", "height"} for f in raw["frames"]), case + " query frame leakage")
        check(all(set(l) <= {"point", "descriptor_base64", "sift_descriptor_base64", "response", "view_id"} for l in raw["landmarks"]), case + " landmark leakage")
        check(raw["room_zones"] == scan["room_zones"] and raw["room_objects"] == scan["room_objects"], case + " query-specific map filtering")
        check(collections.Counter(l["view_id"] for l in raw["landmarks"]) == {f"scan-view-{n}": 400 for n in range(1, 21)}, case + " reference-view coverage")
        landmark_hashes.add(hashlib.sha256(json.dumps(raw["landmarks"], sort_keys=True).encode()).hexdigest())
        check(("intrinsics" in raw) == (item["condition"] == "clean") and "fov_degrees" not in raw, case + " intrinsics condition")
        payload = CameraLocalizationRequest.model_validate(raw)
        result_path = P / "final-results/current" / (case + ".json")
        result = json.loads(result_path.read_text())
        check(result["input_sha256"] == item["sha256"], case + " result/request mismatch")
        check(result["geometry_fingerprint"] == source_hashes and result["geometry_fingerprint_sha256"] == fingerprint, case + " result source mismatch")
        check(result["status"] == "positioned", case + " non-positioned/timeout/error")
        check(0 < result["wall_seconds"] < design["timeout_seconds"], case + " time budget")
        log = result_path.with_suffix(".log").read_text()
        check("Traceback (most recent call last)" not in log and log.rstrip().endswith("positioned"), case + " execution log error")
        started = json.loads(result_path.with_suffix(".started.json").read_text())
        check(started["input_sha256"] == item["sha256"] and started["geometry_fingerprint"] == source_hashes, case + " started metadata")
        gt = queries[item["query"]]
        check(item["split"] == gt["split"], case + " split mismatch")
        pose = np.asarray(result["camera_to_world"], dtype=float)
        expected = np.asarray(gt["camera_to_world"], dtype=float)
        check(pose.shape == (4, 4) and bool(np.isfinite(pose).all()), case + " invalid pose")
        prior = ns["_pose_scene_prior"](pose, payload)
        predicted_room = prior.get("zone_id") if prior["accepted"] else None
        check(prior == result["final_pose_scene_prior"], case + " final-pose zone diagnostic")
        error = float(np.linalg.norm(pose[:3, 3] - expected[:3, 3]))
        angle = math.degrees(math.acos(float(np.clip((np.trace(expected[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1))))
        room_ok = predicted_room == gt["room"]
        row = {"case": case, "split": gt["split"], "condition": item["condition"], "true_room": gt["room"],
            "predicted_room": predicted_room, "status": result["status"], "seconds": result["wall_seconds"],
            "confidence": result["confidence"], "position_error_m": error, "rotation_error_deg": angle,
            "predicted_position": pose[:3, 3].tolist(), "true_position": expected[:3, 3].tolist(), "room_correct": room_ok,
            "tight_success": room_ok and error <= thresholds["tight_m"] and angle <= thresholds["tight_deg"],
            "loose_success": room_ok and error <= thresholds["loose_m"] and angle <= thresholds["loose_deg"]}
        rows.append(row)
    check(len(manifest) == len(rows) == len(stored_rows) == 24, "24 cases required")
    check({(m["query"], m["condition"]) for m in manifest} == {(q, c) for q in queries for c in design["conditions"]}, "Each of 12 queries needs both conditions exactly once")
    check({p.stem for p in (P / "final-results/current").glob("*.json") if not p.name.endswith(".started.json")} == {m["case"] for m in manifest}, "Unexpected or missing raw result files")
    check(len(landmark_hashes) == 1, "Query-specific landmark index")
    check(len({m["query"] for m in manifest if m["split"] == "heldout"}) == 10, "10 heldout poses")
    check(len({m["query"] for m in manifest if m["split"] == "dev"}) == 2, "2 dev poses")
    check(len({tuple(np.asarray(q["camera_to_world"]).ravel()) for q in queries.values()}) == 12, "12 unique query transforms")

    def aggregate(items):
        confusion = {z: {k: 0 for k in [*zones, "abstain"]} for z in zones}
        for row in items:
            confusion[row["true_room"]][row["predicted_room"] or "abstain"] += 1
        answer = {"n": len(items), "positioned": sum(r["status"] == "positioned" for r in items),
            **{k: sum(bool(r[k]) for r in items) for k in ["room_correct", "tight_success", "loose_success"]},
            "statuses": dict(collections.Counter(r["status"] for r in items)), "room_confusion": confusion,
            "metric_denominator": "Positioned only for errors; all requested cases for success and room rates"}
        for key in ["position_error_m", "rotation_error_deg"]:
            values = [r[key] for r in items if r["status"] == "positioned"]
            answer[key] = {"median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "max": max(values)}
        return answer
    independent = {"expected_cases": 24, "completed_cases": len(rows), "all": aggregate(rows),
        "by_condition": {c: aggregate([r for r in rows if r["condition"] == c]) for c in design["conditions"]},
        "by_split": {s: aggregate([r for r in rows if r["split"] == s]) for s in ["dev", "heldout"]}, "rows": rows}
    def compare(a, b, path="summary"):
        if isinstance(a, dict):
            check(set(a) == set(b), path + " keys")
            for key in a:
                if key in b: compare(a[key], b[key], path + "." + key)
        elif isinstance(a, list):
            check(len(a) == len(b), path + " length")
            for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path + f"[{i}]")
        elif isinstance(a, float): check(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8), path + " value")
        else: check(a == b, path + " value")
    compare(independent, stored)
    markdown = (P / "RESULTS.md").read_text()
    markdown_assertions = ["24/24", "17/24 (70.8%)", "13/20 (65%)", "14/24 (58.3%)", "10/20 (50%)",
        "7.98cm / 0.475°", "8.019m", "11.392m", "11.352°", "44.195°", "9/12", "8/12", "17.3cm", "8.15m",
        "12 distinct poses", "0.938145", "87 inliers", "2.47px"]
    for value in markdown_assertions: check(value in markdown, "RESULTS.md missing/changed numeric claim: " + value)
    table_rows = [line for line in markdown.splitlines() if line.startswith("| ") and line.split("|")[1].strip().replace(" ", "_") in zones]
    for line in table_rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        check([int(c) for c in cells[1:]] == list(independent["all"]["room_confusion"][cells[0].replace(" ", "_")].values()), "RESULTS.md confusion " + cells[0])
    check(len(table_rows) == 5, "RESULTS.md five confusion rows")
    report = {"passed": not errors, "errors": errors, "independent_of_summarize_connected_py": True,
        "no_solver_rerun": True, "source_commit": integrity["commit"], "source_files_checked": len(integrity["files"]),
        "request_hashes_verified": len(rows), "result_source_fingerprints_verified": len(rows),
        "same_full_landmark_index_in_all_requests": len(landmark_hashes) == 1,
        "input_contract_and_no_query_pose_room_prior_anchors_checked": True,
        "request_frames_only_rgb_width_height": True, "reference_view_count": 20, "landmarks_per_view": 400,
        "split_cases": dict(collections.Counter(r["split"] for r in rows)), "split_unique_poses": {"heldout": 10, "dev": 2},
        "timeouts_errors_unavailable_abstentions": sum(r["status"] != "positioned" for r in rows),
        "maximum_solver_seconds": max(r["seconds"] for r in rows), "thresholds": thresholds,
        "summary_comparison_tolerance": 1e-8, "markdown_numeric_claims_checked": markdown_assertions,
        "packaged_source_integrity_exists_and_matches": (P / "source_integrity.json").exists() and sha(P / "source_integrity.json") == sha(ROOT / "source_integrity.json"),
        "input_artifact_hashes": {n: sha(P / n) for n in ["manifest.json", "results_summary.json", "RESULTS.md", "fixtures/connected_home/evaluator_groundtruth.json"]},
        "metrics": {k: v for k, v in independent.items() if k != "rows"}, "rows": rows}
    target = ROOT / "audit/independent_final_validation.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "errors": errors, "all": independent["all"], "heldout": independent["by_split"]["heldout"], "saved": str(target)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
