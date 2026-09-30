# Simplified scan / detailed RGB camera-localization benchmark

## What this actually tests

Three deterministic **separate single-room layout variants** (one development,
two held out), not a connected multi-room home. Each has four scan reference
poses and two distinct fixed-camera query poses. Two query exposures have the
same camera pose: this checks repeated static-view photometric stability, not
two independent query viewpoints. There are 24 cases (6 poses × 4 conditions).
This is a small synthetic regression suite, not a real-device acceptance study.

The RGB images are physically lit Blender Cycles renders with detailed furniture,
legs, cushions, floor seams, wall art, books, and light fixtures. The separate
metric Y-up USDZ contains **12 cubes and zero detailed meshes**: 8 semantic
object bounds, floor, and 3 walls. Every furniture box is the exact evaluated
axis-aligned bounding box of its detailed scene group. The JSON object centers,
dimensions, and transforms describe the same bounds as the USDZ.

## Runtime code trace

- `app/main.py:roomplan_visual_landmarks` validates a native RoomPlan map,
  invokes the worker, assigns scan-batch view IDs, merges 3cm per-view voxels,
  and retains at most 8,000 landmarks using response-sorted round-robin views.
- `geometry_service/contracts.py:VisualLandmarkFrame` requires RGB, intrinsics,
  and ARKit camera-to-world. The depth bundle is optional.
- `geometry_service/localization.py:build_visual_landmarks` extracts actual
  ORB and optional SIFT features. `_world_point` backprojects axial depth into
  ARKit +X right, +Y up, -Z forward and then into RoomPlan coordinates.
  `_triangulated_candidates` can obtain points from known scan poses without
  depth. Scan camera poses remain necessary in that fallback.
- `geometry_service/app.py` detects semantic objects/occluders in query frames
  using real YOLO-World. The fixture invokes the same detector runtime with the
  scan-derived label aliases and minimum confidence 0.10, caching its actual
  outputs. No projected ground-truth object boxes or object IDs are supplied as
  query correspondences.
- `localize_camera` matches derived reference landmarks to query JPEGs, solves
  PnP, and uses RoomPlan cuboid/room priors and semantic detections for candidate
  initialization/verification. Exact query camera poses are evaluation-only.
- USDZ is the revision-linked display artifact; structured RoomPlan geometry
  and derived visual landmarks are the solver inputs. A detailed mesh export
  is not a substitute for these inputs.

This agrees with `0xbiel/one-docs` at
`7ec1dffff326a4c274bae4bc7d11206a9ecbd37b`, especially
`pages/architecture/lidar-mapping-visual-guide.md` and `pages/api/spatial-vision.md`.
Some docs discuss approximate ARKit-video maps; this benchmark follows the
native RoomPlan visual-landmark path and does not claim that RGB alone supplies
metric scan poses or depth.

## Frozen conditions and leakage boundary

1. `clean`: exact scan pose/K, synthetic axial depth, exact query intrinsics
2. `noisy_depth_missing_boxes`: independent 2.5cm Gaussian scan-depth noise,
   10% dropped depth pixels, and every second semantic object omitted
3. `no_depth`: reference RGB + exact scan poses/K only; actual pairwise ORB
   triangulation, separately identified scan-pair batches
4. `unknown_intrinsics`: clean scan inputs, but query K/FOV omitted so the
   actual automatic focal-length search is exercised

The renderer declared all layouts, reference poses, query poses, and condition
parameters before final evaluation. No production parameter was tuned on these
held-out cases. A development preflight caught an incorrect synthetic-world
axis conversion; all renders, metadata, landmarks, and request bytes were rebuilt
after correcting it. This was fixture validation, not a production improvement.

The reference RGB-D is an idealized simulated scan, **not measured LiDAR**;
its exact reference poses are an ARKit oracle assumption. Depth noise does not
model ARKit drift or all LiDAR failure modes. Query ground truth never enters
request JSON. It remains in `evaluator_groundtruth.json`, read only by the evaluator
for scoring returned solver outputs. `run_single.py` does not read ground truth
and receives only request/output filenames; it is not OS-sandboxed from those
files. Neither previous query estimates nor search priors are supplied.

Both versions receive byte-identical SHA-256-frozen payloads, with the same
actual descriptors and genuine detector results. Failed/needs-rescan/timeout
cases count in the denominator. Translation/rotation percentiles are explicitly
positioned-only; threshold success rates include every requested case. Thresholds
were fixed at 10cm/2° and 25cm/5°. CPU pose refinement availability differs from
the intended MPS/CUDA deployment and is reported in raw diagnostics.

## Reproduction

Use Blender, USD Python (`usd-core` with validation support), NumPy/OpenCV,
and the repository's Python test dependencies. Real detections additionally
need the production YOLO-World checkpoint and its runtime dependencies.

1. `blender -b --python generate_scene.py` creates `assets/room.blend`
2. `blender -b -t 4 --python render.py -- assets/room.blend`
3. `python package_scans.py`, then `python validate_scan_bounds.py` and
   `VALIDATION_LABEL=candidate PYTHONPATH=CANDIDATE_CHECKOUT python validate_backend_usdz.py`
4. Set production model environment paths and run `python detect.py`
5. Set `PYTHONPATH` to the candidate checkout; run `python prepare.py`
6. `python validate.py`
7. Run `benchmark_serial.py --source BASELINE_CHECKOUT --label baseline`,
   then `benchmark_serial.py --source CANDIDATE_CHECKOUT --label optimized`.
   Never run these simultaneously with each other or other heavy model work.
8. Run `python summarize.py` after both serial runs finish

The retained archive includes original final fixture bytes, input manifests,
raw results, summaries, scripts, code hashes, and environment versions. Rendering
on a different Blender build may not produce byte-identical JPEGs; the retained
payloads are the authoritative paired comparison inputs.

### Additional limits

All three variants use the same 6m × 5m rectangular shell and furniture model
family; two variants rearrange object positions/colors. Held out means the
layout was excluded from development, not unseen architecture or unseen object
categories. All renders use one 480×360 camera model with fx=fy=320px
(horizontal FOV 73.74°); this does not establish generalization across camera
lenses or resolutions. There are no open doorways to another mapped room, severe scan
pose drift, moving furniture, lens distortion, rolling shutter, real camera
noise, or multi-home distractors. Those remain untested.

The evaluated caller is the actual worker function plus its actual detector
runtime, not HTTP upload/authentication/persistence or an iOS/macOS capture.
Known query K is privileged calibration available only in the calibrated
conditions; the unknown-intrinsics condition omits it entirely. The 180-second
per-call harness deadline is fixed. A timeout is a measured CPU-budget failure,
not proof that the solver could never return a pose or that GPU deployment has
the same limit. The initial concurrent prototype was invalidated after a lock/deadline audit.
Final results use exclusive serial solver subprocesses, with the watchdog
starting from a readiness timestamp immediately before localization. There
are no locks or reused provisional outputs in the final driver. Startup
failures and compute timeouts are recorded separately; all remain failures
in the denominator. The separate provisional folder is audit history only.
