> Publication availability: this is a compact QA checkout. Exact replay payloads and most original images/depth are omitted. See [../README.md](../README.md), [../omitted_payloads.json](../omitted_payloads.json) and the fresh-model recipe [../FRESH_RUN.md](../FRESH_RUN.md). Historical results and validation receipts were produced before this publication reduction.

# Connected-home camera-localization pilot

A procedural 14m x 10m home with four deliberately similar furnished rooms, a continuous 2m hallway, four 1.2m x 2.2m doorway openings and 2.8m walls. This is a synthetic regression fixture, not a real scan or a real-device acceptance test.

## Assets and actual runtime

- `connected_home.blend`: detailed source scene, measured in metres with Blender Z-up
- `connected_home_preview.png`: cutaway overview; outside walls are hidden for this overview only
- `fixtures/connected_home/connected_home_simple_boxes.usdz`: simple metric Y-up scan model, 49 cubes, no detailed meshes
- `fixtures/connected_home/normalized_scan.json`: synthetic data conforming to the actual RoomPlan schema; required native-ios/RoomPlan literals are schema discriminators, not claims of native capture. See provenance validation sidecar
- `fixtures/connected_home/scan.json`: all five floor zones and 32 native-style semantic bounds supplied to localization
- `fixtures/connected_home/ref_*.jpg`: 20 rendered scan reference images, with synthetic axial depth and known scan poses
- `fixtures/connected_home/query_*.jpg`: 12 distinct fixed-camera query views including hallway and before/after doorway positions

The pinned code is ONE main at 9c05becdc3af59b15092075c1eb1dc90e2ab9e70. Source is copied byte-exact and checked against Git blob IDs. No production application code was changed. This harness calls actual `build_visual_landmarks`, pretrained YOLO-World detector runtime and `localize_camera`. HTTP/authentication/persistence and physical iPhone capture are not exercised.

## Frozen cases and leakage boundary

The design is frozen before any localization call: 20 reference views, 12 query views, two conditions (`clean`, `unknown_intrinsics`), 24 calls. Two west-south query viewpoints are development; ten other query viewpoints are held out. Held-out means camera views, not unseen architecture: every room is present in the reference scan and all share a furniture family. The intentionally near-repeated layouts are visual room-confusion distractors.

The clean condition supplies calibrated query K. Unknown-intrinsics omits query K/FOV and invokes the actual focal-length search. Scan poses, intrinsics and depths are idealized synthetic truth, not measured ARKit/LiDAR. All camera images use one 480x360 pinhole camera with fx=fy=320px; this does not test lens diversity. Two exposure variants per query have the same pose and do not count as distinct query views. The procedural scene has no ceiling; rendering noise and ideal depth are not a complete physical camera model.

Only scan references supply world poses/depth to the feature builder. All five room polygons, all objects and 8,000 reference landmarks capped with the actual backend's per-view 3cm voxel merge and response-sorted round-robin block enter each cold-start request. No query room, query camera pose, manually specified matches, previous estimate, search_prior or person_anchors enters a solver request. Detector outputs are actual YOLO-World detections, never projected ground-truth boxes. Query truth is retained separately for scoring. The harness is not OS-sandboxed away from ground-truth files.

## What “room identification” means here

ONE estimates a camera pose in the full home map. The actual room-prior helper assigns a zone from that final returned camera center. A camera looking through a doorway remains in the room containing its camera center, not necessarily the room shown in the image. This is a real code-derived zone diagnostic, not an independent learned room classifier. The backend does not automatically write `camera.room_id`; a preassigned camera can restrict its map before solving, so this pilot uses whole-map/unassigned-camera semantics.

The raw `diagnostics.scene_prior` can describe an earlier candidate. Therefore the harness explicitly applies pinned `_pose_scene_prior` to the FINAL returned matrix and records it separately, mirroring backend final-pose validation. A failed, timeout, unavailable or needs-rescan call remains an abstention. No nearest-ground-truth camera lookup is used. Shared boundary ties are order-sensitive in current code; the tested doorway centers are 25cm either side to avoid an exactly shared edge.

## Validation and scoring

The USDZ passes all available USD validators and ONE's package validator. Normalized scan round-trips through actual backend parsing and yields all five floor zones. Semantic bounds are evaluated detailed mesh AABBs. Doorway clearance was checked against all 32 object AABBs. An inherited east-room TV was moved to the outer wall during fixture validation before any localization; the final scene and artifacts were regenerated.

A single serial process evaluates one case at a time; the watchdog starts immediately before localization, with a fixed 180-second budget. The fixture detector exits before pose evaluation; another authorized benchmark may use two CPU threads concurrently with the one-thread pose solver, so timings describe shared CPU execution. Errors and timeouts count in all success-rate denominators. Position and angular percentiles are positioned-only, alongside complete five-zone confusion and abstentions. Frozen thresholds: 10cm/2 degrees and 25cm/5 degrees. CPU timings include this environment's limitations and do not establish MPS/GPU deployment latency.

Remaining limits include real ARKit drift, real LiDAR noise, severe object motion/occlusion, rolling shutter, lens distortion, other camera lenses, map upload persistence, and real-room deployment. This pilot is intentionally bounded.

## Fresh reproduction

Use the published generation and scoring scripts through [FRESH_RUN.md](../FRESH_RUN.md) in an isolated working copy. A fresh render/model run may differ from the measured run. The original byte-exact inputs and raw result payloads are not present in this compact GitHub checkout.
