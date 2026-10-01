# Connected-home localization QA — compact publication

A generated four-room home with a hallway and doorways, tested against ONE main `9c05becdc3af59b15092075c1eb1dc90e2ab9e70` without application changes.

## Held-out result

**13/20 correct room assignments; 10/20 within 10cm and 2°.** These are ten distinct held-out poses under two calibration conditions. Including development cases: 17/24 correct rooms and 14/24 within either 10cm/2° or 25cm/5°. All calls returned `positioned`; worst position error was 11.39m.

This one synthetic stress fixture exposes confident wrong-room placements in repeated-looking rooms and doorway views. Reference scan poses/depth are idealized; these are not real-home acceptance rates. See [measured results](pilot/RESULTS.md), [numeric summary](pilot/results_summary.json), and the [historical independent verification receipt](audit/independent_final_validation.json).

![Measured localization results](pilot/localization_results.png)

## What is published

Reports, numeric summaries, frozen design/hash manifests, generation/scoring scripts, normalized geometry, one simple USDZ and four original camera JPEG examples:

- [49-cuboid USDZ](pilot/fixtures/connected_home/connected_home_simple_boxes.usdz)
- [Normalized synthetic RoomPlan scan](pilot/fixtures/connected_home/normalized_scan.json)
- [Scan provenance and validation](pilot/fixtures/connected_home/normalized_scan_validation.json)
- [Room camera example](pilot/fixtures/connected_home/query_west_south_0.jpg)
- [Repeated-room alias example](pilot/fixtures/connected_home/query_east_south_1.jpg)
- [Hallway example](pilot/fixtures/connected_home/query_transition_0.jpg)
- [Doorway failure example](pilot/fixtures/connected_home/query_transition_3.jpg)

## What is omitted

**Exact no-model replay is unavailable from this GitHub checkout.** The shared landmark/map payload, full reference RGB/depth set, remaining query/exposure images and compressed raw per-call results were not published because binary transfers were too slow. Both replay metadata JSON files are published, but the 109 omitted binary files are required for complete data access. Their paths, sizes and SHA256 hashes are in [omitted_payloads.json](omitted_payloads.json). A hash proves identity if bytes are later supplied; it cannot replace missing bytes.

The full replay remains retained in the original working copy, but is not accessible through this PR. The published historical validation receipts describe checks already performed against that complete original data. They do not mean this compact checkout independently contains everything needed to repeat those checks.

`restore_replay.py` is retained as a documented helper for a future complete payload set. It reports a missing-payload error in this compact checkout. Do not treat `--check-only` as a runnable verification here.

## Reproduce through a fresh model run

The published source can generate a new connected-home fixture, run genuine YOLO-World detection and ONE localization, and score the new outputs. This requires Blender, Python dependencies and model checkpoints; it is a **fresh render/model run**, not restoration of the original byte-exact benchmark. Results can differ with renderer, model, dependency, hardware or source changes.

Follow [FRESH_RUN.md](FRESH_RUN.md), which keeps new artifacts in an isolated working directory and preserves the published measurements. The original application revision, environment and checkpoint hashes are recorded in the manifests.

## Provenance and scope

Scenes, furniture and images were created procedurally with Blender/Cycles; no external photographs or stock art were used. The furniture recipe derives from this repository's earlier `qa/multiroom-localization` generator. No third-party image attribution is required. YOLO-World/CLIP weights are not redistributed.

The normalized scan uses required `native-ios`/`RoomPlan` literals for schema validation only. It is synthetic, not an iPhone/LiDAR capture. Tests call actual worker functions and final-pose room assignment; HTTP persistence was not exercised. The app does not automatically persist `camera.room_id`, and preassigned cameras may filter their maps. No application fixes, merging or deployment were performed.

A next improvement would retain competing room hypotheses and abstain when room evidence is ambiguous, then evaluate new held-out cases.
