# Connected-home results — ONE main 9c05bec

## Result

24/24 calls completed with a `positioned` response; there were zero timeouts, startup errors, unavailable results, needs-rescan responses or abstentions. That response status did not guarantee a correct estimate.

- Held-out correct room: **13/20 (65%)**, representing 10 distinct viewpoints under two calibration conditions
- All-case correct room: **17/24 (70.8%)**, including four development cases
- Within 10cm and 2°: **14/24 (58.3%)** overall; **10/20 (50%)** held-out
- Within 25cm and 5°: **14/24 (58.3%)** overall; **10/20 (50%)** held-out
- Overall median position / rotation error: **7.98cm / 0.475°**
- Position error P90: **8.019m**; maximum: **11.392m**
- Rotation error P90: **11.352°**; maximum: **44.195°**
- Calibrated room accuracy: **9/12**; unknown-intrinsics: **8/12**

The held-out median position error was 17.3cm and P90 was 8.15m. Percentiles include every positioned case; all 24 cases returned a pose. These are 12 distinct poses with two calibration conditions, not 24 independent viewpoints.

## Five-zone room confusion

Rows = true camera-center room. Columns = predicted zone. No ground-truth nearest-camera lookup is used.

| True / predicted | west south | west north | east south | east north | hallway | abstain |
|---|---:|---:|---:|---:|---:|---:|
| west south | 5 | 0 | 0 | 0 | 1 | 0 |
| west north | 0 | 4 | 0 | 0 | 0 | 0 |
| east south | 2 | 0 | 2 | 0 | 0 | 0 |
| east north | 0 | 2 | 0 | 2 | 0 | 0 |
| hallway | 2 | 0 | 0 | 0 | 4 | 0 |

## Failure evidence

Two east-room viewpoints were assigned to copied west rooms under both calibration conditions, almost exactly 8m away. In the audited calibrated east-south case, the solver reported confidence 0.938145, 87 inliers and 2.47px reprojection error. Its position was within 9.1mm of the true camera translated −8m along X. All 20 reference views and all 8,000 landmarks were present.

Doorway queries exposed further problems. The west doorway approach returned 0.66m/6.31° error when calibrated and 1.36m/11.86° with automatic focal length. Just inside that door, automatic focal length placed the camera back in the hallway, 2.02m from truth. An east-north doorway view was placed in west-south: 11.39m/34.10° calibrated and 9.23m/44.19° uncalibrated.

The source confidence calculation has no competing-room uniqueness penalty. Initial per-view PnP is restricted to the three highest-ranked views plus an all-view hypothesis, with no room-diversity requirement. Saved diagnostics do not expose every initial view ranking, so the exact dropped hypotheses are not established. Guided matching did not discard eastern views via an eight-view cap; its reported west-view list consists of matches surviving around the wrong hypothesis.

Consensus diagnostics can count a temporal aggregate as a third frame even though the request contains two physical images. A focused check showed this was not the cause of the audited 8m alias: direct scan-view support already passed the gate, and the recomputed base confidence exceeded either consensus floor.

## Meaning and next step

This stress fixture shows the current full-home localization path can confidently accept a physically plausible pose in the wrong room. It does not establish real-home error rates. Before relying on automatic room placement, a useful next change would compare hypotheses across distinct rooms and abstain or request confirmation when room evidence is ambiguous, followed by new held-out tests. No application fix, repository push or deployment was performed in this task.

Room assignment is a diagnostic derived from the returned camera center using the actual ONE room-prior helper. The app does not automatically write camera.room_id. An already assigned camera may filter the map; these tests use whole-home/unassigned-camera semantics.

## Reproduction and limits

See README.md, frozen_design.json, source_integrity.json and model_provenance.json. The source is byte-exact main 9c05becdc3af59b15092075c1eb1dc90e2ab9e70. Query room/pose/prior/anchors are absent from all request payloads. Actual scan feature construction, actual backend voxel merge block, actual YOLO-World detections and actual worker localization were used. HTTP persistence and iPhone capture were not tested.

The synthetic home has four deliberately near-repeated rooms plus a hallway, one camera model, ideal scan poses/depth, no ceiling and static furniture. Held-out refers to camera views, not an unseen architecture. No production parameters were tuned. Thresholds and the 180s deadline were frozen. CPU solve timing excludes startup and runs with one solver thread; another authorized benchmark may use two CPU threads. No heavy detector instances were concurrent.

Raw request payloads are stored separately in four six-case ZIP parts, each under 8MB, with SHA256 manifests. The final evaluation bundle retains rendered RGB/depth, normalized scan, simple USDZ, full source recipe, derived landmarks, raw results, scoring data and validation evidence.

## Code references

- [Final-pose room geometry check](https://github.com/0xbiel/one/blob/9c05becdc3af59b15092075c1eb1dc90e2ab9e70/geometry_service/localization.py#L3483-L3529)
- [Confidence computation](https://github.com/0xbiel/one/blob/9c05becdc3af59b15092075c1eb1dc90e2ab9e70/geometry_service/localization.py#L3756-L3763)
- [Candidate acceptance](https://github.com/0xbiel/one/blob/9c05becdc3af59b15092075c1eb1dc90e2ab9e70/geometry_service/localization.py#L3877-L3900)
- [Top-three view selection](https://github.com/0xbiel/one/blob/9c05becdc3af59b15092075c1eb1dc90e2ab9e70/geometry_service/localization.py#L4443-L4462)
- [Preassigned-camera map filtering](https://github.com/0xbiel/one/blob/9c05becdc3af59b15092075c1eb1dc90e2ab9e70/app/main.py#L1939-L1958)
