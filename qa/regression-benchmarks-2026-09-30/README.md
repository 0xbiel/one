# ONE regression benchmark (2026-09-30)

Baseline production source: `e6abb03a803f403371626cfc2636c831308d71a3`.
Corrected implementation: `334235b6737e8ed56f0b0085aadea3d3eff53e95` (PR #2).
Rejected implementation: `7d4ad2e4224aba38d461607fc0cac5f43607d70f` (PR #1, unmerged).

## Measured correctness

| Same-input check | Baseline | Rejected #1 | Corrected #2 |
|---|---:|---:|---:|
| Analytics + same-label focused regression cases | 5/12 | 12/12 | 12/12 |
| Independent-review association regression cases | 1/4 | 2/4 | 4/4 |
| Original geometry cases | 57/59 | 59/59 | 59/59 |
| Geometry incl. added regression cases | 57/61 | 61/61 | 61/61 |

The four association cases overlap one test in the first row; do not add these totals.
Geometry production code is byte-identical in #1/#2. Maximum camera-center axis error in the original semantic-seed fixture improved from 0.5289 m to floating-point zero. This is an exact synthetic fixture, not an empirical real-camera accuracy claim. The provisional fit retained six correct inliers and 0.3796 px residual; its fitted subset diagnostic was corrected from seven to six. No original tolerances or acceptance gates were relaxed.

#2 complete backend+geometry run: **152 passed, 2 PostgreSQL tests skipped locally**. Its backend-only subset has 91 passed. CI runs PostgreSQL separately and has passed for #2; final image comparison and raw traces are committed in evidence/. #1 CI passed 151 tests but independent review still caught named-object identity loss, demonstrating that green tests alone were insufficient.

## Identity limits

A unique unclaimed registered label can be associated for backward compatibility, but that is not verified physical identity. Multiple named same-label candidates remain ambiguous. Active tracks from different cameras cannot claim the same object. A capture pause preserves an arriving track's existing binding. Distinct labels are selected before the vocabulary cap.

## Latency

`benchmark_rates.py` runs 20 randomized-order paired batches of 10,000 calls after warmup. Correct trend cases improve 5/8 to 8/8. Observed median trend cost was 0.99 us to 1.38 us per call on a shared CPU, with substantial scheduling noise. This is a correctness fix, **not a speedup claim**. Test-suite runtimes on shared concurrent machines are not performance comparisons.

## Real-image fall evaluation (all nine clips complete)

Real Ultralytics YOLO-World `yolov8s-worldv2.pt`, SHA256 `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`, ran decoded RGB frames through the production LocalServiceDetector/API, stabilization, and fall event path. No ground-truth boxes were injected. CPU clocks were deterministically replayed at clip capture FPS so offline inference time could not expire temporal tracks. Identical JPEG bytes + prompt lists may reuse deterministic image-model outputs; **every timestamp still traverses the API/tracker**.

All nine clips completed, **704 frames**: **0/5 falls signaled**, **0/4 negative clips signaled**. All baseline and corrected downstream outcomes match. The three controlled falls plus two physics falls were all missed. On the side-fall and occluded-fall clips, each had person detections in 26/80 frames and stable person tracks in 24/80 frames; the model lost the person during the fall. These are known misses, not a successful fall detector. The app's fall heuristic was not changed by these fixes, and no fall-detection improvement is claimed. Exact-label counters exclude non-person detections. Memoized-harness timing must not be presented as deployment inference FPS.

Controlled seven-clip generators, labels, manifest, hashes and scorer are under `qa/fall-animations`. Physics-driven clips are separate fixtures with separate FPS/provenance. Animated synthetic evidence is not clinical, emergency, or real-world safety validation.

## Reproduce

Use Python 3.12 and the checked environment locks; CPU Torch can be installed from the official PyTorch CPU wheel index. Run `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest` for backend and geometry. The dependency locks describe the measured environment; future version ranges in the package are not a frozen benchmark environment. Model weights are downloaded from the official Ultralytics assets release and are not committed.

## Physics clip results

Both full Bullet clips completed genuine image inference: lateral72frames and forward72frames, each12fps. Neither produced a fall event (0/2 physics falls signaled). Paired downstream replay of the same real model predictions on corrected code also produced no events. Exact detection coverage and raw results are in evidence/fall-physics-*.jsonl. The detector still loses the person; physically simulated input did not solve the visual-recognition gap.

## Identical expanded-suite comparison

Verified exactly the same154 test IDs on baseline and corrected code, in the same dependency environment: baseline139passed,13failed,2skipped; corrected152passed,0failed,2skipped. The two skips are PostgreSQL-only local checks; GitHub CI supplies PostgreSQL. Evidence/full-suite-comparison.json includes the test-identity fingerprint and regression-file hashes. Test timings are not a speed comparison.

## Separate ideal-box diagnostic

`oracle_box_diagnostic.py` loads projected ground-truth boxes at confidence1 solely to isolate the existing rule/tracker, and marks its output `oracle_ground_truth_boxes`. This is not detector inference or an end-to-end score. The unchanged stability+fall rules signal all5 fall fixtures but also signal the deliberate slow lie-down at4.3s (1/4 negative fixtures falsely signaled under ideal boxes). This shows both an upstream recognition gap in real images and a separate heuristic false-positive limitation. No fall thresholds were changed to fit these fixtures.

The candidate column replays exactly the saved genuine baseline image-model predictions through the corrected API; it is not a second independent image-inference run. Model and fall-rule files are unchanged. Final-image-comparison.json verifies matching source inference hashes, clip/frame/time completeness, and identical raw detections before computing the paired results. Zero false signals on these four clips does not imply good specificity in real use, particularly when person recognition disappears.
