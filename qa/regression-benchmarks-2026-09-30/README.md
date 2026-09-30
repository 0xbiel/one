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
| Geometry incl. added regression cases | not compared | 61/61 | 61/61 |

The four association cases overlap one test in the first row; do not add these totals.
Geometry production code is byte-identical in #1/#2. Maximum camera-center axis error in the original semantic-seed fixture improved from 0.5289 m to floating-point zero. This is an exact synthetic fixture, not an empirical real-camera accuracy claim. The provisional fit retained six correct inliers and 0.3796 px residual; its fitted subset diagnostic was corrected from seven to six. No original tolerances or acceptance gates were relaxed.

#2 backend run: **91 passed, 2 PostgreSQL tests skipped locally**. CI runs PostgreSQL separately and has passed for #2; exact final CI and full benchmark results will be appended. #1 CI passed 151 tests but independent review still caught named-object identity loss, demonstrating that green tests alone were insufficient.

## Identity limits

A unique unclaimed registered label can be associated for backward compatibility, but that is not verified physical identity. Multiple named same-label candidates remain ambiguous. Active tracks from different cameras cannot claim the same object. A capture pause preserves an arriving track's existing binding. Distinct labels are selected before the vocabulary cap.

## Latency

`benchmark_rates.py` runs 20 randomized-order paired batches of 10,000 calls after warmup. Correct trend cases improve 5/8 to 8/8. Observed median trend cost was 0.99 us to 1.38 us per call on a shared CPU, with substantial scheduling noise. This is a correctness fix, **not a speedup claim**. Test-suite runtimes on shared concurrent machines are not performance comparisons.

## Real-image fall evaluation (partial; remaining clips in progress)

Real Ultralytics YOLO-World `yolov8s-worldv2.pt`, SHA256 `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`, ran decoded RGB frames through the production LocalServiceDetector/API, stabilization, and fall event path. No ground-truth boxes were injected. CPU clocks were deterministically replayed at clip capture FPS so offline inference time could not expire temporal tracks. Identical JPEG bytes + prompt lists may reuse deterministic image-model outputs; **every timestamp still traverses the API/tracker**.

Completed side-fall and occluded-fall clips: **0/2 falls signaled**, 160 frames. Each had person detections in 26/80 frames and stable person tracks in 24/80 frames; the model lost the person during the fall. These are known misses, not a successful fall detector. The app's fall heuristic was not changed by these fixes, and no fall-detection improvement is claimed. Exact-label counters exclude non-person detections. Memoized-harness timing must not be presented as deployment inference FPS.

Controlled seven-clip generators, labels, manifest, hashes and scorer are under `qa/fall-animations`. Physics-driven clips are separate fixtures with separate FPS/provenance. Animated synthetic evidence is not clinical, emergency, or real-world safety validation.

## Reproduce

Use Python 3.12 and the checked environment locks; CPU Torch can be installed from the official PyTorch CPU wheel index. Run `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest` for backend and geometry. The dependency locks describe the measured environment; future version ranges in the package are not a frozen benchmark environment. Model weights are downloaded from the official Ultralytics assets release and are not committed.
