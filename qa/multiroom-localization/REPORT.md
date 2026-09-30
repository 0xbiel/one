# Camera-localization benchmark results

## Result

**Optimized: 24/24 positioned. Baseline: 18/24 positioned and 6 timeouts.**
On held-out layouts, completion improved from **12/16 to 16/16**. Every optimized
case meets the predeclared 10cm / 2° gate. The 18 poses that both versions returned
are **exactly unchanged**. This is an improvement in completion within the fixed
180-second solver budget, not better geometric accuracy on previously solved cases.

Held-out optimized camera error:
- Median: **3.64cm / 0.387°**
- 95th percentile: **4.58cm / 0.576°**
- Worst: **4.91cm / 0.576°**

All failed/time-out attempts remain in success-rate denominators. Error percentiles
are positioned-only. The broader 25cm / 5° gate has the same success counts.

| Held-out condition | Baseline positioned | Optimized positioned | Translation p95 | Rotation p95 |
|---|---:|---:|---:|---:|
| Clean RGB-D | 4/4 | 4/4 | 3.95cm | 0.530° |
| Noisy/missing depth and half the boxes missing | 4/4 | 4/4 | 4.75cm | 0.558° |
| No-depth triangulation | 4/4 | 4/4 | 4.40cm | 0.574° |
| Unknown query intrinsics | 0/4 | 4/4 | 3.61cm | 0.361° |

## Runtime and the exact-equivalence change

All six unknown-intrinsics baseline cases exceeded the 180-second budget.
The optimized version completed all six in **61.75–86.04 seconds**. Unknown query
calibration was never supplied. These are CPU worker-function times, excluding
model detection, whose genuine results were frozen and identical for both versions.
Baseline timeouts are censored; no exact old-runtime speedup is claimed.

| Condition, all six poses | Baseline median completed-call time | Optimized median completed-call time |
|---|---:|---:|
| Clean RGB-D | 37.96s | 14.18s |
| Noisy/missing depth and boxes | 9.96s | 7.50s |
| No-depth triangulation | 4.96s | 3.98s |
| Unknown intrinsics | All six timed out | 71.12s |

The focused change caches ORB descriptor pairs passing the existing Hamming ≤56
gate. It retains at most 64 candidates per landmark; a potentially truncated/dense
row uses the original algorithm. Spatial gates, ratio tests, tie order, reverse
uniqueness, learned scoring, PnP and acceptance thresholds are unchanged. Full array
equality prevents stale mutation reuse; each request resets the thread-local cache.

Evidence: 362 exact ordered-match comparisons, 50 additional independent boundary
cases, and 63 passing geometry tests. A development-only actual-descriptor
microbenchmark observed 2.15× on the cold call and 5.18–6.15× on warm calls, with
671,056 retained array bytes. Independent review confirmed exact tuple/order/distance
equivalence. The pure pre-cache development request exceeded 180s, while the
cache version completed in 83.24s at 4.31cm / 0.423° error.

## Protocol and artifacts

Three separate 6×5m room-layout variants: one development, two held out. Four
known-pose scan-reference views and two unseen fixed-camera poses per layout;
two photometric exposures of each static query camera. Conditions: clean depth,
2.5cm Gaussian depth noise + 10% missing depth + half the objects removed,
no-depth triangulation, and unknown query intrinsics. The actual backend builds
ORB/SIFT landmarks from references and solves each query; actual YOLO-World
supplies query detections. Query poses are evaluator-only.

USDZ packages contain 12 simple cubes each and no detailed meshes. All three pass
the actual baseline and replacement backend package validators plus USD validation.
Reopened USDZ furniture bounds match the structured scan within 1 micrometre.
54 independent Blender/K projection checks agree within 0.001 pixel.

The initial concurrent prototype was invalidated after deadline/lock review and is
excluded. Final 48 runs used exclusive serial subprocesses, fresh outputs, no locks
or reuse, a readiness-based 180-second watchdog, all input SHA256 checks and full
geometry-source fingerprints. No code, input, threshold or split changed during
final evaluation. The independent reviewer rechecked all 48 outcomes, pose errors,
threshold flags, failure counts and code/input hashes.

Baseline code: `e6abb03a803f403371626cfc2636c831308d71a3`.
Candidate: reviewed `334235b6737e8ed56f0b0085aadea3d3eff53e95` geometry plus the
reviewed cache patch. The integrated commit is recorded with the final publication.
This full-suite comparison includes the earlier reviewed geometry fixes as well
as the cache; the development preflight and descriptor-equivalence experiments
separately isolate the cache addition.
Complete raw poses, ground truth, detector outputs, payloads, hashes and environment
records are in the retained archive. `results.json`, `summary.json` and
`performance_summary.json` are the machine-readable final summaries.

## Limits

This is a small synthetic regression suite, not real-device acceptance. Scan
poses/depth are idealized or simulated; even no-depth mode retains exact reference
poses. All scenes share one shell/furniture family and one 480×360 camera model
(73.74° horizontal FOV). This does not test connected rooms, unseen lenses, ARKit
drift, distortion, rolling shutter, moving furniture/people, or real iPhone-to-webcam
capture. No HTTP/authentication/persistence flow, Apple USDZ rendering or MPS/CUDA
inference was claimed. New results are a byte-matched comparison on these new
fixtures, not a replay of the historical one-room benchmark.
