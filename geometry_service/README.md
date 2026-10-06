# ONE real local geometry service

This is the host-side service used by the camera sweep. It runs a real
pretrained YOLOv8-World v2 model locally with PyTorch/MPS and returns
camera-relative 2D geometry. The model vocabulary covers furniture and room
features such as beds, sofas, tables, chairs, doors, windows, floors, walls,
and ceilings.

The service also estimates the visible room envelope from Canny/Hough lines in
the same decoded frames. It never inserts example furniture, doors, windows,
or room rectangles. If the model, accelerator, or structural evidence is not
available, the service reports `503`/`422` and the backend keeps the previous
map.

The service does not create a 3D map. A 3D scene must come from a validated
native iPhone/iPad RoomPlan + LiDAR upload handled by the backend. The RGB
service returns `camera-cv-2d` geometry and `metric_scale_known: false`.

For an existing RoomPlan scene, the worker also supports local visual
registration of a separate fixed camera. During the native scan, bounded RGB +
LiDAR depth samples are converted into ORB plus optional scale-robust SIFT
descriptors tied to metric RoomPlan coordinates; the raw RGB/depth samples are
discarded. Later, fixed camera JPEGs are matched to that derived landmark
index. CPU OpenCV RANSAC/PnP generates robust pose hypotheses, while the
learned correspondence scorer and bounded nonlinear pose polish run on the
selected MPS/CUDA accelerator before a result can pass the strict multi-view
gate.

Before ORB/SIFT extraction, genuinely underexposed frames receive a bounded
shadow lift and local-contrast pass; normal frames are left unchanged. The
same prepared view is used for local object detection, and localization
diagnostics report the luminance statistics and gamma used per frame. This is
an illumination aid, not synthetic evidence: the original frame is never
persisted and geometric reprojection and multi-view agreement still decide
whether a pose is publishable.

## Install the real runtime

From the `one` repository:

```bash
python3 -m venv .geometry-venv
.geometry-venv/bin/python -m pip install -r geometry_service/requirements.txt
```

Download the real YOLOv8-World v2 weights into a private local model
directory. The file is intentionally not committed to Git:

```bash
mkdir -p "$HOME/.cache/one-geometry"
cd "$HOME/.cache/one-geometry"
"/path/to/one/.geometry-venv/bin/python" - <<'PY'
from ultralytics import YOLOWorld

# Ultralytics downloads the official pretrained checkpoint when it is absent.
YOLOWorld("yolov8s-worldv2.pt", verbose=False)
PY
```

Start the service with an explicit checkpoint and the checked-in real-model
configuration:

```bash
cd "/path/to/one"
ONE_GEOMETRY_MODE=model \
ONE_GEOMETRY_MODEL_PATH="$HOME/.cache/one-geometry/yolov8s-worldv2.pt" \
ONE_GEOMETRY_MODEL_CONFIG="$PWD/geometry_service/model_config.yolo-world.json" \
ONE_GEOMETRY_DEVICE=mps \
ONE_GEOMETRY_SOLVER_DEVICE=mps \
ONE_GEOMETRY_ALLOW_CPU=false \
ONE_POSITIONING_WORKERS=3 \
.geometry-venv/bin/python -m geometry_service
```

On macOS, the repository launcher performs this setup and verifies the full
Docker-to-host route for you:

```bash
./scripts/start_mac_gpu.sh
```

It starts or reuses the host worker with CPU fallback disabled, starts the
Compose services, and checks from inside the API container that
`host.docker.internal:8090` reports an active `mps` or `cuda` runtime. The
Compose `vision-worker` remains a CPU-only healthy dependency for the default
stack; camera localization is routed to the host worker and does not use that
container. If the host worker is unavailable or reports CPU, the launcher
stops with diagnostics instead of claiming that GPU solving is active.

`ONE_GEOMETRY_DEVICE=auto` selects MPS first and CUDA second; set
`ONE_GEOMETRY_SOLVER_DEVICE` explicitly when the pose solver must use a known
accelerator. CPU inference is disabled unless `ONE_GEOMETRY_ALLOW_CPU=1` is
explicitly set. The service
reports `mode: "model"`, `runtime.framework: "pytorch-ultralytics"`, the active
accelerator, and the real `model_version` from `/health`. A missing checkpoint,
missing dependency, unavailable accelerator, or invalid model configuration
returns `503`; there is no silent alternate implementation.

The default Docker Compose stack runs this service as `vision-worker` and the
API reaches it through `http://vision-worker:8090`. The container defaults to
CPU because Docker Desktop on macOS cannot expose Metal/MPS to Linux
containers. The checkpoint is bind-mounted from `ONE_GEOMETRY_MODEL_PATH`, and
Compose waits for `/health` before starting the API. For faster local MPS
inference, you can still launch the worker on the host and override
`ONE_GEOMETRY_SERVICE_URL=http://host.docker.internal:8090`. The worker remains
private to the local stack; the phone does not connect to it directly.

`ONE_POSITIONING_WORKERS` controls the bounded thread pool used for fixed-camera
positioning and visual-landmark construction. It defaults to `3` and is clamped
to `1..8`. CPU-heavy ORB/SIFT extraction and OpenCV RANSAC/PnP/FOV hypothesis
generation run off the ASGI event loop, so independent cameras in the same room
or different homes/maps can make progress together. The map-specific matcher,
YOLO-World detector, and bounded differentiable finalist pose polish run on the
selected MPS/CUDA accelerator. The GPU polish is accepted only when it preserves
or improves positive-depth inliers and reprojection error; diagnostics expose
its device, baseline, and result. A shared accelerator lock prevents detector
and solver work from racing on MPS, while the CPU solve pool remains parallel.
The positioning queue admits at most twice the worker count. The single-worker
detector has its own bounded waiter queue, and `/health` reports both bounds.

The `POST /v1/visual-landmarks`, `POST /v1/room-layout`, and
`POST /v1/vision/detect` handlers also move their blocking work onto bounded
executors. A long scan or live detection therefore does not hold the ASGI event
loop hostage while another camera localization request is being submitted or
polled. GPU model calls are intentionally serialized; increasing
`ONE_POSITIONING_WORKERS` increases CPU overlap and request capacity, not the
number of simultaneous Metal command streams.

## Real model configuration

[`model_config.yolo-world.json`](./model_config.yolo-world.json) declares the
model backend, pretrained model name, input shape, normalization, confidence
thresholds, and the indoor vocabulary. The Torch tensor is `[batch, 3, height,
width]` with values in `0..1`, which matches the configured zero/one
normalization. `YOLOWorld.set_classes()` installs the vocabulary before
inference.

The output adapter converts detector boxes into normalized map items:

- furniture boxes become labeled `furniture` items with center, size, and
  confidence;
- door/window boxes become `openings` with a normalized segment;
- actual image edge lines become the room polygon and wall segments;
- metrics and diagnostics identify the detector backend, structure method,
  item counts, frame count, and active device.

The coordinates are image-space `0..1`, not meters. Use the caregiver map ruler
to apply a measured reference length, or use the native RoomPlan/LiDAR path for
metric 3D coordinates.

## HTTP contract

`GET /health` returns `200` only after PyTorch, the selected accelerator, the
real checkpoint, the model configuration, and the YOLO-World vocabulary have
loaded. `POST /v1/room-layout` accepts three to twenty bounded JPEG frames and
returns `room-layout-response.v1`.

The worker also exposes `POST /v1/vision/detect`, `POST /v1/visual-landmarks`,
and `POST /v1/camera-localization`. Vision uses the loaded YOLO-World model;
visual-landmark construction and robust hypothesis generation remain local
OpenCV work, while learned matching and pose polish use the same GPU runtime.
A weak feature match returns `needs_rescan` instead of fabricating a pose.

Camera localization never assumes a 60° lens. When browser intrinsics are not
known, the solver searches a bounded plausible horizontal FOV range and keeps
the selected value only when the geometric checks pass. People and movable
chairs are treated as transient occluders rather than stable scene anchors.
The active scene-reference flow captures three short fixed-camera rounds and
uses static RoomPlan/visual landmarks to propose a pose for review.

Raw JPEG bytes and derived tensors are held only for the request and are not
written to disk, returned, or emitted in logs. A low-confidence or structurally
unstable sweep is rejected; it must not replace the previous map.

## Local pose and safety runtime

Fall safety does not use the room detector's person rectangle as its final
signal. The worker loads the explicit `ONE_POSE_MODEL_PATH` COCO keypoint
checkpoint (the local setup uses `yolo11n-pose.pt` on MPS), attaches 17-joint
poses to person detections, and returns `pose_keypoints` plus a pose model
version in `/v1/vision/detect`. The API keeps a bounded per-camera pose
sequence and sends derived COCO-17 windows to `/v1/fall/score`. That endpoint
loads an evaluated ST-GCN plus causal-TCN artifact from
`ONE_FALL_SEQUENCE_MODEL_PATH`; the API then requires the learned probability,
an upright-to-low transition, sufficient visible joints, and persistence across
several frames before creating a review-only event.
After confirmation, a 15-second per-track refractory interval prevents a brief
pose reclassification from generating a duplicate alert for the same episode.
If the pose model is missing, the health response reports it as unavailable;
if the learned artifact is missing, `fall_sequence.status` reports it as
unavailable and production emits no heuristic-only fall event. The fall path
does not silently fall back to bounding-box aspect ratios.

Download the local pose checkpoint with:

```bash
./scripts/download_pose_models.sh
```

This remains a safety review signal, not a medical diagnosis, and should be
calibrated against representative fall and non-fall sequences before clinical
or emergency use. The local artifact is trained by
`scripts/train_fall_sequence_model.py`; its ignored evaluation receipt is
`data/fall-models/fall_sequence_stgcn_tcn.pt.evaluation.json`.

The design follows the evidence-backed two-stage pattern used in skeleton
action recognition: estimate joints first, then model a time window of the
tracked skeleton. ST-GCN learns spatial joint relationships, while the causal
TCN models temporal motion without using future frames. The checked-in runtime
uses the same architecture as the locally trained artifact. On the current
18-fall/18-ADL URFD subset, the grouped eight-sequence holdout is 0.75 balanced
accuracy (3 TP, 3 TN, 1 FP, 1 FN); this is a small research regression result,
not clinical validation. The receipt also records the exact groups, threshold,
device, and limitations.

For face presentation-attack detection, the passive MiniFAS score is paired
with the guided yaw challenge and a narrow face-crop color-diversity quality
gate. The local receipt from `scripts/evaluate_biometrics.py` reports APCER,
BPCER, and ACER over 20 bona-fide camera frames plus 60 in-memory synthetic
screen/print proxies; after the quality gate, this run is APCER 0.00, BPCER
0.00, ACER 0.00. Those proxies are regression checks, not an ISO/IEC
30107-3 certification: performance can shift with lighting, screens, printed
photos, masks, optics, and demographics. The encrypted, keyed projection
protects and revokes templates, but the report's synthetic embedding test is
not a claim of cross-demographic biometric accuracy or irreversible security
under every attack model.

Research references: [ST-GCN](https://arxiv.org/abs/1801.07455), [a fall
detection review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7866979/), [the UR
Fall dataset](https://www.fenix.ur.edu.pl/~mkepski/ds/uf.html), [the face
anti-spoofing survey](https://arxiv.org/abs/2106.14948), [MiniFAS ONNX
implementation](https://github.com/facenox/face-antispoof-onnx), [NIST
biometric guidance](https://pages.nist.gov/800-63-3/sp800-63b.html), and
[cancelable face-template research](https://arxiv.org/abs/2404.16255).

## Local face identity and anti-spoof runtime

Face identity is a separate, consent-gated capability from anonymous person
detection. It uses OpenCV Zoo YuNet for face detection and SFace for derived
embeddings. The worker processes JPEGs in memory and never writes raw
enrollment photos, aligned crops, or live frames. Before encryption, the API
applies a profile-specific, keyed, dimension-reducing cancelable projection
with a fresh enrollment salt. Re-enrollment rotates the salt and invalidates
the previous transform. This is a practical cancelable-template layer, not a
claim of completed biometric certification or cross-demographic accuracy
evaluation.

Enrollment and live identity matching also require the local MiniFAS
anti-spoof model. Enrollment must pass both the real-face score and the
iPhone's ordered center/turn-one-side/turn-opposite-side challenge. A face
that fails liveness is never matched to a cared-for person.

Download the model files into the Compose bind mount with:

```bash
./scripts/download_face_models.sh
```

For a host-side worker, set the explicit face, liveness, and pose paths before
starting it:

```bash
ONE_FACE_DETECTOR_MODEL_PATH="$PWD/data/face-models/face_detection_yunet_2023mar.onnx" \
ONE_FACE_RECOGNIZER_MODEL_PATH="$PWD/data/face-models/face_recognition_sface_2021dec.onnx" \
ONE_FACE_LIVENESS_MODEL_PATH="$PWD/data/face-models/face_antispoof_minifas_v2se.onnx" \
ONE_FACE_LIVENESS_MIN_SATURATION=0.04 \
ONE_POSE_MODEL_PATH="$PWD/data/pose-models/yolo11n-pose.pt" \
ONE_FALL_SEQUENCE_MODEL_PATH="$PWD/data/fall-models/fall_sequence_stgcn_tcn.pt" \
ONE_GEOMETRY_MODEL_PATH="/path/to/yolov8s-worldv2.pt" \
ONE_GEOMETRY_MODEL_CONFIG="$PWD/geometry_service/model_config.yolo-world.json" \
.geometry-venv/bin/python -m geometry_service
```

The `/health` response reports the face runtime separately under `face` and the
learned fall scorer under `fall_sequence`. A worker can therefore continue
serving anonymous presence while face identity remains `unavailable` until the
detector, recognizer, and liveness files are configured; in contrast, the API
will not emit a production fall signal while `fall_sequence` is unavailable.
