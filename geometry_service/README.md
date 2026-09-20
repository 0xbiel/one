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
positioning. It defaults to `3` and is clamped to `1..8`. CPU-heavy ORB/SIFT
extraction and OpenCV RANSAC/PnP/FOV hypothesis generation run off the ASGI
event loop, while the map-specific matcher, YOLO-World detector, and bounded
differentiable finalist pose polish run on the selected MPS/CUDA accelerator.
The GPU polish is accepted only when it preserves or improves positive-depth
inliers and reprojection error; diagnostics expose its device, baseline, and
result. A shared accelerator lock prevents detector and solver work from racing
on MPS. The queue admits at most twice the worker count, and `/health` reports
the active worker count and queue bound.

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
