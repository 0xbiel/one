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
LiDAR depth samples are converted into ORB descriptors tied to metric
RoomPlan coordinates; the raw RGB/depth samples are discarded. Later, fixed
camera JPEGs are matched to that derived landmark index and OpenCV
`solvePnPRansac` estimates the camera's 6-DoF pose in `roomplan-local`.

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
ONE_GEOMETRY_DEVICE=auto \
ONE_GEOMETRY_ALLOW_CPU=false \
.geometry-venv/bin/python -m geometry_service
```

`ONE_GEOMETRY_DEVICE=auto` selects MPS first and CUDA second. CPU inference is
disabled unless `ONE_GEOMETRY_ALLOW_CPU=1` is explicitly set. The service
reports `mode: "model"`, `runtime.framework: "pytorch-ultralytics"`, the active
accelerator, and the real `model_version` from `/health`. A missing checkpoint,
missing dependency, unavailable accelerator, or invalid model configuration
returns `503`; there is no silent alternate implementation.

The Docker API reaches the host worker through
`ONE_GEOMETRY_SERVICE_URL=http://host.docker.internal:8090`. The worker should
remain private to the host; the phone does not connect to it directly.

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
visual-landmark construction and camera localization use OpenCV ORB/PnP and do
not require cloud inference. A weak feature match returns `needs_rescan`
instead of fabricating a pose.

Raw JPEG bytes and derived tensors are held only for the request and are not
written to disk, returned, or emitted in logs. A low-confidence or structurally
unstable sweep is rejected; it must not replace the previous map.
