# ONE local geometry service

This directory is a standalone, host-side service for deriving an approximate
camera-relative 2D room layout from a short RGB sweep. It is deliberately
separate from the backend so the backend can call it at:

```text
http://127.0.0.1:8090/v1/room-layout
```

The service never creates a 3D map. A 3D scene must come from a validated native
iPhone/iPad RoomPlan + LiDAR upload handled by the backend. This service only
returns `camera-cv-2d` geometry with `metric_scale_known: false`.

## Run on Apple Silicon

Create an environment on the Mac host and install the service dependencies:

```bash
cd "/Users/biel/Documents/UNI/2026-2027/Hackathon Detect 2.0/one"
python3 -m venv .venv
.venv/bin/python -m pip install -r geometry_service/requirements.txt
```

Model mode requires both an explicit TorchScript checkpoint and an explicit JSON
configuration. No checkpoint is bundled in this repository. The model config
must follow [`model_config.example.json`](./model_config.example.json), including
`output_contract: "camera-room-2d/v1"`.

```bash
export ONE_GEOMETRY_MODEL_PATH="/absolute/path/to/room_layout.ts"
export ONE_GEOMETRY_MODEL_CONFIG="/absolute/path/to/room_layout.json"
export ONE_GEOMETRY_DEVICE=auto
export ONE_GEOMETRY_MODE=model
export ONE_GEOMETRY_PORT=8090
.venv/bin/python -m geometry_service
```

`ONE_GEOMETRY_DEVICE=auto` selects PyTorch MPS first, then CUDA, and only uses
CPU when `ONE_GEOMETRY_ALLOW_CPU=1` is explicitly set. `mps`, `cuda`, and `cpu`
can be selected explicitly. The service binds to `0.0.0.0:8090` by default;
override the bind address with `ONE_GEOMETRY_HOST`.

If PyTorch, Pillow, NumPy, the accelerator, checkpoint, or config is missing,
the process still starts but reports `503 Service Unavailable` from `/health`
and from `/v1/room-layout`. The response includes a safe reason and never
pretends that mock or fallback geometry is model output.

## Model output contract

The TorchScript module receives a float tensor shaped `[batch, 3, height, width]`
using the dimensions and normalization in the config. It must return a mapping
with this shape (the `geometry` wrapper is also accepted):

```json
{
  "polygons": [
    {
      "id": "room-1",
      "label": "Living room",
      "points": [[0.08, 0.10], [0.92, 0.10], [0.92, 0.88], [0.08, 0.88]],
      "confidence": 0.84
    }
  ],
  "walls": [
    {
      "id": "wall-1",
      "start": [0.08, 0.10],
      "end": [0.92, 0.10],
      "confidence": 0.81
    }
  ],
  "camera_pose": {
    "position": {"x": 0.5, "y": 0.5, "z": 0.0},
    "rotation_degrees": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "confidence": 0.79
  },
  "intrinsics": {},
  "metrics": {
    "confidence": 0.82,
    "reprojection_error_px": 2.1,
    "homography_inlier_ratio": 0.86
  },
  "diagnostics": {
    "motion_stability": 0.9,
    "reprojection_error_px": 2.1
  }
}
```

Polygon and wall points are normalized `0..1` coordinates in the backend's
`camera-relative-image` frame. They are not meters. The service rejects missing,
non-finite, out-of-range, or incomplete geometry rather than clamping it into
a misleading map. A result below `minimum_confidence` returns
`422` with `status: "needs_rescan"`; no map should be persisted by the caller.

## HTTP contract

`GET /health` returns `200` only when the selected runtime is ready. An absent
model, missing dependency, or unavailable required accelerator returns `503` and
`status: "unavailable"`.

`POST /v1/room-layout` accepts a bounded sweep:

```json
{
  "schema_version": "room-layout-request.v1",
  "camera_id": "camera-123",
  "resolution": {"width": 1280, "height": 720},
  "output": {
    "dimension": "2d",
    "coordinate_frame": "camera-relative-image",
    "require_gpu": true
  },
  "frames": [
    {
      "frame_base64": "<base64 JPEG>",
      "width": 1280,
      "height": 720,
      "captured_at": "2026-09-13T12:00:00Z"
    }
  ],
  "orientation": "landscape-right",
  "room_label": "Living room"
}
```

Each frame uses the backend field name `frame_base64`; RGB/image aliases are
intentionally rejected so an adapter mismatch is visible. At most 20 frames
may be submitted (16 is the normal backend sweep size), each decoded JPEG is at
most 3 MB, and the decoded batch is at most 18 MB. Only JPEG bytes and derived
tensors exist in process memory during the request. They are not written to
disk, emitted in logs, or returned to the caller.

A successful response has the following top-level provenance:

```json
{
  "schema_version": "room-layout-response.v1",
  "status": "ready",
  "source": "camera-cv-2d",
  "dimension": "2d",
  "metric_scale_known": false,
  "model_version": "room-layout-example-v1",
  "geometry": {
    "coordinate_frame": "camera-relative-image",
    "polygons": [
      {
        "id": "room-1",
        "label": "Living room",
        "points": [
          {"x": 0.08, "y": 0.10},
          {"x": 0.92, "y": 0.10},
          {"x": 0.92, "y": 0.88},
          {"x": 0.08, "y": 0.88}
        ],
        "confidence": 0.84
      }
    ],
    "walls": [
      {
        "id": "wall-1",
        "start": {"x": 0.08, "y": 0.10},
        "end": {"x": 0.92, "y": 0.10},
        "confidence": 0.81
      }
    ],
    "camera_pose": {
      "coordinate_frame": "camera-relative",
      "position": {"x": 0.5, "y": 0.5, "z": 0.0},
      "rotation_degrees": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
      "confidence": 0.79
    },
    "intrinsics": {},
    "metrics": {
      "confidence": 0.82,
      "reprojection_error_px": 4.0,
      "homography_inlier_ratio": 0.86
    }
  },
  "confidence": 0.82,
  "diagnostics": {"raw_frames_persisted": false}
}
```

The abbreviated arrays above are illustrative; a real `ready` response always
contains at least one validated polygon room and wall and a complete camera
pose. `502` means the model returned an invalid output, `500` means inference
failed, and `422` means the sweep was rejected or needs a rescan.

## Dependency-free contract fixture

For local HTTP wiring checks only, opt into the deterministic fixture explicitly:

```bash
export ONE_GEOMETRY_MODE=mock
.venv/bin/python -m geometry_service
```

Mock mode uses the same request and response schema, hashes the in-memory JPEG
bytes to keep results repeatable, and labels the health response as
`mode: "mock"`. It is never selected automatically when model mode is
unavailable. The pure-Python helper lives in [`mock.py`](./mock.py).
