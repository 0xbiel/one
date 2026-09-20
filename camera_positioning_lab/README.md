# Camera Positioning Lab

Local-only IRL harness for the real fixed-camera → RoomPlan localization path.
It is intentionally separate from `one-frontend` and runs in a localhost browser
window so macOS can grant camera access through `getUserMedia`.

## Start

```bash
./camera_positioning_lab/run_lab.sh
```

The launcher starts the host MPS geometry worker when needed, checks the Docker
API, starts the lab at `http://127.0.0.1:8765`, and opens it in a browser.

Sign in with a local development ONE email, choose the physical camera record,
open the Mac camera, then run a 6-frame localization. Every solve is sent with
`review_only=true`, so the lab can display candidate poses and diagnostics
without replacing the active saved registration.

Raw camera frames are captured in the browser and forwarded in memory to the
local API/geometry worker. The lab does not persist them.
