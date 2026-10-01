# Docker deployment guide

This is the source-level guide for running the ONE backend stack. The public
documentation in one-docs links to this file for the exact Dockerfile and
dependency details.

## What is included

The root Dockerfile builds the FastAPI API with Python 3.12. The Compose stack
starts these services by default:

| Service | Purpose | Exposed port |
| --- | --- | --- |
| api | ONE HTTP API | 8000 |
| postgres | PostgreSQL application database | internal |
| redis | cache and coordination | internal |
| minio | S3-compatible object storage | internal |
| livekit | local development WebRTC server | 7880, 7881, 7882/udp |

The following services are optional Compose profiles:

| Profile | Service | Purpose |
| --- | --- | --- |
| vision | vision-worker | CPU-capable geometry and positioning worker |
| web | frontend and caddy | local web UI and HTTPS proxy |
| demo | demo-service | isolated demo data and sessions; requires the sibling one-demo-service checkout |

The vision worker is not needed to start the API. The API can report a
degraded geometry status until a worker or another configured geometry service
is available.

## Dockerfiles and runtime dependencies

### API image

The root Dockerfile copies pyproject.toml, the app package and the numbered
SQL files in migrations/, then installs the PostgreSQL extra:

~~~text
.[postgres]
~~~

The API runtime therefore includes FastAPI, Uvicorn, Pydantic Settings,
multipart form handling, cryptography and the psycopg PostgreSQL driver. The
development-only dependencies (pytest and httpx) are not installed in the
production image.

### Geometry worker image

geometry_service/Dockerfile is deliberately separate because it is much larger
than the API image. It installs:

- Python 3.12 and the geometry service requirements;
- CPU PyTorch and torchvision wheels;
- FastAPI, Uvicorn, NumPy, Pillow, Ultralytics and headless OpenCV;
- the CLIP implementation required by the YOLO-World model;
- the small system libraries required by OpenCV.

The worker expects a YOLO-World checkpoint and the face-model directory to be
mounted read-only. Docker Desktop Linux containers use CPU by default. The
host-side worker can still be used when a machine-specific GPU setup is
required, but that is a different deployment path.

## First startup

From the one-backend checkout, create the local environment file and replace
the example passwords and secrets before sharing the stack:

~~~powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up --build -d
docker compose ps
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing
~~~

The first build downloads the base images and Python packages. Subsequent
starts reuse the images and named volumes.

The API reads the numbered PostgreSQL migrations from migrations/ during
startup and records applied versions in schema_migrations. This includes
016_outside_location_tracking.sql, so the outside-location tables are created
by the same startup path as the rest of the schema.

The normal stack does not require a model checkpoint. To start the optional
vision worker, set these values in .env first:

~~~dotenv
ONE_GEOMETRY_MODEL_PATH=C:/absolute/path/to/yolov8s-worldv2.pt
ONE_FACE_MODELS_PATH=C:/absolute/path/to/face-models
~~~

Then run:

~~~powershell
docker compose --profile vision up --build -d
docker compose ps
~~~

The face-model directory should contain the configured YuNet and SFace model
files. The worker health endpoint is available inside the Compose network at
http://vision-worker:8090/health.

## Optional web and demo profiles

The web profile builds the sibling one-frontend checkout and starts Caddy:

~~~powershell
docker compose --profile web up --build -d
~~~

The demo profile is only usable when the sibling one-demo-service checkout
exists next to one-backend:

~~~powershell
docker compose --profile demo up --build -d
~~~

Profiles can be combined:

~~~powershell
docker compose --profile vision --profile web up --build -d
~~~

The API listens on 127.0.0.1:8000 and the frontend on 127.0.0.1:4175 by
default. Do not assume the entire stack is loopback-only: LiveKit publishes
its media ports on all host interfaces, and the current Caddy Compose service
also adds IPv6 wildcard (`[::]`) mappings even when its IPv4 bind is loopback.
The `.env` values configure Caddy's port numbers; `ONE_CADDY_BIND` does not
remove those IPv6 mappings. Restrict these ports with the host firewall on
untrusted networks, or update the Compose mappings deliberately before sharing
the machine. Caddy ports are controlled by ONE_CADDY_HTTPS_PORT and
ONE_CADDY_HTTP_PORT.

## Connecting an Android device

For an Android emulator, use 10.0.2.2 to reach the host:

~~~text
http://10.0.2.2:8000/api/v1
~~~

For a physical phone on the same Wi-Fi network, bind the API to the LAN and
use the computer's LAN address:

~~~dotenv
ONE_API_BIND=0.0.0.0
ONE_API_PORT=8000
ONE_LIVEKIT_URL=ws://192.168.1.132:7880
ONE_LIVEKIT_NODE_IP=192.168.1.132
~~~

Replace the example address with the computer's current IPv4 address. Allow
TCP 8000 and the LiveKit ports through the operating-system firewall only for
the trusted private network. The phone must not use 127.0.0.1, because that
address points back to the phone itself.

The Android app must use the same host address in its API base URL. The API
health check from the computer proves only that the local container is healthy;
the LAN check must also succeed from the phone's network.

## Inspecting, restarting and stopping

~~~powershell
docker compose ps
docker compose logs --tail=200 api
docker compose logs --tail=200 postgres
docker compose restart api
docker compose down
~~~

docker compose down removes containers but keeps the named database, object
storage and MinIO volumes. Do not add -v unless deleting local data is
intentional and backed up.

If Docker Desktop shows an old stopped project named one-backend, it may be
from an earlier Compose configuration. Run the commands above from this
checkout to recreate the services from the current files. Check the actual
state with docker compose ps rather than relying on a stale Docker Desktop
row.

## Common failures

- could not reach the ONE API: verify the API health request on the computer,
  then use the computer's LAN IP from a physical phone; do not use
  127.0.0.1 on the phone.
- Port 8000 already in use: change ONE_API_PORT or stop the process holding the
  port.
- Vision worker build or startup failure: leave the vision profile disabled
  while testing the core API, then verify the checkpoint and face-model paths.
- Database startup failure: inspect docker compose logs postgres and confirm
  that the named volume is not being shared by an incompatible local stack.
- LiveKit connects locally but not from a phone: set ONE_LIVEKIT_URL and
  ONE_LIVEKIT_NODE_IP to a host-reachable LAN address and check UDP 7882.

Never commit .env, model checkpoints, face models or generated data/ content.
The repository .gitignore is intended to keep those local assets out of Git.

## Outside location tracking update

After pulling `android-test`, rebuild the API with `docker compose up --build -d api`.
On startup, numbered migration `017_outside_tracking_reliability.sql` adds
shared Home zones, place revisions, street/dwell metadata and a history-clear
watermark to the existing PostgreSQL volume. Keep a database backup before
updating; do not run `docker compose down -v`. The API retains at most seven
days of points while running and deletes expired rows hourly and on history
reads. The Android tracking branch and this API update must be deployed
together; older API builds cannot serve the new pagination and zone contract.

Street search/reverse lookup is optional. Set `ONE_GEOCODER_BASE_URL` in `.env`
to the base URL of a Nominatim-compatible server under your control or an
approved provider; it must be reachable from the API container. No geocoding
service is bundled by this Compose file and the public OSM Nominatim endpoint
is not a supported default. With the variable unset or the service down,
tracking and history still work and Android shows a zone or coordinates. The
API Dockerfile needs no new dependency for this update.
