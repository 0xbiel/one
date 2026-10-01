# Fresh generation, inference and scoring

This recipe starts with the published source and regenerates the missing data. It runs rendering, feature construction, model inference and pose solving. It does not reproduce the original request bytes without the omitted payloads, and it must not be described as exact no-model replay.

## Prerequisites

- Blender 4.3.2 was used for the original RGB/depth generation
- A compatible Python environment with NumPy, OpenCV, PyTorch, Pydantic, SciPy, usd-core (including USD validation), Ultralytics, its CLIP support and Pillow
- The actual YOLO-World checkpoint and CLIP weights, obtained separately; original SHA256s are in `pilot/model_provenance.json`
- The original versions are recorded in `pilot/environment.lock.txt`; this records the measured Linux environment, not a guaranteed cross-platform installer

For the closest comparison, use application source at `9c05becdc3af59b15092075c1eb1dc90e2ab9e70`. Alternatively, explicitly record whichever application revision you want to evaluate. The QA scripts can be copied from the PR while the application source comes from a separate checkout of the measured revision.

## Prepare an isolated run

Set these paths to real local locations. `ONE_SOURCE_ROOT` must contain `app/` and `geometry_service/`; `QA_SOURCE` must be the published `qa/connected-home-localization` directory.

```sh
export ONE_SOURCE_ROOT=/absolute/path/to/evaluated-one-checkout
export QA_SOURCE=/absolute/path/to/qa/connected-home-localization
export RUN_ROOT=/absolute/path/to/new-connected-home-run
mkdir -p "$RUN_ROOT"
cp -R "$QA_SOURCE" "$RUN_ROOT/fixture"
cd "$RUN_ROOT/fixture"
export PYTHONPATH="$ONE_SOURCE_ROOT"
```

Run in this copy, not inside the published QA directory. The generation/preparation scripts overwrite local fixture images, design metadata and the request manifest in the copy; the committed reports remain unchanged.

## Generate all original roles, then detect

```sh
blender -b -t 2 --python pilot/build_connected.py
python pilot/package_connected.py
python audit/build_connected_native_fixture.py
```

This creates four connected furnished rooms, a hallway, 20 reference RGB/depth views and 12 query views; scan and scoring-only poses are recorded separately. The source generator creates its own detailed furniture and does not need the omitted reference images.

Configure actual checkpoint and runtime locations, using appropriate device/cache settings for your machine:

```sh
export ONE_GEOMETRY_MODEL_PATH=/absolute/path/to/yolov8s-worldv2.pt
export ONE_GEOMETRY_MODEL_CONFIG="$ONE_SOURCE_ROOT/geometry_service/model_config.yolo-world.json"
export ONE_GEOMETRY_DEVICE=cpu
export ONE_GEOMETRY_ALLOW_CPU=1
export YOLO_CONFIG_DIR=/absolute/path/to/writable/model-settings
export XDG_CACHE_HOME=/absolute/path/to/writable/model-cache
python pilot/detect.py
python pilot/prepare_connected.py
```

Detection invokes the real pretrained runtime; no ground-truth query boxes are supplied. Preparation derives landmarks only from reference poses/depth and uses the actual backend merge block. It supplies all room zones, with no query room, query pose, previous-pose prior or person anchor in the localization payloads. The `clean` condition supplies query intrinsics; `unknown_intrinsics` omits them.

## Run and score new localization outputs

```sh
cd pilot
python benchmark_serial.py --source "$ONE_SOURCE_ROOT" --label fresh
python summarize_connected.py --label fresh
```

Use a new label for every run. The driver refuses to reuse result paths. The fixed watchdog is 180 seconds per localization call after readiness. Failures remain in the denominator. `fresh_summary.json` contains the new five-zone confusion matrix, position/rotation errors and both threshold counts. The original `results_summary.json` remains the published historical result.

The threshold pairs are 10cm/2° and 25cm/5°. These are 24 condition cases from 12 distinct poses: four development cases from two poses and 20 held-out cases from ten poses. They are not 24 independent viewpoints or an unseen-house evaluation.

## Verification boundaries

`audit/audit_room_assignment.py --source "$ONE_SOURCE_ROOT"` can check contract and room-assignment semantics without the original omitted image payloads. It is not an image-localization accuracy test.

`audit/independent_final_validation.py` is a verifier of the original frozen benchmark and its original claims, not a generic pass/fail gate for a new run. It requires the original omitted requests/results and measured source. Do not run it against fresh outputs and expect the historical hashes or metrics to match.

The original no-model reconstruction helper, `restore_replay.py`, also requires the omitted payloads. Its presence in the repository does not make exact replay available. Use the omitted-payload inventory to identify missing files if those bytes are supplied later.
