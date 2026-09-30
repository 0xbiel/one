# ONE inference runtime inspection
Inspection date: 2026-09-30. Backend source commit: 9c05becdc3af59b15092075c1eb1dc90e2ab9e70. Frontend: 84e21f90f53213c79775168ee0491019ad1f1400.

## Verified setup
- Python environment: /workspace/shared/real_fall_eval/.venv
- Exact package snapshot: runtime-requirements.lock.txt
- Torch 2.14.0+cpu, torchvision 0.29.0+cpu, Ultralytics 8.4.168, NumPy 2.5.2, Pillow 11.3.0, OpenCV 4.14.0
- Repository requires torch>=2.4,<3 and ultralytics>=8.4,<9; exact runtime used in user's deployment is not pinned/verified.
- pip check passed. Initial incompatible torchvision binary was resolved by reinstalling the matching official CPU wheel. No repository source modified.
- No private data transmitted to inference providers; local PyTorch execution only. Ultralytics settings sync=false before any image-model execution.
- Recovered geometry runtime imports only config, gpu_runtime and low_light. detect_jpeg dynamically imports real_vision. All those exact source files plus JSON config were materialized.
- Runtime health reports ready; YOLOWorld detect_jpeg and YOLO11 pose predict ran successfully on uniform-gray smoke input. Empty detections expected. This is execution validation, not evidence of fall accuracy.

## Invocation
Set before any ultralytics import:
  YOLO_CONFIG_DIR=/workspace/shared/real_fall_eval/models
  XDG_CACHE_HOME=/workspace/shared/real_fall_eval/cache
  PYTHONPATH=/workspace/shared/real_fall_eval/one
  ONE_GEOMETRY_MODEL_PATH=/workspace/shared/real_fall_eval/models/yolov8s-worldv2.pt
  ONE_GEOMETRY_DEVICE=cpu
  ONE_GEOMETRY_ALLOW_CPU=1
Use /workspace/shared/real_fall_eval/.venv/bin/python.
In Python set ultralytics.utils.torch_utils.NUM_THREADS=2; torch.set_num_threads(2); torch.set_num_interop_threads(1).
Then RoomLayoutRuntime(ServiceSettings.from_env()). Configuration defaults to one/geometry_service/model_config.yolo-world.json.

## Weights provenance
Official Ultralytics asset downloader, v8.4.0:
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt
  models/yolov8s-worldv2.pt (25,923,032 bytes)
  SHA256 9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt
  models/yolo11n-pose.pt (6,255,593 bytes)
  SHA256 869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0
CLIP text encoder:
  official Ultralytics CLIP fork https://github.com/ultralytics/CLIP.git at a13192f8cb767260d7dfd98c843b0716593169e7
  cached pretrained OpenAI ViT-B/32 models/clip/ViT-B-32.pt (353,976,522 bytes)
  SHA256 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
CLIP uses Ultralytics SETTINGS.weights_dir/clip, not XDG_CACHE_HOME, so preserve the YOLO_CONFIG_DIR settings to avoid redundant downloads.

## Actual detection and temporal contract
- World checkpoint: yolov8s-worldv2.pt; model version yolov8s-worldv2-indoor-objects-v1
- Model input imgsz=(384,640), minimum detection confidence0.20, max_det100
- JPEG decoded BGR. Exact low_light.py preprocessor runs before inference; normal frames unchanged, dark frames gamma/CLAHE corrected.
- Each request sets requested vocabulary, predicts, restores indoor-map vocabulary.
- main.py vision_candidate_labels: no requested labels -> person plus unique enabled tracked-object labels (up to20); person-only expands to person, keys, glasses, mobile phone, remote control, cup, bottle, book, medication box, cane, walker. This 11-label configuration is the empty-home default; populated homes are different.
- TemporalStabilityTracker: min_hits3, IoU>=0.20, <=4s association window. Greedy matching; one match per track per frame. Its cleanup uses wall clock, so offline replays must control app.vision.datetime or use a faithfully offset replay clock.
- FallDetectionTracker: person only, confidence>=0.55, stable positive track ID, both bbox width/height>=8% of frame, upright aspect(height/width)>=1.20, low<=0.95, center_y drop>=0.07, 3 low confirmations in<=4s, stale_after12s, one event until upright reset.
- Contrary to docstring simplification, FallDetectionTracker can establish a candidate using previous last_center_y when no upright reference was ever observed. Preserve source for baseline; do not assume upright prerequisite beyond actual code.
- Main receives actual frontend captured_at and calls fall update only on stabilized person output.

## Frontend-equivalent input
src/camera/CameraSetupCard.tsx: immediate tick then every1400ms; skips if previous request running; captures maximum dimension640, browser canvas JPEG quality0.58.
src/camera/roomSweep.ts: scale=min(1,640/max(width,height)), rounded dimensions; preserve captured_at.
src/api/client.ts: submitVisionFrame sends candidate_labels:[].
A 5Hz replay is a diagnostic higher-frequency input, not current frontend cadence. 1.4s replay is idealized frontend cadence without variable request overruns; codec differences remain between OpenCV JPEG quality58 and browser canvas0.58.

## Performance caveat
Four repeated blank-frame current-runtime calls:2.18,2.15,2.03,2.01 seconds CPU; pose smoke0.27s. This is not a representative deployment benchmark. Runtime calls set_classes twice each frame; current Ultralytics version recomputes text embeddings and clears predictor every call. Exact-vocabulary embedding caching can reduce harness cost if equivalence is verified; do not report cached harness timing as production throughput.

## Keypoint diagnostic
YOLO11n-pose is an independent pretrained person/17-keypoint model, with no claim of fall-specific training or accuracy. Compare detector boxes separately from temporal keypoint features; replacing the detector and classifier together otherwise confounds diagnosis.
COCO order:0nose,1left_eye,2right_eye,3left_ear,4right_ear,5left_shoulder,6right_shoulder,7left_elbow,8right_elbow,9left_wrist,10right_wrist,11left_hip,12right_hip,13left_knee,14right_knee,15left_ankle,16right_ankle.
Sources: https://docs.ultralytics.com/models/yolo11 ; https://docs.ultralytics.com/datasets/pose/coco ; https://docs.ultralytics.com/models/yolo-world

