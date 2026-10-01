#!/bin/zsh

# Download the small COCO pose checkpoint used by the local GPU worker.
# Model weights are intentionally not committed to this repository.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROOT_DIR="${SCRIPT_DIR:h}"
MODEL_DIR="${ONE_POSE_MODELS_PATH:-$ROOT_DIR/data/pose-models}"
DESTINATION="$MODEL_DIR/yolo11n-pose.pt"

mkdir -p "$MODEL_DIR"
if [[ ! -s "$DESTINATION" ]]; then
  temporary="${DESTINATION}.part"
  rm -f "$temporary"
  curl --fail --location --retry 3 --output "$temporary" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt"
  [[ -s "$temporary" ]] || { print -u2 -- "Downloaded pose checkpoint is empty"; exit 1; }
  mv "$temporary" "$DESTINATION"
else
  print -- "Already present: $DESTINATION"
fi

print -- "Pose model is ready at $DESTINATION"
