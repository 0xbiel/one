#!/bin/zsh

# Download the OpenCV Zoo YuNet/SFace models into a private local directory.
# Model weights are intentionally not committed to this repository.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROOT_DIR="${SCRIPT_DIR:h}"
MODEL_DIR="${ONE_FACE_MODELS_PATH:-$ROOT_DIR/data/face-models}"

mkdir -p "$MODEL_DIR"

download() {
  local destination="$1"
  local url="$2"
  if [[ -s "$destination" ]]; then
    print -- "Already present: $destination"
    return
  fi
  local temporary="${destination}.part"
  rm -f "$temporary"
  print -- "Downloading $(basename "$destination")"
  curl --fail --location --retry 3 --output "$temporary" "$url"
  [[ -s "$temporary" ]] || { print -u2 -- "Downloaded file is empty: $destination"; exit 1; }
  mv "$temporary" "$destination"
}

download \
  "$MODEL_DIR/face_detection_yunet_2023mar.onnx" \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
download \
  "$MODEL_DIR/face_recognition_sface_2021dec.onnx" \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

print -- "Face models are ready in $MODEL_DIR"
