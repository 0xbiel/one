FROM python:3.12-slim

WORKDIR /app

COPY geometry_service/requirements.txt ./geometry_service/requirements.txt
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.14.0+cpu" "torchvision==0.29.0+cpu" \
    && python -m pip install --no-cache-dir -r geometry_service/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# YOLO-World loads Ultralytics' CLIP fork when custom class prompts are set.
# Pin the same revision used by the working host geometry environment so the
# container never performs a runtime package install or depends on mutable HEAD.
RUN python -m pip install --no-cache-dir \
    "https://github.com/ultralytics/CLIP/archive/a13192f8cb767260d7dfd98c843b0716593169e7.zip"

# YOLO-World's prompt encoder otherwise downloads ViT-B/32 the first time the
# worker starts. Bake the pinned runtime weight into the image so container
# startup stays local and deterministic after the image has been built.
RUN python -c "import torch; from ultralytics.nn.text_model import build_text_model; build_text_model('clip:ViT-B/32', device=torch.device('cpu'))"

COPY geometry_service ./geometry_service

EXPOSE 8090

CMD ["python", "-m", "geometry_service"]
