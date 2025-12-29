FROM mcr.microsoft.com/devcontainers/python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# System deps + Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl ca-certificates \
    libgl1 libglib2.0-0 \
    tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Detectron2 from source (CPU build)
RUN pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

# LayoutParser + OCR helper
RUN pip install -U "pillow<10" \
 && pip install layoutparser opencv-python-headless matplotlib pytesseract

RUN pip install -U google-genai numpy


# ---- Bake PubLayNet model files into the image (NO Dropbox, NO ?dl=1) ----
ENV LP_MODEL_DIR=/opt/layoutparser_models/PubLayNet/faster_rcnn_R_50_FPN_3x
RUN mkdir -p ${LP_MODEL_DIR} \
 && curl -L --fail -o ${LP_MODEL_DIR}/config.yml \
    https://huggingface.co/nlpconnect/PubLayNet-faster_rcnn_R_50_FPN_3x/resolve/main/config.yml \
 && curl -L --fail -o ${LP_MODEL_DIR}/model_final.pth \
    https://huggingface.co/nlpconnect/PubLayNet-faster_rcnn_R_50_FPN_3x/resolve/main/model_final.pth \
 && test -s ${LP_MODEL_DIR}/config.yml \
 && test -s ${LP_MODEL_DIR}/model_final.pth \
 && chmod -R a+rX /opt/layoutparser_models

WORKDIR /workspaces
