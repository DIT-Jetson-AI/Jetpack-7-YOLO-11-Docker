# ==========================================================
#  YOLO11 + USB Camera inference on Jetson Orin Nano
#  JetPack 7.2 (L4T r39.2 / Ubuntu 24.04 / CUDA 13.2 / TRT 10.16 / Python 3.12)
# ==========================================================
# 핵심 원칙: 호스트 드라이버가 CUDA 13.2 이므로
#            컨테이너 내부도 반드시 CUDA 13.x 로 맞춘다.
#            (CUDA 12.x 컨테이너 → CUDA error 801 "operation not supported")
# ----------------------------------------------------------
FROM nvcr.io/nvidia/base/ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    YOLO_CONFIG_DIR=/workspace/.yolo \
    OPENCV_VIDEOIO_PRIORITY_V4L2=1

# ---------- 1) 기본 OS 패키지 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git wget curl ca-certificates gnupg \
        v4l-utils libgl1 libglib2.0-0 \
        libsm6 libxext6 libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------- 2) (선택) NVIDIA Jetson APT 저장소 = TensorRT / cuDNN ----------
# TensorRT 엔진(.engine) 변환까지 컨테이너 안에서 하려면 USE_TENSORRT=1 로 빌드.
# PyTorch(CUDA) 추론만 할 거면 0 으로 두면 이미지가 훨씬 가볍고 빌드가 빠름.
ARG USE_TENSORRT=1
ARG L4T_REPO=r39.2
RUN if [ "$USE_TENSORRT" = "1" ]; then \
      echo "deb https://repo.download.nvidia.com/jetson/common ${L4T_REPO} main" \
          > /etc/apt/sources.list.d/nvidia-l4t.list && \
      echo "deb https://repo.download.nvidia.com/jetson/som ${L4T_REPO} main" \
          >> /etc/apt/sources.list.d/nvidia-l4t.list && \
      curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
          -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
          tensorrt libnvinfer-bin python3-libnvinfer && \
      rm -rf /var/lib/apt/lists/* ; \
    fi

# ---------- 3) PyTorch (CUDA 13.0 빌드, aarch64) ----------
# JetPack 7.2(CUDA 13)에서는 PyPI 기본 torch 가 아니라 cu130 인덱스를 써야 한다.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu130

# ---------- 4) Ultralytics (YOLO11) ----------
# torch 가 재설치되지 않도록 의존성 없이 설치 후 필요한 것만 채운다.
RUN python3 -m pip install "ultralytics>=8.3.0" --no-deps && \
    python3 -m pip install \
        opencv-python-headless numpy pillow pyyaml requests scipy \
        matplotlib pandas tqdm psutil py-cpuinfo ultralytics-thop polars

# ---------- 5) 작업 디렉터리 / 모델 ----------
WORKDIR /workspace
RUN mkdir -p /workspace/models /workspace/outputs /workspace/.yolo
# 빌드 시점에 가중치를 미리 받아두면 현장에서 오프라인 실행 가능
ARG PREFETCH_MODEL=yolo11n.pt
RUN wget -q -O /workspace/models/${PREFETCH_MODEL} \
      https://github.com/ultralytics/assets/releases/download/v8.3.0/${PREFETCH_MODEL} || \
    echo "model prefetch skipped (network unavailable)"

COPY src/usbcam_infer.py /workspace/usbcam_infer.py

ENV ULTRALYTICS_WEIGHTS=/workspace/models/yolo11n.pt
CMD ["python3", "/workspace/usbcam_infer.py", "--help"]
