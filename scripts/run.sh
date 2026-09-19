#!/usr/bin/env bash
# YOLO11 on JetPack 7.2 - 빌드 & 실행 헬퍼
set -euo pipefail

# 레포지토리 루트 (scripts/ 의 상위)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE=${IMAGE:-yolo11-jp72:latest}
CAM=${CAM:-/dev/video0}
PORT=${PORT:-8080}

usage() {
  cat <<'EOF'
사용법:
  ./run.sh build              # 이미지 빌드 (TensorRT 포함)
  ./run.sh build-slim         # 이미지 빌드 (PyTorch만, 가볍고 빠름)
  ./run.sh check              # 컨테이너 안에서 GPU/카메라 확인
  ./run.sh stream             # USB 카메라 추론 → http://<jetson-ip>:8080
  ./run.sh view               # USB 카메라 추론 → X11 창
  ./run.sh engine             # TensorRT 엔진 변환 후 스트림 추론
  ./run.sh shell              # 컨테이너 셸 진입
환경변수: IMAGE, CAM(기본 /dev/video0), PORT(기본 8080)
EOF
}

common_opts=(
  --rm -it
  --runtime nvidia
  --ipc=host
  --network host
  --device "${CAM}"
  -v "$(pwd)/models:/workspace/models"
  -v "$(pwd)/outputs:/workspace/outputs"
)

case "${1:-}" in
  build)       docker build --build-arg USE_TENSORRT=1 -t "${IMAGE}" "${ROOT}" ;;
  build-slim)  docker build --build-arg USE_TENSORRT=0 -t "${IMAGE}" "${ROOT}" ;;
  check)
    docker run "${common_opts[@]}" "${IMAGE}" bash -lc '
      python3 -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")";
      v4l2-ctl --list-devices || true;
      v4l2-ctl -d '"${CAM}"' --list-formats-ext | head -40 || true' ;;
  stream)
    docker run "${common_opts[@]}" "${IMAGE}" \
      python3 /workspace/usbcam_infer.py --device "${CAM}" --stream "${PORT}" --mjpg --half ;;
  view)
    xhost +local:docker >/dev/null 2>&1 || true
    docker run "${common_opts[@]}" \
      -e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix \
      "${IMAGE}" python3 /workspace/usbcam_infer.py --device "${CAM}" --view --mjpg --half ;;
  engine)
    docker run "${common_opts[@]}" "${IMAGE}" \
      python3 /workspace/usbcam_infer.py --device "${CAM}" --engine --stream "${PORT}" --mjpg ;;
  shell)       docker run "${common_opts[@]}" "${IMAGE}" bash ;;
  *)           usage ;;
esac
