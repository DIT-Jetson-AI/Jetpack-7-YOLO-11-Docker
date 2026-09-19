#!/usr/bin/env bash
# YOLO11 on JetPack 7.2 — 빌드 & 실행 헬퍼
#
#   ./scripts/run.sh build     이미지 빌드 (TensorRT 포함)
#   ./scripts/run.sh check     GPU·카메라 인식 확인
#   ./scripts/run.sh image     샘플 이미지 1장 추론
#   ./scripts/run.sh batch     images/ 폴더 일괄 추론
#   ./scripts/run.sh camera    USB 카메라 실시간 추론
#   ./scripts/run.sh shell     컨테이너 셸 진입
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # 레포 루트로 이동

IMAGE=${IMAGE:-yolo11-jp72:latest}       # 이미지 태그
CAM=${CAM:-/dev/video0}                  # 카메라 장치

mkdir -p images outputs

# 모든 실행에 공통으로 붙는 docker 옵션
#   --runtime nvidia : GPU 사용 (이게 빠지면 CPU로 돈다)
#   -v ...:/workspace/... : 호스트 폴더를 컨테이너 안으로 연결
DOCKER_OPTS=(
  --rm -it
  --runtime nvidia
  --ipc=host
  -v "$PWD/images:/workspace/images"
  -v "$PWD/outputs:/workspace/outputs"
  -w /workspace/outputs                  # 결과물이 outputs/ 에 쌓이도록
)

case "${1:-}" in
  build)
    docker build -t "${IMAGE}" .
    ;;
  build-slim)   # TensorRT 없이 빠르게 빌드 (PyTorch 추론만)
    docker build --build-arg USE_TENSORRT=0 -t "${IMAGE}" .
    ;;
  check)
    docker run "${DOCKER_OPTS[@]}" --device "${CAM}" "${IMAGE}" bash -lc '
      python3 -c "import torch; print(\"GPU:\", torch.cuda.is_available())"
      v4l2-ctl --list-devices || true'
    ;;
  image)
    docker run "${DOCKER_OPTS[@]}" "${IMAGE}" \
      python3 /workspace/predict_image.py "${2:-}"
    ;;
  batch)
    docker run "${DOCKER_OPTS[@]}" "${IMAGE}" \
      python3 /workspace/predict_batch.py "${2:-/workspace/images}"
    ;;
  camera)
    xhost +local:docker >/dev/null 2>&1 || true   # 컨테이너가 화면에 창을 띄울 수 있게
    docker run "${DOCKER_OPTS[@]}" \
      --device "${CAM}" \
      -e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix \
      "${IMAGE}" python3 /workspace/usbcam_infer.py
    ;;
  shell)
    docker run "${DOCKER_OPTS[@]}" --device "${CAM}" "${IMAGE}" bash
    ;;
  *)
    sed -n '2,9p' "$0"   # 위 주석을 사용법으로 출력
    ;;
esac
