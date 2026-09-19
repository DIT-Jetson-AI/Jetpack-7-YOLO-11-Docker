# Jetpack-7-YOLO-11-Docker

**NVIDIA Jetson Orin Nano (JetPack 7.2) 에서 YOLO11 + USB 카메라 실시간 추론을 위한 Docker 환경**

[![JetPack](https://img.shields.io/badge/JetPack-7.2-76B900)](https://developer.nvidia.com/embedded/jetpack)
[![L4T](https://img.shields.io/badge/L4T-r39.2-76B900)](https://developer.nvidia.com/embedded/jetson-linux)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## 제1절. 개요

JetPack 7.2 로 올라오면서 Jetson 의 CUDA 가 **12.x → 13.2** 로 바뀌었습니다.
그 결과 기존에 널리 쓰이던 JetPack 6 용 컨테이너들이 **그대로는 동작하지 않습니다.**

> 콘센트 비유 — 호스트(벽)는 이제 220V(CUDA 13.2)인데,
> 기존 컨테이너(기기)는 110V(CUDA 12.x) 플러그입니다.
> 어댑터로 해결되는 문제가 아니라 기기 자체를 바꿔야 합니다.

| 항목 | 값 |
|---|---|
| JetPack | 7.2 (L4T **r39.2**) |
| OS | Ubuntu 24.04 / Python 3.12 |
| CUDA | **13.2** |
| cuDNN | 9.20 |
| TensorRT | 10.16.2 |
| 대상 보드 | Jetson Orin Nano (Super) 8GB — sm_87 |
| 모델 | Ultralytics YOLO11 (n/s/m/l/x) |

### 동작하지 않는 조합 (주의)

| 이미지 | 결과 |
|---|---|
| `dustynv/*` (JetPack 6 계열) | ❌ CUDA error 801 |
| `ultralytics/ultralytics:latest-jetson-jetpack6` | ❌ CUDA error 801 |
| `nvcr.io/nvidia/l4t-*:r36.x` | ❌ 드라이버 버전 불일치 |
| **본 레포지토리 (CUDA 13 베이스 직접 빌드)** | ✅ |

---

## 제2절. 레포지토리 구성

```
Jetpack-7-YOLO-11-Docker/
├── Dockerfile              # CUDA 13.2 베이스 + torch(cu130) + Ultralytics + TensorRT(선택)
├── src/
│   └── usbcam_infer.py     # USB 카메라 실시간 추론 (X11 / MJPEG 웹스트림 / mp4 저장)
├── scripts/
│   └── run.sh              # 빌드·실행 헬퍼
├── docs/
│   ├── setup.md            # 호스트 사전 준비 상세
│   └── troubleshooting.md  # 문제 해결 모음
├── .dockerignore
├── .gitignore
└── LICENSE
```

---

## 제3절. 호스트 사전 준비 (Jetson 에서 1회)

```bash
sudo apt-get update
sudo apt-get install -y docker.io nvidia-container-toolkit v4l-utils git

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER" && newgrp docker

# 최대 성능 모드
sudo nvpmodel -q          # 사용 가능한 모드 확인
sudo nvpmodel -m 2        # Orin Nano Super: MAXN_SUPER
sudo jetson_clocks

# USB 카메라 확인
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext    # MJPG 지원 여부 확인
```

자세한 내용은 [`docs/setup.md`](docs/setup.md) 참고.

---

## 제4절. 빌드

```bash
git clone https://github.com/DIT-Jetson-AI/Jetpack-7-YOLO-11-Docker.git
cd Jetpack-7-YOLO-11-Docker
chmod +x scripts/run.sh
mkdir -p models outputs

./scripts/run.sh build        # TensorRT 포함 (엔진 변환까지 컨테이너 내부에서)
# 또는
./scripts/run.sh build-slim   # PyTorch 만 — 빌드 빠르고 이미지 작음
```

| 빌드 인자 | 기본값 | 설명 |
|---|---|---|
| `USE_TENSORRT` | `1` | `0` 이면 Jetson APT 저장소/TensorRT 설치를 건너뜀 |
| `L4T_REPO` | `r39.2` | JetPack 버전이 다르면 이 값을 맞춰줄 것 |
| `PREFETCH_MODEL` | `yolo11n.pt` | 빌드 시 미리 받아둘 가중치 (오프라인 실행용) |

---

## 제5절. 동작 확인

```bash
./scripts/run.sh check
```

아래 두 가지가 확인되면 정상입니다.

```
torch 2.x.x cuda True Orin        ← GPU 인식
/dev/video0 ...                   ← 카메라 인식
```

---

## 제6절. 실행

```bash
# (A) 헤드리스 — 다른 PC 브라우저에서 http://<jetson-ip>:8080
./scripts/run.sh stream

# (B) Jetson 에 모니터가 연결된 경우 — X11 창
./scripts/run.sh view

# (C) TensorRT 엔진 변환 후 추론 (최초 1회 수 분 소요)
./scripts/run.sh engine

# (D) 컨테이너 셸 진입
./scripts/run.sh shell
```

docker 명령을 직접 쓰는 경우:

```bash
docker run --rm -it --runtime nvidia --ipc=host --network host \
  --device /dev/video0 \
  -v "$PWD/models:/workspace/models" \
  -v "$PWD/outputs:/workspace/outputs" \
  yolo11-jp72:latest \
  python3 /workspace/usbcam_infer.py --device 0 --stream 8080 --mjpg --half
```

### 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--model /workspace/models/yolo11s.pt` | 모델 교체 (커스텀 가중치 포함) |
| `--imgsz 640` | 입력 해상도. 320 으로 낮추면 FPS 크게 상승 |
| `--conf 0.25` | 신뢰도 임계값 |
| `--mjpg` | 카메라 MJPG 모드 강제 — 720p 이상에서 FPS 확보 |
| `--half` | FP16 추론 |
| `--engine` | TensorRT 엔진 자동 변환 후 사용 |
| `--stream 8080` | MJPEG 웹 스트림 |
| `--view` | X11 창 표시 |
| `--save /workspace/outputs/rec.mp4` | 결과 영상 저장 |

### 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `IMAGE` | `yolo11-jp72:latest` | 이미지 태그 |
| `CAM` | `/dev/video0` | 카메라 장치 |
| `PORT` | `8080` | 스트림 포트 |

---

## 제7절. 성능 참고치

Orin Nano 8GB / 640px / MAXN_SUPER 기준 대략치입니다.

| 실행 방식 | 예상 FPS |
|---|---|
| PyTorch FP32 | 15 ~ 25 |
| PyTorch FP16 (`--half`) | 25 ~ 35 |
| TensorRT FP16 (`--engine`) | 45 ~ 60 |

실측치는 전원 모드·카메라 포맷·모델 크기에 따라 달라집니다.
카메라가 YUYV 로만 동작하면 USB 대역폭 한계로 720p 에서 5~10 FPS 에 묶이므로 `--mjpg` 를 반드시 사용하세요.

---

## 제8절. 문제 해결

대표 증상만 옮깁니다. 전체는 [`docs/troubleshooting.md`](docs/troubleshooting.md) 참고.

| 증상 | 원인 / 조치 |
|---|---|
| `CUDA error 801 operation not supported` | 컨테이너가 CUDA 12.x (JetPack 6 용 이미지). 본 Dockerfile 로 재빌드 |
| `torch.cuda.is_available() == False` | `--runtime nvidia` 누락 또는 `nvidia-ctk runtime configure` 미실행 |
| 카메라를 열 수 없음 | `--device /dev/video0` 누락, 또는 장치 번호가 다름 |
| X11 창이 뜨지 않음 | 호스트에서 `xhost +local:docker`, `DISPLAY` 확인 |
| 빌드 중 Jetson APT 저장소 실패 | `--build-arg USE_TENSORRT=0` 으로 슬림 빌드 |
| 엔진 변환 중 OOM | swap 8~16GB 확대 후 재시도, 또는 `--imgsz 480` |

---

## 제9절. 주의사항

- **TensorRT 엔진(.engine) 은 장치·버전 종속적입니다.** 다른 Jetson 이나 다른 JetPack 버전으로
  파일을 그대로 옮기면 동작하지 않습니다. 배포 대상 보드마다 각각 변환하세요.
- `models/`, `outputs/` 디렉터리는 git 추적 대상에서 제외되어 있습니다.
- 본 구성은 **추론(inference)** 을 전제로 합니다. 학습은 Orin Nano 에서도 가능하나 실용적이지 않습니다.

---

## 제10절. 참고 자료

- [JetPack 7.2 Deep Dive — Seeed Studio Wiki](https://wiki.seeedstudio.com/jetpack72_deep_dive/)
- [Setting Up the Jetson Orin Nano Super Dev Kit on JetPack 7.2 — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/setting-up-the-nvidia-jetson-orin-nano-super-dev-kit-on-jetpack-7-2-a-practical-guide-june-2026/372490)
- [Docker image L4T JetPack 7.2.1 / L4T r39.2.1 — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/docker-image-l4t-jetpack-7-2-1-l4t-r39-2-1/380545)
- [NVIDIA Jetson 배포 가이드 — Ultralytics Docs](https://docs.ultralytics.com/guides/nvidia-jetson)
- [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)

---

## License

[MIT](LICENSE)
