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
│   ├── predict_image.py    # 이미지 1장 추론        (35줄)
│   ├── predict_batch.py    # 폴더·동영상 일괄 추론  (55줄)
│   └── usbcam_infer.py     # USB 카메라 실시간 추론 (66줄)
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
mkdir -p images outputs

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
GPU: True          ← GPU 인식 (False면 8절 참고)
/dev/video0 ...    ← 카메라 인식
```

---

## 제6절. 실행

세 가지 스크립트가 전부이고, 각각 한 가지 일만 합니다.

| 스크립트 | 하는 일 |
|---|---|
| `predict_image.py` | 이미지 1장 — 동작 확인용 |
| `predict_batch.py` | 폴더·동영상 일괄 — 결과를 `results.csv` 로 |
| `usbcam_infer.py` | USB 카메라 실시간 — 화면 표시 + mp4 저장 |

헬퍼 스크립트로 실행하는 것이 가장 간단합니다.

```bash
./scripts/run.sh image      # 샘플 이미지 1장 (인터넷에서 자동으로 받아옴)
./scripts/run.sh batch      # images/ 폴더 전부
./scripts/run.sh camera     # USB 카메라
./scripts/run.sh shell      # 컨테이너 셸
```

`images/` 와 `outputs/` 폴더가 컨테이너 안으로 연결되므로, 넣을 파일은 `images/` 에 두고
결과는 `outputs/` 에서 확인하시면 됩니다.

### 직접 실행

```bash
docker run --rm -it --runtime nvidia --ipc=host \
  -v "$PWD/images:/workspace/images" -v "$PWD/outputs:/workspace/outputs" \
  -w /workspace/outputs yolo11-jp72:latest \
  python3 /workspace/predict_batch.py /workspace/images
```

### 인자

옵션 파싱을 걷어내고 위치 인자 두 개만 받습니다. 순서는 **입력, 모델** 입니다.

```bash
python3 predict_image.py my.jpg yolo11s.pt
python3 predict_batch.py /workspace/images yolo11n-seg.pt
python3 usbcam_infer.py  0 yolo11n.pt          # 0 = /dev/video0
```

모델은 이름만 바꾸면 종류가 바뀌고, 없으면 자동으로 내려받습니다.

| 가중치 | 결과 |
|---|---|
| `yolo11n.pt` / `s` / `m` / `l` / `x` | 객체 탐지 (뒤로 갈수록 정확하지만 느림) |
| `yolo11n-seg.pt` | 세그멘테이션 (픽셀 단위 윤곽) |
| `yolo11n-pose.pt` | 포즈 (사람 관절 17개) |
| `yolo11n-obb.pt` | 회전 박스 |

임계값·해상도처럼 자주 건드리지 않는 값은 스크립트 안에 그대로 적혀 있으니, 필요하면
`model.predict(...)` 줄에 `conf=0.4` 같은 인자를 직접 넣으시면 됩니다.

```python
result = model.predict(image, save=True, conf=0.4, imgsz=960)
```

### 모니터가 없을 때

`usbcam_infer.py` 상단의 `SHOW = True` 를 `False` 로 바꾸면 창을 띄우지 않고
`usbcam_out.mp4` 로만 저장합니다.

---

## 제7절. 성능 참고치

Orin Nano 8GB / 640px / MAXN_SUPER 기준 대략치입니다.

| 모델 | 예상 FPS |
|---|---|
| `yolo11n.pt` | 15 ~ 25 |
| `yolo11s.pt` | 10 ~ 15 |

실측치는 전원 모드·카메라 포맷·모델 크기에 따라 달라집니다. 더 빠르게 하려면
`model.predict(...)` 에 `half=True` 를 넣거나(FP16), `imgsz=320` 으로 해상도를 낮추세요.
TensorRT 엔진(`model.export(format="engine")`)까지 쓰면 2~3배까지 올라가지만, 코드가
복잡해지므로 이 레포에서는 빼두었습니다.

카메라 영상이 끊긴다면 `usbcam_infer.py` 의 MJPG 설정을 확인하세요. 무압축(YUYV)으로
동작하면 USB 대역폭 한계로 720p 에서 5~10 FPS 에 묶입니다.

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
| 메모리 부족 | swap 8~16GB 확대, 또는 헤드리스 부팅으로 1GB 이상 확보 |

---

## 제9절. 주의사항

- 세 스크립트는 **읽고 고치기 쉬운 것**을 최우선으로 두었습니다. 옵션 파싱과 예외 처리를
  덜어냈으므로, 필요한 기능은 `model.predict(...)` 인자에 직접 추가해서 쓰시면 됩니다.
- **TensorRT 엔진(.engine) 은 장치·버전 종속적입니다.** 다른 Jetson 이나 다른 JetPack 버전으로
  파일을 그대로 옮기면 동작하지 않습니다. 배포 대상 보드마다 각각 변환하세요.
- `images/`, `outputs/` 디렉터리는 git 추적 대상에서 제외되어 있습니다.
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
