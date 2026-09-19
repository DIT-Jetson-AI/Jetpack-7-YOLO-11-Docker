# 문제 해결

## GPU 관련

### `CUDA error 801: operation not supported`

가장 흔한 증상입니다. **컨테이너 내부 CUDA 가 12.x 인데 호스트 드라이버가 CUDA 13.2** 인 경우 발생합니다.

원인이 되는 이미지: `dustynv/*` (JetPack 6 계열), `ultralytics/ultralytics:latest-jetson-jetpack6`,
`nvcr.io/nvidia/l4t-*:r36.x`, Ollama 공식 바이너리 등.

조치: 본 레포지토리의 Dockerfile(CUDA 13 베이스)로 재빌드합니다.

```bash
./scripts/run.sh build
```

### `torch.cuda.is_available()` 가 False

순서대로 확인합니다.

```bash
# 1. 런타임 지정 여부
docker run --rm --runtime nvidia yolo11-jp72:latest \
  python3 -c "import torch; print(torch.cuda.is_available())"

# 2. 호스트 툴킷 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. torch 가 CPU 빌드로 깔린 경우 (cu130 인덱스 누락)
docker run --rm yolo11-jp72:latest python3 -c "import torch; print(torch.__version__)"
# → '+cpu' 가 붙어 있으면 Dockerfile 의 --index-url 확인
```

### PyTorch 결과가 NaN 으로 나옴

SBSA 휠의 sm_87 커널 커버리지 문제가 보고된 바 있습니다.
YOLO11 추론에서는 대체로 재현되지 않지만, 의심되면 TensorRT 경로(`--engine`)로 전환하세요.

---

## 카메라 관련

### 카메라를 열 수 없음

```bash
# 장치 번호 확인 — /dev/video0 이 메타데이터 장치인 경우가 있음
v4l2-ctl --list-devices

# 컨테이너에 장치 전달 확인
docker run ... --device /dev/video0 ...

# 여러 장치를 한꺼번에 넘기려면
--device /dev/video0 --device /dev/video1 --device /dev/bus/usb
```

### FPS 가 5~10 으로 묶임

카메라가 YUYV(무압축)로 동작 중일 가능성이 큽니다. USB 대역폭 한계입니다.

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext | grep -A3 MJPG
```

MJPG 를 지원하면 `--mjpg` 옵션을 붙이고, 지원하지 않으면 해상도를 낮추세요(`--width 640 --height 480`).

### 화면 지연(latency) 이 큼

`--imgsz` 를 낮추거나(`320`), `--engine` 으로 전환합니다.
스크립트는 `CAP_PROP_BUFFERSIZE=1` 로 버퍼를 최소화해 두었습니다.

---

## 표시 / 스트림

### X11 창이 뜨지 않음

```bash
xhost +local:docker
echo $DISPLAY            # 보통 :0
docker run ... -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
```

SSH 접속 중이라면 X11 forwarding 대신 `--stream` (MJPEG) 을 쓰는 편이 훨씬 안정적입니다.

### 브라우저에서 스트림이 안 보임

```bash
# 컨테이너가 --network host 로 실행됐는지 확인
# 방화벽 확인
sudo ufw allow 8080/tcp
# Jetson IP 확인
hostname -I
```

---

## 빌드

### Jetson APT 저장소(TensorRT) 설치 실패

네트워크 또는 저장소 버전 불일치입니다. 두 가지 선택지가 있습니다.

```bash
# 1) TensorRT 없이 슬림 빌드 (PyTorch 추론만)
./scripts/run.sh build-slim

# 2) L4T 저장소 버전을 실제 값으로 맞춤
cat /etc/nv_tegra_release        # R39 REVISION: 2.x → r39.2
docker build --build-arg L4T_REPO=r39.2 -t yolo11-jp72:latest .
```

### 빌드 중 디스크 부족

```bash
docker system prune -a           # 사용하지 않는 이미지/캐시 정리
df -h
```

---

## TensorRT

### 엔진 변환 중 OOM 또는 멈춤

```bash
# swap 확대 후 재시도 (docs/setup.md 4절)
# 또는 해상도를 낮춰 변환
python3 /workspace/usbcam_infer.py --engine --imgsz 480
```

변환은 최초 1회만 수행되며, 결과 `.engine` 파일은 `models/` 에 남습니다(볼륨 마운트 기준).

### 다른 보드에서 `.engine` 이 동작하지 않음

정상입니다. TensorRT 엔진은 GPU 아키텍처·TensorRT 버전·드라이버에 종속됩니다.
배포 대상 보드에서 각각 변환하세요. CI 로 굽는다면 동일 JetPack 이미지의 Jetson 러너가 필요합니다.

---

## 진단용 한 줄 명령

```bash
docker run --rm -it --runtime nvidia --device /dev/video0 yolo11-jp72:latest bash -lc '
  python3 -c "import torch,cv2;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())";
  python3 -c "import tensorrt as t;print(\"trt\",t.__version__)" 2>/dev/null || echo "trt: not installed";
  v4l2-ctl --list-devices'
```
