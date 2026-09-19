# 호스트 사전 준비 (JetPack 7.2 / Jetson Orin Nano)

## 1. JetPack 7.2 설치 확인

```bash
cat /etc/nv_tegra_release        # R39 (release), REVISION: 2.x 확인
nvcc --version                   # CUDA 13.2
dpkg -l | grep -E 'tensorrt|cudnn'
```

`nvcc` 가 없으면 PATH 를 확인합니다.

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 2. Docker + NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y docker.io nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER" && newgrp docker
```

`/etc/docker/daemon.json` 에 다음이 들어가 있어야 합니다.

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

기본 런타임을 nvidia 로 두려면 `"default-runtime": "nvidia"` 를 추가합니다.
(이 경우 `--runtime nvidia` 를 생략할 수 있습니다.)

## 3. 전원 / 클럭

```bash
sudo nvpmodel -q              # 현재 모드
sudo nvpmodel -m 2            # Orin Nano Super: MAXN_SUPER (모드 번호는 보드별 확인)
sudo jetson_clocks            # 클럭 고정
sudo jetson_clocks --show
```

부팅 시 자동 적용하려면 systemd 서비스로 등록합니다.

```bash
sudo tee /etc/systemd/system/jetson-perf.service >/dev/null <<'EOF'
[Unit]
Description=Jetson max performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/nvpmodel -m 2
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now jetson-perf.service
```

## 4. 메모리 확보 (8GB 모델 권장)

```bash
# 헤드리스 부팅 — 1GB 이상 확보
sudo systemctl set-default multi-user.target

# swap 확대 (NVMe 권장)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 5. USB 카메라 확인

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

출력에서 확인할 것:

- `MJPG` 포맷 지원 여부 → 지원하면 `--mjpg` 사용 (720p 이상에서 FPS 확보의 핵심)
- 지원 해상도/프레임레이트 → `--width/--height/--fps` 값을 여기에 맞춤

일부 카메라는 `/dev/video0` 이 메타데이터 장치이고 실제 영상은 `/dev/video1` 인 경우가 있습니다.
`v4l2-ctl --list-devices` 출력의 장치 번호를 그대로 사용하세요.

권한 문제가 나면:

```bash
sudo usermod -aG video "$USER" && newgrp video
```

## 6. 모니터링

```bash
sudo pip3 install jetson-stats --break-system-packages
sudo jtop        # GPU 사용률, 전력, 온도, 메모리 실시간 확인
```
