#!/usr/bin/env python3
"""
YOLO11 — USB 카메라 실시간 추론

  python3 usbcam_infer.py            # /dev/video0, 화면 표시
  python3 usbcam_infer.py 1          # /dev/video1
  python3 usbcam_infer.py 0 yolo11s.pt

종료: 창을 클릭한 상태에서 q 또는 Esc.
모니터 없이 쓸 때는 SHOW를 False로 바꾸면 결과가 mp4로만 저장된다.
"""
import sys
import time

import cv2
from ultralytics import YOLO

CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else "yolo11n.pt"
SHOW = True          # 모니터가 없으면 False
OUT = "usbcam_out.mp4"

model = YOLO(WEIGHTS)

# ── 카메라 열기 ─────────────────────────────────────────
cap = cv2.VideoCapture(CAM)
# MJPG로 받아야 720p에서도 프레임이 안 떨어진다. (무압축 YUYV는 USB 대역폭에 걸림)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    sys.exit(f"카메라를 열 수 없습니다: /dev/video{CAM}  (docker run에 --device 옵션 확인)")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
print(f"카메라 {w}x{h} — 종료하려면 Ctrl+C (또는 창에서 q)")

# ── 추론 루프 ───────────────────────────────────────────
frames, t0 = 0, time.time()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # verbose=False: 프레임마다 찍히는 로그를 끈다
        result = model.predict(frame, verbose=False)[0]
        annotated = result.plot()          # 박스가 그려진 이미지

        writer.write(annotated)
        if SHOW:
            cv2.imshow("YOLO11", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        frames += 1
        if frames % 30 == 0:
            print(f"{frames / (time.time() - t0):.1f} FPS  객체 {len(result.boxes)}개")
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n{frames}프레임 처리 — 저장: {OUT}")
