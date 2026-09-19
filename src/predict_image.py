#!/usr/bin/env python3
"""
YOLO11 — 이미지 1장 추론 (가장 단순한 예제)

  python3 predict_image.py                    # 샘플 이미지로 동작 확인
  python3 predict_image.py my.jpg             # 내 이미지
  python3 predict_image.py my.jpg yolo11s.pt  # 다른 모델
"""
import sys

from ultralytics import YOLO

# ── 설정 ────────────────────────────────────────────────
# 명령행 인자를 그대로 받는다. 없으면 기본값을 쓴다.
image = sys.argv[1] if len(sys.argv) > 1 else "https://ultralytics.com/images/bus.jpg"
weights = sys.argv[2] if len(sys.argv) > 2 else "yolo11n.pt"

# ── 추론 ────────────────────────────────────────────────
# YOLO()는 가중치가 없으면 자동으로 내려받는다.
model = YOLO(weights)

# save=True면 박스를 그린 이미지가 runs/detect/predict/ 에 저장된다.
# [0] 인 이유: predict()는 여러 장을 받을 수 있어 항상 리스트를 돌려준다.
result = model.predict(image, save=True)[0]

# ── 결과 출력 ───────────────────────────────────────────
print(f"\n탐지 {len(result.boxes)}개  (추론 {result.speed['inference']:.1f}ms)\n")

for box in result.boxes:
    name = result.names[int(box.cls)]          # 클래스 번호 → 이름 (0 → person)
    conf = float(box.conf)                     # 신뢰도 0~1
    x1, y1, x2, y2 = box.xyxy[0].tolist()      # 좌상단, 우하단 좌표
    print(f"  {name:<12} {conf:.2f}   ({x1:.0f}, {y1:.0f}) ~ ({x2:.0f}, {y2:.0f})")

print(f"\n결과 이미지: {result.save_dir}")
