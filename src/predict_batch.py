#!/usr/bin/env python3
"""
YOLO11 — 폴더 / 동영상 일괄 추론, 결과를 CSV로 저장

  python3 predict_batch.py /workspace/images
  python3 predict_batch.py /workspace/videos/test.mp4
  python3 predict_batch.py /workspace/images yolo11n-seg.pt   # 세그멘테이션
  python3 predict_batch.py /workspace/images yolo11n-pose.pt  # 포즈

입력이 이미지든 폴더든 동영상이든 Ultralytics가 알아서 구분하므로
우리가 따로 분기할 필요가 없다.
"""
import csv
import sys
from pathlib import Path

from ultralytics import YOLO

source = sys.argv[1] if len(sys.argv) > 1 else "https://ultralytics.com/images/bus.jpg"
weights = sys.argv[2] if len(sys.argv) > 2 else "yolo11n.pt"
out_csv = Path("results.csv")

model = YOLO(weights)

# stream=True: 결과를 한꺼번에 메모리에 쌓지 않고 하나씩 넘겨준다.
#              동영상이나 이미지 수천 장에서도 메모리가 터지지 않는다.
# save=True  : 박스를 그린 이미지·영상을 runs/ 아래에 자동 저장한다.
results = model.predict(source, stream=True, save=True)

rows = []
save_dir = None
for frame_no, result in enumerate(results):
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append({
            "file": Path(result.path).name,        # 어느 이미지에서 나왔는지
            "frame": frame_no,                     # 동영상일 때의 프레임 번호
            "class": result.names[int(box.cls)],
            "conf": round(float(box.conf), 3),
            "x1": round(x1), "y1": round(y1),
            "x2": round(x2), "y2": round(y2),
        })
    print(f"[{frame_no + 1:>5}] {Path(result.path).name:<30} 객체 {len(result.boxes)}개")
    save_dir = result.save_dir   # 마지막 값을 아래 출력에 쓴다

# ── CSV 저장 ────────────────────────────────────────────
# encoding="utf-8-sig": 엑셀에서 한글이 깨지지 않도록 BOM을 붙인다.
with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["file"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n총 {len(rows)}건 탐지")
print(f"CSV    : {out_csv.resolve()}")
print(f"이미지 : {save_dir}")
