#!/usr/bin/env python3
"""
YOLO11 사전학습 모델 — 단일 이미지 추론 최소 예제

사용 예)
  python3 predict_image.py bus.jpg
  python3 predict_image.py bus.jpg --model yolo11s.pt --conf 0.4
  python3 predict_image.py bus.jpg --out /workspace/outputs/result.jpg

모델(yolo11n.pt 등)은 지정한 이름이 없으면 Ultralytics가 자동으로 내려받습니다.
"""
import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="YOLO11 single image inference")
    p.add_argument("image", help="입력 이미지 경로 (없으면 샘플 이미지 자동 사용)", nargs="?")
    p.add_argument("--model", default="yolo11n.pt", help="가중치 (n/s/m/l/x 또는 커스텀 .pt)")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    p.add_argument("--imgsz", type=int, default=640, help="입력 해상도")
    p.add_argument("--out", default="result.jpg", help="결과 이미지 저장 경로")
    args = p.parse_args()

    # 1) 디바이스 확인 — GPU가 잡히지 않으면 CPU로 진행
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] torch {torch.__version__} / device = "
          f"{torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 2) 입력 이미지 (없으면 Ultralytics 샘플 URL 사용)
    source = args.image or "https://ultralytics.com/images/bus.jpg"
    if args.image and not Path(args.image).exists():
        sys.exit(f"[ERROR] 이미지를 찾을 수 없습니다: {args.image}")

    # 3) 모델 로드 — 로컬에 없으면 자동 다운로드
    print(f"[INFO] 모델 로드: {args.model}")
    model = YOLO(args.model)

    # 4) 추론
    result = model.predict(source, imgsz=args.imgsz, conf=args.conf,
                           device=device, verbose=False)[0]

    # 5) 탐지 결과 출력
    print(f"\n[RESULT] 탐지 객체 {len(result.boxes)}개  (전처리/추론/후처리 ms: "
          f"{result.speed['preprocess']:.1f} / {result.speed['inference']:.1f} / "
          f"{result.speed['postprocess']:.1f})\n")
    print(f"{'#':>3}  {'class':<15} {'conf':>6}   x1    y1    x2    y2")
    print("-" * 56)
    for i, box in enumerate(result.boxes):
        name = result.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"{i:>3}  {name:<15} {float(box.conf):>6.3f}  "
              f"{x1:>5.0f} {y1:>5.0f} {x2:>5.0f} {y2:>5.0f}")

    # 6) 박스가 그려진 이미지 저장
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(out))
    print(f"\n[INFO] 결과 이미지 저장: {out.resolve()}")


if __name__ == "__main__":
    main()
