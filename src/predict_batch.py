#!/usr/bin/env python3
"""
YOLO11 배치 추론 — 이미지 / 폴더 / 동영상 + detect·segment·pose 지원

사용 예)
  # 폴더 안 이미지 전부
  python3 predict_batch.py /workspace/images --out /workspace/outputs

  # 동영상 (결과 mp4 저장)
  python3 predict_batch.py /workspace/videos/test.mp4 --out /workspace/outputs

  # 세그멘테이션 / 포즈
  python3 predict_batch.py /workspace/images --model yolo11n-seg.pt
  python3 predict_batch.py /workspace/images --model yolo11n-pose.pt

모델 종류는 가중치 이름으로 자동 판별됩니다.
  yolo11n.pt      → detect   (박스)
  yolo11n-seg.pt  → segment  (박스 + 마스크 면적)
  yolo11n-pose.pt → pose     (박스 + 키포인트 17개)
  yolo11n-obb.pt  → obb      (회전 박스)
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


# ---------------------------------------------------------------- 인자
def parse_args():
    p = argparse.ArgumentParser(description="YOLO11 batch inference")
    p.add_argument("source", help="이미지 파일 / 폴더 / 동영상 파일 경로")
    p.add_argument("--model", default="yolo11n.pt", help="가중치 (-seg, -pose, -obb 포함)")
    p.add_argument("--out", default="outputs", help="결과 저장 폴더")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 임계값")
    p.add_argument("--imgsz", type=int, default=640, help="입력 해상도")
    p.add_argument("--half", action="store_true", help="FP16 추론")
    p.add_argument("--classes", default="", help="특정 클래스만 (예: 0,2,3)")
    p.add_argument("--stride", type=int, default=1, help="동영상 N프레임마다 1장 처리")
    p.add_argument("--no-save-image", action="store_true", help="결과 이미지/영상 저장 생략")
    return p.parse_args()


# ---------------------------------------------------------------- 입력 수집
def collect_sources(src: Path):
    """(종류, 경로목록) 반환 — 종류는 'images' 또는 'video'"""
    if not src.exists():
        sys.exit(f"[ERROR] 경로를 찾을 수 없습니다: {src}")
    if src.is_dir():
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXT)
        if not files:
            sys.exit(f"[ERROR] 폴더에 이미지가 없습니다: {src}")
        return "images", files
    if src.suffix.lower() in VID_EXT:
        return "video", [src]
    if src.suffix.lower() in IMG_EXT:
        return "images", [src]
    sys.exit(f"[ERROR] 지원하지 않는 형식입니다: {src.suffix}")


# ------------------------------------------------------- 결과 → 행 목록 변환
def rows_from_result(result, frame_id, source_name):
    """탐지 1건을 dict 1개로. task에 따라 추가 필드를 붙인다."""
    rows = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return rows

    masks = getattr(result, "masks", None)
    kpts = getattr(result, "keypoints", None)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
        row = {
            "source": source_name,
            "frame": frame_id,
            "index": i,
            "class_id": int(box.cls),
            "class_name": result.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": round(x2 - x1, 1),
            "height": round(y2 - y1, 1),
        }
        # segment — 마스크 픽셀 면적
        if masks is not None and i < len(masks.data):
            row["mask_area_px"] = int(masks.data[i].sum().item())
        # pose — 키포인트 (x, y, conf) 목록
        if kpts is not None and i < len(kpts.data):
            row["keypoints"] = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                                for x, y, c in kpts.data[i].tolist()]
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 메인
def main():
    args = parse_args()
    src = Path(args.source)
    out_dir = Path(args.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] torch {torch.__version__} / device = "
          f"{torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(args.model)
    task = getattr(model, "task", "detect")
    print(f"[INFO] 모델 {args.model}  (task = {task})")

    cls_filter = [int(c) for c in args.classes.split(",") if c.strip()] or None
    predict_kw = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                      half=args.half, device=device, classes=cls_filter,
                      verbose=False)

    kind, files = collect_sources(src)
    all_rows, t_start = [], time.time()

    # ---------------- 이미지 / 폴더 ----------------
    if kind == "images":
        print(f"[INFO] 이미지 {len(files)}장 처리 시작\n")
        for n, f in enumerate(files, 1):
            result = model.predict(str(f), **predict_kw)[0]
            all_rows += rows_from_result(result, frame_id=0, source_name=f.name)
            if not args.no_save_image:
                result.save(filename=str(out_dir / "annotated" / f.name))
            print(f"[{n:>4}/{len(files)}] {f.name:<40} 객체 {len(result.boxes):>3}개  "
                  f"{result.speed['inference']:.1f}ms")

    # ---------------- 동영상 ----------------
    else:
        video = files[0]
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            sys.exit(f"[ERROR] 동영상을 열 수 없습니다: {video}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] 동영상 {video.name}  {w}x{h} @ {fps:.0f}fps  "
              f"총 {total}프레임 (stride={args.stride})\n")

        writer = None
        if not args.no_save_image:
            writer = cv2.VideoWriter(str(out_dir / "annotated" / f"{video.stem}_out.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / args.stride, (w, h))
        fid = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if fid % args.stride:
                    fid += 1
                    continue
                result = model.predict(frame, **predict_kw)[0]
                all_rows += rows_from_result(result, frame_id=fid, source_name=video.name)
                if writer:
                    vis = result.plot()
                    if vis.shape[1] != w or vis.shape[0] != h:
                        vis = cv2.resize(vis, (w, h))   # 크기가 어긋나면 무음 실패하므로 방어
                    writer.write(vis)
                if fid % (50 * args.stride) == 0:
                    print(f"[{fid:>7}/{total}] 객체 {len(result.boxes):>3}개  "
                          f"{result.speed['inference']:.1f}ms")
                fid += 1
        except KeyboardInterrupt:
            print("\n[INFO] 사용자 중단")
        finally:
            cap.release()
            if writer:
                writer.release()

    # ---------------- 결과 저장 ----------------
    elapsed = time.time() - t_start
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    csv_cols = ["source", "frame", "index", "class_id", "class_name", "confidence",
                "x1", "y1", "x2", "y2", "width", "height", "mask_area_px"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer_csv = csv.DictWriter(fp, fieldnames=csv_cols, extrasaction="ignore")
        writer_csv.writeheader()
        writer_csv.writerows(all_rows)

    # ---------------- 요약 ----------------
    counts = Counter(r["class_name"] for r in all_rows)
    units = len(files) if kind == "images" else len({r["frame"] for r in all_rows})
    print("\n" + "=" * 52)
    print(f"처리 {'이미지' if kind == 'images' else '프레임'} {units}건 / 탐지 {len(all_rows)}건 "
          f"/ 소요 {elapsed:.1f}초")
    print("-" * 52)
    for name, cnt in counts.most_common():
        print(f"  {name:<20} {cnt:>6}")
    print("=" * 52)
    print(f"[INFO] CSV  : {csv_path.resolve()}")
    print(f"[INFO] JSON : {json_path.resolve()}")
    if not args.no_save_image:
        print(f"[INFO] 이미지: {(out_dir / 'annotated').resolve()}")


if __name__ == "__main__":
    main()
