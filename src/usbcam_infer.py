#!/usr/bin/env python3
"""
YOLO11 USB 카메라 실시간 추론 (Jetson Orin Nano / JetPack 7.2)

사용 예)
  python3 usbcam_infer.py --device 0 --view          # X11 창으로 보기
  python3 usbcam_infer.py --device 0 --stream 8080   # 브라우저로 보기(헤드리스)
  python3 usbcam_infer.py --device 0 --engine        # TensorRT 엔진 자동 변환 후 추론
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLO11 USB camera inference")
    p.add_argument("--device", default="0", help="/dev/videoN 의 N 또는 경로")
    p.add_argument("--model", default=os.environ.get("ULTRALYTICS_WEIGHTS", "yolo11n.pt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--mjpg", action="store_true", help="카메라 MJPG 포맷 강제(고해상도 FPS 확보)")
    p.add_argument("--half", action="store_true", help="FP16 추론")
    p.add_argument("--engine", action="store_true", help="TensorRT .engine 으로 변환 후 사용")
    p.add_argument("--view", action="store_true", help="X11 창 표시")
    p.add_argument("--stream", type=int, default=0, help="MJPEG 웹 스트림 포트 (0=비활성)")
    p.add_argument("--save", default="", help="결과 영상 저장 경로(.mp4)")
    return p.parse_args()


def build_model(args):
    weights = args.model
    if args.engine:
        eng = Path(weights).with_suffix(".engine")
        if not eng.exists():
            print(f"[INFO] TensorRT 엔진 생성 중... ({eng})  최초 1회 수 분 소요")
            YOLO(weights).export(format="engine", imgsz=args.imgsz, half=True, device=0)
        weights = str(eng)
    print(f"[INFO] 모델 로드: {weights}")
    return YOLO(weights)


def open_camera(args):
    src = int(args.device) if str(args.device).isdigit() else args.device
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"[ERROR] 카메라를 열 수 없습니다: {args.device}  "
                 f"(docker run 에 --device /dev/video0 을 넣었는지 확인)")
    if args.mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 카메라 열림: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
    return cap, w, h


class MjpegServer:
    """의존성 없는 초경량 MJPEG 스트리머 (http://<jetson-ip>:PORT/)"""

    def __init__(self, port):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self._frame = None
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with outer._lock:
                            buf = outer._frame
                        if buf is None:
                            time.sleep(0.01)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(buf)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        print(f"[INFO] MJPEG 스트림: http://<jetson-ip>:{port}/")

    def update(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._lock:
                self._frame = buf.tobytes()


def main():
    args = parse_args()
    print(f"[INFO] torch {torch.__version__} / CUDA available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] GPU를 못 찾았습니다. --runtime nvidia 옵션과 CUDA 버전을 확인하세요.")

    model = build_model(args)
    cap, w, h = open_camera(args)

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (w, h))
    server = MjpegServer(args.stream) if args.stream else None

    n, t0, fps = 0, time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 프레임 수신 실패")
                break

            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                half=args.half, device=0 if torch.cuda.is_available() else "cpu",
                                verbose=False)[0]
            out = res.plot()

            n += 1
            if n % 10 == 0:
                fps = 10.0 / (time.time() - t0)
                t0 = time.time()
            cv2.putText(out, f"{fps:5.1f} FPS  obj={len(res.boxes)}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if writer:
                writer.write(out)
            if server:
                server.update(out)
            if args.view:
                cv2.imshow("YOLO11 - USB Camera", out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif not server and not writer and n % 30 == 0:
                print(f"[{n:6d}] {fps:5.1f} FPS  objects={len(res.boxes)}")
    except KeyboardInterrupt:
        print("\n[INFO] 종료")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
