#!/usr/bin/env python3
import os
import sys
import time
import cv2

def main():
    backend = os.environ.get('DX_CAMERA_BACKEND', 'LIBCAMERA').upper()
    w = int(os.environ.get('DX_CAMERA_WIDTH', 1280))
    h = int(os.environ.get('DX_CAMERA_HEIGHT', 720))
    fps = int(os.environ.get('DX_CAMERA_FPS', 30))

    print(f"[RPI] Tentando abrir câmera - backend={backend} {w}x{h}@{fps}")

    cap = None
    if backend in ('AUTO', 'LIBCAMERA'):
        pipelines = [
            f"libcamerasrc ! video/x-raw,width={w},height={h},framerate={max(1,fps)}/1 ! videoconvert ! appsink",
            f"libcamerasrc ! image/jpeg,width={w},height={h},framerate={max(1,fps)}/1 ! jpegdec ! videoconvert ! appsink",
        ]
        for pipe in pipelines:
            print(f"[RPI] Pipeline: {pipe}")
            cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if cap and cap.isOpened():
                break
            if cap:
                cap.release()
                cap = None
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2 if backend == 'V4L2' else 0)

    if not cap or not cap.isOpened():
        print("[RPI] Falha ao abrir a câmera")
        sys.exit(1)

    print("[RPI] Câmera aberta. Capturando 5 frames...")
    ok = False
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and getattr(frame, 'size', 0) > 0:
            ok = True
            print(f"[RPI] Frame {i+1}: {frame.shape[1]}x{frame.shape[0]}")
            # salva um snapshot para validação manual
            if i == 0:
                cv2.imwrite("/tmp/rpi_camera_snapshot.jpg", frame)
            time.sleep(0.05)
        else:
            time.sleep(0.1)

    cap.release()
    print("[RPI] Resultado:", "OK" if ok else "FALHA")
    sys.exit(0 if ok else 2)

if __name__ == '__main__':
    main()


