import time

import cv2

from src_control.camera_setup import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    open_all_cameras,
    read_camera_frame,
)


def main() -> None:
    cap1, cap2, cap3, caps = open_all_cameras()
    windows = ("Camera 1", "Camera 2", "Camera 3")

    for idx, name in enumerate(windows):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, CAMERA_WIDTH, CAMERA_HEIGHT)
        cv2.moveWindow(name, 40 + idx * 680, 40)

    stats = {name: {"ok": 0, "fail": 0} for name in windows}
    last_log = 0.0

    try:
        while True:
            frames = (
                read_camera_frame(cap1),
                read_camera_frame(cap2),
                read_camera_frame(cap3),
            )

            for name, frame in zip(windows, frames):
                if frame is None:
                    stats[name]["fail"] += 1
                    continue
                stats[name]["ok"] += 1
                cv2.imshow(name, frame)

            now = time.monotonic()
            if now - last_log >= 1.0:
                last_log = now
                print(
                    " | ".join(
                        f"{name}: ok={stats[name]['ok']} fail={stats[name]['fail']}"
                        for name in windows
                    ),
                    end="\r",
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n종료 명령(q) 감지!")
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n강제 종료 감지!")
    finally:
        for cap in caps:
            if cap is not None and cap.isOpened():
                cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
