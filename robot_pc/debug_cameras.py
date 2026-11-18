import glob
import os
import re
import time

import cv2

from src_control.camera_setup import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    _configure_camera,
    _fourcc_to_str,
    _open_configs,
)


def video_index(path: str) -> int | None:
    match = re.search(r"video(\d+)$", path)
    return int(match.group(1)) if match else None


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "-"


def symlink_labels(index: int) -> list[str]:
    labels = []
    for root in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        for path in sorted(glob.glob(f"{root}/*")):
            if video_index(os.path.realpath(path)) == index:
                labels.append(f"{root}: {os.path.basename(path)}")
    return labels


def probe(index: int) -> None:
    path = f"/dev/video{index}"
    name = read_text(f"/sys/class/video4linux/video{index}/name")
    sys_index = read_text(f"/sys/class/video4linux/video{index}/index")
    labels = symlink_labels(index)
    label_text = " | ".join(labels) if labels else "-"

    opened_any = False
    tried = []
    for fourcc, width_req, height_req, fps_req in _open_configs(name):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        opened = cap.isOpened()
        opened_any = opened_any or opened
        requested_fourcc = fourcc or "AUTO"
        if not opened:
            cap.release()
            continue
        _configure_camera(cap, fourcc, width_req, height_req, fps_req)
        ok_count = 0
        shape = "-"
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                ok_count += 1
                shape = f"{frame.shape[1]}x{frame.shape[0]}"
            time.sleep(0.03)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = f"{cap.get(cv2.CAP_PROP_FPS):.1f}"
        actual_fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

        tried.append(f"{requested_fourcc} {width_req}x{height_req}@{fps_req}:{ok_count}/20")
        if ok_count > 0:
            print(
                f"/dev/video{index} opened=1 frames={ok_count}/20 "
                f"sys_index={sys_index} name='{name}' "
                f"req={requested_fourcc} {width_req}x{height_req}@{fps_req} "
                f"actual={actual_fourcc} {width}x{height}@{fps} "
                f"frame={shape} labels={label_text}"
            )
            return

    sample = ", ".join(tried[:8])
    if len(tried) > 8:
        sample += f", ... total={len(tried)}"
    print(
        f"/dev/video{index} opened={int(opened_any)} frames=0/20 "
        f"sys_index={sys_index} name='{name}' labels={label_text} tried={sample or '-'}"
    )


def main() -> None:
    paths = sorted(glob.glob("/dev/video*"), key=lambda p: video_index(p) or -1)
    if not paths:
        print("No /dev/video* devices found.")
        return

    print(f"Requested config: {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS} MJPG")
    show_all = os.environ.get("CAPSTONE_DEBUG_ALL_VIDEO_NODES", "0") == "1"
    for path in paths:
        index = video_index(path)
        if index is not None:
            sys_index = read_text(f"/sys/class/video4linux/video{index}/index")
            if sys_index != "0" and not show_all:
                name = read_text(f"/sys/class/video4linux/video{index}/name")
                print(f"/dev/video{index} skipped sys_index={sys_index} name='{name}'")
                continue
            probe(index)


if __name__ == "__main__":
    main()
