import argparse
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def video_index(path: str) -> int | None:
    match = re.search(r"video(\d+)$", path)
    return int(match.group(1)) if match else None


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return "-"


def find_jieli_index() -> int | None:
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: video_index(p) or -1):
        index = video_index(path)
        if index is None:
            continue
        sys_index = read_text(f"/sys/class/video4linux/video{index}/index")
        name = read_text(f"/sys/class/video4linux/video{index}/name").lower()
        links = []
        for root in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
            for link in glob.glob(f"{root}/*"):
                if video_index(os.path.realpath(link)) == index:
                    links.append(os.path.basename(link).lower())
        haystack = " ".join([name] + links)
        if sys_index == "0" and any(token in haystack for token in ("jieli", "usb composite", "hcam")):
            return index
    return None


def video_interface_path(index: int) -> Path:
    return Path(f"/sys/class/video4linux/video{index}/device").resolve()


def find_usb_device_name(interface_path: Path) -> str | None:
    for parent in [interface_path] + list(interface_path.parents):
        name = parent.name
        if re.fullmatch(r"\d+-\d+(?:\.\d+)*", name):
            return name
    return None


def run_fuser(index: int) -> None:
    paths = [f"/dev/video{index}", f"/dev/video{index + 1}"]
    print("점유 프로세스 확인:")
    try:
        subprocess.run(["fuser", "-v", *paths], check=False)
    except FileNotFoundError:
        print("  fuser 명령이 없습니다. sudo apt install psmisc 로 설치할 수 있습니다.")


def write_sysfs(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def reset_authorized(usb_name: str) -> bool:
    authorized = Path("/sys/bus/usb/devices") / usb_name / "authorized"
    if not authorized.exists():
        print(f"authorized 경로를 찾지 못했습니다: {authorized}")
        return False
    print(f"USB authorized reset: {authorized}")
    write_sysfs(authorized, "0")
    time.sleep(1.0)
    write_sysfs(authorized, "1")
    time.sleep(1.0)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Jieli/USB camera 상태 확인 및 USB reset")
    parser.add_argument("--video", type=int, default=None, help="대상 /dev/videoN 번호")
    parser.add_argument("--reset", action="store_true", help="USB 장치를 authorized 0/1로 리셋")
    args = parser.parse_args()

    index = args.video if args.video is not None else find_jieli_index()
    if index is None:
        print("Jieli/HCAM 카메라를 찾지 못했습니다. --video 2 처럼 직접 지정하세요.")
        return

    name = read_text(f"/sys/class/video4linux/video{index}/name")
    sys_index = read_text(f"/sys/class/video4linux/video{index}/index")
    interface_path = video_interface_path(index)
    usb_name = find_usb_device_name(interface_path)

    print(f"대상: /dev/video{index} name='{name}' sys_index={sys_index}")
    print(f"interface: {interface_path}")
    print(f"usb device: {usb_name or '-'}")
    run_fuser(index)

    if not args.reset:
        if usb_name:
            print("\n리셋하려면:")
            print(f"  sudo {sys.executable} debug_camera_reset.py --video {index} --reset")
        return

    if os.geteuid() != 0:
        print("USB reset은 root 권한이 필요합니다. sudo로 다시 실행하세요.")
        raise SystemExit(1)
    if not usb_name:
        print("USB device 이름을 찾지 못해 reset할 수 없습니다.")
        raise SystemExit(1)

    if reset_authorized(usb_name):
        print("USB reset 완료. 2초 뒤 python debug_cameras.py 또는 debug_camera_live.py로 확인하세요.")


if __name__ == "__main__":
    main()
