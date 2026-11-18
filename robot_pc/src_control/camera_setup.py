# 두 진입점이 공유하는 웹캠 3대 초기화 로직.
# open_all_cameras() 호출 시 약 2초 블로킹(내부 time.sleep × 2) 발생.

import time
import glob
import os
import re
import sys
import cv2

CAMERA_WIDTH = int(os.environ.get("CAPSTONE_CAMERA_WIDTH", "320"))
CAMERA_HEIGHT = int(os.environ.get("CAPSTONE_CAMERA_HEIGHT", "240"))
CAMERA_FPS = int(os.environ.get("CAPSTONE_CAMERA_FPS", "10"))
CAMERA_VERBOSE_PROBE = os.environ.get("CAPSTONE_CAMERA_VERBOSE_PROBE", "0") == "1"
CAMERA_EXTENSIVE_PROBE = os.environ.get("CAPSTONE_CAMERA_EXTENSIVE_PROBE", "0") == "1"
CAMERA_READ_ATTEMPTS = int(os.environ.get("CAPSTONE_CAMERA_READ_ATTEMPTS", "3"))
JIELI_READ_ATTEMPTS = int(os.environ.get("CAPSTONE_JIELI_READ_ATTEMPTS", "1"))


def _unique(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _is_jieli_camera(name: str) -> bool:
    lower_name = name.lower()
    return any(token in lower_name for token in ("jieli", "usb composite", "hcam"))


def _open_configs(name: str = ""):
    if not CAMERA_EXTENSIVE_PROBE:
        configs = []
        if _is_jieli_camera(name):
            configs.extend([
                ("MJPG", 480, 320, 30),
                ("MJPG", 480, 320, CAMERA_FPS),
            ])
        configs.extend([
            ("MJPG", CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS),
            (None, 0, 0, 0),
        ])
        return _unique(configs)

    resolutions = _unique([
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        (640, 480),
        (320, 240),
        (160, 120),
    ])
    fps_values = _unique([CAMERA_FPS, 5, 15, 30])
    formats = ["MJPG", "YUYV", None]

    configs = []
    for width, height in resolutions:
        for fps in fps_values:
            for fourcc in formats:
                configs.append((fourcc, width, height, fps))
    configs.append((None, 0, 0, 0))
    return configs


def _fourcc_to_str(value) -> str:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "-"
    chars = "".join(chr((value >> 8 * i) & 0xFF) for i in range(4))
    return chars if chars.strip("\x00") else "-"


def _configure_camera(cap: cv2.VideoCapture, fourcc: str | None, width: int, height: int, fps: int) -> None:
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)


def open_camera(index: int, name: str = ""):
    """
    단일 카메라를 열고 MJPG/640×480/15fps 로 설정한다. 열기 실패 시 None 반환.

    Windows 환경에서 DirectShow(CAP_DSHOW)가 'index 캡처'를 지원하지 않는 경우가 있어
    MSMF(CAP_MSMF) → DSHOW(CAP_DSHOW) → 자동(0) 순으로 시도한다.
    """
    if sys.platform.startswith("win"):
        sources = ((index, cv2.CAP_MSMF), (index, cv2.CAP_DSHOW), (index, 0))
    elif sys.platform.startswith("linux"):
        sources = ((index, cv2.CAP_V4L2), (index, 0))
    else:
        sources = ((index, 0),)

    failed_configs = []
    for source, backend in sources:
        for fourcc, width_req, height_req, fps_req in _open_configs(name):
            requested_fourcc = fourcc or "AUTO"
            print(
                f"🔎 카메라 시도: /dev/video{index} {name} "
                f"(source={source}, backend={backend}, req={requested_fourcc} {width_req}x{height_req}@{fps_req})",
                flush=True,
            )
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                _configure_camera(cap, fourcc, width_req, height_req, fps_req)
                read_attempts = JIELI_READ_ATTEMPTS if _is_jieli_camera(name) else CAMERA_READ_ATTEMPTS
                for _ in range(max(1, read_attempts)):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        actual_fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
                        requested_fourcc = fourcc or "AUTO"
                        print(
                            f"✅ 카메라 열림: /dev/video{index} {name} "
                            f"(req={requested_fourcc} {width_req}x{height_req}@{fps_req}, "
                            f"actual={actual_fourcc} {width}x{height}@{fps:.1f})"
                        )
                        return cap
                    time.sleep(0.04)
                failed_configs.append(f"{requested_fourcc} {width_req}x{height_req}@{fps_req}")
                if CAMERA_VERBOSE_PROBE:
                    print(
                        f"⚠️ 프레임 없음: /dev/video{index} {name} "
                        f"(req={requested_fourcc} {width_req}x{height_req}@{fps_req})"
                    )
            try:
                cap.release()
            except Exception:
                pass
    sample = ", ".join(failed_configs[:8])
    if len(failed_configs) > 8:
        sample += f", ... 총 {len(failed_configs)}개 조합"
    print(f"⚠️ 카메라 열기 실패: /dev/video{index} {name} (프레임 실패 조합: {sample or '-'})")
    return None


def read_camera_frame(cap: cv2.VideoCapture | None, label: str = ""):
    if cap is None:
        return None
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            if width != CAMERA_WIDTH or height != CAMERA_HEIGHT:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
            return frame
        time.sleep(0.005)
    if label:
        print(f"\n⚠️ {label} 프레임 읽기 실패")
    return None


def _video_index(path: str) -> int | None:
    match = re.search(r"video(\d+)$", path)
    return int(match.group(1)) if match else None


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "-"


def _linux_video_links(index: int) -> list[str]:
    links = []
    for root in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        for path in sorted(glob.glob(f"{root}/*")):
            if _video_index(os.path.realpath(path)) == index:
                links.append(f"{root}: {os.path.basename(path)}")
    return links


def _linux_usb_branch(index: int) -> str | None:
    for link in _linux_video_links(index):
        match = re.search(r"(pci-[^:]+:\d+\.\d+-usb-\d+:\d+)", link)
        if match:
            return match.group(1)
    return None


def _linux_raw_video_nodes() -> list[str]:
    nodes = []
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: _video_index(p) or -1):
        index = _video_index(path)
        if index is None:
            continue
        name = _read_text(f"/sys/class/video4linux/video{index}/name")
        sys_index = _read_text(f"/sys/class/video4linux/video{index}/index")
        links = " | ".join(_linux_video_links(index)) or "-"
        nodes.append(f"{path}: sys_index={sys_index}, name={name}, links={links}")
    return nodes


def _is_capture_video_node(index: int, name: str = "") -> bool:
    if "video-index1" in name.lower():
        return False
    index_path = f"/sys/class/video4linux/video{index}/index"
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read().strip() == "0"
    except OSError:
        return True


def _linux_camera_devices() -> list[tuple[int, str]]:
    devices_by_index = {}
    by_id_paths = sorted(glob.glob("/dev/v4l/by-id/*"))
    for path in by_id_paths:
        index = _video_index(os.path.realpath(path))
        if index is None or index in devices_by_index:
            continue
        name = os.path.basename(path).replace("_", " ")
        if not _is_capture_video_node(index, name):
            continue
        devices_by_index[index] = name

    for path in sorted(glob.glob("/dev/video*"), key=lambda p: _video_index(p) or -1):
        index = _video_index(path)
        if index is None or index in devices_by_index:
            continue
        if not _is_capture_video_node(index):
            continue
        name = _read_text(f"/sys/class/video4linux/video{index}/name")
        if name == "-":
            name = f"Camera {index}"
        devices_by_index[index] = name

    if not devices_by_index:
        devices_by_index = {index: f"Camera {index}" for index in range(3)}

    return sorted(devices_by_index.items())


def _select_camera_indices(devices: list[tuple[int, str]]) -> list[int]:
    forced_order = os.environ.get("CAPSTONE_CAMERA_ORDER")
    if forced_order:
        try:
            indices = [int(item.strip()) for item in forced_order.split(",") if item.strip()]
        except ValueError:
            print(f"⚠️ CAPSTONE_CAMERA_ORDER 형식 오류: {forced_order}")
        else:
            return indices[:3]

    available = []
    for index, name in devices:
        lower_name = name.lower()
        if "lg camera" in lower_name or "virtual" in lower_name:
            continue
        available.append((index, lower_name))

    selected_indices = []
    for index, lower_name in available:
        if "abko" in lower_name:
            selected_indices.append(index)
            break

    logi_candidates = [
        (index, lower_name) for index, lower_name in available
        if any(token in lower_name for token in ("logi", "c270", "046d"))
        and index not in selected_indices
    ]
    logi_candidates.sort(key=lambda item: (0 if "usb-" in item[1] else 1, item[0]))
    selected_indices.extend(index for index, _ in logi_candidates)
    selected_indices.extend(
        index for index, _ in available
        if index not in selected_indices
    )
    return selected_indices[:3]


def _get_camera_devices() -> list[tuple[int, str]]:
    if sys.platform.startswith("win"):
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        return list(enumerate(graph.get_input_devices()))
    if sys.platform.startswith("linux"):
        return _linux_camera_devices()
    return [(index, f"Camera {index}") for index in range(3)]


def _camera_open_priority(index: int, name: str) -> tuple[int, int]:
    lower_name = name.lower()
    if any(token in lower_name for token in ("jieli", "usb composite", "hcam")):
        return (0, index)
    return (1, index)


def _camera_open_order_jobs(open_jobs: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    forced_order = os.environ.get("CAPSTONE_CAMERA_OPEN_ORDER")
    if forced_order:
        try:
            ordered_indices = [int(item.strip()) for item in forced_order.split(",") if item.strip()]
        except ValueError:
            print(f"⚠️ CAPSTONE_CAMERA_OPEN_ORDER 형식 오류: {forced_order}")
        else:
            rank = {index: rank for rank, index in enumerate(ordered_indices)}
            return sorted(open_jobs, key=lambda item: (rank.get(item[1], len(rank)), item[0]))

    return sorted(open_jobs, key=lambda item: _camera_open_priority(item[1], item[2]))


def _warn_linux_usb_branch_conflicts(selected_indices: list[int]) -> None:
    branches: dict[str, list[int]] = {}
    for index in selected_indices:
        branch = _linux_usb_branch(index)
        if branch:
            branches.setdefault(branch, []).append(index)

    for branch, indices in branches.items():
        if len(indices) > 1:
            joined = ", ".join(f"/dev/video{index}" for index in indices)
            print(
                f"⚠️ 같은 USB branch에 카메라가 같이 있습니다: {branch} -> {joined}. "
                "동시 스트림 실패 시 둘 중 하나를 다른 물리 USB 포트/허브로 옮기세요."
            )


def open_all_cameras():
    """
    FilterGraph 로 장치 목록을 조회하고 ABKO 1대 + Logi 2대를 순서대로 선택한 뒤
    (cap1, cap2, cap3, caps) 를 반환한다.
    caps 는 None 이 제거된 실제 연결 카메라 리스트.
    약 2초 블로킹 발생(DirectShow 초기화 대기).
    """
    print("📷 웹캠 3대 연결 시도: Linux는 /dev/video* V4L2, Windows는 MSMF/DSHOW 순으로 시도합니다...")

    devices = _get_camera_devices()
    detected_labels = [f"/dev/video{index}: {name}" for index, name in devices]
    print(f"감지된 카메라: {detected_labels}")

    selected_indices = _select_camera_indices(devices)
    device_names = dict(devices)
    if sys.platform.startswith("linux"):
        raw_nodes = _linux_raw_video_nodes()
        print(f"전체 /dev/video* 노드: {raw_nodes if raw_nodes else '없음'}")
        _warn_linux_usb_branch_conflicts(selected_indices)

    selected_labels = [f"/dev/video{index}: {device_names.get(index, '')}" for index in selected_indices]
    print(f"선택된 카메라: {selected_labels}")
    print(f"수동 고정 예: CAPSTONE_CAMERA_ORDER={','.join(str(i) for i in selected_indices)} DYNAMIXEL_PORT=/dev/ttyUSB0 python control_gamepad.py")
    if len(selected_indices) < 3:
        print("⚠️ 캡처 가능한 카메라가 3대 미만으로 감지됐습니다.")
        print("   위쪽 카메라가 전체 /dev/video* 노드에도 없으면 코드 문제가 아니라 USB 인식/허브/포트/전원 문제입니다.")

    caps_by_slot = [None, None, None]
    open_jobs = [
        (slot, index, device_names.get(index, ""))
        for slot, index in enumerate(selected_indices[:3])
    ]
    open_jobs = _camera_open_order_jobs(open_jobs)
    open_order_labels = [f"Camera {slot + 1}=/dev/video{index}" for slot, index, _ in open_jobs]
    print(f"카메라 열기 순서: {open_order_labels}")

    for slot, index, name in open_jobs:
        caps_by_slot[slot] = open_camera(index, name)
        time.sleep(0.5)

    cap1, cap2, cap3 = caps_by_slot

    caps = [cap for cap in [cap1, cap2, cap3] if cap is not None]
    if len(caps) < 3:
        print(f"⚠️ 경고: 카메라 {3 - len(caps)}대가 연결되지 않았습니다. 인덱스 매핑/가상카메라/점유 여부를 확인하세요.")

    return cap1, cap2, cap3, caps
