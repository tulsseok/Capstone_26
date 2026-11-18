import itertools
import os
import time

from src_control.camera_setup import open_camera, read_camera_frame


def parse_indices() -> list[int]:
    raw = os.environ.get("CAPSTONE_CAMERA_ORDER", "0,4,2")
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def test_combo(indices: tuple[int, ...], seconds: float = 2.0) -> None:
    print(f"\n=== combo {indices} ===", flush=True)
    caps = []
    try:
        for index in indices:
            cap = open_camera(index, f"combo-test-{index}")
            caps.append(cap)

        stats = {index: {"ok": 0, "fail": 0} for index in indices}
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            for index, cap in zip(indices, caps):
                frame = read_camera_frame(cap)
                if frame is None:
                    stats[index]["fail"] += 1
                else:
                    stats[index]["ok"] += 1

        for index in indices:
            ok = stats[index]["ok"]
            fail = stats[index]["fail"]
            print(f"/dev/video{index}: ok={ok} fail={fail}")
    finally:
        for cap in caps:
            if cap is not None and cap.isOpened():
                cap.release()


def main() -> None:
    indices = parse_indices()
    print(f"Testing simultaneous camera combos from: {indices}")
    for size in range(2, len(indices) + 1):
        for combo in itertools.combinations(indices, size):
            test_combo(combo)


if __name__ == "__main__":
    main()
