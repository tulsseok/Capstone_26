# 두 진입점(control_gamepad.py / control_keyboard.py)이 공유하는 하드웨어 상수.
# 이 파일만 수정하면 두 진입점에 동시 반영된다.

import glob
import os
import sys

SERIAL_PORT_PATTERNS = ("/dev/serial/by-id/*", "/dev/ttyUSB*", "/dev/ttyACM*")


def available_device_names() -> list[str]:
    ports = []
    seen = set()
    for pattern in SERIAL_PORT_PATTERNS:
        for port in sorted(glob.glob(pattern)):
            real_port = os.path.realpath(port)
            if real_port in seen:
                continue
            ports.append(real_port)
            seen.add(real_port)
    return ports


def _default_device_name() -> str:
    env_port = os.environ.get("DYNAMIXEL_PORT")
    if env_port:
        return env_port
    if sys.platform.startswith("win"):
        return "COM3"
    ports = available_device_names()
    if ports:
        return ports[0]
    return "/dev/ttyUSB0"


DEVICENAME = _default_device_name()
BAUDRATE = 1000000

# 제어 루프 주기(초). 두 진입점에서 공통 사용.
SLEEP_DT = 0.01

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
PROTOCOL_VERSION = 2.0

DXL_IDs = [11, 22, 23, 24, 25]

# 팀에서 직접 측정한 관절별 절대 안전 가동 범위
LIMITS = {
    11: (0, 4095),     # 베이스
    22: (1000, 3000),  # 어깨
    23: (1000, 3000),  # 팔꿈치
    24: (1000, 3000),  # 손목
    25: (1000, 2700)   # 그리퍼
}
