# 두 진입점이 공유하는 Dynamixel 드라이버 객체 및 유틸리티 함수.

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

from src_control.robot_constants import (
    DEVICENAME, BAUDRATE, PROTOCOL_VERSION,
    DXL_IDs, ADDR_TORQUE_ENABLE, ADDR_PRESENT_POSITION,
    available_device_names,
)

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)


def _is_permission_error(exc: Exception) -> bool:
    return getattr(exc, "errno", None) == 13 or "Permission denied" in str(exc)


def _print_permission_help() -> None:
    print(f"   포트는 감지됐지만 권한이 없습니다: {DEVICENAME}")
    print("   영구 해결: sudo usermod -a -G dialout $USER")
    print("   적용: 로그아웃 후 다시 로그인하거나 터미널에서 newgrp dialout")
    print(f"   임시 해결: sudo chmod a+rw {DEVICENAME}")


def _print_port_help() -> None:
    ports = available_device_names()
    if ports:
        print(f"   감지된 포트: {', '.join(ports)}")
        print(f"   예: DYNAMIXEL_PORT={ports[0]} python control_keyboard.py")
    else:
        print("   감지된 시리얼 포트가 없습니다: /dev/serial/by-id/*, /dev/ttyUSB*, /dev/ttyACM*")
        print("   USB 연결/전원/WSL USB 패스스루 여부를 확인한 뒤 다시 실행하세요.")


def init_robot():
    try:
        port_opened = portHandler.openPort()
    except Exception as exc:
        print(f"❌ 포트를 열 수 없습니다: {DEVICENAME} ({exc})")
        if _is_permission_error(exc):
            _print_permission_help()
        else:
            _print_port_help()
        raise SystemExit(1)
    if not port_opened:
        print(f"❌ 포트를 열 수 없습니다: {DEVICENAME}")
        _print_port_help()
        raise SystemExit(1)
    if not portHandler.setBaudRate(BAUDRATE):
        print("❌ 보드레이트 설정 실패!")
        raise SystemExit(1)
    for dxl_id in DXL_IDs:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
    print("🤖 로봇팔 토크 ON! 통신 에러 방패 적용 완료.")


def close_robot():
    for dxl_id in DXL_IDs:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()
    print("💤 로봇 연결 종료. 수고하셨습니다!")


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def read_current_positions() -> dict:
    """모든 DXL_IDs 의 현재 위치를 읽어 dict 로 반환. 통신 실패 시 해당 ID 는 -1."""
    current_pos = {}
    for dxl_id in DXL_IDs:
        try:
            pos, comm, _ = packetHandler.read4ByteTxRx(
                portHandler, dxl_id, ADDR_PRESENT_POSITION
            )
            current_pos[dxl_id] = pos if comm == COMM_SUCCESS else -1
        except Exception:
            current_pos[dxl_id] = -1
    return current_pos
