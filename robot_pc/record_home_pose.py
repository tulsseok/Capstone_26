#!/usr/bin/env python3
"""홈 자세 측정 — 손으로 자세 잡고 SPACE. `control_gamepad.py` 의 녹화 종료 복귀용.

**토크를 켜지 않는다.** 시작할 때 오히려 확실히 끄고 관절값을 읽기만 한다. 서보에
아무것도 쓰지 않으므로 팔이 스스로 움직일 일이 없다. (`claude_arm/tools/jog_anchor.py`
와 같은 방식이지만 `src_control` 모듈만 쓴다 — numpy·dxl_writer 불필요.)

흐름
----
1. 실행 → 토크 OFF (팔이 힘없이 움직인다. **손으로 받쳐라**)
2. 팔을 매 에피소드가 끝났을 때 있었으면 하는 **종료 자세**로 손으로 잡는다 → SPACE
3. 그리퍼를 손으로 **활짝 벌린다** → SPACE
4. `home_pose.json` 저장 후 종료

3번이 필요한 이유: `LIMITS[25] = (1000, 2700)` 중 어느 끝이 "열림"인지 코드 어디에도
없다. 벌린 상태의 실측값이 어느 끝에 가까운지로 방향을 결정한다 — 추측하지 않는다.

키
--
    SPACE   현재 값 기록 → 다음 단계
    u       직전 기록 취소
    0 / ESC 중단 (저장하지 않음)

SPACE 1회당 5번 읽어 관절별 **중앙값**을 쓴다. 손으로 붙잡은 동안의 흔들림을 줄인다.
입력은 터미널 cbreak + ECHO OFF — **이 스크립트를 실행한 터미널에 포커스**가 있어야 한다.

실행
----
    python record_home_pose.py
    CAPSTONE_HOME_POSE=my_home.json python record_home_pose.py
"""
from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path
from statistics import median

from src_control.home_return import default_home_pose_path
from src_control.robot_constants import (
    ADDR_TORQUE_ENABLE,
    BAUDRATE,
    DEVICENAME,
    DXL_IDs,
    LIMITS,
)
from src_control.robot_driver import (
    _print_permission_help,
    _print_port_help,
    packetHandler,
    portHandler,
    read_current_positions,
)

GRIPPER_ID = 25

STEPS: list[tuple[str, str]] = [
    ("home", "팔 전체를 원하는 **종료 자세**로 손으로 잡아라"),
    ("gripper_open", "그리퍼를 손으로 **활짝 벌려라** (팔 자세는 상관없다)"),
]


class RawTerminal:
    """cbreak + ECHO OFF. ECHO 를 끄지 않으면 눌린 키가 화면에 찍혀 상태 줄이 깨진다."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.saved: list | None = None

    def enter(self) -> None:
        if self.saved is None:
            self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        attrs = termios.tcgetattr(self.fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)

    def restore(self) -> None:
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read_keys(self, timeout_s: float) -> list[str]:
        keys: list[str] = []
        if select.select([sys.stdin], [], [], timeout_s)[0]:
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                keys.append(ch)
        return keys


def open_port_torque_off() -> bool:
    """포트를 열고 보드레이트를 맞춘 뒤 **토크를 끈다**. init_robot() 은 토크를 켜므로 쓰지 않는다."""
    try:
        opened = portHandler.openPort()
    except Exception as exc:
        print(f"❌ 포트를 열 수 없습니다: {DEVICENAME} ({exc})")
        if getattr(exc, "errno", None) == 13 or "Permission denied" in str(exc):
            _print_permission_help()
        else:
            _print_port_help()
        return False
    if not opened:
        print(f"❌ 포트를 열 수 없습니다: {DEVICENAME}")
        _print_port_help()
        return False
    if not portHandler.setBaudRate(BAUDRATE):
        print("❌ 보드레이트 설정 실패!")
        return False
    for dxl_id in DXL_IDs:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    return True


def read_median(samples: int = 5, gap_s: float = 0.02) -> dict[int, int] | None:
    """관절별 중앙값. 통신 실패(-1)한 샘플은 버린다. 한 관절이라도 유효 샘플이 없으면 None."""
    buckets: dict[int, list[int]] = {dxl_id: [] for dxl_id in DXL_IDs}
    for _ in range(samples):
        pos = read_current_positions()
        for dxl_id in DXL_IDs:
            value = pos.get(dxl_id, -1)
            if value is not None and value >= 0:
                buckets[dxl_id].append(int(value))
        time.sleep(gap_s)
    out: dict[int, int] = {}
    for dxl_id in DXL_IDs:
        if not buckets[dxl_id]:
            return None
        out[dxl_id] = int(round(median(buckets[dxl_id])))
    return out


def build_pose(home: dict[int, int], gripper_open_sample: int) -> tuple[dict, list[str]]:
    """측정값 → 저장할 dict. 범위를 벗어난 값은 clamp 하고 사유를 함께 돌려준다."""
    notes: list[str] = []
    home_out: dict[str, int] = {}
    for dxl_id in DXL_IDs:
        lo, hi = LIMITS[dxl_id]
        value = home[dxl_id]
        if not lo <= value <= hi:
            clamped = max(lo, min(value, hi))
            notes.append(f"ID{dxl_id} {value} → {clamped} (안전 범위 {lo}~{hi} 밖이라 clamp)")
            value = clamped
        home_out[str(dxl_id)] = value

    lo, hi = LIMITS[GRIPPER_ID]
    if abs(gripper_open_sample - lo) <= abs(gripper_open_sample - hi):
        open_target, close_target = lo, hi
    else:
        open_target, close_target = hi, lo

    return (
        {
            "measured_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "home": home_out,
            "gripper_open_sample": int(gripper_open_sample),
            "gripper_open_target": int(open_target),
            "gripper_close_target": int(close_target),
        },
        notes,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="홈 자세 측정 (토크 OFF, 손으로 자세 잡고 SPACE)")
    ap.add_argument("--out", type=Path, default=None, help="저장 경로 (기본: CAPSTONE_HOME_POSE 또는 home_pose.json)")
    ap.add_argument("--samples", type=int, default=5, help="SPACE 1회당 읽기 횟수 (중앙값)")
    ap.add_argument("--hz", type=float, default=20.0, help="상태줄 갱신 주기")
    args = ap.parse_args()

    out_path = args.out if args.out is not None else default_home_pose_path()

    if not sys.stdin.isatty():
        print("[err] 터미널(tty)에서 실행해야 합니다. 파이프/리다이렉트 불가.")
        return 2

    print(__doc__)
    print(f"저장 경로 : {out_path}" + ("  (⚠️ 이미 있음 — 덮어씁니다)" if out_path.exists() else ""))
    print(f"DXL 포트  : {DEVICENAME} @ {BAUDRATE}\n")

    if not open_port_torque_off():
        return 1

    term = RawTerminal()
    period = 1.0 / max(args.hz, 1.0)
    recorded: dict[str, dict[int, int]] = {}
    order: list[str] = []
    cursor = 0
    aborted = False

    print("🔓 토크 OFF — 팔이 힘없이 움직입니다. 손으로 받치세요.")
    print("   서보에 아무것도 쓰지 않습니다 (읽기 전용).\n")
    print(f"👉 [1/{len(STEPS)}] {STEPS[0][1]} → SPACE\n")

    try:
        term.enter()
        msg = ""
        while cursor < len(STEPS):
            for ch in term.read_keys(period):
                if ch in ("0", "\x1b"):
                    aborted = True
                    raise KeyboardInterrupt

                if ch == " ":
                    ticks = read_median(args.samples)
                    if ticks is None:
                        msg = "읽기 실패 — 배선/전원 확인 후 다시 SPACE"
                        continue
                    key = STEPS[cursor][0]
                    recorded[key] = ticks
                    order.append(key)
                    cursor += 1
                    term.restore()
                    print(f"\n✅ [{cursor}/{len(STEPS)}] {key} = {[ticks[i] for i in DXL_IDs]}")
                    if cursor < len(STEPS):
                        print(f"\n👉 [{cursor + 1}/{len(STEPS)}] {STEPS[cursor][1]} → SPACE\n")
                    term.enter()
                    msg = ""

                elif ch == "u":
                    if not order:
                        msg = "취소할 기록이 없습니다"
                    else:
                        last = order.pop()
                        recorded.pop(last, None)
                        cursor = [k for k, _ in STEPS].index(last)
                        term.restore()
                        print(f"\n↩️ 취소: {last}")
                        print(f"\n👉 [{cursor + 1}/{len(STEPS)}] {STEPS[cursor][1]} → SPACE\n")
                        term.enter()

            if cursor >= len(STEPS):
                break

            cur = read_current_positions()
            body = " ".join(f"{i}:{cur.get(i, -1):5d}" for i in DXL_IDs)
            print(f"\r[{cursor}/{len(STEPS)}] {body} | {msg:38s}", end="", flush=True)

        term.restore()

    except KeyboardInterrupt:
        term.restore()
        aborted = True
        print("\n\n중단됨 — 저장하지 않습니다.")
    finally:
        term.restore()
        try:
            for dxl_id in DXL_IDs:
                packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
            portHandler.closePort()
        except Exception as exc:
            print(f"[warn] 포트 정리 실패: {exc}")

    if aborted or len(recorded) < len(STEPS):
        print("측정이 완료되지 않아 파일을 쓰지 않았습니다.")
        return 1

    pose, notes = build_pose(recorded["home"], recorded["gripper_open"][GRIPPER_ID])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pose, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n🎉 저장 완료 → {out_path}")
    print(f"   home            : {[pose['home'][str(i)] for i in DXL_IDs]}")
    print(f"   그리퍼 열림 실측 : {pose['gripper_open_sample']} "
          f"→ 열림 목표 {pose['gripper_open_target']} / 닫힘 목표 {pose['gripper_close_target']}")
    for note in notes:
        print(f"   ⚠️ {note}")
    print("\n다음: python control_gamepad.py — X 로 녹화 종료하면 이 자세로 복귀합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
