"""녹화 종료 시 홈 자세 복귀 — control_gamepad.py 에서 사용.

X(녹화 종료)를 누르면 세션을 즉시 닫지 않고 다음 4단계를 거친 뒤 닫는다.

    open_gripper → move_arm → close_gripper → hold(1초) → 종료

이 구간 전체가 데이터셋에 그대로 기록된다("작업이 끝나면 홈으로 돌아간다"를 학습시키기
위한 것). 팔과 그리퍼를 **동시에** 움직이지 않는 이유는 claude_arm 의 phase 규칙과 같다 —
그리퍼가 여닫히는 동안 팔이 움직이면 물체를 쳐낸다.

보간은 **속도 상한 방식**이다. 단계 진입 시점의 자세를 시작값으로 고정해 두고,
`경과시간 × 속도` 만큼만 목표 쪽으로 진행한 goal 을 매 루프 전송한다.
현재 관절값이 아니라 시작값 기준으로 궤적을 계산한다 — 현재값 기준이면 서보 추종
지연이 매 루프 누적되어 목표에 영영 닿지 못한다.

홈 자세는 `record_home_pose.py` 로 미리 측정해 `home_pose.json` 에 저장해 둔다.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

try:
    from src_control.robot_constants import DXL_IDs, LIMITS
except ModuleNotFoundError:  # 파일을 직접 실행한 경우 (python src_control/home_return.py)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src_control.robot_constants import DXL_IDs, LIMITS

ARM_IDS: tuple[int, ...] = (11, 22, 23, 24)
GRIPPER_ID = 25

PHASES: tuple[str, ...] = ("open_gripper", "move_arm", "close_gripper", "hold")
PHASE_LABELS: dict[str, str] = {
    "open_gripper": "그리퍼 개방",
    "move_arm": "팔 복귀",
    "close_gripper": "그리퍼 폐쇄",
    "hold": "정지 유지",
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"⚠️ {name} 형식 오류. 기본값 {default} 사용.")
        return default
    if value <= 0:
        print(f"⚠️ {name} 는 0보다 커야 합니다. 기본값 {default} 사용.")
        return default
    return value


def default_home_pose_path() -> Path:
    return Path(os.environ.get("CAPSTONE_HOME_POSE", "home_pose.json"))


# ── 홈 자세 파일 ────────────────────────────────────────────────────────────

def load_home_pose(path: Path | str | None = None) -> dict[str, Any] | None:
    """`home_pose.json` 을 읽어 검증한다. 실패하면 경고 후 None (호출부는 폴백)."""
    path = Path(path) if path is not None else default_home_pose_path()

    if not path.exists():
        print(f"⚠️ 홈 자세 파일이 없습니다: {path}")
        print("   python record_home_pose.py 로 먼저 측정하세요.")
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"⚠️ 홈 자세 파일을 읽을 수 없습니다: {path} ({exc})")
        return None

    home_raw = raw.get("home")
    if not isinstance(home_raw, dict):
        print(f"⚠️ 홈 자세 파일에 'home' 항목이 없습니다: {path}")
        return None

    home: dict[int, int] = {}
    for dxl_id in DXL_IDs:
        value = home_raw.get(str(dxl_id), home_raw.get(dxl_id))
        if value is None:
            print(f"⚠️ 홈 자세에 ID{dxl_id} 값이 없습니다: {path}")
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            print(f"⚠️ 홈 자세 ID{dxl_id} 값이 정수가 아닙니다: {value!r}")
            return None
        lo, hi = LIMITS[dxl_id]
        if not lo <= value <= hi:
            print(f"⚠️ 홈 자세 ID{dxl_id}={value} 가 안전 범위 {lo}~{hi} 밖입니다.")
            return None
        home[dxl_id] = value

    lo, hi = LIMITS[GRIPPER_ID]
    targets: dict[str, int] = {}
    for key in ("gripper_open_target", "gripper_close_target"):
        value = raw.get(key)
        try:
            value = int(value)
        except (TypeError, ValueError):
            print(f"⚠️ 홈 자세 파일에 '{key}' 가 없거나 정수가 아닙니다: {path}")
            return None
        if not lo <= value <= hi:
            print(f"⚠️ {key}={value} 가 그리퍼 범위 {lo}~{hi} 밖입니다.")
            return None
        targets[key] = value

    print(
        f"🏠 홈 자세 로드: {path}\n"
        f"   home={[home[i] for i in DXL_IDs]} "
        f"그리퍼 열림={targets['gripper_open_target']} "
        f"닫힘={targets['gripper_close_target']}"
    )
    return {
        "home": home,
        "gripper_open_target": targets["gripper_open_target"],
        "gripper_close_target": targets["gripper_close_target"],
        "path": str(path),
    }


# ── 복귀 상태 머신 ──────────────────────────────────────────────────────────

class HomeReturn:
    """4단계 복귀 상태 머신. 제어 루프에서 매 회 `step()` 을 한 번 부른다.

    `send_goal(dxl_id, goal_tick)` 은 호출부가 주입한다 (로봇 없이 테스트 가능).
    """

    def __init__(
        self,
        pose: dict[str, Any],
        send_goal: Callable[[int, int], None],
        *,
        arm_speed: float | None = None,
        grip_speed: float | None = None,
        hold_sec: float | None = None,
        tol: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
    ) -> None:
        self._home: dict[int, int] = dict(pose["home"])
        self._open_target: int = int(pose["gripper_open_target"])
        self._close_target: int = int(pose["gripper_close_target"])
        self._send = send_goal
        self._clock = clock
        self._log = log

        self.arm_speed = arm_speed if arm_speed is not None else _env_float("CAPSTONE_HOME_ARM_SPEED", 400.0)
        self.grip_speed = grip_speed if grip_speed is not None else _env_float("CAPSTONE_HOME_GRIP_SPEED", 800.0)
        self.hold_sec = hold_sec if hold_sec is not None else _env_float("CAPSTONE_HOME_HOLD_SEC", 1.0)
        self.tol = int(tol if tol is not None else _env_float("CAPSTONE_HOME_TOL", 20.0))

        self._active = False
        self._phase = ""
        self._phase_idx = -1
        self._targets: dict[int, int] = {}
        self._starts: dict[int, int] = {}
        self._last_goal: dict[int, int] = {}
        self._speed = self.arm_speed
        self._need = 0.0
        self._deadline = 0.0
        self._phase_start = 0.0
        self.timed_out_phases: list[str] = []

    # ── 상태 조회 ──
    @property
    def active(self) -> bool:
        return self._active

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def phase_label(self) -> str:
        if not self._active:
            return "완료"
        step_no = self._phase_idx + 1
        return f"{PHASE_LABELS.get(self._phase, self._phase)} {step_no}/{len(PHASES)}"

    # ── 진행 ──
    def start(self, current_pos: dict[int, int]) -> None:
        self._active = True
        self._phase_idx = -1
        self._last_goal.clear()
        self.timed_out_phases.clear()
        self._advance(current_pos)

    def step(self, current_pos: dict[int, int]) -> bool:
        """진행 중이면 True, 전 단계가 끝났으면 False. 매 제어 루프에서 1회 호출."""
        if not self._active:
            return False

        elapsed = self._clock() - self._phase_start

        if self._phase == "hold":
            # 정지 유지 — 서보에 아무것도 쓰지 않는다. 이 1초도 그대로 녹화된다.
            if elapsed >= self.hold_sec:
                self._advance(current_pos)
            return self._active

        max_delta = self._speed * elapsed
        commanded_done = True
        for dxl_id, target in self._targets.items():
            start = self._starts[dxl_id]
            dist = target - start
            if abs(dist) <= max_delta:
                goal = target
            else:
                goal = int(round(start + math.copysign(max_delta, dist)))
                commanded_done = False
            lo, hi = LIMITS[dxl_id]
            goal = max(lo, min(goal, hi))
            self._last_goal[dxl_id] = goal
            self._send(dxl_id, goal)

        if commanded_done and self._settled(current_pos):
            self._advance(current_pos)
            return self._active

        if elapsed > self._deadline:
            self.timed_out_phases.append(self._phase)
            self._log(
                f"\n⚠️ 홈 복귀 '{PHASE_LABELS.get(self._phase, self._phase)}' 타임아웃"
                f" ({elapsed:.1f}s > {self._deadline:.1f}s) — 다음 단계로 진행합니다."
            )
            self._advance(current_pos)

        return self._active

    def abort(self) -> None:
        self._active = False
        self._phase = ""

    # ── 내부 ──
    def _settled(self, current_pos: dict[int, int]) -> bool:
        """실제 관절이 목표 ±tol 안에 들어왔는지. 통신 실패(-1) 관절은 판정에서 뺀다
        (여기서 막으면 통신이 끊긴 순간 복귀가 영영 안 끝난다 — 타임아웃이 받아준다)."""
        for dxl_id, target in self._targets.items():
            cur = current_pos.get(dxl_id, -1)
            if cur is None or cur < 0:
                continue
            if abs(int(cur) - target) > self.tol:
                return False
        return True

    def _targets_for(self, phase: str) -> dict[int, int]:
        if phase == "open_gripper":
            return {GRIPPER_ID: self._open_target}
        if phase == "move_arm":
            return {i: self._home[i] for i in ARM_IDS}
        if phase == "close_gripper":
            return {GRIPPER_ID: self._close_target}
        return {}

    def _advance(self, current_pos: dict[int, int]) -> None:
        self._phase_idx += 1
        if self._phase_idx >= len(PHASES):
            self._active = False
            self._phase = ""
            return

        self._phase = PHASES[self._phase_idx]
        self._targets = self._targets_for(self._phase)
        self._speed = self.grip_speed if self._phase in ("open_gripper", "close_gripper") else self.arm_speed

        self._starts = {}
        for dxl_id, target in self._targets.items():
            cur = current_pos.get(dxl_id, -1)
            if cur is None or cur < 0:
                # 통신 실패 — 마지막으로 보낸 goal, 그것도 없으면 목표에서 출발(=즉시 도달)
                cur = self._last_goal.get(dxl_id, target)
            self._starts[dxl_id] = int(cur)

        if self._targets:
            max_dist = max(abs(self._targets[i] - self._starts[i]) for i in self._targets)
            self._need = max_dist / self._speed
        else:
            self._need = self.hold_sec
        self._deadline = self._need * 1.5 + 1.5
        self._phase_start = self._clock()


# ── 로봇 없이 자체 검산 ─────────────────────────────────────────────────────

def _selftest() -> int:
    """가짜 시계 + 가짜 서보로 상태 머신을 검산한다. 로봇/시리얼 불필요."""
    pose = {
        "home": {11: 2048, 22: 2200, 23: 1600, 24: 2100, 25: 2600},
        "gripper_open_target": 1000,
        "gripper_close_target": 2700,
    }
    dt = 0.02
    failures = 0

    def run(tracking: bool, start_pos: dict[int, int], comm_fail: bool = False):
        """tracking=True 면 서보가 goal 을 즉시 따라간다. False 면 꿈쩍도 안 한다."""
        clock = {"t": 0.0}
        pos = dict(start_pos)
        logs: list[str] = []

        def send(dxl_id: int, goal: int) -> None:
            if tracking:
                pos[dxl_id] = goal

        hr = HomeReturn(
            pose, send, arm_speed=400.0, grip_speed=800.0, hold_sec=1.0, tol=20,
            clock=lambda: clock["t"], log=logs.append,
        )
        marks: list[tuple[str, float]] = []
        hr.start(pos)
        prev_phase = hr.phase
        phase_begin = 0.0
        for _ in range(20000):
            view = {i: -1 for i in pos} if comm_fail else pos
            if not hr.step(view):
                break
            clock["t"] += dt
            if hr.phase != prev_phase:
                marks.append((prev_phase, clock["t"] - phase_begin))
                phase_begin = clock["t"]
                prev_phase = hr.phase
        marks.append((prev_phase, clock["t"] - phase_begin))
        return marks, clock["t"], pos, logs, hr

    print("=" * 64)
    print("[1] 정상 추종 — 4단계 순서와 단계별 소요 시간")
    start = {11: 1048, 22: 2600, 23: 1200, 24: 2100, 25: 2000}
    marks, total, pos, logs, hr = run(True, start)
    # 기대: 개방 |2000-1000|/800=1.25s, 팔 max(|2048-1048|,|2200-2600|,|1600-1200|,0)/400=2.5s,
    #       폐쇄 |2700-1000|/800=2.125s, 유지 1.0s
    expect = {"open_gripper": 1.25, "move_arm": 2.5, "close_gripper": 2.125, "hold": 1.0}
    seen = [m[0] for m in marks]
    if seen != list(PHASES):
        print(f"  ❌ 단계 순서 {seen} != {list(PHASES)}")
        failures += 1
    else:
        print(f"  ✅ 단계 순서 {seen}")
    for phase, took in marks:
        want = expect[phase]
        ok = abs(took - want) <= dt * 2
        print(f"  {'✅' if ok else '❌'} {phase:14s} {took:5.2f}s (기대 {want:.2f}s ±{dt*2:.2f})")
        failures += 0 if ok else 1
    ok = all(pos[i] == pose["home"][i] for i in ARM_IDS) and pos[25] == pose["gripper_close_target"]
    print(f"  {'✅' if ok else '❌'} 최종 자세 {[pos[i] for i in DXL_IDs]} "
          f"(기대 팔=홈, 그리퍼={pose['gripper_close_target']})")
    failures += 0 if ok else 1
    print(f"  총 {total:.2f}s, 타임아웃 {hr.timed_out_phases or '없음'}")

    print("\n[2] 서보가 안 움직임 — 단계마다 타임아웃 후 강제 진행")
    marks, total, pos, logs, hr = run(False, start)
    ok = hr.timed_out_phases == ["open_gripper", "move_arm", "close_gripper"]
    print(f"  {'✅' if ok else '❌'} 타임아웃 단계 {hr.timed_out_phases}")
    failures += 0 if ok else 1
    ok = not hr.active and total < 30.0
    print(f"  {'✅' if ok else '❌'} {total:.2f}s 만에 스스로 종료 (무한 대기 아님)")
    failures += 0 if ok else 1
    print(f"  경고 {len(logs)}건")

    print("\n[3] 전 관절 통신 실패(-1) — 판정을 보류하지 않고 진행")
    marks, total, pos, logs, hr = run(True, start, comm_fail=True)
    ok = not hr.active and not hr.timed_out_phases
    print(f"  {'✅' if ok else '❌'} 타임아웃 없이 {total:.2f}s 에 종료 "
          f"(타임아웃 {hr.timed_out_phases or '없음'})")
    failures += 0 if ok else 1

    print("\n[4] 이미 홈에 있을 때 — 즉시 유지 단계로")
    at_home = dict(pose["home"])
    at_home[25] = pose["gripper_open_target"]
    marks, total, pos, logs, hr = run(True, at_home)
    want = 1.0 + abs(pose["gripper_close_target"] - pose["gripper_open_target"]) / 800
    ok = abs(total - want) <= dt * 4
    print(f"  {'✅' if ok else '❌'} 총 {total:.2f}s (기대 {want:.2f}s = 폐쇄+유지, 개방·팔은 0초)")
    failures += 0 if ok else 1

    print("=" * 64)
    print("🎉 전부 통과" if failures == 0 else f"❌ 실패 {failures}건")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
