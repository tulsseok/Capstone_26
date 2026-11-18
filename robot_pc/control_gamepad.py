import glob
import os
import struct
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame
import cv2

from src_control.robot_constants import (
    ADDR_GOAL_POSITION,
    DXL_IDs,
    LIMITS,
    SLEEP_DT,
)
from src_control.robot_driver import (
    portHandler, packetHandler,
    init_robot, close_robot, clamp, read_current_positions,
)
from src_control.camera_setup import read_camera_frame, open_all_cameras
from src_control.save_data import DatasetRecorder
from src_control.home_return import HomeReturn, load_home_pose

DEADZONE = 0.12
JOY_EVENT_FORMAT = "IhBB"
JOY_EVENT_SIZE = struct.calcsize(JOY_EVENT_FORMAT)
CAMERA_ONLY = os.environ.get("CAPSTONE_CAMERA_ONLY", "0") == "1"
GAMEPAD_RUMBLE = os.environ.get("GAMEPAD_RUMBLE", "1") != "0"
RUMBLE_DURATION_MS = int(os.environ.get("GAMEPAD_RUMBLE_DURATION_MS", "250"))
RUMBLE_INTERVAL_SEC = float(os.environ.get("GAMEPAD_RUMBLE_INTERVAL_SEC", "1.0"))


def _apply_deadzone(value: float) -> float:
    return value if abs(value) > DEADZONE else 0.0


def _parse_index_map(env_name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError:
        print(f"⚠️ {env_name} 형식 오류. 기본값 {default} 사용.")
        return default
    if len(values) != len(default):
        print(f"⚠️ {env_name} 항목 수 오류. 기본값 {default} 사용.")
        return default
    return values


def _default_axis_map(axis_count: int) -> tuple[int, int, int, int]:
    default = (0, 1, 3, 4) if axis_count >= 5 else (0, 1, 2, 3)
    return _parse_index_map("GAMEPAD_AXIS_MAP", default)


def _default_button_map() -> tuple[int, int, int, int]:
    default = (0, 1, 2, 3)
    raw = os.environ.get("GAMEPAD_BUTTON_MAP")
    if not raw:
        return default
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError:
        print(f"⚠️ GAMEPAD_BUTTON_MAP 형식 오류. 기본값 {default} 사용.")
        return default
    if len(values) == 3:
        return values + (3,)
    if len(values) != 4:
        print(f"⚠️ GAMEPAD_BUTTON_MAP 항목 수 오류. 기본값 {default} 사용.")
        return default
    return values


class GamepadInput:
    def __init__(self) -> None:
        self.backend = ""
        self.joystick = None
        self.axis_map = (0, 1, 3, 4)
        self.button_map = _default_button_map()
        self.js_fd: int | None = None
        self.js_path = ""
        self.js_axes = [0.0] * 8
        self.js_buttons = [0] * 16
        self._rumble_events: list[tuple[float, float, float, int]] = []

    def open(self) -> bool:
        try:
            pygame.display.init()
            try:
                pygame.display.set_caption("Gamepad Input")
                pygame.display.set_mode((320, 90))
            except pygame.error:
                pass
            pygame.joystick.init()
            count = pygame.joystick.get_count()
        except pygame.error as exc:
            print(f"⚠️ pygame 조이스틱 초기화 실패: {exc}")
            count = 0

        if count > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.axis_map = _default_axis_map(self.joystick.get_numaxes())
            self.backend = "pygame"
            print(f"🎮 컨트롤러 연결됨: {self.joystick.get_name()}")
            print(f"   axes={self.joystick.get_numaxes()} buttons={self.joystick.get_numbuttons()} axis_map={self.axis_map} button_map={self.button_map}")
            return True

        return self._open_linux_joydev()

    def _open_linux_joydev(self) -> bool:
        paths = []
        seen = set()
        for pattern in ("/dev/input/by-id/*-joystick", "/dev/input/by-path/*-joystick", "/dev/input/js*"):
            for path in sorted(glob.glob(pattern)):
                real_path = os.path.realpath(path)
                if real_path in seen:
                    continue
                seen.add(real_path)
                paths.append(real_path)

        denied = []
        for path in paths:
            try:
                self.js_fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError:
                denied.append(path)
                continue
            except OSError:
                continue
            self.js_path = path
            self.axis_map = _parse_index_map("GAMEPAD_AXIS_MAP", (0, 1, 3, 4))
            self.backend = "joydev"
            print(f"🎮 Linux joydev 컨트롤러 연결됨: {path}")
            print(f"   axis_map={self.axis_map} button_map={self.button_map}")
            return True

        print("🎮 엑스박스 컨트롤러가 연결되지 않았거나 접근할 수 없습니다.")
        if denied:
            print(f"   권한 없음: {', '.join(denied)}")
            print("   필요 시: sudo usermod -a -G input $USER  후 로그아웃/로그인")
        else:
            print("   감지 경로 없음: /dev/input/js*, /dev/input/by-id/*-joystick")
            print("   /dev/input/event*만 보이면: sudo modprobe joydev 후 다시 실행")
        return False

    def read(self) -> tuple[dict[str, float], dict[str, bool]]:
        self.pump_rumble()
        if self.backend == "pygame":
            pygame.event.pump()
            return self._read_pygame()
        if self.backend == "joydev":
            self._pump_joydev()
            return self._read_joydev()
        return self._empty_state()

    def schedule_rumble(self, count: int, interval_sec: float = RUMBLE_INTERVAL_SEC) -> None:
        if not GAMEPAD_RUMBLE or self.backend != "pygame" or self.joystick is None:
            return
        now = time.monotonic()
        for idx in range(count):
            self._rumble_events.append((now + idx * interval_sec, 0.7, 1.0, RUMBLE_DURATION_MS))
        self.pump_rumble()

    def pump_rumble(self) -> None:
        if not self._rumble_events:
            return
        if not GAMEPAD_RUMBLE or self.backend != "pygame" or self.joystick is None:
            self._rumble_events.clear()
            return
        now = time.monotonic()
        pending = []
        for due_time, low, high, duration_ms in self._rumble_events:
            if due_time <= now:
                try:
                    self.joystick.rumble(low, high, duration_ms)
                except Exception:
                    pass
            else:
                pending.append((due_time, low, high, duration_ms))
        self._rumble_events = pending

    def _read_pygame(self) -> tuple[dict[str, float], dict[str, bool]]:
        axis_count = self.joystick.get_numaxes()
        button_count = self.joystick.get_numbuttons()

        def axis(index: int) -> float:
            if index >= axis_count:
                return 0.0
            return _apply_deadzone(float(self.joystick.get_axis(index)))

        def button(index: int) -> bool:
            return index < button_count and bool(self.joystick.get_button(index))

        left_x, left_y, right_x, right_y = self.axis_map
        btn_a, btn_b, btn_x, btn_y = self.button_map
        return (
            {
                "left_x": axis(left_x),
                "left_y": axis(left_y),
                "right_x": axis(right_x),
                "right_y": axis(right_y),
            },
            {
                "a": button(btn_a),
                "b": button(btn_b),
                "x": button(btn_x),
                "y": button(btn_y),
            },
        )

    def _pump_joydev(self) -> None:
        if self.js_fd is None:
            return
        while True:
            try:
                data = os.read(self.js_fd, JOY_EVENT_SIZE * 32)
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break

            usable_len = len(data) - (len(data) % JOY_EVENT_SIZE)
            for offset in range(0, usable_len, JOY_EVENT_SIZE):
                _, value, event_type, number = struct.unpack_from(JOY_EVENT_FORMAT, data, offset)
                event_type &= ~0x80
                if event_type == 0x02 and number < len(self.js_axes):
                    self.js_axes[number] = _apply_deadzone(max(-1.0, min(1.0, value / 32767.0)))
                elif event_type == 0x01 and number < len(self.js_buttons):
                    self.js_buttons[number] = int(value)

    def _read_joydev(self) -> tuple[dict[str, float], dict[str, bool]]:
        def axis(index: int) -> float:
            return self.js_axes[index] if index < len(self.js_axes) else 0.0

        def button(index: int) -> bool:
            return index < len(self.js_buttons) and bool(self.js_buttons[index])

        left_x, left_y, right_x, right_y = self.axis_map
        btn_a, btn_b, btn_x, btn_y = self.button_map
        return (
            {
                "left_x": axis(left_x),
                "left_y": axis(left_y),
                "right_x": axis(right_x),
                "right_y": axis(right_y),
            },
            {
                "a": button(btn_a),
                "b": button(btn_b),
                "x": button(btn_x),
                "y": button(btn_y),
            },
        )

    def _empty_state(self) -> tuple[dict[str, float], dict[str, bool]]:
        return (
            {"left_x": 0.0, "left_y": 0.0, "right_x": 0.0, "right_y": 0.0},
            {"a": False, "b": False, "x": False, "y": False},
        )

    def close(self) -> None:
        if self.backend == "pygame" and self.joystick is not None:
            try:
                self.joystick.stop_rumble()
            except Exception:
                pass
        if self.js_fd is not None:
            try:
                os.close(self.js_fd)
            except OSError:
                pass
            self.js_fd = None


def main():
    gamepad = None
    if not CAMERA_ONLY:
        gamepad = GamepadInput()
        if not gamepad.open():
            return

        init_robot()
    else:
        print("📷 카메라 전용 모드: 로봇/게임패드 초기화를 건너뜁니다.")

    SPEED_STEP_BASE = 48
    GRIPPER_STEP = 64

    recorder = DatasetRecorder()

    def send_home_goal(dxl_id: int, goal: int) -> None:
        try:
            packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, goal)
        except Exception:
            pass

    # 홈 자세가 없으면(미측정/파일 손상) 복귀 없이 기존 동작(X = 즉시 종료)으로 폴백한다.
    home_pose = None if CAMERA_ONLY else load_home_pose()
    home_return = HomeReturn(home_pose, send_home_goal) if home_pose is not None else None
    if home_return is None and not CAMERA_ONLY:
        print("   → X 는 기존대로 녹화를 즉시 종료합니다 (홈 복귀 없음).")

    cap1, cap2, cap3, caps = open_all_cameras()

    print("\n--- 조작법 ---")
    print("왼쪽 스틱: 밑동/어깨 | 오른쪽 스틱: 팔꿈치/손목 | A/B 버튼: 그리퍼")
    if home_return is not None:
        print(f"🔥 X 버튼: 녹화 시작 / 녹화 종료(→ 그리퍼 개방 → 홈 복귀 → 그리퍼 폐쇄 → "
              f"{home_return.hold_sec:.0f}초 유지 후 저장)")
        print("   진동 1회=녹화 시작, 2회=홈 복귀 시작, 3회=복귀 완료·저장 | 복귀 중에는 스틱/버튼 입력이 무시됩니다")
    else:
        print("🔥 X 버튼: 녹화 시작/중지(시작 진동 1회, 종료 진동 2회)")
    print("🗑 Y 버튼: 마지막 녹화 삭제(Y 1회 후 X 전까지 잠금)")
    print("종료하려면 터미널에서 Ctrl+C 또는 카메라 창 클릭 후 'q'를 누르세요.\n")
    print("축이 이상하면 예: GAMEPAD_AXIS_MAP=0,1,2,3 DYNAMIXEL_PORT=/dev/ttyUSB0 python control_gamepad.py")
    print("버튼이 이상하면 예: GAMEPAD_BUTTON_MAP=0,1,2,3 DYNAMIXEL_PORT=/dev/ttyUSB0 python control_gamepad.py\n")

    # 버튼 중복 입력을 막기 위한 이전 상태 저장 변수
    prev_btn_x = False
    prev_btn_y = False
    delete_locked_until_x = False

    try:
        while True:
            # 🎮 1. 조이스틱 이벤트 처리
            if CAMERA_ONLY:
                axes = {"left_x": 0.0, "left_y": 0.0, "right_x": 0.0, "right_y": 0.0}
                btn_a = btn_b = btn_x = btn_y = False
            else:
                axes, buttons = gamepad.read()
                btn_a = buttons["a"]
                btn_b = buttons["b"]
                btn_x = buttons["x"]
                btn_y = buttons["y"]

            # 🤖 2. 로봇 제어 및 현재 위치 읽기
            current_pos = {dxl_id: 0 for dxl_id in DXL_IDs} if CAMERA_ONLY else read_current_positions()

            returning = home_return is not None and home_return.active
            if returning:
                rec_tag = f"🏠 {home_return.phase_label} ts={recorder.last_timestep}"
            elif recorder.recording:
                rec_tag = f"녹화중 ts={recorder.last_timestep}"
            else:
                rec_tag = "대기"
            if CAMERA_ONLY:
                print(f"📷 [카메라 전용] 💾 {rec_tag}", end='\r')
            else:
                print(f"📊 [관절값] ID11:{current_pos.get(11,-1):4d} | ID22:{current_pos.get(22,-1):4d} | ID23:{current_pos.get(23,-1):4d} | ID24:{current_pos.get(24,-1):4d} | ID25:{current_pos.get(25,-1):4d} | 💾 {rec_tag}", end='\r')

            # 안전 제어 전송 함수
            def send_goal_safe(dxl_id, stick_value):
                if stick_value == 0 or current_pos.get(dxl_id, -1) == -1: return
                step = SPEED_STEP_BASE
                if dxl_id == 22: step *= 2
                goal = int(current_pos[dxl_id] + (stick_value * step))
                goal = clamp(goal, LIMITS[dxl_id][0], LIMITS[dxl_id][1])
                try:
                    packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, goal)
                except Exception:
                    pass

            if returning:
                # 복귀 중에는 스틱/그리퍼 입력을 전부 무시한다. 상태 머신만 서보에 쓴다.
                home_return.step(current_pos)
            elif not CAMERA_ONLY:
                send_goal_safe(11, axes['left_x'])
                send_goal_safe(22, axes['left_y'])
                send_goal_safe(23, axes['right_y'])
                send_goal_safe(24, axes['right_x'])

            if not returning and not CAMERA_ONLY and (btn_a or btn_b) and current_pos.get(25, -1) != -1:
                offset = -GRIPPER_STEP if btn_a else GRIPPER_STEP
                goal = clamp(current_pos[25] + offset, LIMITS[25][0], LIMITS[25][1])
                try:
                    packetHandler.write4ByteTxRx(portHandler, 25, ADDR_GOAL_POSITION, goal)
                except Exception:
                    pass

            # 📷 3. 카메라 프레임 읽기
            frame1 = read_camera_frame(cap1, "Camera 1")
            frame2 = read_camera_frame(cap2, "Camera 2")
            frame3 = read_camera_frame(cap3, "Camera 3")

            if frame1 is not None: cv2.imshow('Camera 1', frame1)
            if frame2 is not None: cv2.imshow('Camera 2', frame2)
            if frame3 is not None: cv2.imshow('Camera 3', frame3)

            # 복귀 구간과 유지 구간도 이 호출로 그대로 기록된다.
            recorder.maybe_record_frame((frame1, frame2, frame3), current_pos)

            # 복귀 4단계가 전부 끝난 뒤에야 세션을 닫는다 (유지 구간 마지막 프레임 포함).
            if returning and not home_return.active:
                recorder.toggle_recording()
                if gamepad is not None:
                    gamepad.schedule_rumble(3)
                delete_locked_until_x = False
                returning = False

            if btn_x and not prev_btn_x and not returning:
                if not recorder.recording:
                    recorder.toggle_recording()
                    if gamepad is not None:
                        gamepad.schedule_rumble(1)
                    delete_locked_until_x = False
                elif home_return is not None:
                    home_return.start(current_pos)
                    if gamepad is not None:
                        gamepad.schedule_rumble(2)
                    print(f"\n🏠 홈 복귀 시작 — 완료 후 {home_return.hold_sec:.1f}초 유지하고 저장합니다.")
                else:
                    recorder.toggle_recording()
                    if gamepad is not None:
                        gamepad.schedule_rumble(2)
                    delete_locked_until_x = False
            if btn_y and not prev_btn_y and not returning:
                if delete_locked_until_x:
                    print("\n🛡 Y 삭제 잠금: 다음 X 입력 후 다시 삭제할 수 있습니다.")
                elif recorder.delete_last_recording():
                    delete_locked_until_x = True

            prev_btn_x = btn_x
            prev_btn_y = btn_y

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\n종료 명령(q) 감지!")
                break
                
            time.sleep(SLEEP_DT)

    except KeyboardInterrupt:
        print("\n\n강제 종료 감지!")
    finally:
        recorder.finalize()
        for cap in caps:
            if cap is not None and cap.isOpened(): 
                cap.release()
        cv2.destroyAllWindows()
        if gamepad is not None:
            gamepad.close()
        if not CAMERA_ONLY:
            close_robot()
        pygame.quit()

if __name__ == '__main__':
    main()
