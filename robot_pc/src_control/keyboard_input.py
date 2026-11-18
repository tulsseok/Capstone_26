"""
키보드 + 마우스 입력 상태를 리스너 스레드로 수집하고
메인 루프에서 델타 값을 소비하는 클래스.
"""

import os
import glob
import struct
import sys
import time

DEBUG_INPUT = True  # 마우스/키보드 입력 진단 로그
MOVEMENT_PULSE_FRAMES = 8


class KeyboardMouseState:
    """
    - A/D: base(좌/우)
    - W/S: shoulder(위/아래)
    - Q/E: elbow(위/아래)
    - X: 녹화 시작/중지 토글(상승 에지 1회 — 메인 루프에서 소비)
    """

    def __init__(self) -> None:
        self.left_down: bool = False
        self.right_down: bool = False
        self.space_down: bool = False
        self.ctrl_down: bool = False
        self.shift_down: bool = False

        self.a_down: bool = False
        self.d_down: bool = False
        self.w_down: bool = False
        self.s_down: bool = False
        self.shift_w_down: bool = False
        self.shift_s_down: bool = False
        self.q_down: bool = False
        self.e_down: bool = False

        self._prev_x: int | None = None
        self._prev_y: int | None = None

        # 메인 루프에서 소비할 델타(누적)
        self.mouse_dx: int = 0
        self.mouse_dy: int = 0
        self.wheel_dy: int = 0

        self.save_toggle_edge: bool = False  # X 키 물리적 눌림 1회(메인에서 토글 소비 후 False)
        self._x_down: bool = False  # OS 키 반복(on_press 다중)으로 토글 중복 방지
        self.quit_edge: bool = False
        self._q_cv2_frames = 0
        self._e_cv2_frames = 0
        self._q_cv2_active = False
        self._e_cv2_active = False
        self._cv2_space_toggle = False
        self._pulse_keys: dict[str, int] = {}
        self._pulse_attrs_active: set[str] = set()
        self._last_key_event = "none"
        self._last_key_time = 0.0

        self._backend = "pynput"
        self._pygame = None
        self._pygame_started = False
        self._kb_listener = None
        self._mouse_listener = None
        self._terminal_started = False
        self._terminal_attrs = None
        self._select = None
        self._evdev_fds: list[tuple[int, str]] = []
        self._evdev_event_size = struct.calcsize("llHHI")

        try:
            from pynput import mouse, keyboard  # 지연 import (환경에 따라 설치 여부 다름)
            from pynput.mouse import Button
            from pynput.keyboard import Key

            self._Button = Button
            self._Key = Key

            self._kb_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._mouse_listener = mouse.Listener(
                on_move=self._on_move,
                on_click=self._on_click,
                on_scroll=self._on_scroll,
            )
        except Exception as exc:
            self._init_pygame_backend(exc)

    def _init_pygame_backend(self, exc: Exception) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            import pygame
        except Exception as pygame_exc:
            raise RuntimeError(
                "pynput 입력 초기화에 실패했고 pygame fallback도 사용할 수 없습니다. "
                f"pynput 오류: {exc}"
            ) from pygame_exc
        self._backend = "pygame"
        self._pygame = pygame
        self._pynput_error = exc

    def start(self) -> None:
        if self._backend == "pynput":
            self._kb_listener.start()
            self._mouse_listener.start()
            self._start_evdev_input()
            self._start_terminal_input()
            return

        pygame = self._pygame
        pygame.display.init()
        pygame.display.set_caption("Robot Input Capture")
        pygame.display.set_mode((360, 120))
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.key.set_repeat(1, 50)
        pygame.mouse.get_rel()
        self._pygame_started = True
        print("\n[pynput 사용 불가] pygame 입력 창을 사용합니다. 마우스/키보드는 'Robot Input Capture' 창에 포커스하세요.")
        self._start_evdev_input()
        self._start_terminal_input()

    def stop(self) -> None:
        self._stop_terminal_input()
        self._stop_evdev_input()
        if self._backend == "pynput":
            self._kb_listener.stop()
            self._mouse_listener.stop()
            return
        if self._pygame_started:
            self._pygame.event.set_grab(False)
            self._pygame.display.quit()
            self._pygame_started = False

    def _start_terminal_input(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            self._terminal_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._select = select
            self._terminal_started = True
            print("[입력] 터미널 키 입력도 활성화됨: 터미널 포커스에서 q/e/x/p/space 사용 가능")
        except Exception as exc:
            print(f"[입력] 터미널 키 입력 활성화 실패: {exc}")

    def _stop_terminal_input(self) -> None:
        if not self._terminal_started or self._terminal_attrs is None:
            return
        try:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._terminal_attrs)
        except Exception:
            pass
        self._terminal_started = False
        self._terminal_attrs = None

    def _start_evdev_input(self) -> None:
        paths = []
        seen = set()
        for pattern in ("/dev/input/by-id/*-event-kbd", "/dev/input/by-path/*-event-kbd"):
            for path in sorted(glob.glob(pattern)):
                real_path = os.path.realpath(path)
                if real_path in seen:
                    continue
                seen.add(real_path)
                paths.append(real_path)

        denied = []
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError:
                denied.append(path)
                continue
            except OSError:
                continue
            self._evdev_fds.append((fd, path))

        if self._evdev_fds:
            names = ", ".join(path for _, path in self._evdev_fds)
            print(f"[입력] Linux 키보드 직접 입력 활성화: {names}")
        elif denied:
            print("[입력] Linux 키보드 직접 입력 권한 없음.")
            print("       필요 시: sudo usermod -a -G input $USER  후 로그아웃/로그인")

    def _stop_evdev_input(self) -> None:
        for fd, _ in self._evdev_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._evdev_fds.clear()

    def pump(self) -> None:
        self._tick_cv2_pulses()

        if self._backend != "pygame" or not self._pygame_started:
            self._apply_pulse_key_states()
            self._pump_evdev_keys()
            self._pump_terminal_keys()
            self._decay_pulse_keys()
            return

        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_edge = True
            elif event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                self.mouse_dx += dx
                self.mouse_dy += dy
            elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                pressed = event.type == pygame.MOUSEBUTTONDOWN
                if event.button == 1:
                    self.left_down = pressed
                elif event.button == 3:
                    self.right_down = pressed
                elif pressed and event.button == 4:
                    self.wheel_dy += 1
                elif pressed and event.button == 5:
                    self.wheel_dy -= 1
            elif event.type == pygame.MOUSEWHEEL:
                self.wheel_dy += event.y
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                self._on_pygame_key(event.key, event.type == pygame.KEYDOWN)
        self._poll_pygame_keys()
        self._pump_evdev_keys()
        self._pump_terminal_keys()
        self._decay_pulse_keys()

    def _pump_terminal_keys(self) -> None:
        if not self._terminal_started or self._select is None:
            return
        try:
            while self._select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                self.handle_cv2_key(ord(ch), source="terminal")
        except KeyboardInterrupt:
            raise
        except Exception:
            return

    def _pump_evdev_keys(self) -> None:
        for fd, path in list(self._evdev_fds):
            while True:
                try:
                    data = os.read(fd, self._evdev_event_size * 16)
                except BlockingIOError:
                    break
                except OSError:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    self._evdev_fds = [(item_fd, item_path) for item_fd, item_path in self._evdev_fds if item_fd != fd]
                    break

                if not data:
                    break

                usable_len = len(data) - (len(data) % self._evdev_event_size)
                for offset in range(0, usable_len, self._evdev_event_size):
                    _, _, event_type, code, value = struct.unpack_from("llHHI", data, offset)
                    if event_type == 1:
                        self._handle_linux_key(code, value, path)

    def _mark_key_event(self, source: str, key_name: str, pressed: bool) -> None:
        state = "down" if pressed else "up"
        self._last_key_event = f"{source}:{key_name}:{state}"
        self._last_key_time = time.monotonic()

    def debug_snapshot(self) -> str:
        key_age = time.monotonic() - self._last_key_time if self._last_key_time else -1.0
        pulses = ",".join(f"{key}:{frames}" for key, frames in sorted(self._pulse_keys.items()))
        if not pulses:
            pulses = "-"
        return (
            f"backend={self._backend} pygame={int(self._pygame_started)} "
            f"evdev={len(self._evdev_fds)} terminal={int(self._terminal_started)} "
            f"last_key={self._last_key_event} age={key_age:.1f}s pulses={pulses}"
        )

    def _handle_linux_key(self, code: int, value: int, source_path: str = "evdev") -> None:
        pressed = value in (1, 2)
        released = value == 0
        source = f"evdev:{os.path.basename(source_path)}"
        key_names = {
            57: "space",
            29: "ctrl",
            97: "ctrl",
            42: "shift",
            54: "shift",
            30: "a",
            32: "d",
            17: "w",
            31: "s",
            16: "q",
            18: "e",
            45: "x",
            25: "p",
            1: "esc",
        }
        if code in key_names and value in (0, 1, 2):
            self._mark_key_event(source, key_names[code], pressed)

        if code == 57:
            self.space_down = pressed or self._cv2_space_toggle
        elif code in (29, 97):
            self.ctrl_down = pressed
        elif code in (42, 54):
            self.shift_down = pressed
        elif code == 30:
            self.a_down = pressed
        elif code == 32:
            self.d_down = pressed
        elif code == 17:
            self.w_down = pressed
        elif code == 31:
            self.s_down = pressed
        elif code == 16:
            self.q_down = pressed
        elif code == 18:
            self.e_down = pressed
        elif code == 45 and value == 1:
            self.save_toggle_edge = True
        elif code in (1, 25) and value == 1:
            self.quit_edge = True

        if released and code == 57:
            self.space_down = self._cv2_space_toggle
        self.shift_w_down = self.shift_down and self.w_down
        self.shift_s_down = self.shift_down and self.s_down

    def _poll_pygame_keys(self) -> None:
        pygame = self._pygame
        keys = pygame.key.get_pressed()
        keep_external = bool(self._evdev_fds)
        self.space_down = bool(keys[pygame.K_SPACE]) or self._cv2_space_toggle or (keep_external and self.space_down)
        self.ctrl_down = bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]) or (keep_external and self.ctrl_down)
        self.shift_down = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) or (keep_external and self.shift_down)
        self.a_down = bool(keys[pygame.K_a]) or self._pulse_key_active("a") or (keep_external and self.a_down)
        self.d_down = bool(keys[pygame.K_d]) or self._pulse_key_active("d") or (keep_external and self.d_down)
        self.w_down = bool(keys[pygame.K_w]) or self._pulse_key_active("w") or (keep_external and self.w_down)
        self.s_down = bool(keys[pygame.K_s]) or self._pulse_key_active("s") or (keep_external and self.s_down)
        self.shift_w_down = (self.shift_down and self.w_down) or self._pulse_key_active("shift_w")
        self.shift_s_down = (self.shift_down and self.s_down) or self._pulse_key_active("shift_s")
        self.q_down = bool(keys[pygame.K_q]) or self._q_cv2_frames > 0 or (keep_external and self.q_down)
        self.e_down = bool(keys[pygame.K_e]) or self._e_cv2_frames > 0 or (keep_external and self.e_down)

        x_down = bool(keys[pygame.K_x])
        if x_down and not self._x_down:
            self.save_toggle_edge = True
        self._x_down = x_down

        if keys[pygame.K_p] or keys[pygame.K_ESCAPE]:
            self.quit_edge = True

    def _tick_cv2_pulses(self) -> None:
        if self._q_cv2_frames > 0:
            self._q_cv2_frames -= 1
            self.q_down = True
            self._q_cv2_active = True
        elif self._q_cv2_active:
            self.q_down = False
            self._q_cv2_active = False

        if self._e_cv2_frames > 0:
            self._e_cv2_frames -= 1
            self.e_down = True
            self._e_cv2_active = True
        elif self._e_cv2_active:
            self.e_down = False
            self._e_cv2_active = False

    def _pulse_key(self, name: str, frames: int = 2) -> None:
        self._pulse_keys[name] = max(frames, self._pulse_keys.get(name, 0))
        self._apply_pulse_key_states()

    def _pulse_key_active(self, name: str) -> bool:
        return self._pulse_keys.get(name, 0) > 0

    def _apply_pulse_key_states(self) -> None:
        pulse_attrs = {
            "a": "a_down",
            "d": "d_down",
            "w": "w_down",
            "s": "s_down",
            "shift_w": "shift_w_down",
            "shift_s": "shift_s_down",
        }
        for name, attr in pulse_attrs.items():
            if self._pulse_key_active(name):
                setattr(self, attr, True)
                self._pulse_attrs_active.add(attr)
            elif attr in self._pulse_attrs_active:
                setattr(self, attr, False)
                self._pulse_attrs_active.remove(attr)

    def _decay_pulse_keys(self) -> None:
        for name, frames in list(self._pulse_keys.items()):
            if frames <= 1:
                del self._pulse_keys[name]
            else:
                self._pulse_keys[name] = frames - 1

    def _on_pygame_key(self, key: int, pressed: bool) -> None:
        pygame = self._pygame
        key_names = {
            pygame.K_SPACE: "space",
            pygame.K_LCTRL: "ctrl",
            pygame.K_RCTRL: "ctrl",
            pygame.K_LSHIFT: "shift",
            pygame.K_RSHIFT: "shift",
            pygame.K_a: "a",
            pygame.K_d: "d",
            pygame.K_w: "w",
            pygame.K_s: "s",
            pygame.K_q: "q",
            pygame.K_e: "e",
            pygame.K_x: "x",
            pygame.K_p: "p",
            pygame.K_ESCAPE: "esc",
        }
        if key in key_names:
            self._mark_key_event("pygame", key_names[key], pressed)
        if key == pygame.K_SPACE:
            self.space_down = pressed
        elif key in (pygame.K_LCTRL, pygame.K_RCTRL):
            self.ctrl_down = pressed
        elif key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
            self.shift_down = pressed
        elif key == pygame.K_a:
            self.a_down = pressed
        elif key == pygame.K_d:
            self.d_down = pressed
        elif key == pygame.K_w:
            self.w_down = pressed
            self.shift_w_down = self.shift_down and pressed
        elif key == pygame.K_s:
            self.s_down = pressed
            self.shift_s_down = self.shift_down and pressed
        elif key == pygame.K_q:
            self.q_down = pressed
        elif key == pygame.K_e:
            self.e_down = pressed
        elif key == pygame.K_x:
            if pressed and not self._x_down:
                self._x_down = True
                self.save_toggle_edge = True
            elif not pressed:
                self._x_down = False
        elif pressed and key in (pygame.K_p, pygame.K_ESCAPE):
            self.quit_edge = True

    def handle_cv2_key(self, key: int, source: str = "cv2") -> None:
        if key in (-1, 255):
            return
        if key in (ord("a"), ord("A")):
            self._mark_key_event(source, "a", True)
            self._pulse_key("a", MOVEMENT_PULSE_FRAMES)
        elif key in (ord("d"), ord("D")):
            self._mark_key_event(source, "d", True)
            self._pulse_key("d", MOVEMENT_PULSE_FRAMES)
        elif key == ord("w"):
            self._mark_key_event(source, "w", True)
            self._pulse_key("w", MOVEMENT_PULSE_FRAMES)
        elif key == ord("s"):
            self._mark_key_event(source, "s", True)
            self._pulse_key("s", MOVEMENT_PULSE_FRAMES)
        elif key == ord("W"):
            self._mark_key_event(source, "shift_w", True)
            self._pulse_key("shift_w", MOVEMENT_PULSE_FRAMES)
        elif key == ord("S"):
            self._mark_key_event(source, "shift_s", True)
            self._pulse_key("shift_s", MOVEMENT_PULSE_FRAMES)
        elif key in (ord("q"), ord("Q")):
            self._mark_key_event(source, "q", True)
            self.q_down = True
            self._q_cv2_frames = MOVEMENT_PULSE_FRAMES
        elif key in (ord("e"), ord("E")):
            self._mark_key_event(source, "e", True)
            self.e_down = True
            self._e_cv2_frames = MOVEMENT_PULSE_FRAMES
        elif key in (ord("x"), ord("X")):
            self._mark_key_event(source, "x", True)
            self.save_toggle_edge = True
        elif key == ord(" "):
            self._mark_key_event(source, "space", True)
            self._cv2_space_toggle = not self._cv2_space_toggle
            self.space_down = self._cv2_space_toggle
            state = "ON" if self._cv2_space_toggle else "OFF"
            print(f"\n[입력] Space 동시제어 토글: {state}")
        elif key in (ord("p"), ord("P"), 27):
            self._mark_key_event(source, "p", True)
            self.quit_edge = True

    # --- pynput callbacks (리스너 스레드) ---
    def _on_key_press(self, key) -> None:
        if key == self._Key.space:
            self.space_down = True
            return
        if key in (self._Key.ctrl, self._Key.ctrl_l, self._Key.ctrl_r):
            self.ctrl_down = True
            return
        if key in (self._Key.shift, self._Key.shift_l, self._Key.shift_r):
            self.shift_down = True
            self.shift_w_down = self.w_down
            self.shift_s_down = self.s_down
            return
        try:
            ch = key.char.lower()
        except Exception:
            return
        if ch == "a":
            self._mark_key_event("pynput", "a", True)
            self.a_down = True
        elif ch == "d":
            self._mark_key_event("pynput", "d", True)
            self.d_down = True
        elif ch == "w":
            self._mark_key_event("pynput", "w", True)
            self.w_down = True
            self.shift_w_down = self.shift_down
        elif ch == "s":
            self._mark_key_event("pynput", "s", True)
            self.s_down = True
            self.shift_s_down = self.shift_down
        elif ch == "q":
            self._mark_key_event("pynput", "q", True)
            self.q_down = True
        elif ch == "e":
            self._mark_key_event("pynput", "e", True)
            self.e_down = True
        elif ch == "x":
            self._mark_key_event("pynput", "x", True)
            if not self._x_down:
                self._x_down = True
                self.save_toggle_edge = True
        elif ch == "p":
            self._mark_key_event("pynput", "p", True)
            self.quit_edge = True

    def _on_key_release(self, key) -> None:
        if key == self._Key.space:
            self.space_down = False
            return
        if key in (self._Key.ctrl, self._Key.ctrl_l, self._Key.ctrl_r):
            self.ctrl_down = False
            return
        if key in (self._Key.shift, self._Key.shift_l, self._Key.shift_r):
            self.shift_down = False
            self.shift_w_down = False
            self.shift_s_down = False
            return
        try:
            ch = key.char.lower()
        except Exception:
            return
        if ch == "a":
            self._mark_key_event("pynput", "a", False)
            self.a_down = False
        elif ch == "d":
            self._mark_key_event("pynput", "d", False)
            self.d_down = False
        elif ch == "w":
            self._mark_key_event("pynput", "w", False)
            self.w_down = False
            self.shift_w_down = False
        elif ch == "s":
            self._mark_key_event("pynput", "s", False)
            self.s_down = False
            self.shift_s_down = False
        elif ch == "q":
            self._mark_key_event("pynput", "q", False)
            self.q_down = False
        elif ch == "e":
            self._mark_key_event("pynput", "e", False)
            self.e_down = False
        elif ch == "x":
            self._x_down = False

    def _on_move(self, x: int, y: int) -> None:
        if self._prev_x is None:
            self._prev_x, self._prev_y = x, y
            return
        dx = x - self._prev_x
        dy = y - self._prev_y
        self._prev_x, self._prev_y = x, y
        if dx == 0 and dy == 0:
            return
        self.mouse_dx += dx
        self.mouse_dy += dy

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        if DEBUG_INPUT:
            try:
                btn_name = str(button)
            except Exception:
                btn_name = "<unknown_button>"
            print(
                f"\n[DEBUG_INPUT] on_click button={btn_name} pressed={pressed} "
                f"(before) left_down={self.left_down} right_down={self.right_down}"
            )
        if button == self._Button.left:
            self.left_down = pressed
        elif button == self._Button.right:
            self.right_down = pressed
        if DEBUG_INPUT:
            print(
                f"[DEBUG_INPUT] (after) left_down={self.left_down} right_down={self.right_down}\n"
            )

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self.wheel_dy += dy

    # --- 메인 루프 소비용 ---
    def consume_mouse_delta(self) -> tuple[int, int]:
        dx, dy = self.mouse_dx, self.mouse_dy
        self.mouse_dx = 0
        self.mouse_dy = 0
        return dx, dy

    def consume_wheel(self) -> int:
        dy = self.wheel_dy
        self.wheel_dy = 0
        return dy

    def consume_save_toggle_edge(self) -> bool:
        """X 키 눌림 1회만 True 로 반환한다."""
        if self.save_toggle_edge:
            self.save_toggle_edge = False
            return True
        return False

    def consume_quit_edge(self) -> bool:
        if self.quit_edge:
            self.quit_edge = False
            return True
        return False
