import os
import time
import cv2
from dynamixel_sdk import COMM_SUCCESS

from src_control.robot_constants import (
    ADDR_GOAL_POSITION,
    DXL_IDs, LIMITS,
    SLEEP_DT,
)
from src_control.robot_driver import (
    portHandler, packetHandler,
    init_robot, close_robot, clamp, read_current_positions,
)
from src_control.camera_setup import CAMERA_HEIGHT, CAMERA_WIDTH, open_all_cameras
from src_control.save_data import DatasetRecorder
from src_control.keyboard_input import KeyboardMouseState


DEBUG_METRICS = os.environ.get("CAPSTONE_DEBUG_METRICS", "1") != "0"
CONTROL_METRIC_INTERVAL = float(os.environ.get("CAPSTONE_CONTROL_METRIC_INTERVAL", "1.0"))
CAMERA_METRIC_INTERVAL = float(os.environ.get("CAPSTONE_CAMERA_METRIC_INTERVAL", "2.0"))


def main():
    init_robot()

    # 입력 스케일(실로봇용 Dynamixel tick 단위)
    BASE_TICKS_PER_KEY_FRAME = 48     # ID11 (A/D 홀드 시, 기존 3배)
    SHOULDER_TICKS_PER_KEY_FRAME =48 # ID22 (W/S 홀드 시, 기존 3배)
    ELBOW_TICKS_PER_KEY_FRAME = 48    # ID23 (Q/E 홀드 시, 기존 3배)
    WRIST_TICKS_PER_PX = 6            # ID24 (마우스 앞/뒤 이동, 기존 3배)
    GRIPPER_STEP = 192                # ID25 (좌/우클릭 홀드 시, 기존 3배)

    recorder = DatasetRecorder()

    cap1, cap2, cap3, caps = open_all_cameras()

    print("\n--- 조작법 ---")
    print("A / D: 밑동(ID11) 좌/우 (반전 적용, 3배 속도)")
    print("W / S: ID22 위/아래")
    print("Q / E: ID23 위/아래")
    print("마우스 앞/뒤 이동: ID24 밀기/당기기")
    print("좌클릭 / 우클릭 홀드: ID25")
    print("X 키: 녹화 시작/중지 (10Hz, dataset/<Task>/<시간>/)")
    print("종료하려면 터미널에서 Ctrl+C 또는 카메라 창 클릭 후 'p'를 누르세요.\n")

    for idx, x in enumerate((20, 700, 1380), start=1):
        cv2.namedWindow(f'Camera {idx}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Camera {idx}', CAMERA_WIDTH, CAMERA_HEIGHT)
        cv2.moveWindow(f'Camera {idx}', x, 40)

    km = KeyboardMouseState()
    km.start()

    frame_warned = set()
    goal_cache = {}
    last_goal_reports = {}
    last_control_metric_ts = 0.0
    last_camera_metric_ts = 0.0
    camera_stats = {
        "Camera 1": {"ok": 0, "fail": 0},
        "Camera 2": {"ok": 0, "fail": 0},
        "Camera 3": {"ok": 0, "fail": 0},
    }

    def read_camera_frame(cap, label):
        stats = camera_stats[label]
        if not cap:
            stats["fail"] += 1
            return None
        for _ in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                stats["ok"] += 1
                return frame
            time.sleep(0.005)
        stats["fail"] += 1
        if label not in frame_warned:
            frame_warned.add(label)
            print(f"\n⚠️ {label} 프레임을 읽지 못했습니다.")
        return None

    def format_goal_report(report):
        if not report:
            return "no-command"
        parts = [
            f"reason={report.get('reason', '-')}",
            f"pos={report.get('current', '-')}",
            f"base={report.get('base', '-')}",
            f"raw={report.get('raw_goal', '-')}",
            f"goal={report.get('goal', '-')}",
            f"comm={report.get('comm', '-')}",
            f"err={report.get('dxl_error', '-')}",
        ]
        if report.get("exception"):
            parts.append(f"exception={report['exception']}")
        return " ".join(parts)

    def cap_opened(cap):
        return bool(cap is not None and cap.isOpened())

    try:
        while True:
            km.pump()
            if km.consume_quit_edge():
                print("\n\n종료 명령(p) 감지!")
                break

            current_pos = read_current_positions()

            rec_tag = f"녹화중 ts={recorder.last_timestep}" if recorder.recording else "대기"
            active_inputs = []
            if km.a_down: active_inputs.append("A")
            if km.d_down: active_inputs.append("D")
            if km.w_down: active_inputs.append("W")
            if km.s_down: active_inputs.append("S")
            if km.q_down: active_inputs.append("Q")
            if km.e_down: active_inputs.append("E")
            if km.left_down: active_inputs.append("LClick")
            if km.right_down: active_inputs.append("RClick")
            input_tag = ",".join(active_inputs) if active_inputs else "-"
            print(f"📊 [관절값] ID11:{current_pos.get(11,-1):4d} | ID22:{current_pos.get(22,-1):4d} | ID23:{current_pos.get(23,-1):4d} | ID24:{current_pos.get(24,-1):4d} | ID25:{current_pos.get(25,-1):4d} | 🎮 {input_tag} | 💾 {rec_tag}", end='\r')

            def send_goal_delta_safe(dxl_id: int, delta_ticks: int) -> dict:
                current = current_pos.get(dxl_id, -1)
                report = {
                    "id": dxl_id,
                    "delta": delta_ticks,
                    "current": current,
                    "base": "-",
                    "raw_goal": "-",
                    "goal": "-",
                    "comm": "-",
                    "dxl_error": "-",
                    "reason": "skip",
                }
                if delta_ticks == 0:
                    report["reason"] = "zero_delta"
                    last_goal_reports[dxl_id] = report
                    return report
                if current == -1:
                    report["reason"] = "present_position_read_fail"
                    last_goal_reports[dxl_id] = report
                    return report
                base = goal_cache.get(dxl_id, current_pos[dxl_id])
                raw_goal = int(base + delta_ticks)
                goal = clamp(raw_goal, LIMITS[dxl_id][0], LIMITS[dxl_id][1])
                report["base"] = base
                report["raw_goal"] = raw_goal
                report["goal"] = goal
                goal_cache[dxl_id] = goal
                if goal == base:
                    report["reason"] = "limit_clamped_no_move"
                    last_goal_reports[dxl_id] = report
                    return report
                try:
                    comm, dxl_error = packetHandler.write4ByteTxRx(
                        portHandler, dxl_id, ADDR_GOAL_POSITION, goal
                    )
                    report["comm"] = comm
                    report["dxl_error"] = dxl_error
                    report["reason"] = "sent" if comm == COMM_SUCCESS and dxl_error == 0 else "write_error"
                except Exception as exc:
                    report["reason"] = "write_exception"
                    report["exception"] = str(exc)
                last_goal_reports[dxl_id] = report
                return report

            # 1) 키보드/마우스 이동: ID11 / ID22 / ID23 / ID24
            _, mdy = km.consume_mouse_delta()
            km.consume_wheel()

            base_delta = (int(km.a_down) - int(km.d_down)) * BASE_TICKS_PER_KEY_FRAME
            if base_delta != 0:
                send_goal_delta_safe(11, base_delta)

            if mdy != 0:
                send_goal_delta_safe(24, int((-mdy) * WRIST_TICKS_PER_PX))

            shoulder_delta = (int(km.w_down) - int(km.s_down)) * SHOULDER_TICKS_PER_KEY_FRAME
            if shoulder_delta != 0:
                send_goal_delta_safe(22, shoulder_delta)

            elbow_delta = (int(km.q_down) - int(km.e_down)) * ELBOW_TICKS_PER_KEY_FRAME
            if elbow_delta != 0:
                send_goal_delta_safe(23, elbow_delta)

            # 2) 클릭 홀드: ID25
            if km.left_down:
                send_goal_delta_safe(25, -GRIPPER_STEP)
            elif km.right_down:
                send_goal_delta_safe(25, +GRIPPER_STEP)

            # 카메라 프레임 읽기
            frame1 = frame2 = frame3 = None
            frame1 = read_camera_frame(cap1, "Camera 1")
            frame2 = read_camera_frame(cap2, "Camera 2")
            frame3 = read_camera_frame(cap3, "Camera 3")

            if frame1 is not None: cv2.imshow('Camera 1', frame1)
            if frame2 is not None: cv2.imshow('Camera 2', frame2)
            if frame3 is not None: cv2.imshow('Camera 3', frame3)

            if DEBUG_METRICS:
                now = time.monotonic()
                if now - last_control_metric_ts >= CONTROL_METRIC_INTERVAL:
                    last_control_metric_ts = now
                    id22_report = last_goal_reports.get(22)
                    print(
                        "\n[METRIC-WS] "
                        f"w={int(km.w_down)} s={int(km.s_down)} "
                        f"id22_pos={current_pos.get(22, -1)} "
                        f"delta={shoulder_delta} "
                        f"{format_goal_report(id22_report)} "
                        f"input=({km.debug_snapshot()})"
                    )
                if now - last_camera_metric_ts >= CAMERA_METRIC_INTERVAL:
                    last_camera_metric_ts = now
                    print(
                        "\n[METRIC-CAM] "
                        f"C1 opened={int(cap_opened(cap1))} ok={camera_stats['Camera 1']['ok']} fail={camera_stats['Camera 1']['fail']} | "
                        f"C2 opened={int(cap_opened(cap2))} ok={camera_stats['Camera 2']['ok']} fail={camera_stats['Camera 2']['fail']} | "
                        f"C3 opened={int(cap_opened(cap3))} ok={camera_stats['Camera 3']['ok']} fail={camera_stats['Camera 3']['fail']}"
                    )

            recorder.maybe_record_frame((frame1, frame2, frame3), current_pos)

            if km.consume_save_toggle_edge():
                recorder.toggle_recording()

            key = cv2.waitKey(1) & 0xFF
            km.handle_cv2_key(key, source="cv2")
            if km.consume_quit_edge():
                print("\n\n종료 명령(p) 감지!")
                break

            time.sleep(SLEEP_DT)

    except KeyboardInterrupt:
        print("\n\n강제 종료 감지!")
    finally:
        recorder.finalize()
        try:
            km.stop()
        except Exception:
            pass
        for cap in caps:
            if cap is not None and cap.isOpened():
                cap.release()
        cv2.destroyAllWindows()
        close_robot()


if __name__ == '__main__':
    main()
