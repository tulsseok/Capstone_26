#!/usr/bin/env python3
"""SmolVLA local client -- the process that runs ON THE ROBOT PC.

Cameras -> JPEG -> HTTP /predict -> tick chunk -> (optional 1xK FORWARD Gaussian blur along
time on the ARM joints only, --smooth-actions: row t is averaged with t+1..t+K-1, peak weight
on t, so it never lags) -> safety -> Dynamixel, at 10 Hz, plus a full
per-step session recording so a rollout can be re-analysed afterwards instead of being judged
from a terminal scrollback.

Derived from `gr00t_client_local.py` (which stays untouched and still drives the GR00T server).
Everything below that is NOT in that file is here because it was measured to matter:

1. CAMERAS ARE BOUND BY /dev/v4l/by-id, NOT BY INDEX.
   `/dev/videoN` numbering changes across USB re-enumeration, and SmolVLA has no per-camera
   embedding -- swapping camera1 and camera2 produces no error, just a worse policy. The three
   slots are resolved from stable identifiers:
       camera1 (right) = Logitech C270, USB serial 200901010001   MJPG 320x240@10
       camera2 (left)  = the OTHER C270 ("C270 HD WEBCAM")        MJPG 320x240@10
       camera3 (top)   = Jieli USB Composite Device               MJPG 480x320@30
   Unresolved slots are a hard error with the full device listing printed. `--cam-devices`
   accepts indices or paths for a manual override.

2. THE FRAME PIPELINE MATCHES THE RECORDER EXACTLY.
   `dex_ref/src_control/camera_setup.py:152-153` resizes any frame whose size is not 320x240
   with `cv2.resize(..., INTER_AREA)` -- a NON-UNIFORM resize, not a crop. camera3 is 480x320
   (aspect 1.5 vs 1.333), so its training frames are horizontally squeezed. This client applies
   the identical resize, because the policy only ever saw squeezed camera3 frames. Same MJPG
   fourcc and CAP_PROP_BUFFERSIZE=1 as the recorder.

3. DEFAULTS COME FROM MEASUREMENTS, NOT FROM CAUTION.
   `--max-step-ticks 150`: `vla/log.md`'s only real rollout executed 178 steps of which 168
   (94.4%) were rate-clipped at 25 ticks -- the motion in that run was the limiter's, not the
   policy's. The requested move per step there had median 66, p90 150, max 412 ticks; 150 is that
   p90. It is a limit derived from a LOG, not from a hardware spec: confirm it on the real arm.
   `--step-dt-ms 100` = the dataset's 10 Hz. `--exec-steps 8` = the checkpoint's n_action_steps.
   Gripper ticks are bounded to [1000,2700] (confirmed against the h5); the four arm joints are
   left at [0,4095] because their real travel is NOT confirmed anywhere on this machine.

4. EVERY ROLLOUT IS RECORDED.  `rollout_<timestamp>/` gets steps.jsonl (present / requested /
   post-clip / actually-reached ticks, latencies, flags), chunks.jsonl (+ .npy) with the server's
   response meta, summary.json (clip rates, per-channel move distribution, tracking error, RTT)
   and, with --label, a human success/failure label so rollouts can be counted later.

5. TERMINATION IS A DECISION, NOT A SIDE EFFECT (--stop-on-done auto|on|off).
   A 6-D checkpoint (5 motors + done_flag, the runs/d6_* variants) makes the server answer
   /predict with `done`, `done_mean`, `done_streak`; a 5-D one (runs/d5_G and everything
   before it) has no such channel and the field stays None. The threshold and the
   consecutive-hit count are the SERVER's (--done-threshold / --done-consecutive) -- this
   client only obeys the boolean, so the two processes cannot disagree about what "done"
   means. 'auto' (default) stops on done=true and does nothing on a 5-D checkpoint, 'on'
   also refuses to run against a 5-D one, 'off' never stops early. The chunk that carries
   done is recorded but NOT executed: it is already ~--done-consecutive requests late, and
   the arm holds its last commanded pose. done_mean is printed and recorded in every mode,
   so a rollout run with 'off' can still be used to pick a threshold afterwards.

6. THE ROLLOUT STARTS WHERE THE RECORDINGS ENDED, AND YOU CAN WATCH IT.
   Before step 0 the arm is driven to `control_gamepad.py`'s default pose via that file's own
   `src_control.home_return.HomeReturn` (gripper open -> arm home -> gripper fully closed ->
   hold), so step 0 is inside the training distribution instead of wherever the arm happened
   to be left. `--no-home-start` skips it; `--home-pose` points at another measurement.
   The three camera slots are also shown live in cv2 windows, like the recorder's
   (`--no-show-cams` to suppress, `--cam-preview-scale` to enlarge). The windows display the
   frames that were ACTUALLY SENT, so they refresh once per chunk -- re-reading the cameras
   mid-chunk would block on a 10 fps capture and blow the step budget. 'q' in a window stops
   the rollout cleanly.

SAFETY
  * `--dry-run` is ON by default; `--no-dry-run` is required to move a motor.
  * gate 1 = absolute per-joint tick window, gate 2 = per-step |dtick| budget. Both counted.
  * the server's `execute=false` (its own --dry-run) is honoured -- writes are suppressed.
  * Ctrl-C / exception -> torque off, port closed, summary still written.

DEPENDENCIES (robot PC): numpy required; opencv-python for cameras; pillow or cv2 for JPEG;
requests optional (urllib fallback); dynamixel-sdk required only for --no-dry-run. lerobot is
NOT required -- the motor order and tick limits are imported from `openmani4_follower` when it
is importable and mirrored locally (and cross-checked) when it is not.

HTTP contract (identical to gr00t_server_openmani.py / dex_ref/client_live.py)
  POST /init    {"lang": str, "reasoning_cache_n": int}
  POST /predict {"images": [b64 JPEG xN], "qpos": [5 ticks]}
             -> {"joint_ticks_b64", "chunk_size", "joint_dim", "preprocess_ms", "inference_ms",
                 "postprocess_ms", + SmolVLA meta: execute, n_action_steps, max_step_ticks_*,
                 + on a 6-D checkpoint: done, done_score, done_mean, done_streak, ...}

Examples
    python smolvla_client_local.py --offline-self-test            # no server, no robot
    python smolvla_client_local.py --list-cameras                 # what is plugged in
    python smolvla_client_local.py --self-test --dummy-cam --url http://SERVER:8000
    python smolvla_client_local.py --url http://SERVER:8000 \
        --lang "place the tennis ball on the left" --out-dir runs/rollouts --label
    # ... watch the dry run, then:
    python smolvla_client_local.py --url http://SERVER:8000 --lang "..." --no-dry-run
    # 6-D (done_flag) checkpoint, but keep running past done to see how the flag behaves:
    python smolvla_client_local.py --url http://SERVER:8000 --lang "..." --stop-on-done off
"""

import argparse
import base64
import glob
import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---- optional deps: degrade loudly, never silently ---------------------------------------
try:
    import cv2

    _CV2_OK, _CV2_ERR = True, None
except Exception as _e:  # pragma: no cover - env dependent
    cv2, _CV2_OK, _CV2_ERR = None, False, _e

try:
    from PIL import Image

    _PIL_OK, _PIL_ERR = True, None
except Exception as _e:  # pragma: no cover
    Image, _PIL_OK, _PIL_ERR = None, False, _e

try:
    import requests

    _REQ_OK, _REQ_ERR = True, None
except Exception as _e:  # pragma: no cover
    requests, _REQ_OK, _REQ_ERR = None, False, _e

try:
    from dynamixel_sdk import (
        COMM_SUCCESS,
        GroupSyncRead,
        GroupSyncWrite,
        PacketHandler,
        PortHandler,
    )

    _DXL_OK, _DXL_ERR = True, None
except Exception as _e:  # pragma: no cover
    _DXL_OK, _DXL_ERR = False, _e

# The teleop recorder's homing state machine, reused verbatim so this client starts every
# rollout in the SAME pose `control_gamepad.py` leaves the arm in at the end of every recorded
# episode. Duplicating the sequence here would let the two drift apart silently, which is the
# one thing that would quietly move the start state out of the training distribution.
# It pulls in `src_control.robot_constants` only (no dynamixel_sdk), so it is safe to import
# even in --dry-run on a machine with no SDK.
try:
    from src_control.home_return import (  # noqa: E402
        PHASE_LABELS as _HOME_PHASE_LABELS,
        HomeReturn,
        default_home_pose_path,
        load_home_pose,
    )

    _HOME_OK, _HOME_ERR = True, None
except Exception as _e:  # pragma: no cover
    HomeReturn = load_home_pose = default_home_pose_path = None
    _HOME_PHASE_LABELS = {}
    _HOME_OK, _HOME_ERR = False, _e


# ============================================================================
# Hardware constants.
# `openmani4_follower` is the single source of truth, but importing it pulls in lerobot, which
# is NOT installed on the robot PC. So: mirror the values, import when possible, and CROSS-CHECK
# (a silent divergence between the deployment limits and the robot class would be exactly the
# kind of bug this file exists to prevent).
# ============================================================================
ASSUMED_DXL_IDS = (11, 22, 23, 24, 25)   # policy channel order == dataset state/action order
ASSUMED_PORT = "/dev/ttyUSB0"
ASSUMED_BAUD = 1_000_000
ASSUMED_PROTOCOL = 2.0                   # X-series
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4

JOINT_DIM = 5
TICK_MIN, TICK_MAX = 0, 4095
GRIPPER_TICK_MIN, GRIPPER_TICK_MAX = 1000, 2700   # confirmed against the h5
GRIPPER_SLOT = 4                                  # id25 is the last channel
MOTOR_LABELS = ["id11", "id22", "id23", "id24", "id25/grip"]
DEFAULT_CAMS = 3
_CONST_SOURCE = "local mirror (openmani4_follower not importable)"

try:  # only works where lerobot is installed (this dev machine, not the robot PC)
    from openmani4_follower import (  # noqa: E402
        GRIPPER_TICK_MAX as _G_HI,
        GRIPPER_TICK_MIN as _G_LO,
        MOTOR_IDS as _MOTOR_IDS,
        TICK_MAX as _T_HI,
        TICK_MIN as _T_LO,
    )

    _mismatch = []
    if tuple(_MOTOR_IDS.values()) != ASSUMED_DXL_IDS:
        _mismatch.append(f"motor ids {tuple(_MOTOR_IDS.values())} != {ASSUMED_DXL_IDS}")
    if (_T_LO, _T_HI) != (TICK_MIN, TICK_MAX):
        _mismatch.append(f"arm ticks ({_T_LO},{_T_HI}) != ({TICK_MIN},{TICK_MAX})")
    if (_G_LO, _G_HI) != (GRIPPER_TICK_MIN, GRIPPER_TICK_MAX):
        _mismatch.append(f"gripper ticks ({_G_LO},{_G_HI}) != "
                         f"({GRIPPER_TICK_MIN},{GRIPPER_TICK_MAX})")
    if _mismatch:
        raise SystemExit("openmani4_follower disagrees with this file's mirrored constants: "
                         + "; ".join(_mismatch) + ". Fix the mirror before running the robot.")
    _CONST_SOURCE = "openmani4_follower (cross-checked against the local mirror: identical)"
except SystemExit:
    raise
except Exception:
    pass


# ============================================================================
# Camera identity -- MEASURED rig spec (user-confirmed 2026-08-16).
# ============================================================================

class _CamGrabber(threading.Thread):
    """Keeps reading one capture so the newest frame is always in hand.

    Without this the loop pays for its own staleness. A V4L2 queue hands back the frame that
    was current at the LAST read, so a synchronous reader has to throw one frame away and then
    block a full capture period -- ~200 ms per chunk, measured on the robot PC 2026-09-20.
    A thread that never stops reading keeps the queue empty, so `latest()` returns at once with
    a frame at most one capture period old.
    """

    MAX_CONSECUTIVE_MISSES = 30

    def __init__(self, cap, slot: int):
        super().__init__(daemon=True, name=f"cam-grab{slot}")
        self._cap = cap
        self.slot = slot
        self._lock = threading.Lock()
        self._frame = None
        self._stamp_ms = -1.0
        self._seq = 0
        self._err: str | None = None
        self._stop_evt = threading.Event()

    def run(self) -> None:
        misses = 0
        while not self._stop_evt.is_set():
            try:
                ok, bgr = self._cap.read()
            except Exception as e:                      # a yanked USB cable raises here
                with self._lock:
                    self._err = f"slot{self.slot}: capture raised {type(e).__name__}: {e}"
                return
            if not ok or bgr is None:
                misses += 1
                if misses >= self.MAX_CONSECUTIVE_MISSES:
                    with self._lock:
                        self._err = (f"slot{self.slot}: {misses} consecutive read failures "
                                     f"-- the camera stopped streaming")
                    return
                time.sleep(0.005)
                continue
            misses = 0
            try:
                ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            except Exception:
                ts = 0
            with self._lock:
                self._frame = bgr
                self._stamp_ms = float(ts) if ts and ts > 0 else -1.0
                self._seq += 1

    def latest(self, timeout_s: float = 3.0):
        """(frame, age_ms, seq). Blocks only while the very first frame is still in flight."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._err:
                    raise RuntimeError(self._err)
                if self._frame is not None:
                    age = (time.monotonic() * 1000.0 - self._stamp_ms
                           if self._stamp_ms > 0 else -1.0)
                    return self._frame, age, self._seq
            if time.monotonic() > deadline:
                raise RuntimeError(f"camera slot{self.slot}: no frame within {timeout_s:.1f}s")
            time.sleep(0.002)

    def stop(self, join_s: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive():
            self.join(join_s)


class CamSpec:
    """One client image slot: how to recognise the device and how to open it."""

    def __init__(self, slot, key, role, width, height, fps, serial=None, tokens=(),
                 fallback_tokens=()):
        self.slot = slot
        self.key = key            # observation.images.<key> on the server side
        self.role = role
        self.width = width        # what the DRIVER is asked for (native capture format)
        self.height = height
        self.fps = fps
        self.serial = serial
        self.tokens = tuple(tokens)                    # strong identifiers
        self.fallback_tokens = tuple(fallback_tokens)  # weaker (model family)

    def __repr__(self):
        return f"<CamSpec slot{self.slot} {self.key}/{self.role} {self.width}x{self.height}@{self.fps}>"


# The recorder normalises EVERY frame to 320x240 (camera_setup.CAMERA_WIDTH/HEIGHT).
SEND_W, SEND_H = 320, 240

# Upper bound on reads spent dropping frames the V4L2 queue held while the loop was busy.
MAX_DRAIN_READS = 6

CAM_SPECS = [
    CamSpec(0, "camera1", "right", 320, 240, 10, serial="200901010001",
            fallback_tokens=("c270", "046d", "logi")),
    CamSpec(1, "camera2", "left", 320, 240, 10,
            fallback_tokens=("c270", "046d", "logi", "webcam")),
    CamSpec(2, "camera3", "top", 480, 320, 30,
            tokens=("jieli", "usb_composite", "usb composite", "hcam")),
]


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _video_index(path: str):
    m = re.search(r"video(\d+)$", path)
    return int(m.group(1)) if m else None


def enumerate_video_devices() -> list[dict]:
    """Every V4L2 CAPTURE node with a stable name, preferring /dev/v4l/by-id.

    Only `index == 0` nodes are kept: a UVC camera claims a capture node and a metadata node,
    and the metadata node opens fine but never yields a frame (same rule as
    camera_setup._is_capture_video_node).

    by-id alone is not enough. Logitech C270s ship a hardcoded serial (200901010001), so two
    of them collide on one by-id name and udev keeps a single symlink -- the second camera then
    has NO by-id entry and would be invisible here. /dev/v4l/by-path is per-USB-port and never
    collides, so it is scanned as a fallback for any capture node by-id missed. Observed on the
    robot PC 2026-08-16: video0 (by-id, port -1) + video4 (by-path only, port -2.3).
    """
    out = []
    seen = set()

    def _add(link: str, source: str) -> None:
        real = os.path.realpath(link)
        idx = _video_index(real)
        if idx is None or idx in seen:
            return
        name = os.path.basename(link)
        if "video-index1" in name.lower():
            return
        sys_index = _read_text(f"/sys/class/video4linux/video{idx}/index")
        if sys_index not in ("", "0"):
            return
        seen.add(idx)
        out.append({
            "by_id": link,
            "dev": real,
            "index": idx,
            "by_id_name": name,
            "sys_name": _read_text(f"/sys/class/video4linux/video{idx}/name"),
            "id_source": source,
        })

    for p in sorted(glob.glob("/dev/v4l/by-id/*")):
        _add(p, "by-id")
    for p in sorted(glob.glob("/dev/v4l/by-path/*")):
        _add(p, "by-path")
    return out


def _haystack(entry: dict) -> str:
    return f"{entry.get('by_id_name', '')} {entry.get('sys_name', '')}".lower().replace("_", " ")


def assign_cameras(entries: list[dict], specs=CAM_SPECS) -> tuple[list[dict], list[str]]:
    """Resolve each slot to a device. Returns (assigned per slot, notes). Raises when ambiguous.

    Order matters: the strongest identifier wins first (serial, then the Jieli token), and
    camera2 is whatever C270 is left over. That is the only slot that cannot be identified
    positively, so it is assigned last and only when exactly one candidate remains.
    """
    notes = []
    assigned: list[dict | None] = [None] * len(specs)
    used = set()

    def take(i, e, why):
        assigned[i] = dict(e, matched_by=why)
        used.add(e["index"])
        notes.append(f"slot{i}/{specs[i].key} <- /dev/video{e['index']} ({why}) {e['by_id_name']}")

    for i, sp in enumerate(specs):          # 1. serial numbers
        if not sp.serial:
            continue
        hits = [e for e in entries if sp.serial.lower() in _haystack(e) and e["index"] not in used]
        if len(hits) > 1:
            raise RuntimeError(f"{sp.key}: serial {sp.serial} matches {len(hits)} devices "
                               f"{[h['by_id_name'] for h in hits]}")
        if hits:
            take(i, hits[0], f"serial {sp.serial}")

    for i, sp in enumerate(specs):          # 2. strong model tokens (Jieli)
        if assigned[i] is not None or not sp.tokens:
            continue
        hits = [e for e in entries
                if any(t in _haystack(e) for t in sp.tokens) and e["index"] not in used]
        if len(hits) > 1:
            raise RuntimeError(f"{sp.key}: {sp.tokens} matches {len(hits)} devices "
                               f"{[h['by_id_name'] for h in hits]}")
        if hits:
            take(i, hits[0], f"token {[t for t in sp.tokens if t in _haystack(hits[0])][0]!r}")

    for i, sp in enumerate(specs):          # 3. leftover, by model family
        if assigned[i] is not None:
            continue
        hits = [e for e in entries
                if e["index"] not in used
                and (not sp.fallback_tokens or any(t in _haystack(e) for t in sp.fallback_tokens))]
        if len(hits) == 1:
            take(i, hits[0], "only remaining device of its model family")
        elif len(hits) > 1:
            raise RuntimeError(
                f"{sp.key} is ambiguous: {len(hits)} unassigned devices match "
                f"{sp.fallback_tokens} -> {[h['by_id_name'] for h in hits]}. This slot has no "
                f"positive identifier, so it cannot be guessed. Pass --cam-devices explicitly.")

    missing = [specs[i].key for i, a in enumerate(assigned) if a is None]
    if missing:
        listing = "\n".join(f"    /dev/video{e['index']:<3d} {e['by_id_name']}  [{e['sys_name']}]"
                            for e in entries) or "    (no /dev/v4l/by-id capture nodes at all)"
        raise RuntimeError(
            f"could not resolve camera slot(s) {missing}.\n  devices seen:\n{listing}\n"
            f"  Expected: a C270 with serial 200901010001 (camera1), a second C270 (camera2) and "
            f"a Jieli USB Composite Device (camera3).\n"
            f"  Override with --cam-devices '<cam1>,<cam2>,<cam3>' (indices or /dev paths).")
    return assigned, notes


def resolve_manual(tokens: list[str], entries: list[dict]) -> list[dict]:
    """--cam-devices override: each token is an index, a /dev/videoN, or a by-id path/substring."""
    by_index = {e["index"]: e for e in entries}
    out = []
    for tok in tokens:
        tok = tok.strip()
        e = None
        if re.fullmatch(r"\d+", tok):
            idx = int(tok)
            e = by_index.get(idx, {"index": idx, "dev": f"/dev/video{idx}",
                                   "by_id": "", "by_id_name": "(manual index)",
                                   "sys_name": _read_text(f"/sys/class/video4linux/video{idx}/name")})
        elif tok.startswith("/dev/"):
            real = os.path.realpath(tok)
            idx = _video_index(real)
            if idx is None:
                raise RuntimeError(f"--cam-devices '{tok}' does not resolve to a /dev/videoN node")
            e = by_index.get(idx, {"index": idx, "dev": real, "by_id": tok,
                                   "by_id_name": os.path.basename(tok),
                                   "sys_name": _read_text(f"/sys/class/video4linux/video{idx}/name")})
        else:
            hits = [x for x in entries if tok.lower() in _haystack(x)]
            if len(hits) != 1:
                raise RuntimeError(f"--cam-devices '{tok}' matched {len(hits)} devices; be specific")
            e = hits[0]
        out.append(dict(e, matched_by="manual --cam-devices"))
    return out


# ============================================================================
# Pure safety / payload logic (covered by --offline-self-test).
# ============================================================================

def default_joint_limits() -> str:
    return ",".join(
        f"{GRIPPER_TICK_MIN}:{GRIPPER_TICK_MAX}" if i == GRIPPER_SLOT else f"{TICK_MIN}:{TICK_MAX}"
        for i in range(JOINT_DIM))


def parse_joint_limits(spec: str, n: int = JOINT_DIM) -> tuple[np.ndarray, np.ndarray]:
    """'0:4095,...' -> (lo[n], hi[n]) int32. A single 'lo:hi' applies to all joints."""
    parts = [p for p in spec.split(",") if p.strip() != ""]
    if len(parts) == 1:
        parts = parts * n
    if len(parts) != n:
        raise ValueError(f"--joint-limits needs 1 or {n} 'lo:hi' entries, got {len(parts)}: {spec}")
    lo, hi = [], []
    for p in parts:
        a, sep, b = p.partition(":")
        if not sep:
            raise ValueError(f"--joint-limits entry '{p}' is not 'lo:hi'")
        a, b = int(a), int(b)
        if a >= b:
            raise ValueError(f"--joint-limits entry '{p}' has lo >= hi")
        if a < TICK_MIN or b > TICK_MAX:
            raise ValueError(f"--joint-limits entry '{p}' outside [{TICK_MIN},{TICK_MAX}]")
        lo.append(a)
        hi.append(b)
    return np.array(lo, dtype=np.int32), np.array(hi, dtype=np.int32)


def parse_max_step(spec: str, n: int = JOINT_DIM) -> np.ndarray:
    """'150' or '150,150,150,150,200' -> int32[n] per-step tick budget."""
    parts = [p for p in spec.split(",") if p.strip() != ""]
    if len(parts) == 1:
        parts = parts * n
    if len(parts) != n:
        raise ValueError(f"--max-step-ticks needs 1 or {n} entries, got {len(parts)}: {spec}")
    v = np.array([int(p) for p in parts], dtype=np.int32)
    if np.any(v <= 0):
        raise ValueError(f"--max-step-ticks must be positive, got {v.tolist()}")
    return v


def clip_limits(ticks, lo, hi) -> np.ndarray:
    """Gate 1: absolute per-joint tick window."""
    return np.clip(np.asarray(ticks, dtype=np.int64), lo, hi).astype(np.int32)


def _ftdi_latency_ms(port: str):
    """The serial adapter's latency timer in ms, or None when it is not an FTDI/usb-serial."""
    name = Path(port).name
    txt = _read_text(f"/sys/bus/usb-serial/devices/{name}/latency_timer")
    try:
        return int(txt)
    except (TypeError, ValueError):
        return None


def substep_goals(prev_cmd, cmd, n_sub: int) -> list[np.ndarray]:
    """Split one control step's move into `n_sub` evenly spaced goal writes.

    The servos run with Profile Velocity = 0 (verified on the robot PC 2026-09-22: register
    112 reads 0 on all five ids), i.e. NO internal trajectory generation -- every goal write
    is a dash at maximum speed. One 100 ms step therefore arrives as a rush followed by a
    wait, which is the visible stutter. `HomeReturn` looks smooth for exactly this reason: it
    writes ~8 ticks every 20 ms instead of ~42 ticks every 100 ms, at the SAME average
    velocity (400 vs 420 ticks/s, measured over 822 steps of rollout_20260922_184848).

    The endpoint is untouched -- goal `n_sub` is `cmd` exactly -- so the pose at each 100 ms
    boundary, and with it the 10 Hz distribution the policy was trained on, does not change.
    Per-write displacement is bounded by |cmd - prev_cmd| / n_sub, so --max-step-ticks needs
    no scaling: the split divides the already-limited move.
    """
    n_sub = max(1, int(n_sub))
    tgt = np.asarray(cmd, dtype=np.int32)
    if n_sub == 1:
        return [tgt]
    base = np.asarray(prev_cmd, dtype=np.float64)
    span = np.asarray(cmd, dtype=np.float64) - base
    out = [np.rint(base + span * (k / n_sub)).astype(np.int32) for k in range(1, n_sub)]
    out.append(tgt)             # land on cmd exactly, never on a rounded approximation
    return out


def rate_limit(prev_cmd, target, max_step) -> np.ndarray:
    """Gate 2: bound |target - prev_cmd| per joint per control step.

    Matches `Openmani4Follower._apply_safety` (clamp first, then rate-limit against the previous
    COMMAND, seeded from the measured present position). It deliberately LAGS the policy on a big
    jump -- that lag is the safety. Every clipped step is counted and written to steps.jsonl.
    """
    prev = np.asarray(prev_cmd, dtype=np.int64)
    tgt = np.asarray(target, dtype=np.int64)
    ms = np.asarray(max_step, dtype=np.int64)
    return (prev + np.clip(tgt - prev, -ms, ms)).astype(np.int32)


def gaussian_kernel_1d(ksize: int, sigma: float = 0.0) -> np.ndarray:
    """Normalised 1-D Gaussian. sigma<=0 -> the cv2.getGaussianKernel auto rule.

    ksize=3, sigma=0 gives sigma=0.8 -> weights [0.239, 0.522, 0.239]: a light blur that
    keeps most of the original row.
    """
    k = int(ksize)
    if k < 3 or k % 2 == 0:
        raise ValueError(f"--smooth-kernel must be an ODD integer >= 3, got {ksize}")
    s = float(sigma)
    if s <= 0:
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8      # == cv2.getGaussianKernel(k, 0)
    x = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    w = np.exp(-(x ** 2) / (2.0 * s * s))
    return w / w.sum()


def forward_taps(w: np.ndarray) -> np.ndarray:
    """대칭 가우시안 w를 FORWARD(선행) 창 t..t+K-1 용으로 재배치한다.

    반환 f[i] 는 a[t+i] 에 곱해질 가중치다. 최대 가중치(중심 탭)는 t 에 그대로 남고,
    나머지 탭은 가까운 쌍부터 순서대로 미래 쪽으로 펼쳐진다:

        ksize=3 -> f = [0.522, 0.239, 0.239]   (t, t+1, t+2)
        ksize=5 -> f = [0.383, 0.242, 0.242, 0.061, 0.061]

    가우시안이 대칭이므로 w[pad+i] == w[pad-i] 이고, 그 결과 f 는 단조 감소한다.
    가중치 집합 자체는 기존 커널과 동일 -- 놓이는 위치만 미래로 바뀐 것이다.
    """
    pad = (w.size - 1) // 2
    order = [pad]
    for i in range(1, pad + 1):
        order += [pad + i, pad - i]
    return w[order]


def smooth_chunk(ticks: np.ndarray, ksize: int = 3, sigma: float = 0.0,
                 exclude: tuple = (GRIPPER_SLOT,)) -> np.ndarray:
    """Gaussian blur ALONG TIME (rows) of a [chunk_size, joint_dim] tick chunk, per joint.

    FORWARD(선행) 창을 쓴다. row t 는 t 자신과 그 뒤 K-1 스텝만 본다:

        기존:  out[t] = 0.239*a[t-1] + 0.522*a[t] + 0.239*a[t+1]      (창 중심 = t)
        현재:  out[t] = 0.522*a[t]   + 0.239*a[t+1] + 0.239*a[t+2]    (최대 가중치 = t)

    가중치는 같은 가우시안이고 위치만 미래로 옮겼다 (`forward_taps` 참조). 과거 샘플을
    전혀 쓰지 않으므로 위상 지연이 없고, 오히려 선형 구간에서는 다음 스텝 쪽으로 살짝
    당겨진다 (1x3, 100 tick/step 램프 기준 +72 tick). 지터는 여전히 눌린다.

    This smooths BETWEEN the action steps the server returned; it does NOT touch the joint
    axis. 창이 청크 끝을 넘어가는 마지막 K-1 행은 replicate padding 으로 마지막 행을
    반복해 채우므로 0 으로 끌려가지 않고, 상수 청크는 정확히 상수로 남는다
    (verified in --offline-self-test).

    `exclude` lists channels that are COPIED THROUGH UNCHANGED. It defaults to the gripper
    (GRIPPER_SLOT): a grasp is a near-step command whose whole point is the fast edge, and
    blurring it both rounds the closing edge and shrinks an already low-variance channel.
    The 4 arm joints are the ones whose jitter this is meant to take out.

    Caveat that remains: it only smooths INSIDE one chunk. The seam between chunk k's last
    executed row and chunk k+1's row0 is untouched -- that discontinuity is the rate limiter's
    job. 선행 창이라 청크 꼬리 쪽 K-1 행은 볼 미래가 없어 실질 블러가 약해진다.
    """
    a = np.asarray(ticks)
    w = gaussian_kernel_1d(ksize, sigma)
    if a.shape[0] < 2:
        return a.astype(np.int32)
    ncol = a.shape[1]
    skip = {int(j) + ncol if int(j) < 0 else int(j) for j in (exclude or ())}
    skip = {j for j in skip if 0 <= j < ncol}   # out-of-range index is ignored, NOT wrapped
    f = forward_taps(w)
    # 창이 t..t+K-1 이므로 패딩은 뒤쪽에만 K-1 행. 'valid' convolve 는 커널을 뒤집으므로
    # f[::-1] 을 넘겨 out[t] = sum_i f[i]*padded[t+i] 가 되게 한다.
    padded = np.pad(a.astype(np.float64), ((0, f.size - 1), (0, 0)), mode="edge")
    out = a.astype(np.float64).copy()
    for j in range(a.shape[1]):
        if j in skip:
            continue                                  # gripper: passed through untouched
        out[:, j] = np.convolve(padded[:, j], f[::-1], mode="valid")
    return np.rint(out).astype(np.int32)


def decode_chunk(b64: str, chunk_size: int, joint_dim: int) -> np.ndarray:
    """base64 -> int32 [chunk_size, joint_dim] (C-order). Mirrors the server exactly."""
    flat = np.frombuffer(base64.b64decode(b64), dtype=np.int32)
    expected = chunk_size * joint_dim
    if flat.size != expected:
        raise ValueError(
            f"joint_ticks_b64 decodes to {flat.size} int32 but chunk_size*joint_dim="
            f"{chunk_size}*{joint_dim}={expected}. Server/client dtype or order mismatch.")
    return flat.reshape(chunk_size, joint_dim)


def encode_jpeg_b64(rgb: np.ndarray, quality: int) -> str:
    """RGB uint8 (H,W,3) -> base64 JPEG. PIL preferred (== the recorder), cv2 fallback."""
    if rgb.dtype != np.uint8:
        raise TypeError(f"frame must be uint8, got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"frame must be (H,W,3) RGB, got {rgb.shape}")
    if _PIL_OK:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    if _CV2_OK:
        ok, enc = cv2.imencode(".jpg", rgb[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(enc.tobytes()).decode("ascii")
    raise RuntimeError(f"neither PIL ({_PIL_ERR}) nor cv2 ({_CV2_ERR}) is importable; "
                       f"cannot JPEG-encode frames")


def match_recorder_size(frame_bgr_or_rgb: np.ndarray) -> np.ndarray:
    """The recorder's rule, verbatim (camera_setup.read_camera_frame:151-154).

    `if width != 320 or height != 240: cv2.resize(frame, (320,240), INTER_AREA)`.
    Non-uniform: camera3's 480x320 (aspect 1.5) is squeezed to 1.333, which is what the policy
    was trained on. Doing anything else here -- a crop, a letterbox -- would put the deployment
    frames off the training distribution.
    """
    h, w = frame_bgr_or_rgb.shape[:2]
    if w == SEND_W and h == SEND_H:
        return frame_bgr_or_rgb
    if not _CV2_OK:
        raise RuntimeError(f"a {w}x{h} frame must be resized to {SEND_W}x{SEND_H} to match the "
                           f"recorder, but opencv is not importable ({_CV2_ERR})")
    return cv2.resize(frame_bgr_or_rgb, (SEND_W, SEND_H), interpolation=cv2.INTER_AREA)


# ============================================================================
# Cameras
# ============================================================================

class CameraRig:
    """N V4L2 captures bound by identity, or synthetic frames with --dummy-cam.

    Frames come back as RGB uint8 320x240 in SLOT order (slot i == the client's image_i, which
    the server maps onto camera{i+1} via its --cam-indices).
    """

    def __init__(self, n_slots: int = DEFAULT_CAMS, dummy: bool = False,
                 manual: list[str] | None = None, specs=CAM_SPECS, verbose: bool = True,
                 swap01: bool = False, threaded: bool = True):
        self.n_slots = int(n_slots)
        self.swap01 = bool(swap01)
        self.dummy = bool(dummy)
        self.specs = list(specs)[: self.n_slots]
        self.binding: list[dict] = []
        self._caps = []
        self._tick = 0
        self.last_ages_ms: list[float] = []
        self.last_discards: list[int] = []
        self.threaded = bool(threaded) and not dummy
        self._grabbers: list[_CamGrabber] = []
        if dummy:
            if verbose:
                print(f"[cam] --dummy-cam: {self.n_slots} synthetic {SEND_H}x{SEND_W} frames "
                      f"(NO real cameras, NO identity check).")
            self.binding = [{"index": -1, "by_id_name": "(dummy)", "sys_name": "(dummy)",
                             "matched_by": "dummy", "dev": "(dummy)"} for _ in range(self.n_slots)]
            return
        if not _CV2_OK:
            raise RuntimeError(f"opencv is not importable ({_CV2_ERR}); use --dummy-cam or "
                               f"install opencv-python on the robot host")
        entries = enumerate_video_devices()
        if manual:
            if len(manual) != self.n_slots:
                raise RuntimeError(f"--cam-devices needs {self.n_slots} entries, got {len(manual)}")
            self.binding = resolve_manual(manual, entries)
            notes = [f"slot{i} <- /dev/video{b['index']} (manual)"
                     for i, b in enumerate(self.binding)]
        else:
            self.binding, notes = assign_cameras(entries, self.specs)
        # --swap-cam01: 슬롯에 묶인 '장치'만 맞바꾼다. specs(키/역할/해상도)는 그대로다.
        # slot0/slot1 은 둘 다 320x240@10 이라 장치 교환이 안전하다.
        if self.swap01 and self.n_slots >= 2:
            self.binding[0], self.binding[1] = self.binding[1], self.binding[0]
            notes = [f"slot0 <- /dev/video{self.binding[0]['index']} (--swap-cam01)",
                     f"slot1 <- /dev/video{self.binding[1]['index']} (--swap-cam01)"] + notes[2:]
            notes.append("image_0 <-> image_1 SWAPPED by --swap-cam01")
        if verbose:
            for n in notes:
                print(f"[cam] {n}")
        self._open_all(verbose=verbose)
        if self.threaded:
            self._grabbers = [_CamGrabber(cap, i) for i, cap in enumerate(self._caps)]
            try:
                for g in self._grabbers:
                    g.start()
                for g in self._grabbers:
                    g.latest()      # surface a dead camera here, not mid-rollout
            except Exception:
                # A half-built rig would leave grabber threads spinning on captures nobody
                # will ever release, and the /dev/video nodes locked against the next run.
                self.close()
                raise
            if verbose:
                print(f"[cam] {len(self._grabbers)} background grabber(s) running: every read "
                      f"returns the newest frame with no capture wait (--no-cam-threads to "
                      f"go back to reading inline).")

    # -- open --------------------------------------------------------------
    def _open_all(self, verbose: bool = True):
        for i, (sp, b) in enumerate(zip(self.specs, self.binding)):
            cap = cv2.VideoCapture(b["index"], cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(b["index"])
            if not cap.isOpened():
                self.close()
                raise RuntimeError(f"slot{i}/{sp.key}: failed to open /dev/video{b['index']} "
                                   f"({b['by_id_name']})")
            # Same knobs, same order, as camera_setup._configure_camera.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, sp.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sp.height)
            cap.set(cv2.CAP_PROP_FPS, sp.fps)
            self._caps.append(cap)
            ok, frame = False, None
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.04)
            if not ok or frame is None:
                self.close()
                raise RuntimeError(f"slot{i}/{sp.key}: /dev/video{b['index']} opened but produced "
                                   f"no frame at MJPG {sp.width}x{sp.height}@{sp.fps}")
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps = float(cap.get(cv2.CAP_PROP_FPS))
            fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fcc_s = "".join(chr((fcc >> 8 * k) & 0xFF) for k in range(4)).strip("\x00") or "-"
            b.update({"actual_w": aw, "actual_h": ah, "actual_fps": afps, "fourcc": fcc_s,
                      "requested": f"MJPG {sp.width}x{sp.height}@{sp.fps}",
                      "first_frame_shape": list(frame.shape)})
            if verbose:
                warn = ""
                if (aw, ah) != (sp.width, sp.height):
                    warn = f"  <-- driver gave {aw}x{ah}, NOT the recorder's {sp.width}x{sp.height}"
                print(f"[cam] slot{i} {sp.key:<8s} ({sp.role:<5s}) /dev/video{b['index']:<3d} "
                      f"{b['sys_name'] or b['by_id_name']!r:<34s} req={sp.width}x{sp.height}@{sp.fps} "
                      f"actual={fcc_s} {aw}x{ah}@{afps:.0f}{warn}")
                if (aw, ah) != (SEND_W, SEND_H):
                    print(f"[cam]        -> resized to {SEND_W}x{SEND_H} with INTER_AREA, exactly "
                          f"like camera_setup.read_camera_frame (non-uniform for camera3)")

    # -- read --------------------------------------------------------------
    def frame_age_ms(self, cap) -> float:
        """How old the frame just read is, in ms. <0 when the driver has no timestamp.

        V4L2 reports CAP_PROP_POS_MSEC on the monotonic clock, so the age is a plain
        subtraction. Measured on the robot PC 2026-09-20: a fresh frame is 0-30 ms old.
        """
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ts or ts <= 0:
            return -1.0
        return time.monotonic() * 1000.0 - float(ts)

    def read(self, fresh: bool = False) -> list[np.ndarray]:
        """One frame per slot, in slot order (slot i == the client's image_i).

        fresh=True drops whatever the driver queued while the loop was busy. A V4L2 capture
        keeps handing back the frame that was current at the LAST read, so the first read
        after a gap is as old as the gap itself -- measured on the robot PC 2026-09-20:
        3.9 s stale after a 4 s pause, i.e. after the home start the policy would otherwise
        see the arm BEFORE it was homed. grab() does not advance the queue; only read() does.
        """
        if self.dummy:
            self._tick += 1
            rng = np.random.default_rng(1234 + self._tick)
            frames = []
            for i in range(self.n_slots):
                f = np.zeros((SEND_H, SEND_W, 3), dtype=np.uint8)
                f[:, :, i % 3] = 60
                x = (self._tick * 7 + i * 40) % max(1, SEND_W - 20)
                f[:, x:x + 20, :] = 200
                f = np.clip(f.astype(np.int16) + rng.integers(0, 16, f.shape), 0, 255).astype(np.uint8)
                frames.append(f)
            return frames
        if self.threaded:
            # The grabbers already keep the queue empty, so `fresh` costs nothing here.
            frames, self.last_ages_ms, self.last_discards = [], [], []
            for g in self._grabbers:
                bgr, age, _seq = g.latest()
                self.last_ages_ms.append(age)
                self.last_discards.append(0)
                frames.append(match_recorder_size(bgr)[:, :, ::-1].copy())
            return frames
        frames = []
        self.last_ages_ms = []
        self.last_discards = []
        for i, cap in enumerate(self._caps):
            # A frame older than two capture periods was queued while we were away.
            fps = float(self.binding[i].get("actual_fps") or 0.0) or float(self.specs[i].fps) or 10.0
            max_age_ms = 2000.0 / fps
            bgr, age, dropped = None, -1.0, 0
            for attempt in range(MAX_DRAIN_READS if fresh else 1):
                got = None
                for _ in range(3):     # camera_setup.read_camera_frame retries 3x
                    ok, got = cap.read()
                    if ok and got is not None:
                        break
                    time.sleep(0.005)
                    got = None
                if got is None:
                    break
                bgr, age = got, self.frame_age_ms(cap)
                if not fresh:
                    break
                if age < 0:                 # no driver timestamp: one blind discard, then take it
                    if attempt >= 1:
                        break
                    dropped += 1
                    continue
                if age <= max_age_ms:
                    break
                dropped += 1
            if bgr is None:
                raise RuntimeError(f"camera slot{i} (/dev/video{self.binding[i]['index']}) "
                                   f"read failed 3x")
            self.last_ages_ms.append(age)
            self.last_discards.append(dropped)
            bgr = match_recorder_size(bgr)
            frames.append(bgr[:, :, ::-1].copy())    # BGR -> RGB (server does PIL convert("RGB"))
        return frames

    def close(self):
        for g in self._grabbers:        # stop reading BEFORE releasing the capture
            try:
                g.stop()
            except Exception:
                pass
        self._grabbers = []
        for c in self._caps:
            try:
                c.release()
            except Exception:
                pass
        self._caps = []


# ============================================================================
# HTTP
# ============================================================================

class CameraPreview:
    """Live cv2 windows, one per camera slot, like `control_gamepad.py`'s.

    It displays THE FRAMES THAT WERE SENT to the server, not a second independent capture, so
    what is on screen is exactly what the policy saw. That also fixes the refresh rate: once
    per CHUNK (~--exec-steps x --step-dt-ms), not once per control step. Re-reading the
    cameras mid-chunk would block on a 10 fps capture and blow the 100 ms step budget, which
    is the one thing the preview must not do.

    Never fatal. A headless host (no DISPLAY) disables it with a single warning and the
    rollout continues untouched.
    """

    def __init__(self, labels, enabled: bool = True, scale: float = 1.0):
        self.labels = list(labels)
        self.scale = float(scale)
        self.enabled = bool(enabled) and _CV2_OK
        self.reason = ""
        if enabled and not _CV2_OK:
            self.reason = f"opencv is not importable ({_CV2_ERR})"
        self._opened = False

    def _disable(self, exc: Exception) -> None:
        self.enabled = False
        self.reason = f"{type(exc).__name__}: {exc}"
        print(f"[cam][preview] disabled -- {self.reason}\n"
              f"[cam][preview] (no DISPLAY? run with --no-show-cams, or export DISPLAY=:0)")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def show(self, frames_rgb, chunk_idx: int) -> None:
        if not self.enabled:
            return
        try:
            for i, rgb in enumerate(frames_rgb):
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                if self.scale != 1.0:
                    bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale,
                                     interpolation=cv2.INTER_NEAREST)
                cv2.putText(bgr, f"chunk {chunk_idx}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(bgr, f"chunk {chunk_idx}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1, cv2.LINE_AA)
                title = self.labels[i] if i < len(self.labels) else f"slot{i}"
                cv2.imshow(title, bgr)
                if not self._opened:
                    cv2.moveWindow(title, 30 + i * (bgr.shape[1] + 26), 40)
            self._opened = True
            cv2.waitKey(1)
        except Exception as e:
            self._disable(e)

    def poll(self) -> int:
        """Pump the GUI event loop between frames. Returns the pressed key code (-1 = none)."""
        if not self.enabled:
            return -1
        try:
            return cv2.waitKey(1) & 0xFF
        except Exception as e:
            self._disable(e)
            return -1

    def close(self) -> None:
        if not (self.enabled and self._opened):
            return
        try:
            cv2.destroyAllWindows()
            for _ in range(4):
                cv2.waitKey(1)
        except Exception:
            pass


class Http:
    def __init__(self, url: str, timeout_s: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self._sess = requests.Session() if _REQ_OK else None

    def _urllib(self, path: str, obj):
        import urllib.error
        import urllib.request

        data = None if obj is None else json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}{path}", data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(body)
            except Exception:
                return e.code, {"detail": body}

    def request(self, path: str, obj=None) -> tuple[int, dict]:
        """Returns (status_code, parsed_json). Does NOT raise on 4xx/5xx."""
        if self._sess is None:
            return self._urllib(path, obj)
        if obj is None:
            r = self._sess.get(f"{self.url}{path}", timeout=self.timeout_s)
        else:
            r = self._sess.post(f"{self.url}{path}", data=json.dumps(obj).encode("utf-8"),
                                headers={"Content-Type": "application/json"},
                                timeout=self.timeout_s)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"detail": r.text}

    _prev_rtt_ms = None          # measured on the client; the server cannot see it

    def must(self, path: str, obj=None) -> dict:
        code, payload = self.request(path, obj)
        if code != 200:
            raise RuntimeError(f"{path} -> HTTP {code}: {payload.get('detail', payload)}")
        return payload

    def predict(self, images_b64, qpos_ticks, step_dt_ms: float | None = None
                ) -> tuple[np.ndarray, dict, float]:
        """The PREVIOUS round trip is reported to the server with this request.

        Only the client can see the round trip: the server's own clock stops at the response,
        so JPEG encode and the network are invisible to it. Sending the previous measurement
        (and the client's step period) lets the server print the slack -- how much of the
        execution budget was left when the plan arrived -- which is the number that decides
        whether --emit-steps can be lowered.
        """
        t0 = time.perf_counter()
        body = {"images": images_b64, "qpos": [float(x) for x in qpos_ticks]}
        if self._prev_rtt_ms is not None:
            body["prev_rtt_ms"] = float(self._prev_rtt_ms)
        if step_dt_ms is not None:
            body["step_dt_ms"] = float(step_dt_ms)
        j = self.must("/predict", body)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        self._prev_rtt_ms = rtt_ms
        ticks = decode_chunk(j["joint_ticks_b64"], j["chunk_size"], j["joint_dim"])
        return ticks, j, rtt_ms


# ============================================================================
# Dynamixel
# ============================================================================

class DxlBus:
    """GroupSyncRead(present) / GroupSyncWrite(goal). Raises clearly if the SDK is absent."""

    simulated = False

    def __init__(self, ids, port: str, baud: int):
        if not _DXL_OK:
            raise RuntimeError(
                f"dynamixel_sdk is not importable ({_DXL_ERR}). Install it on the robot host "
                f"(`pip install dynamixel-sdk`) or stay in --dry-run.")
        self.ids = list(ids)
        self.ph = PortHandler(port)
        self.pk = PacketHandler(ASSUMED_PROTOCOL)
        if not self.ph.openPort():
            raise RuntimeError(f"failed to open DXL port {port}")
        if not self.ph.setBaudRate(baud):
            raise RuntimeError(f"failed to set baud {baud} on {port}")
        self._w = GroupSyncWrite(self.ph, self.pk, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        self._r = GroupSyncRead(self.ph, self.pk, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        for i in self.ids:
            if not self._r.addParam(i):
                raise RuntimeError(f"GroupSyncRead addParam failed for id {i}")

    def read_present(self) -> np.ndarray:
        if self._r.txRxPacket() != COMM_SUCCESS:
            raise RuntimeError("GroupSyncRead txRxPacket failed (check power/wiring/baud)")
        return np.array([self._r.getData(i, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                         for i in self.ids], dtype=np.int32)

    def write_goal(self, ticks: np.ndarray):
        self._w.clearParam()
        for i, t in zip(self.ids, np.asarray(ticks, dtype=np.int64)):
            t = int(t)
            if not self._w.addParam(i, bytes([t & 0xFF, (t >> 8) & 0xFF,
                                              (t >> 16) & 0xFF, (t >> 24) & 0xFF])):
                raise RuntimeError(f"GroupSyncWrite addParam failed for id {i}")
        if self._w.txPacket() != COMM_SUCCESS:
            raise RuntimeError("GroupSyncWrite txPacket failed")

    def torque(self, on: bool):
        for i in self.ids:
            self.pk.write1ByteTxRx(self.ph, i, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def safe_start(self) -> np.ndarray:
        """GOAL := present, THEN torque on -- avoids the stale-goal snap at power-on."""
        present = self.read_present()
        self.write_goal(present)
        self.torque(True)
        return present

    def close(self):
        try:
            self.ph.closePort()
        except Exception:
            pass


class SimBus:
    """A first-order fake arm for dry runs with no hardware. NEVER touches a real motor.

    `actual` lags `cmd` by at most `speed` ticks per step, so the tracking-error columns in
    steps.jsonl exercise the same code path they will on the real robot -- but every record it
    produces is marked `simulated: true` so nobody mistakes it for a measurement.
    """

    simulated = True

    def __init__(self, start_ticks, speed: int = 90):
        self.pos = np.asarray(start_ticks, dtype=np.int64).copy()
        self.goal = self.pos.copy()
        self.speed = int(speed)

    def read_present(self) -> np.ndarray:
        d = np.clip(self.goal - self.pos, -self.speed, self.speed)
        self.pos = self.pos + d
        return self.pos.astype(np.int32)

    def write_goal(self, ticks):
        self.goal = np.asarray(ticks, dtype=np.int64).copy()

    def torque(self, on: bool):
        pass

    def safe_start(self) -> np.ndarray:
        return self.pos.astype(np.int32)

    def close(self):
        pass


def home_start_move(bus, dxl_ids, lo, hi, pose, *, loop_dt: float = 0.02,
                    wall_cap_s: float = 45.0, log=print) -> np.ndarray:
    """Drive the arm to `control_gamepad.py`'s default pose BEFORE the rollout starts.

    Runs the recorder's own `HomeReturn` state machine (gripper open -> arm to home ->
    gripper fully closed -> brief hold), so the pose the policy sees at step 0 is the pose
    every training episode started from. Returns the present ticks after the move.

    `HomeReturn` emits one goal per joint; this bridges that to the bus's 5-joint
    GroupSyncWrite by holding the last commanded value on the joints the current phase does
    not touch -- which is exactly the state machine's "hold" contract.
    """
    idx = {int(i): k for k, i in enumerate(dxl_ids)}
    present = np.asarray(bus.read_present(), dtype=np.int64)
    pending = present.copy()

    def send(dxl_id: int, goal: int) -> None:
        pending[idx[int(dxl_id)]] = int(goal)

    hr = HomeReturn(pose, send, log=lambda m: log(m.strip() or m))
    cur = {int(i): int(present[k]) for i, k in idx.items()}
    hr.start(cur)
    log(f"[home] -> {[int(pose['home'][i]) for i in dxl_ids]} "
        f"(gripper open {pose['gripper_open_target']} -> close {pose['gripper_close_target']}), "
        f"arm {hr.arm_speed:.0f} grip {hr.grip_speed:.0f} ticks/s, hold {hr.hold_sec:.1f}s")

    t0 = time.time()
    phase = ""
    while hr.active:
        if time.time() - t0 > wall_cap_s:
            log(f"[home][warn] wall-clock cap {wall_cap_s:.0f}s hit in "
                f"'{_HOME_PHASE_LABELS.get(hr.phase, hr.phase)}' -- giving up on homing and "
                f"starting the rollout from wherever the arm is.")
            hr.abort()
            break
        raw = np.asarray(bus.read_present(), dtype=np.int64)
        cur = {int(i): int(raw[k]) for i, k in idx.items()}
        if hr.phase != phase:
            phase = hr.phase
            log(f"[home] {_HOME_PHASE_LABELS.get(phase, phase)} ... present={raw.tolist()}")
        hr.step(cur)
        bus.write_goal(clip_limits(pending, lo, hi))
        time.sleep(loop_dt)

    present = np.asarray(bus.read_present(), dtype=np.int32)
    want = np.array([int(pose["home"][i]) for i in dxl_ids], dtype=np.int64)
    want[idx[25]] = int(pose["gripper_close_target"])
    err = np.abs(present.astype(np.int64) - want)
    log(f"[home] done in {time.time() - t0:4.1f}s: present={present.tolist()} "
        f"target={want.tolist()} |err|={err.tolist()}"
        + (f"  timed_out={hr.timed_out_phases}" if hr.timed_out_phases else ""))
    return present


# ============================================================================
# Session recording -- the point of this client.
# ============================================================================

def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


class SessionRecorder:
    """rollout_<ts>/ : steps.jsonl + chunks.jsonl + chunk_*.npy + meta.json + summary.json."""

    def __init__(self, root: str, args, save_frames: str = "off", jpeg_quality: int = 90):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root).expanduser() / f"rollout_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_frames = save_frames
        self.jpeg_quality = int(jpeg_quality)
        self.t0 = time.time()
        self._steps = (self.dir / "steps.jsonl").open("w", buffering=1)
        self._chunks = (self.dir / "chunks.jsonl").open("w", buffering=1)
        self.rows: list[dict] = []
        self.chunk_rows: list[dict] = []
        self._pending_frames: list[np.ndarray] | None = None
        self.meta = {"argv": sys.argv, "args": {k: (list(v) if isinstance(v, tuple) else v)
                                                for k, v in vars(args).items()},
                     "started": stamp, "const_source": _CONST_SOURCE,
                     "deps": {"cv2": _CV2_OK, "PIL": _PIL_OK, "requests": _REQ_OK,
                              "dynamixel_sdk": _DXL_OK},
                     "motor_labels": MOTOR_LABELS}
        if save_frames == "all":
            print(f"[rec][warn] --save-frames all: ~3 JPEGs per /predict. At 10 Hz with "
                  f"--exec-steps 8 that is ~4 frames/s ~= 60 KB/s ~= 3.5 MB/min. Use 'flagged' "
                  f"to keep only the interesting ones.")

    # -- setup ---------------------------------------------------------------
    def add_meta(self, **kw):
        self.meta.update(kw)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    # -- per chunk -----------------------------------------------------------
    def chunk(self, idx: int, ticks_chunk: np.ndarray, resp: dict, rtt_ms: float,
              present: np.ndarray, frames: list[np.ndarray] | None,
              raw_chunk: np.ndarray | None = None, smooth: dict | None = None):
        """`ticks_chunk` is what will be EXECUTED. If the Gaussian smoother ran, `raw_chunk` is
        the server's untouched chunk and is stored next to it as chunk_<i>_raw.npy."""
        np.save(self.dir / f"chunk_{idx:04d}.npy", ticks_chunk)
        meta = {k: v for k, v in resp.items() if k != "joint_ticks_b64"}
        row = {"chunk": idx, "t_rel": round(time.time() - self.t0, 4),
               "rtt_ms": round(rtt_ms, 2), "present_ticks": present.astype(int).tolist(),
               "row0": ticks_chunk[0].astype(int).tolist(),
               "shape": list(ticks_chunk.shape), "server": meta}
        if raw_chunk is not None:
            raw = np.asarray(raw_chunk)
            np.save(self.dir / f"chunk_{idx:04d}_raw.npy", raw)
            row["row0_raw"] = raw[0].astype(int).tolist()
        if smooth:
            row["smooth"] = smooth
        self.chunk_rows.append(row)
        self._chunks.write(json.dumps(row) + "\n")
        self._pending_frames = frames

    def flush_frames(self, idx: int, flagged: bool):
        """Frames are grabbed once per /predict, so they are saved per chunk, not per step."""
        if self._pending_frames is None or self.save_frames == "off":
            return
        if self.save_frames == "flagged" and not flagged:
            self._pending_frames = None
            return
        d = self.dir / "frames"
        d.mkdir(exist_ok=True)
        for i, f in enumerate(self._pending_frames):
            try:
                p = d / f"c{idx:04d}_cam{i + 1}.jpg"
                if _PIL_OK:
                    Image.fromarray(f).save(p, quality=self.jpeg_quality)
                elif _CV2_OK:
                    cv2.imwrite(str(p), f[:, :, ::-1])
            except Exception as e:
                print(f"[rec][warn] frame save failed: {type(e).__name__}: {e}")
                break
        self._pending_frames = None

    # -- per step ------------------------------------------------------------
    def step(self, **kw):
        row = {"t_rel": round(time.time() - self.t0, 4), **kw}
        self.rows.append(row)
        self._steps.write(json.dumps(row) + "\n")

    # -- end -----------------------------------------------------------------
    def summarize(self, print_it: bool = True) -> dict:
        n = len(self.rows)
        if n == 0:
            s = {"steps": 0, "chunks": len(self.chunk_rows), "note": "no steps executed"}
            (self.dir / "summary.json").write_text(json.dumps(s, indent=2))
            return s
        A = lambda k: np.asarray([r[k] for r in self.rows], dtype=np.float64)  # noqa: E731
        req = np.abs(A("requested") - A("prev_cmd"))          # what the policy asked for
        dcmd = np.abs(A("cmd") - A("prev_cmd"))               # what was actually commanded
        terr = np.abs(A("actual") - A("cmd"))                 # tracking error
        cl = np.asarray([r["clip_limit_ch"] for r in self.rows], dtype=bool)
        cr = np.asarray([r["clip_rate_ch"] for r in self.rows], dtype=bool)
        dt = A("dt_ms")
        rtt = np.asarray([c["rtt_ms"] for c in self.chunk_rows], dtype=np.float64)
        inf = np.asarray([float(c["server"].get("inference_ms", np.nan))
                          for c in self.chunk_rows], dtype=np.float64)
        per = lambda M: {  # noqa: E731
            "per_channel_p50": [round(_pct(M[:, i], 50), 1) for i in range(JOINT_DIM)],
            "per_channel_p90": [round(_pct(M[:, i], 90), 1) for i in range(JOINT_DIM)],
            "per_channel_max": [round(float(M[:, i].max()), 1) for i in range(JOINT_DIM)],
            "max_over_channels_p50": round(_pct(M.max(axis=1), 50), 1),
            "max_over_channels_p90": round(_pct(M.max(axis=1), 90), 1),
            "max_over_channels_max": round(float(M.max()), 1),
        }
        s = {
            "dir": str(self.dir),
            "steps": n,
            "chunks": len(self.chunk_rows),
            "duration_s": round(time.time() - self.t0, 1),
            "motor_labels": MOTOR_LABELS,
            "clip": {
                "rate_clipped_steps": int(cr.any(axis=1).sum()),
                "rate_clipped_frac": round(float(cr.any(axis=1).mean()), 4),
                "limit_clipped_steps": int(cl.any(axis=1).sum()),
                "limit_clipped_frac": round(float(cl.any(axis=1).mean()), 4),
                "rate_clipped_per_channel": cr.sum(axis=0).astype(int).tolist(),
                "limit_clipped_per_channel": cl.sum(axis=0).astype(int).tolist(),
            },
            "requested_move_ticks": per(req),
            "commanded_move_ticks": per(dcmd),
            "tracking_error_ticks": per(terr),
            "timing_ms": {"step_dt_p50": round(_pct(dt, 50), 1),
                          "step_dt_p90": round(_pct(dt, 90), 1),
                          "step_dt_max": round(float(dt.max()), 1),
                          "rtt_p50": round(_pct(rtt, 50), 1) if rtt.size else None,
                          "rtt_p90": round(_pct(rtt, 90), 1) if rtt.size else None,
                          "rtt_max": round(float(rtt.max()), 1) if rtt.size else None,
                          "server_inference_p50": (round(_pct(inf[np.isfinite(inf)], 50), 1)
                                                   if np.isfinite(inf).any() else None)},
            "simulated_bus": bool(self.meta.get("simulated_bus")),
        }
        (self.dir / "summary.json").write_text(json.dumps(s, indent=2))
        if print_it:
            self.print_summary(s)
        return s

    @staticmethod
    def print_summary(s: dict):
        if not s.get("steps"):
            print(f"[summary] no steps executed ({s.get('chunks', 0)} chunks)")
            return
        c = s["clip"]
        print("=" * 78)
        print(f"[summary] {s['steps']} steps / {s['chunks']} chunks in {s['duration_s']}s"
              + ("   (SIMULATED BUS -- 'actual' is not a measurement)"
                 if s.get("simulated_bus") else ""))
        print(f"  RATE-CLIPPED steps  : {c['rate_clipped_steps']}/{s['steps']} "
              f"({100 * c['rate_clipped_frac']:.1f}%)   per channel {c['rate_clipped_per_channel']}")
        print(f"  LIMIT-CLIPPED steps : {c['limit_clipped_steps']}/{s['steps']} "
              f"({100 * c['limit_clipped_frac']:.1f}%)  per channel "
              f"{c['limit_clipped_per_channel']}")
        if c["rate_clipped_frac"] > 0.5:
            print("  [!] more than half the steps were rate-clipped: the arm is following the "
                  "LIMITER, not the policy. Raise --max-step-ticks (watching) or accept that "
                  "this rollout does not test the policy. (log.md's failed run was 94.4%.)")
        for label, key in (("requested move |plan-prev|", "requested_move_ticks"),
                           ("commanded move |cmd-prev|", "commanded_move_ticks"),
                           ("tracking err |actual-cmd|", "tracking_error_ticks")):
            m = s[key]
            print(f"  {label:<26s} p50/p90/max over channels = "
                  f"{m['max_over_channels_p50']}/{m['max_over_channels_p90']}/"
                  f"{m['max_over_channels_max']} ticks")
            print(f"    {'per-channel p90':<24s} {m['per_channel_p90']}   "
                  f"({', '.join(MOTOR_LABELS)})")
        t = s["timing_ms"]
        print(f"  step dt p50/p90/max : {t['step_dt_p50']}/{t['step_dt_p90']}/{t['step_dt_max']} ms")
        print(f"  server RTT p50/p90/max: {t['rtt_p50']}/{t['rtt_p90']}/{t['rtt_max']} ms "
              f"(inference p50 {t['server_inference_p50']} ms)")
        print(f"  -> {s['dir']}")
        print("=" * 78)

    def label(self, value: str | None, interactive: bool) -> None:
        lab = value
        if lab is None and interactive:
            try:
                print("\n[label] outcome? s=success  f=fail  p=partial  a=aborted/invalid "
                      "(anything else = skip)")
                key = input("[label] > ").strip().lower()
                lab = {"s": "success", "f": "fail", "p": "partial", "a": "aborted"}.get(key)
                if lab is None:
                    print("[label] skipped")
                    return
                note = input("[label] note (optional) > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[label] skipped (no tty)")
                return
        else:
            note = ""
        if lab is None:
            return
        (self.dir / "label.json").write_text(json.dumps(
            {"label": lab, "note": note, "t": time.strftime("%Y-%m-%d %H:%M:%S"),
             "steps": len(self.rows), "chunks": len(self.chunk_rows)}, indent=2))
        print(f"[label] {lab!r} -> {self.dir / 'label.json'}")

    def close(self):
        for f in (self._steps, self._chunks):
            try:
                f.close()
            except Exception:
                pass


# ============================================================================
# Control loop
# ============================================================================

def run_loop(args) -> int:
    lo, hi = parse_joint_limits(args.joint_limits)
    max_step = parse_max_step(args.max_step_ticks)

    dry = args.dry_run
    if not dry and not _DXL_OK:
        print(f"[safety] --no-dry-run requested but dynamixel_sdk is not importable "
              f"({_DXL_ERR}) -> FORCING --dry-run. Nothing will be written to any motor.")
        dry = True

    print("=" * 78)
    print(f"  url            : {args.url}")
    print(f"  lang           : {args.lang!r}")
    print(f"  dry_run        : {dry}   {'(no motor writes)' if dry else '(REAL MOTOR WRITES)'}")
    print(f"  joint limits   : lo={lo.tolist()} hi={hi.tolist()}")
    print(f"                   gripper [{GRIPPER_TICK_MIN},{GRIPPER_TICK_MAX}] is CONFIRMED; "
          f"the 4 arm joints are UNCONFIRMED (full encoder range)")
    print(f"  max step ticks : {max_step.tolist()}  @ {args.step_dt_ms} ms/step "
          f"({1000 / max(args.step_dt_ms, 1):.0f} Hz)")
    print(f"  exec steps     : first {args.exec_steps} rows of each chunk (receding horizon)")
    smooth_exclude = () if args.smooth_gripper else (GRIPPER_SLOT,)
    smooth_where = ("ALL {0} channels".format(JOINT_DIM) if args.smooth_gripper
                    else "the {0} arm joints ({1})".format(
                        JOINT_DIM - 1, ", ".join(MOTOR_LABELS[:GRIPPER_SLOT])))
    if args.smooth_actions:
        _w = gaussian_kernel_1d(args.smooth_kernel, args.smooth_sigma)
        print(f"  action smooth  : ON  1x{args.smooth_kernel} FORWARD Gaussian along TIME "
              f"(t..t+{args.smooth_kernel - 1}, peak on t), "
              f"sigma={args.smooth_sigma if args.smooth_sigma > 0 else 'auto'}, "
              f"weights={[round(float(x), 4) for x in _w]} (edge-replicated)")
        print(f"                   applied to {smooth_where}; "
              f"{MOTOR_LABELS[GRIPPER_SLOT]} is "
              f"{'ALSO blurred (--smooth-gripper)' if args.smooth_gripper else 'passed through RAW'}")
    else:
        print(f"  action smooth  : off (--smooth-actions to enable)")
    print(f"  stop on done   : {args.stop_on_done}  (6-D checkpoints only; "
          f"the threshold itself lives on the server)")

    home_pose = None
    if not args.home_start:
        print("  home start     : off (--no-home-start): the rollout begins from whatever pose "
              "the arm is already in")
    elif not _HOME_OK:
        print(f"  home start     : UNAVAILABLE -- src_control.home_return is not importable "
              f"({type(_HOME_ERR).__name__}: {_HOME_ERR}). Starting from the current pose.")
    else:
        home_pose = load_home_pose(args.home_pose)
        if home_pose is None:
            print("  home start     : NO POSE FILE -- run `python record_home_pose.py` once. "
                  "Starting from the current pose.")
        else:
            print(f"  home start     : on -- arm goes to control_gamepad's default pose "
                  f"{[int(home_pose['home'][i]) for i in args.dxl_ids]} "
                  f"(gripper closed to {home_pose['gripper_close_target']}) before step 0")
    print(f"  constants from : {_CONST_SOURCE}")
    print("=" * 78)

    http = Http(args.url, args.request_timeout_s)
    try:
        health = http.must("/health")
    except Exception as e:
        print(f"[health] cannot reach {args.url}: {type(e).__name__}: {e}\n"
              f"  * is smolvla_server_openmani.py running on that host?\n"
              f"  * the server needs `sudo ufw allow 8000/tcp` (this blocked the last attempt)\n"
              f"  * `--offline-self-test` needs no server at all")
        return 1
    print(f"[health] {json.dumps(health)}")
    n_cams = int(health.get("client_cams", args.cams))
    if n_cams != args.cams:
        print(f"[cam][warn] server expects {n_cams} images per request but --cams={args.cams}. "
              f"Following the SERVER ({n_cams}).")
    if health.get("dropped_client_slots"):
        print(f"[cam][warn] the server DROPS client slot(s) {health['dropped_client_slots']}.")
    server_has_done = bool(health.get("has_done"))
    stop_on_done = server_has_done and args.stop_on_done != "off"
    if args.stop_on_done == "on" and not server_has_done:
        print(f"[done][FAIL] --stop-on-done on, but the server reports has_done=false: the "
              f"loaded checkpoint has a {health.get('joint_dim')}-D action and carries no "
              f"done_flag channel, so no termination judgement is possible. Load a 6-D "
              f"checkpoint, or pass --stop-on-done auto|off.")
        return 1
    if stop_on_done:
        print(f"[done] termination ENABLED: the loop ends when the server reports done=true "
              f"(server threshold={health.get('done_threshold')}, "
              f"consecutive={health.get('done_consecutive')}). The chunk that carries done is "
              f"NOT executed.")
    elif server_has_done:
        print("[done] the checkpoint HAS a done_flag channel but --stop-on-done off: done_mean "
              "is printed and recorded only; the loop runs to --max-chunks.")
    else:
        print("[done] the checkpoint has NO done_flag channel (5-D action): no termination "
              "judgement is possible; the loop runs to --max-chunks.")

    nas = health.get("n_action_steps")
    if nas and args.exec_steps > int(nas):
        print(f"[safety][warn] --exec-steps {args.exec_steps} > the checkpoint's n_action_steps "
              f"{nas}: the tail rows are further from the observation than the policy was "
              f"configured to execute open-loop. Lower it.")
    # RTC (server-side) aligns the previous plan's unexecuted tail so that its step 0 IS this
    # chunk's step 0, which assumes the arm executes exactly as many rows as the server emits.
    # A mismatch shifts the guidance by the difference without any error, so it is reported here
    # on every run. Nothing to turn on client-side: --rtc / --coda-rtc live on the server.
    if health.get("rtc"):
        emit = int(health.get("chunk_size") or args.exec_steps)
        extra = ""
        if health.get("coda_rtc") not in (None, "off"):
            extra = f", coda_rtc={health['coda_rtc']} (the Coda correction AE's stage is guided too)"
        print(f"[rtc] the server guides each chunk with the previous one's unexecuted tail "
              f"(execution_horizon={health.get('rtc_execution_horizon')}, "
              f"inference_delay={health.get('rtc_inference_delay')}, "
              f"max_guidance_weight={health.get('rtc_max_guidance_weight')}{extra}).")
        if int(args.exec_steps) != emit:
            print(f"[rtc][warn] --exec-steps {args.exec_steps} != the {emit} rows the server "
                  f"emits: RTC assumes the arm executes exactly what was emitted, so the "
                  f"guidance is misaligned by {abs(int(args.exec_steps) - emit)} steps. Match "
                  f"them (or restart the server with --emit-steps {args.exec_steps}).")
    fps = health.get("fps")
    if fps and abs(1000.0 / args.step_dt_ms - float(fps)) > 0.51:
        print(f"[timing][warn] the dataset is {fps} Hz but --step-dt-ms {args.step_dt_ms} runs at "
              f"{1000 / args.step_dt_ms:.1f} Hz. The policy's tick deltas are per DATASET step.")

    # Grabber threads make stale frames impossible, so --stale-frames (reproduce the old
    # behaviour) only means anything with inline reads. Honour the flag the user actually set.
    _threaded = args.cam_threads and args.fresh_frames
    if args.cam_threads and not args.fresh_frames:
        print("[cam] --stale-frames forces inline reads: grabber threads cannot serve a stale "
              "frame. Pass neither flag for the fast, fresh default.")
    cam = CameraRig(n_cams, dummy=args.dummy_cam, manual=args.cam_devices,
                    swap01=args.swap_cam01, threaded=_threaded)
    _labels = [f"slot{i} {sp.key} ({sp.role})" for i, sp in enumerate(cam.specs)]
    while len(_labels) < n_cams:
        _labels.append(f"slot{len(_labels)}")
    preview = CameraPreview(_labels, enabled=args.show_cams, scale=args.cam_preview_scale)
    if args.show_cams:
        print(f"[cam][preview] {'ON' if preview.enabled else 'OFF (' + preview.reason + ')'} "
              f"-- {len(_labels)} windows, refreshed once per chunk with the frames actually "
              f"sent. 'q' in a window stops the rollout (torque off, summary still written).")
    rec = None
    bus = None
    if args.out_dir:
        rec = SessionRecorder(args.out_dir, args, save_frames=args.save_frames,
                              jpeg_quality=args.jpeg_quality)
        print(f"[rec] {rec.dir}")

    step_dt = args.step_dt_ms / 1000.0
    chunk_idx = 0
    n_clip_limit = n_clip_rate = 0
    server_blocked_writes = False
    stop_reason = "max_chunks"
    quit_requested = False
    try:
        init = http.must("/init", {"lang": args.lang, "reasoning_cache_n": 0})
        print(f"[init] {json.dumps(init)}")
        if init.get("in_training_distribution") is False:
            print(f"[lang][WARN] {args.lang!r} is NOT one of the {init.get('n_training_tasks')} "
                  f"sentences this policy was trained on. Expect out-of-distribution behaviour.")

        if dry:
            if _DXL_OK:
                try:
                    bus = DxlBus(args.dxl_ids, args.port, args.baud)
                    present = bus.read_present()
                    print(f"[dxl] dry-run: READ ONLY. present={present.tolist()} (torque untouched)")
                except Exception as e:
                    print(f"[dxl] dry-run: could not open the bus ({type(e).__name__}: {e}); "
                          f"simulating the arm from --sim-start-ticks.")
                    bus = None
            if bus is None:
                bus = SimBus(args.sim_start_ticks, speed=args.sim_speed)
                present = bus.read_present()
                print(f"[dxl] dry-run: SIMULATED arm (SimBus, {args.sim_speed} ticks/step), "
                      f"present={present.tolist()}. 'actual' below is NOT a measurement.")
        else:
            print(f"\n*** REAL MOTOR WRITES in {args.arm_delay_s:.0f}s -- Ctrl-C to abort ***\n")
            time.sleep(args.arm_delay_s)
            bus = DxlBus(args.dxl_ids, args.port, args.baud)
            present = bus.safe_start()
            print(f"[dxl] safe_start: GOAL:=present {present.tolist()}, torque ON")

        # Start every rollout from the pose control_gamepad.py leaves the arm in, so step 0 is
        # inside the training distribution. Skipped in --dry-run against a REAL bus (that path
        # is read-only by contract); the SimBus runs it so a dry run still exercises the code.
        homed = False
        if home_pose is not None and (not dry or getattr(bus, "simulated", False)):
            present = home_start_move(bus, args.dxl_ids, lo, hi, home_pose)
            homed = True
        elif home_pose is not None:
            print("[home] skipped: --dry-run against a real bus is read-only "
                  "(--no-dry-run to actually home, or --dummy-cam/SimBus to rehearse it).")

        # The home start takes seconds, and a V4L2 queue hands back the frame that was current
        # at the LAST read -- so without this the first observation is a picture of the arm
        # BEFORE it was homed. Drop those, so step 0 sees the pose the policy was trained on.
        if args.fresh_frames and not args.dummy_cam:
            cam.read(fresh=True)
            _ages = " ".join(f"slot{i}={a:.0f}ms" for i, a in enumerate(cam.last_ages_ms))
            _how = ("grabber threads kept the queue empty" if cam.threaded
                    else f"dropped {sum(cam.last_discards)} stale frame(s) inline")
            print(f"[cam] post-home: {_how}; frame age now {_ages} "
                  f"-- the first observation is taken AFTER the home start.")

        n_sub = max(1, int(args.substeps))
        read_reserve = 0.0          # EMA of read_present(); seeded on the first step
        if n_sub > 1:
            print(f"[substep] {n_sub} goal writes per step: one every "
                  f"{args.step_dt_ms / n_sub:.1f} ms, each moving at most "
                  f"{int(np.ceil(max_step.max() / n_sub))} ticks. Step boundaries and "
                  f"--max-step-ticks are unchanged.")
            _lat = _ftdi_latency_ms(args.port)
            if _lat is not None and _lat > 4:
                print(f"[substep][warn] {args.port} latency_timer is {_lat} ms, so "
                      f"read_present() costs about that much and eats "
                      f"{_lat / (args.step_dt_ms / n_sub):.1f} substep slot(s). Lower it: "
                      f"echo 1 | sudo tee /sys/bus/usb-serial/devices/"
                      f"{Path(args.port).name}/latency_timer")

        if rec:
            rec.add_meta(health=health, init=init, cameras=cam.binding,
                         joint_limits={"lo": lo.tolist(), "hi": hi.tolist()},
                         max_step_ticks=max_step.tolist(), dry_run=bool(dry),
                         simulated_bus=bool(getattr(bus, "simulated", False)),
                         input_mode=args.input_mode,
                         lang=args.lang,
                         substeps=n_sub,
                         ftdi_latency_ms=_ftdi_latency_ms(args.port),
                         stop_on_done=args.stop_on_done,
                         stop_on_done_active=bool(stop_on_done),
                         server_has_done=bool(server_has_done),
                         home_start={"requested": bool(args.home_start),
                                     "path": str(args.home_pose),
                                     "performed": bool(homed),
                                     "pose": (None if home_pose is None else
                                              {"home": {str(k): int(v) for k, v in
                                                        home_pose["home"].items()},
                                               "gripper_open_target":
                                                   int(home_pose["gripper_open_target"]),
                                               "gripper_close_target":
                                                   int(home_pose["gripper_close_target"])})})

        prev_cmd = clip_limits(present, lo, hi)
        if not np.array_equal(prev_cmd, present):
            print(f"[safety][warn] present {present.tolist()} is OUTSIDE --joint-limits; "
                  f"seeding prev_cmd at the clipped {prev_cmd.tolist()}.")
        ramped = False

        while chunk_idx < args.max_chunks:
            # Same staleness applies at every chunk boundary: exec_steps * step_dt_ms elapses
            # between grabs, so an undrained read shows the arm one chunk in the past.
            frames = cam.read(fresh=args.fresh_frames)
            preview.show(frames, chunk_idx + 1)
            images_b64 = [encode_jpeg_b64(f, args.jpeg_quality) for f in frames]
            present = bus.read_present()
            ticks_chunk, resp, rtt_ms = http.predict(images_b64, present,
                                                     step_dt_ms=args.step_dt_ms)
            chunk_idx += 1

            if ticks_chunk.shape[1] != JOINT_DIM:
                raise RuntimeError(f"server joint_dim={ticks_chunk.shape[1]}, expected {JOINT_DIM}")

            raw_chunk = None
            smooth_meta = None
            if args.smooth_actions:
                raw_chunk = ticks_chunk
                ticks_chunk = smooth_chunk(ticks_chunk, args.smooth_kernel, args.smooth_sigma,
                                           exclude=smooth_exclude)
                d_max = int(np.abs(ticks_chunk.astype(np.int64)
                                   - raw_chunk.astype(np.int64)).max())
                smooth_meta = {"kernel": int(args.smooth_kernel),
                               "sigma": float(args.smooth_sigma),
                               "exclude": list(smooth_exclude),
                               "max_abs_delta_ticks": d_max}
                if chunk_idx == 1:
                    print(f"[smooth] 1x{args.smooth_kernel} Gaussian applied along time to "
                          f"{smooth_where}; chunk 1 moved by at most {d_max} ticks")
            if resp.get("execute") is False and not server_blocked_writes:
                server_blocked_writes = True
                print("[safety] the server answered execute=false (its own --dry-run): motor "
                      "writes are suppressed for the rest of this run.")
            allow_write = (not dry) and resp.get("execute") is not False

            n_exec = min(args.exec_steps, ticks_chunk.shape[0])
            span = int(np.abs(np.diff(ticks_chunk[:n_exec], axis=0)).max()) if n_exec >= 2 else 0
            jump0 = int(np.abs(ticks_chunk[0].astype(np.int64) - present.astype(np.int64)).max())
            _sl = resp.get("slack_ms")
            print(f"[chunk {chunk_idx:04d}] rtt={rtt_ms:7.1f} ms "
                  f"(infer={resp.get('inference_ms', float('nan')):7.1f}"
                  + ("" if _sl is None else f" slack={_sl:+.0f}") + ") "
                  f"chunk={tuple(ticks_chunk.shape)} exec={n_exec} max|drow|={span} jump0={jump0}")
            if chunk_idx == 1 and max(span, jump0) > int(max_step.min()):
                print(f"  [safety][note] the policy's own step size ({max(span, jump0)} ticks) "
                      f"exceeds --max-step-ticks {max_step.tolist()}, so the rate limiter will "
                      f"clip and the arm will LAG the policy. Watch the clip rate in the summary.")
            if rec:
                rec.chunk(chunk_idx, ticks_chunk, resp, rtt_ms, present, frames,
                          raw_chunk=raw_chunk, smooth=smooth_meta)

            done_mean = resp.get("done_mean")
            if done_mean is not None:
                print(f"  [done] mean={done_mean:.3f} "
                      f"streak={resp.get('done_streak')}/{resp.get('done_consecutive')} "
                      f"flag={resp.get('done')}"
                      + ("" if stop_on_done else "   (--stop-on-done off: ignored)"))
            if stop_on_done and resp.get("done") is True:
                stop_reason = "done"
                print(f"[done] the server reports done=true after {chunk_idx} chunks "
                      f"(done_mean={done_mean:.3f}). Stopping WITHOUT executing this chunk; "
                      f"the arm holds its last commanded pose.")
                if rec:
                    rec.flush_frames(chunk_idx, False)
                break

            plan_rows = [(row, "chunk") for row in ticks_chunk[:n_exec]]
            if not ramped and args.ramp_steps > 0:
                first = clip_limits(ticks_chunk[0], lo, hi).astype(np.float64)
                base = prev_cmd.astype(np.float64)
                plan_rows = [(np.rint(base + (first - base) * (k / args.ramp_steps)).astype(np.int32),
                              "ramp") for k in range(1, args.ramp_steps + 1)] + plan_rows
                print(f"[ramp] {prev_cmd.tolist()} -> {ticks_chunk[0].tolist()} in "
                      f"{args.ramp_steps} steps (rate limit still applies)")
                ramped = True

            chunk_flagged = False
            for s, (plan, src) in enumerate(plan_rows):
                t0 = time.perf_counter()
                lim = clip_limits(plan, lo, hi)
                cl_ch = (lim != np.asarray(plan, dtype=np.int32))
                cmd = rate_limit(prev_cmd, lim, max_step)
                cr_ch = (cmd != lim)
                n_clip_limit += int(cl_ch.any())
                n_clip_rate += int(cr_ch.any())
                chunk_flagged = chunk_flagged or bool(cr_ch.any() or cl_ch.any())

                # Spread the writes across the step, but leave the measured cost of
                # read_present() at the end so the step still closes on --step-dt-ms. An
                # unreserved read pushes a 100 ms step to 116 ms on this FTDI adapter and
                # drags the whole rollout off the 10 Hz rate the policy was trained at.
                goals = substep_goals(prev_cmd, cmd, n_sub)
                write_window = max(0.0, step_dt - read_reserve)
                if allow_write or getattr(bus, "simulated", False):
                    for k, g in enumerate(goals):
                        if k:
                            slot = t0 + write_window * k / len(goals)
                            rem = slot - time.perf_counter()
                            if rem > 0:
                                time.sleep(rem)
                        bus.write_goal(g)    # sim bus follows too, dry run or not
                t_rd = time.perf_counter()
                actual = bus.read_present()
                read_reserve = 0.75 * read_reserve + 0.25 * (time.perf_counter() - t_rd)

                flags = ("LIM" if cl_ch.any() else "---") + "/" + ("RATE" if cr_ch.any() else "----")
                sub_tag = ""
                if n_sub > 1:
                    _first = int(np.abs(np.asarray(goals[0], dtype=np.int64)
                                        - prev_cmd.astype(np.int64)).max())
                    sub_tag = f" sub={len(goals)}x<={_first}"
                sim = "(sim)" if getattr(bus, "simulated", False) else ""
                print(f"  s{s:02d}{'*' if src == 'ramp' else ' '} "
                      f"plan={np.asarray(plan).tolist()} cmd={cmd.tolist()} "
                      f"d={(cmd.astype(int) - prev_cmd.astype(int)).tolist()} "
                      f"actual={actual.tolist()}{sim} "
                      f"err={(actual.astype(int) - cmd.astype(int)).tolist()} clip={flags}{sub_tag}")

                if preview.poll() == ord('q'):
                    stop_reason = "user_quit"
                    quit_requested = True
                    print("\n[loop] 'q' pressed in a preview window -- stopping after this step; "
                          "the arm holds its last commanded pose.")

                busy = time.perf_counter() - t0
                if busy < step_dt:
                    time.sleep(step_dt - busy)
                elif busy > step_dt * 1.5:
                    print(f"  [timing][warn] step took {busy * 1000:.0f} ms > {args.step_dt_ms} ms "
                          f"budget (bus read/write is the usual cause)")
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if rec:
                    rec.step(chunk=chunk_idx, step=s, src=src,
                             present=present.astype(int).tolist(),
                             requested=np.asarray(plan, dtype=np.int64).astype(int).tolist(),
                             after_limit=lim.astype(int).tolist(),
                             prev_cmd=prev_cmd.astype(int).tolist(),
                             cmd=cmd.astype(int).tolist(),
                             actual=actual.astype(int).tolist(),
                             clip_limit_ch=cl_ch.astype(int).tolist(),
                             clip_rate_ch=cr_ch.astype(int).tolist(),
                             wrote=bool(allow_write), dt_ms=round(dt_ms, 2),
                             substeps=len(goals))
                prev_cmd = cmd
                if quit_requested:
                    break
            if rec:
                rec.flush_frames(chunk_idx, chunk_flagged)
            if quit_requested:
                break

    except KeyboardInterrupt:
        stop_reason = "interrupt"
        print("\n[loop] interrupted by user")
    except Exception as e:
        stop_reason = "error"
        print(f"\n[loop] ERROR {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return _finish(bus, cam, rec, args, dry, chunk_idx, n_clip_limit, n_clip_rate, 1,
                       stop_reason=stop_reason, preview=preview)
    return _finish(bus, cam, rec, args, dry, chunk_idx, n_clip_limit, n_clip_rate, 0,
                   stop_reason=stop_reason, preview=preview)


def _finish(bus, cam, rec, args, dry, chunk_idx, n_clip_limit, n_clip_rate, code,
            stop_reason="max_chunks", preview=None) -> int:
    """Motors -> cameras -> records. Torque off even if setup half-failed."""
    if preview is not None:
        preview.close()
    if bus is not None:
        try:
            if not dry:
                bus.torque(False)
                print("[dxl] torque disabled")
        except Exception as e:
            print(f"[dxl][warn] torque disable failed: {e}")
        try:
            bus.close()
        except Exception as e:
            print(f"[dxl][warn] close failed: {e}")
    if cam is not None:
        cam.close()
    print(f"[loop] chunks={chunk_idx} limit-clips={n_clip_limit} rate-clips={n_clip_rate} "
          f"stop={stop_reason}")
    if rec is not None:
        try:
            rec.add_meta(stop_reason=stop_reason)
        except Exception as e:
            print(f"[rec][warn] stop_reason not recorded: {type(e).__name__}: {e}")
        try:
            rec.summarize()
            rec.label(args.label_value, args.label)
        except Exception as e:
            print(f"[rec][warn] summary/label failed: {type(e).__name__}: {e}")
        rec.close()
    return code


# ============================================================================
# Self-tests
# ============================================================================

def _checker():
    state = {"ok": True}

    def check(name, cond, extra=""):
        state["ok"] = state["ok"] and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")

    return state, check


_FAKE_DEVICES = [
    {"index": 0, "dev": "/dev/video0", "by_id": "/dev/v4l/by-id/x",
     "by_id_name": "usb-046d_081b_200901010001-video-index0", "sys_name": "UVC Camera (046d:081b)"},
    {"index": 4, "dev": "/dev/video4", "by_id": "/dev/v4l/by-id/y",
     "by_id_name": "usb-C270_HD_WEBCAM_C270_HD_WEBCAM-video-index0", "sys_name": "C270 HD WEBCAM"},
    {"index": 2, "dev": "/dev/video2", "by_id": "/dev/v4l/by-id/z",
     "by_id_name": "usb-Jieli_Technology_USB_Composite_Device_123-video-index0",
     "sys_name": "USB Composite Device"},
]


def offline_self_test(args) -> int:
    """No server, no robot, no cameras: safety math, payload bytes, camera assignment, recorder."""
    print("== smolvla_client_local offline self-test (no server, no robot) ==")
    print(f"  deps: cv2={_CV2_OK} PIL={_PIL_OK} requests={_REQ_OK} dynamixel_sdk={_DXL_OK}")
    print(f"  constants: {_CONST_SOURCE}")
    st, check = _checker()

    # --- safety math -----------------------------------------------------------------
    lo, hi = parse_joint_limits(args.joint_limits)
    ms = parse_max_step(args.max_step_ticks)
    check("default --joint-limits parses to 5 pairs", lo.shape == (5,) and hi.shape == (5,),
          f"lo={lo.tolist()} hi={hi.tolist()}")
    check("the gripper channel is bounded to the CONFIRMED [1000,2700]",
          int(lo[GRIPPER_SLOT]) == GRIPPER_TICK_MIN and int(hi[GRIPPER_SLOT]) == GRIPPER_TICK_MAX,
          f"{int(lo[GRIPPER_SLOT])}:{int(hi[GRIPPER_SLOT])}")
    check("default --max-step-ticks is the 150 from log.md's p90", int(ms.min()) == 150,
          f"{ms.tolist()}")
    huge = np.full(5, 999999, dtype=np.int64)
    check("limit clip respects hi", bool(np.all(clip_limits(huge, lo, hi) <= hi)))
    check("limit clip respects lo", bool(np.all(clip_limits(-huge, lo, hi) >= lo)))
    prev = np.array([2048, 2048, 2048, 2048, 2000], dtype=np.int32)
    stepped = rate_limit(prev, huge, ms)
    check("rate limit caps |d| to max_step",
          bool(np.all(np.abs(stepped.astype(int) - prev.astype(int)) <= ms)),
          f"d={(stepped.astype(int) - prev.astype(int)).tolist()}")
    walk = prev.copy()
    for _ in range(10):
        walk = rate_limit(walk, huge, ms)
    check("repeated rate limit converges toward the target (no oscillation)",
          bool(np.all(walk.astype(int) >= prev.astype(int))), f"after 10: {walk.tolist()}")
    check("clip-then-rate never leaves the joint window",
          bool(np.all(rate_limit(prev, clip_limits(huge, lo, hi), ms) <= hi)))
    # the 25 vs 150 story, reproduced from log.md's numbers
    moves = np.array([66, 150, 412], dtype=np.int64)
    check("a 25-tick budget clips even the MEDIAN requested move from log.md",
          bool((moves[0] > 25) and (moves[0] <= 150)), "median 66 > 25, <= 150")

    # --- wire format -----------------------------------------------------------------
    fake = (np.arange(8 * 5, dtype=np.int32) * 3 + 1000).reshape(8, 5)
    b64 = base64.b64encode(fake.tobytes()).decode("ascii")
    check("int32 C-order base64 round-trip", bool(np.array_equal(decode_chunk(b64, 8, 5), fake)))
    try:
        decode_chunk(b64, 9, 5)
        check("wrong chunk_size is rejected", False)
    except ValueError:
        check("wrong chunk_size is rejected", True)

    # --- action smoothing ------------------------------------------------------------
    w3 = gaussian_kernel_1d(3, 0.0)
    check("1x3 Gaussian kernel is normalised and symmetric",
          abs(float(w3.sum()) - 1.0) < 1e-12 and abs(float(w3[0] - w3[2])) < 1e-12,
          f"weights={[round(float(x), 4) for x in w3]}")
    check("the centre tap dominates (a LIGHT blur)", float(w3[1]) > 0.5, f"{float(w3[1]):.3f}")
    for bad in (2, 4, 1):
        try:
            gaussian_kernel_1d(bad)
            check(f"kernel {bad} is rejected", False)
        except ValueError:
            check(f"kernel {bad} is rejected", True)
    const = np.full((8, 5), 2048, dtype=np.int32)
    check("a constant chunk is unchanged (edge padding, no drift to zero)",
          bool(np.array_equal(smooth_chunk(const), const)))
    f3 = forward_taps(w3)
    check("forward taps keep the peak on t and reuse the same weights",
          abs(float(f3[0]) - float(w3[1])) < 1e-12 and float(f3[0]) > float(f3[1])
          and abs(float(f3.sum()) - 1.0) < 1e-12,
          f"taps={[round(float(x), 4) for x in f3]} (t, t+1, t+2)")
    check("forward taps stay monotone for a wider kernel (no weight jumps back up)",
          bool(np.all(np.diff(forward_taps(gaussian_kernel_1d(5, 0.0))) <= 1e-12)),
          f"k5 taps={[round(float(x), 4) for x in forward_taps(gaussian_kernel_1d(5, 0.0))]}")
    ramp = (np.arange(8, dtype=np.int32)[:, None] * 100 + 1000).repeat(5, axis=1).astype(np.int32)
    d_ramp = smooth_chunk(ramp).astype(int) - ramp.astype(int)
    lead = float(sum(i * f3[i] for i in range(f3.size))) * 100.0   # == +71.7 ticks for 1x3
    d_arm = d_ramp[:, 0]                                   # col 0 = an ARM joint; the gripper
    check("the excluded gripper column does not move on a ramp either",   # is excluded by default
          int(np.abs(d_ramp[:, GRIPPER_SLOT]).max()) == 0)
    check("a linear ramp LEADS by a constant offset on its interior rows (no lag)",
          int(np.abs(d_arm[:-2] - int(round(lead))).max()) <= 1,
          f"d={d_arm[0]} ticks vs predicted lead {lead:.1f} on a 100-tick/step ramp")
    check("the lead is forward, never backward (the window never looks at the past)",
          int(d_arm.min()) >= 0, f"min d={int(d_arm.min())}")
    check("only the last K-1 rows fall short of the full lead (tail replication)",
          int(d_arm[-1]) == 0 and 0 < int(d_arm[-2]) < int(round(lead)),
          f"tail d={d_arm[-2]}/{d_arm[-1]}")
    spike = np.full((8, 5), 2000, dtype=np.int32)
    spike[4] = 2600
    sm = smooth_chunk(spike)
    check("a one-step spike is attenuated but not erased",
          2000 < int(sm[4, 0]) < 2600, f"2600 -> {int(sm[4, 0])}")
    check("smoothing reduces step-to-step variation",
          float(np.abs(np.diff(sm, axis=0)).sum()) < float(np.abs(np.diff(spike, axis=0)).sum()))
    check("smoothing preserves shape and int32 dtype",
          sm.shape == spike.shape and sm.dtype == np.int32, f"{sm.shape} {sm.dtype}")
    # the gripper channel must survive a blur bit-exact by default
    grip = np.full((8, JOINT_DIM), 2000, dtype=np.int32)
    grip[:, GRIPPER_SLOT] = [2700, 2700, 2700, 1000, 1000, 1000, 1000, 1000]   # a hard close
    grip[4, 0] = 2600                                                          # arm spike
    g_sm = smooth_chunk(grip)
    check("the gripper channel is bit-exact by default (its edge is NOT rounded)",
          bool(np.array_equal(g_sm[:, GRIPPER_SLOT], grip[:, GRIPPER_SLOT])),
          f"{g_sm[:, GRIPPER_SLOT].tolist()}")
    check("the arm channels ARE still smoothed in the same call",
          int(g_sm[4, 0]) != int(grip[4, 0]), f"2600 -> {int(g_sm[4, 0])}")
    check("--smooth-gripper (exclude=()) does blur the gripper",
          not np.array_equal(smooth_chunk(grip, exclude=())[:, GRIPPER_SLOT],
                             grip[:, GRIPPER_SLOT]))
    check("excluding every channel is a no-op",
          bool(np.array_equal(smooth_chunk(grip, exclude=tuple(range(JOINT_DIM))), grip)))
    jit = np.array([[2000], [2100], [2000], [2100], [2000]], dtype=np.int32)
    check("a wider kernel smooths more than 1x3",
          float(np.abs(np.diff(smooth_chunk(jit, 5), axis=0)).sum())
          < float(np.abs(np.diff(smooth_chunk(jit, 3), axis=0)).sum()))
    check("a 1-row chunk survives the smoother",
          smooth_chunk(np.full((1, 5), 2048, dtype=np.int32)).shape == (1, 5))

    # --- camera identity (no hardware needed) ----------------------------------------
    assigned, notes = assign_cameras(_FAKE_DEVICES)
    check("camera1 <- the C270 with serial 200901010001", assigned[0]["index"] == 0,
          assigned[0]["matched_by"])
    check("camera2 <- the other C270", assigned[1]["index"] == 4, assigned[1]["matched_by"])
    check("camera3 <- the Jieli device", assigned[2]["index"] == 2, assigned[2]["matched_by"])
    shuffled = [_FAKE_DEVICES[2], _FAKE_DEVICES[0], _FAKE_DEVICES[1]]
    a2, _ = assign_cameras(shuffled)
    check("assignment is independent of enumeration order (index churn is survivable)",
          [a["index"] for a in a2] == [0, 4, 2], f"{[a['index'] for a in a2]}")
    try:
        assign_cameras([d for d in _FAKE_DEVICES if d["index"] != 2])
        check("a missing camera is a hard error", False)
    except RuntimeError as e:
        check("a missing camera is a hard error", "camera3" in str(e))
    try:
        assign_cameras(_FAKE_DEVICES + [dict(_FAKE_DEVICES[1], index=6,
                                             by_id_name="usb-C270_HD_WEBCAM_other-video-index0")])
        check("an ambiguous 3rd C270 is a hard error", False)
    except RuntimeError as e:
        check("an ambiguous 3rd C270 is a hard error", "ambiguous" in str(e))

    # --- recorder-matching resize -----------------------------------------------------
    if _CV2_OK:
        cam3 = np.zeros((320, 480, 3), dtype=np.uint8)
        cam3[:, 240:] = 255                       # right half white
        r = match_recorder_size(cam3)
        check("a 480x320 camera3 frame becomes 320x240 (squeeze, not crop)",
              r.shape == (SEND_H, SEND_W, 3), f"{r.shape}")
        check("the resize is NON-UNIFORM: the right half stays the right half",
              bool(r[:, :SEND_W // 2].mean() < 10 and r[:, SEND_W // 2:].mean() > 245),
              f"left={r[:, :SEND_W // 2].mean():.0f} right={r[:, SEND_W // 2:].mean():.0f}")
        already = np.zeros((240, 320, 3), dtype=np.uint8)
        check("a 320x240 frame is passed through untouched",
              match_recorder_size(already) is already)

    # --- payload ----------------------------------------------------------------------
    cam = CameraRig(args.cams, dummy=True, verbose=False)
    frames = cam.read()
    check(f"dummy rig produced {args.cams} RGB {SEND_H}x{SEND_W} frames",
          len(frames) == args.cams and all(f.shape == (SEND_H, SEND_W, 3) for f in frames),
          f"{frames[0].shape}")
    imgs = [encode_jpeg_b64(f, args.jpeg_quality) for f in frames]
    check("JPEG base64 encodes", all(isinstance(s, str) and len(s) > 100 for s in imgs),
          f"len0={len(imgs[0])}")
    if _PIL_OK:
        back = np.asarray(Image.open(io.BytesIO(base64.b64decode(imgs[0]))).convert("RGB"))
        check("server-side decode reproduces the shape", back.shape == frames[0].shape)
        check("channel order survives the round trip (a BGR bug would show here)",
              int(np.argmax(back.reshape(-1, 3).mean(axis=0))) ==
              int(np.argmax(frames[0].reshape(-1, 3).mean(axis=0))))
    raw = json.dumps({"images": imgs, "qpos": [float(x) for x in args.sim_start_ticks]})
    check("payload is JSON-serializable", len(raw) > 0, f"{len(raw) / 1024:.0f} KiB/request")
    cam.close()

    # --- recorder + summary math -------------------------------------------------------
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rec = SessionRecorder(td, args, save_frames="off")
        rec.add_meta(simulated_bus=True)
        rec.chunk(1, np.zeros((8, 5), dtype=np.int32), {"inference_ms": 40.0}, 55.0,
                  np.full(5, 2048, dtype=np.int32), None)
        for i in range(10):
            clipped = i % 2 == 0
            rec.step(chunk=1, step=i, src="chunk",
                     present=[2048] * 5, requested=[2048 + (300 if clipped else 10)] * 5,
                     after_limit=[2048 + (300 if clipped else 10)] * 5, prev_cmd=[2048] * 5,
                     cmd=[2048 + (150 if clipped else 10)] * 5, actual=[2048 + 5] * 5,
                     clip_limit_ch=[0] * 5, clip_rate_ch=[int(clipped)] * 5,
                     wrote=False, dt_ms=100.0)
        s = rec.summarize(print_it=False)
        rec.close()
        check("summary counts the rate-clip rate correctly",
              s["clip"]["rate_clipped_steps"] == 5 and abs(s["clip"]["rate_clipped_frac"] - 0.5) < 1e-9,
              f"{s['clip']['rate_clipped_steps']}/10")
        check("summary reports the REQUESTED move, not the clipped one",
              s["requested_move_ticks"]["max_over_channels_max"] == 300.0
              and s["commanded_move_ticks"]["max_over_channels_max"] == 150.0,
              f"req max {s['requested_move_ticks']['max_over_channels_max']}, "
              f"cmd max {s['commanded_move_ticks']['max_over_channels_max']}")
        check("summary reports the tracking error",
              s["tracking_error_ticks"]["max_over_channels_max"] == 145.0)
        for f in ("steps.jsonl", "chunks.jsonl", "summary.json", "meta.json", "chunk_0001.npy"):
            check(f"{f} was written", (Path(s['dir']) / f).exists())
        lines = (Path(s["dir"]) / "steps.jsonl").read_text().strip().splitlines()
        check("steps.jsonl is one valid JSON object per step",
              len(lines) == 10 and all(json.loads(x)["chunk"] == 1 for x in lines))

    # --- simulated bus ------------------------------------------------------------------
    sb = SimBus([2048] * 5, speed=90)
    sb.write_goal(np.array([2500] * 5))
    p1 = sb.read_present()
    check("SimBus lags the goal by at most `speed` ticks per step",
          bool(np.all(np.abs(p1.astype(int) - 2048) <= 90)), f"{p1.tolist()}")
    check("SimBus is marked simulated", SimBus.simulated is True and DxlBus.simulated is False)

    # --- home start: the per-joint state machine bridged onto the 5-joint sync write --------
    if not _HOME_OK:
        check("src_control.home_return is importable (needed for the home start)", False,
              f"{type(_HOME_ERR).__name__}: {_HOME_ERR}")
    else:
        pose = {"home": {11: 2060, 22: 2099, 23: 1825, 24: 3000, 25: 2634},
                "gripper_open_target": 1000, "gripper_close_target": 2700}
        # The speed ramp itself is covered by `python src_control/home_return.py`; here the
        # point is the BRIDGE (per-joint send -> one sync write, untouched joints hold) and the
        # final pose, so run the ramp uncapped to keep the self-test instant.
        _saved = {k: os.environ.get(k) for k in
                  ("CAPSTONE_HOME_ARM_SPEED", "CAPSTONE_HOME_GRIP_SPEED", "CAPSTONE_HOME_HOLD_SEC")}
        os.environ.update({"CAPSTONE_HOME_ARM_SPEED": "1000000",
                           "CAPSTONE_HOME_GRIP_SPEED": "1000000",
                           "CAPSTONE_HOME_HOLD_SEC": "0.05"})
        try:
            sb2 = SimBus([1048, 2600, 1200, 2100, 2000], speed=90)
            seen = []
            got = home_start_move(sb2, list(ASSUMED_DXL_IDS), lo, hi, pose,
                                  loop_dt=0.0, log=seen.append)
        finally:
            for k, v in _saved.items():
                os.environ.pop(k, None) if v is None else os.environ.update({k: v})
        want = [2060, 2099, 1825, 3000, 2700]
        check("home start lands on control_gamepad's default pose (SimBus)",
              got.tolist() == want, f"{got.tolist()} vs {want}")
        check("home start closes the gripper to the LIMITS end, not the measured tick",
              int(got[GRIPPER_SLOT]) == GRIPPER_TICK_MAX, f"{int(got[GRIPPER_SLOT])}")
        check("home start runs all four phases in order",
              [ph for ph in ("그리퍼 개방", "팔 복귀", "그리퍼 폐쇄", "정지 유지")
               if any(ph in m for m in seen)] ==
              ["그리퍼 개방", "팔 복귀", "그리퍼 폐쇄", "정지 유지"],
              " | ".join(m for m in seen if m.startswith("[home] ")))
        check("home start never commands outside --joint-limits",
              bool(np.all(got >= lo) and np.all(got <= hi)), f"{got.tolist()}")

    # --- background camera grabber ---------------------------------------------------------
    class _FakeCap:
        """A 100 fps capture whose frames carry their own index, so staleness is provable."""

        def __init__(self):
            self.n = 0
            self.released = False

        def read(self):
            self.n += 1
            time.sleep(0.01)
            return True, np.full((4, 4, 3), min(self.n, 255), np.uint8)

        def get(self, prop):
            return time.monotonic() * 1000.0        # every frame is brand new

    fc = _FakeCap()
    g = _CamGrabber(fc, 0)
    g.start()
    f0, age0, seq0 = g.latest()
    check("the grabber hands back a frame without the caller waiting on the capture",
          f0 is not None and age0 >= 0.0, f"age={age0:.1f}ms")
    time.sleep(0.25)
    f1, _age1, seq1 = g.latest()
    check("the grabber keeps advancing, so a later read sees a NEWER frame",
          seq1 > seq0 and int(f1[0, 0, 0]) > int(f0[0, 0, 0]),
          f"seq {seq0}->{seq1}, pixel {int(f0[0,0,0])}->{int(f1[0,0,0])}")
    t_lat = time.perf_counter()
    for _ in range(20):
        g.latest()
    check("latest() never blocks on the capture period",
          (time.perf_counter() - t_lat) * 1000 < 20.0,
          f"20 calls took {(time.perf_counter()-t_lat)*1000:.2f}ms")
    g.stop()
    check("stop() actually joins the grabber thread", not g.is_alive())

    class _DeadCap:
        def read(self):
            return False, None

        def get(self, prop):
            return 0

    gd = _CamGrabber(_DeadCap(), 1)
    gd.MAX_CONSECUTIVE_MISSES = 3
    gd.start()
    _dead_err = ""
    try:
        gd.latest(timeout_s=2.0)
    except RuntimeError as e:
        _dead_err = str(e)
    check("a camera that stops streaming raises instead of hanging the rollout",
          "consecutive read failures" in _dead_err or "no frame within" in _dead_err, _dead_err)
    gd.stop()

    class _BoomCap:
        def read(self):
            raise OSError("USB cable yanked")

        def get(self, prop):
            return 0

    gb = _CamGrabber(_BoomCap(), 2)
    gb.start()
    _boom_err = ""
    try:
        gb.latest(timeout_s=2.0)
    except RuntimeError as e:
        _boom_err = str(e)
    check("a capture that RAISES is reported, not swallowed into a hang",
          "OSError" in _boom_err or "no frame within" in _boom_err, _boom_err)
    gb.stop()

    class _SilentCap:
        """Alive, but never yields a frame -- the case that used to leak a grabber thread."""

        def __init__(self):
            self.released = False

        def read(self):
            time.sleep(0.02)
            return False, None

        def get(self, prop):
            return 0

        def release(self):
            self.released = True

    _silent = [_SilentCap(), _SilentCap()]
    _orig_open = CameraRig._open_all
    CameraRig._open_all = lambda self, verbose=True: setattr(self, "_caps", _silent)
    try:
        CameraRig(2, verbose=False, threaded=True)
        _init_err = ""
    except RuntimeError as e:
        _init_err = str(e)
    finally:
        CameraRig._open_all = _orig_open
    # Either exit is correct: the miss counter usually trips before latest()'s timeout does.
    check("a rig that cannot start gives up instead of hanging",
          ("no frame within" in _init_err) or ("consecutive read failures" in _init_err),
          _init_err)
    check("a failed __init__ leaks no grabber thread",
          not [t for t in threading.enumerate() if t.name.startswith("cam-grab")],
          f"{[t.name for t in threading.enumerate() if t.name.startswith('cam-grab')]}")
    check("a failed __init__ releases the /dev/video nodes it opened",
          all(c.released for c in _silent), f"{[c.released for c in _silent]}")

    rig_d = CameraRig(2, dummy=True, verbose=False)
    check("--dummy-cam starts no grabber threads", rig_d.threaded is False and not rig_d._grabbers)
    check("a dummy rig still reports one age per slot", len(rig_d.read(fresh=True)) == 2)
    rig_d.close()
    check("no grabber thread outlives close()",
          not [t for t in threading.enumerate() if t.name.startswith("cam-grab")],
          f"{[t.name for t in threading.enumerate() if t.name.startswith('cam-grab')]}")

    # --- camera preview: must never be able to kill a rollout -----------------------------
    pv_off = CameraPreview(["a"], enabled=False)
    pv_off.show([np.zeros((4, 4, 3), np.uint8)], 1)
    check("a disabled preview is a silent no-op", pv_off.poll() == -1 and pv_off.close() is None)

    class _BoomCv2:
        FONT_HERSHEY_SIMPLEX = LINE_AA = INTER_NEAREST = 0

        def __getattr__(self, name):
            def boom(*a, **k):
                raise RuntimeError("cannot open display")
            return boom

    _real_cv2, _real_ok = globals()["cv2"], globals()["_CV2_OK"]
    globals()["cv2"], globals()["_CV2_OK"] = _BoomCv2(), True
    try:
        pv = CameraPreview(["a"], enabled=True)
        pv.show([np.zeros((8, 8, 3), np.uint8)], 1)
        check("a headless host disables the preview instead of raising",
              pv.enabled is False and "cannot open display" in pv.reason, pv.reason)
        check("a disabled-mid-run preview keeps answering poll()/close()",
              pv.poll() == -1 and pv.close() is None)
    finally:
        globals()["cv2"], globals()["_CV2_OK"] = _real_cv2, _real_ok

    print(f"== offline self-test {'PASSED' if st['ok'] else 'FAILED'} ==")
    return 0 if st["ok"] else 1


def self_test(args) -> int:
    """Round trip against a running server. No robot, no cameras (--dummy-cam)."""
    print("== smolvla_client_local self-test (server round-trip, no robot) ==")
    print(f"  url={args.url}  deps: cv2={_CV2_OK} PIL={_PIL_OK} requests={_REQ_OK} "
          f"dynamixel_sdk={_DXL_OK}")
    st, check = _checker()
    http = Http(args.url, args.request_timeout_s)
    try:
        health = http.must("/health")
    except Exception as e:
        print(f"  [FAIL] GET /health: {type(e).__name__}: {e}")
        print("  (start smolvla_server_openmani.py first, or use --offline-self-test)")
        print("== self-test FAILED ==")
        return 1
    print(f"  /health -> {json.dumps(health)}")
    n_cams = int(health.get("client_cams", args.cams))
    check("server reports joint_dim==5", health.get("joint_dim") == JOINT_DIM)
    _has_done = bool(health.get("has_done"))
    print(f"  done channel: has_done={_has_done} threshold={health.get('done_threshold')} "
          f"consecutive={health.get('done_consecutive')} -> --stop-on-done "
          f"{args.stop_on_done} would be "
          f"{'ACTIVE' if (_has_done and args.stop_on_done != 'off') else 'inactive'}")
    check("--stop-on-done on is only used against a checkpoint that has the channel",
          not (args.stop_on_done == "on" and not _has_done),
          "--stop-on-done on with has_done=false would abort the rollout at startup")
    check("server reports a positive chunk size", int(health.get("chunk_size", 0)) > 0,
          f"chunk_size={health.get('chunk_size')}")
    check("server is in joint space / raw ticks (matches this client's assumption)",
          health.get("space") in (None, "joint") and health.get("unit") in (None, "raw_ticks"),
          f"space={health.get('space')} unit={health.get('unit')}")
    check("server motor order matches this client's --dxl-ids",
          health.get("motor_order") in (None, [f"id{i}" for i in args.dxl_ids]),
          f"{health.get('motor_order')} vs {[f'id{i}' for i in args.dxl_ids]}")
    tl = health.get("tick_limits") or {}
    if tl:
        g = tl.get("gripper")
        check("server gripper tick limit == this client's",
              g is None or tuple(g) == (GRIPPER_TICK_MIN, GRIPPER_TICK_MAX), f"{g}")

    init = http.must("/init", {"lang": args.lang, "reasoning_cache_n": 0})
    print(f"  /init -> {json.dumps(init)}")
    if init.get("in_training_distribution") is False:
        print(f"  [note] {args.lang!r} is outside the training distribution; that is a real "
              f"warning for a rollout but does not fail this test.")

    cam = CameraRig(n_cams, dummy=True, verbose=False)
    lo, hi = parse_joint_limits(args.joint_limits)
    max_step = parse_max_step(args.max_step_ticks)
    qpos = np.array(args.sim_start_ticks, dtype=np.int32)
    prev_cmd = clip_limits(qpos, lo, hi)
    last = None
    for it in range(args.self_test_iters):
        imgs = [encode_jpeg_b64(f, args.jpeg_quality) for f in cam.read()]
        code, payload = http.request("/predict", {"images": imgs,
                                                  "qpos": [float(x) for x in qpos]})
        if code != 200:
            check(f"iter{it}: /predict returned 200", False, f"HTTP {code}: {payload}")
            break
        ticks = decode_chunk(payload["joint_ticks_b64"], payload["chunk_size"],
                             payload["joint_dim"])
        last = (payload, ticks)
        check(f"iter{it}: dtype int32 and shape [chunk,{JOINT_DIM}]",
              ticks.dtype == np.int32 and ticks.ndim == 2 and ticks.shape[1] == JOINT_DIM,
              f"{ticks.shape}")
        check(f"iter{it}: every tick is inside the joint window",
              bool(np.all(ticks >= lo) and np.all(ticks <= hi)),
              f"min={ticks.min(axis=0).tolist()} max={ticks.max(axis=0).tolist()}")
        check(f"iter{it}: chunk_size >= --exec-steps ({args.exec_steps})",
              ticks.shape[0] >= args.exec_steps, f"chunk_size={ticks.shape[0]}")
        again = np.frombuffer(base64.b64decode(payload["joint_ticks_b64"]),
                              dtype=np.int32).reshape(payload["chunk_size"], payload["joint_dim"])
        check(f"iter{it}: base64 decode is deterministic", bool(np.array_equal(again, ticks)))
        if ticks.shape[0] >= 2:
            d = np.abs(np.diff(ticks.astype(np.int64), axis=0))
            print(f"    per-DOF max|drow|={d.max(axis=0).tolist()} "
                  f"mean|drow|={np.round(d.mean(axis=0), 1).tolist()}")
        print(f"    server meta: max_step_in_chunk={payload.get('max_step_ticks_in_chunk')} "
              f"first_jump={payload.get('first_step_jump_ticks')} "
              f"n_over_rate_limit={payload.get('n_steps_over_rate_limit')} "
              f"clamped={payload.get('n_clamped_values')} execute={payload.get('execute')}")
        print(f"    timings ms: pre={payload['preprocess_ms']:.1f} "
              f"infer={payload['inference_ms']:.1f} post={payload['postprocess_ms']:.1f}")
        if _has_done:
            check(f"iter{it}: done is a bool on a 6-D checkpoint",
                  isinstance(payload.get("done"), bool), f"done={payload.get('done')!r}")
            check(f"iter{it}: done_mean is a number",
                  isinstance(payload.get("done_mean"), (int, float)),
                  f"done_mean={payload.get('done_mean')!r}")
            print(f"    done: flag={payload.get('done')} mean={payload.get('done_mean')} "
                  f"streak={payload.get('done_streak')}/{payload.get('done_consecutive')}")
        else:
            check(f"iter{it}: done is None on a 5-D checkpoint",
                  payload.get("done") is None, f"done={payload.get('done')!r}")

        n_exec = min(args.exec_steps, ticks.shape[0])
        cl = cr = 0
        p = prev_cmd
        for row in ticks[:n_exec]:
            limr = clip_limits(row, lo, hi)
            cl += int(not np.array_equal(limr, row))
            cmd = rate_limit(p, limr, max_step)
            cr += int(not np.array_equal(cmd, limr))
            p = cmd
        print(f"    safety dry-run over {n_exec} steps: limit-clips={cl} rate-clips={cr} "
              f"({100 * cr / max(n_exec, 1):.0f}% rate) final={p.tolist()}")
        check(f"iter{it}: the safety pipeline keeps commands inside the joint window",
              bool(np.all(p >= lo) and np.all(p <= hi)))
        if cr > 0.5 * n_exec:
            print(f"    [!] over half the steps would be rate-clipped at "
                  f"--max-step-ticks {max_step.tolist()}")
        prev_cmd = p
        qpos = p

    # contract: wrong shapes must be REJECTED, not silently truncated
    code, _ = http.request("/predict", {"images": [], "qpos": [2048.0] * JOINT_DIM})
    check("empty image list is rejected (HTTP 4xx)", 400 <= code < 500, f"HTTP {code}")
    one = encode_jpeg_b64(cam.read()[0], args.jpeg_quality)
    code, payload = http.request("/predict", {"images": [one] * (n_cams + 1),
                                              "qpos": [2048.0] * JOINT_DIM})
    check(f"{n_cams + 1} images (server wants {n_cams}) is rejected (HTTP 4xx)",
          400 <= code < 500, f"HTTP {code}")
    code, _ = http.request("/predict", {"images": [one] * n_cams,
                                        "qpos": [2048.0] * (JOINT_DIM - 1)})
    check("wrong qpos length is rejected (HTTP 4xx)", 400 <= code < 500, f"HTTP {code}")
    cam.close()
    check("at least one successful /predict round trip", last is not None)
    print(f"== self-test {'PASSED' if st['ok'] else 'FAILED'} ==")
    return 0 if st["ok"] else 1


def list_cameras(args) -> int:
    entries = enumerate_video_devices()
    print(f"== V4L2 capture nodes with a stable by-id/by-path name ({len(entries)}) ==")
    for e in entries:
        print(f"  /dev/video{e['index']:<3d} {e['sys_name']!r:<36s} "
              f"[{e.get('id_source', '?'):<7s}] {e['by_id_name']}")
    if not entries:
        print("  (none -- check `ls /dev/v4l/by-id/ /dev/v4l/by-path/`, USB power, and that no "
              "other process holds the cameras)")
        return 1
    try:
        # honour the same override the rollout path uses, so --list-cameras verifies the
        # exact binding the next run will get instead of re-deriving a different one
        if args.cam_devices:
            assigned = resolve_manual(args.cam_devices.split(","), entries)
            if len(assigned) != len(CAM_SPECS):
                print(f"\n[FAIL] --cam-devices needs {len(CAM_SPECS)} entries, got {len(assigned)}")
                return 1
        else:
            assigned, _notes = assign_cameras(entries)
    except RuntimeError as e:
        print(f"\n[FAIL] {e}")
        return 1
    print("\n== slot assignment ==")
    for i, (sp, a) in enumerate(zip(CAM_SPECS, assigned)):
        print(f"  slot{i} {sp.key:<8s} ({sp.role:<5s}) <- /dev/video{a['index']:<3d} "
              f"[{a['matched_by']}]  will request MJPG {sp.width}x{sp.height}@{sp.fps}, "
              f"send {SEND_W}x{SEND_H}")
    print("\nNOTE: this only proves the DEVICES are distinct and stable. It cannot prove that "
          "camera1 is\nphysically the RIGHT camera. Compare one frame per slot against "
          "scratchpad/camcheck/cam{1,2,3}_from_image_{0,1,2}.png before the first rollout.")
    return 0


# ============================================================================
# Entry
# ============================================================================

def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def resolve_voice_instruction(args) -> str:
    """--input-mode voice: 마이크 -> STT -> 54문장 스냅 -> 확인. 확정된 문장을 돌려준다.

    서버는 받은 문장이 학습 집합 안인지만 알려주고(`in_training_distribution`) 집합 자체는
    돌려주지 않으므로, 후보 생성과 스냅은 클라이언트가 한다. 서버 쪽 변경은 전혀 없다 --
    /init 에는 평소와 똑같이 완성된 영어 문장 하나가 간다.

    기본값이 '확인받고 진행' 인 이유: 스냅이 틀리면 팔이 다른 물체를 집는다. 텍스트 모드는
    사람이 이미 문장을 보고 쳤지만 음성 모드에는 그 단계가 없다.
    """
    try:
        import stt_lang
    except ImportError as e:
        raise SystemExit(f"[voice] stt_lang.py 를 import 할 수 없습니다: {e}")

    mics = [] if args.voice_wav else stt_lang.list_mics()
    if args.voice_wav:
        print(f"[voice] 파일 입력 모드: {args.voice_wav} (마이크를 쓰지 않습니다)")
    print(f"[voice] 캡처 장치 {len(mics)}개" + (f": {mics[0]}" if mics else " -- arecord 가 "
          "장치를 못 찾았습니다. --voice-device 로 지정하거나 마이크를 확인하세요."))
    print(f"[voice] 스냅 대상 {len(stt_lang.TASKS)}문장 "
          f"(python stt_lang.py --list-tasks 로 전체 확인)")

    attempts = max(1, int(args.voice_retries))
    for attempt in range(1, attempts + 1):
        try:
            if args.voice_wav:
                result, heard = stt_lang.snap_wav(
                    args.voice_wav, model_size=args.voice_model,
                    language=args.voice_language)
            else:
                result, heard = stt_lang.listen_and_snap(
                    seconds=args.voice_seconds, device=args.voice_device,
                    model_size=args.voice_model, language=args.voice_language)
        except (RuntimeError, OSError) as e:
            raise SystemExit(f"[voice] {e}")
        if args.voice_wav and not result.ok:
            # 같은 파일을 다시 읽으면 결과가 같다. 재시도는 의미가 없다.
            raise SystemExit(f"[voice] {result.explain()}")

        if not result.ok:
            print(f"[voice] {result.explain()}")
            print('[voice] 예: "작은테이프를 나한테 놓아줘" / "큰테이프를 오른쪽으로 밀어줘"')
            if attempt < attempts:
                print(f"[voice] 다시 시도 ({attempt + 1}/{attempts}) ...")
                continue
            raise SystemExit("[voice] 스냅에 계속 실패했습니다. --input-mode text 로 "
                             "--lang 을 직접 주세요.")

        print(f"[voice] 스냅: {result.explain()}")
        if args.voice_yes:
            return result.sentence
        try:
            ans = input(f'[voice] 이 명령으로 실행합니까? "{result.sentence}"  [Enter=예 / n=다시] ')
        except EOFError:          # 파이프로 돌릴 때는 되묻지 않는다
            print("[voice] stdin 이 없어 확인을 건너뜁니다 (--voice-yes 와 동일).")
            return result.sentence
        if ans.strip().lower() not in ("n", "no", "ㅜ", "아니"):
            return result.sentence
        if attempt < attempts:
            print(f"[voice] 다시 말하세요 ({attempt + 1}/{attempts}) ...")
    raise SystemExit("[voice] 확정되지 않았습니다. --input-mode text 로 --lang 을 직접 주세요.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="SmolVLA server base URL")
    ap.add_argument("--lang", default="place the tennis ball on the left",
                    help="task instruction sent once via /init. MUST be one of the 36 training "
                         "sentences; /init reports in_training_distribution. Ignored when "
                         "--input-mode voice supplies one.")
    ap.add_argument("--input-mode", choices=("text", "voice"), default="text",
                    help="where the instruction comes from. text (default) = --lang verbatim, "
                         "exactly as before. voice = record the mic, transcribe it, and SNAP the "
                         "result onto one of the 36 training sentences (stt_lang.py).")
    ap.add_argument("--voice-seconds", type=float, default=5.0,
                    help="--input-mode voice: how long to record one utterance")
    ap.add_argument("--voice-device", default="plughw:Device_1,0",
                    help="--input-mode voice: ALSA capture device. Default plughw:Device_1,0 -- "
                         "the JMTek USB adapter, the only device on this PC that actually "
                         "picked up speech (robot PC 2026-10-07; PCH and the webcam mics "
                         "transcribed to empty). python stt_lang.py --mics lists them.")
    ap.add_argument("--voice-model", default="small",
                    help="--input-mode voice: whisper model size (tiny/base/small/medium/large)")
    ap.add_argument("--voice-language", default=None,
                    help="--input-mode voice: force the spoken language, e.g. ko or en. "
                         "Default: let whisper detect it")
    ap.add_argument("--voice-wav", default=None,
                    help="--input-mode voice: transcribe THIS wav file instead of recording the "
                         "mic. For testing over SSH, where no one is at the microphone.")
    ap.add_argument("--voice-yes", action="store_true",
                    help="--input-mode voice: accept the snapped sentence without asking. OFF by "
                         "default -- a mis-snap moves the arm to the wrong object, so the "
                         "default is to show the snap and wait for confirmation.")
    ap.add_argument("--voice-retries", type=int, default=3,
                    help="--input-mode voice: how many times to re-record when the snap fails "
                         "or is rejected")
    ap.add_argument("--request-timeout-s", type=float, default=120.0)
    ap.add_argument("--out-dir", default="runs/rollouts",
                    help="parent dir for rollout_<timestamp>/ session records ('' disables)")

    # ---- cameras ----
    ap.add_argument("--cams", type=int, default=DEFAULT_CAMS,
                    help="client image slots (the server's /health overrides this)")
    ap.add_argument("--cam-devices", type=_parse_str_list, default=None,
                    help="MANUAL override, 3 comma-separated entries: /dev/video indices, "
                         "/dev paths, or by-id substrings. Default: automatic by-id binding.")
    ap.add_argument("--substeps", type=int, default=50,
                    help="split each control step into N goal writes so the arm sweeps instead "
                         "of dashing (the servos have Profile Velocity=0, so a single write is "
                         "a full-speed dash). 1 = off. 10 puts a write every 10 ms at "
                         "--step-dt-ms 100, matching the home return's smoothness. The pose at "
                         "each step boundary is unchanged, so the 10 Hz training distribution "
                         "is untouched; the arm reaches each waypoint ~half a step later.")
    ap.add_argument("--no-cam-threads", dest="cam_threads", action="store_false",
                    help="read the cameras inline instead of from background grabber threads. "
                         "Correct either way, but inline costs ~200 ms per chunk waiting for a "
                         "fresh frame. Default: threads.")
    ap.set_defaults(cam_threads=True)
    ap.add_argument("--stale-frames", dest="fresh_frames", action="store_false",
                    help="do NOT drop frames the V4L2 queue held while the loop was busy "
                         "(old behaviour: the first observation after the home start, and "
                         "after every chunk, is as old as the gap). Default: drop them.")
    ap.set_defaults(fresh_frames=True)
    ap.add_argument("--swap-cam01", action="store_true",
                    help="swap the devices bound to image_0 and image_1 (slot0 <-> slot1). "
                         "OFF by default. Use when the two C270s changed places.")
    ap.add_argument("--dummy-cam", action="store_true",
                    help="synthetic frames instead of real cameras (no hardware needed)")
    ap.add_argument("--jpeg-quality", type=int, default=90,
                    help="90 == dex_ref/client_live.py, i.e. what the recorder used")
    ap.add_argument("--save-frames", choices=("off", "all", "flagged"), default="off",
                    help="store the sent frames: 'flagged' keeps only chunks where a clip fired")

    # ---- dynamixel ----
    ap.add_argument("--dxl-ids", type=_parse_int_list, default=list(ASSUMED_DXL_IDS),
                    help="motor IDs in policy order [j1,j2,j3,j4,gripper]")
    ap.add_argument("--port", default=ASSUMED_PORT)
    ap.add_argument("--baud", type=int, default=ASSUMED_BAUD)

    # ---- control ----
    ap.add_argument("--exec-steps", type=int, default=8,
                    help="execute the first N rows of each chunk, then re-infer. 8 == the "
                         "checkpoint's n_action_steps.")
    ap.add_argument("--step-dt-ms", type=int, default=100,
                    help="ms per executed step. MUST match the dataset rate (10 Hz = 100 ms)")
    ap.add_argument("--ramp-steps", type=int, default=5,
                    help="first chunk only: linear ramp present -> chunk[0] (anti-jerk). 0=off")
    ap.add_argument("--max-chunks", type=int, default=10_000)
    ap.add_argument("--smooth-actions", action=argparse.BooleanOptionalAction, default=False,
                    help="DEFAULT OFF. Gaussian-blur the server's chunk ALONG TIME (between "
                         "action steps, per joint) before it is executed. FORWARD window: "
                         "row t mixes t..t+K-1 with the peak weight on t, so there is no "
                         "phase lag. The raw chunk is still recorded as chunk_<i>_raw.npy.")
    ap.add_argument("--smooth-gripper", action=argparse.BooleanOptionalAction, default=False,
                    help=f"DEFAULT OFF: the gripper channel ({MOTOR_LABELS[GRIPPER_SLOT]}) is "
                         f"NOT blurred, only the {JOINT_DIM - 1} arm joints are. Pass "
                         f"--smooth-gripper to blur all {JOINT_DIM} channels.")
    ap.add_argument("--smooth-kernel", type=int, default=3,
                    help="odd kernel length in STEPS for --smooth-actions (default 3 = 1x3, "
                         "i.e. row t mixes t, t+1, t+2)")
    ap.add_argument("--smooth-sigma", type=float, default=0.0,
                    help="Gaussian sigma in steps; <=0 uses the cv2 auto rule (0.8 for "
                         "kernel 3 -> weights [0.239, 0.522, 0.239], laid out FORWARD as "
                         "[0.522, 0.239, 0.239] on t, t+1, t+2)")

    # ---- termination (done_flag) ----
    ap.add_argument("--stop-on-done", choices=("auto", "on", "off"), default="auto",
                    help="what to do with the server's `done` field, which exists ONLY on a 6-D "
                         "checkpoint (5 motor channels + done_flag). 'auto' (default) ends the "
                         "rollout when the server reports done=true, and is a no-op against a "
                         "5-D checkpoint, which has no such channel; 'on' is the same but "
                         "REFUSES to run against a 5-D checkpoint; 'off' never ends early. The "
                         "threshold and the consecutive-hit count live on the SERVER "
                         "(--done-threshold / --done-consecutive) -- this client only obeys the "
                         "boolean. done_mean is printed and recorded either way.")

    # ---- safety ----
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                    help="DEFAULT ON. Never writes to a motor; prints/records what it would "
                         "send. Pass --no-dry-run to actually drive the arm.")
    ap.add_argument("--joint-limits", default=default_joint_limits(),
                    help="per-joint 'lo:hi' tick window (1 entry = all joints, or 5 comma-sep). "
                         "Default: arm 0:4095 (UNCONFIRMED), gripper 1000:2700 (confirmed).")
    ap.add_argument("--max-step-ticks", default="150",
                    help="max |dtick| per control step. Default 150 = the p90 requested move in "
                         "log.md; 25 clipped 94.4%% of that rollout's steps.")
    ap.add_argument("--arm-delay-s", type=float, default=3.0,
                    help="countdown before torque-on when --no-dry-run")
    ap.add_argument("--sim-start-ticks", type=_parse_int_list, default=[2048] * JOINT_DIM,
                    help="pose the simulated arm starts from when no bus is reachable")
    ap.add_argument("--sim-speed", type=int, default=90,
                    help="ticks/step the SIMULATED arm can move (dry-run without hardware)")

    # ---- start pose ----
    ap.add_argument("--home-pose", type=Path,
                    default=(default_home_pose_path() if _HOME_OK else Path("home_pose.json")),
                    help="home_pose.json measured by record_home_pose.py; the pose "
                         "control_gamepad.py returns to at the end of every recorded episode")
    ap.add_argument("--no-show-cams", dest="show_cams", action="store_false",
                    help="do NOT open the live camera windows (headless hosts, or when the "
                         "GUI cost is unwanted). Default is to show them, like control_gamepad.py")
    ap.add_argument("--cam-preview-scale", type=float, default=1.0,
                    help="upscale factor for the preview windows only (the frames SENT to the "
                         "server are always 320x240). 2.0 = 640x480 like the recorder's windows")
    ap.set_defaults(show_cams=True)
    ap.add_argument("--no-home-start", dest="home_start", action="store_false",
                    help="do NOT move to the home pose first; start the rollout from wherever "
                         "the arm is (default is to home first, so step 0 matches training)")
    ap.set_defaults(home_start=True)

    # ---- labelling ----
    ap.add_argument("--label", action="store_true",
                    help="prompt for a success/fail label at the end of the rollout")
    ap.add_argument("--label-value", default=None,
                    help="non-interactive label (success|fail|partial|aborted)")

    # ---- tests ----
    ap.add_argument("--self-test", action="store_true",
                    help="round-trip against a running server with synthetic frames (no robot)")
    ap.add_argument("--offline-self-test", action="store_true",
                    help="no server, no robot: safety math, wire format, camera assignment, recorder")
    ap.add_argument("--self-test-iters", type=int, default=2)
    ap.add_argument("--list-cameras", action="store_true",
                    help="enumerate /dev/v4l/by-id and show the slot assignment, then exit")

    args = ap.parse_args()
    if args.input_mode == "voice":
        args.lang = resolve_voice_instruction(args)
    if len(args.dxl_ids) != JOINT_DIM:
        ap.error(f"--dxl-ids needs {JOINT_DIM} ids, got {args.dxl_ids}")
    if len(args.sim_start_ticks) != JOINT_DIM:
        ap.error(f"--sim-start-ticks needs {JOINT_DIM} values, got {args.sim_start_ticks}")
    if args.label_value and args.label_value not in ("success", "fail", "partial", "aborted"):
        ap.error("--label-value must be success|fail|partial|aborted")

    if args.list_cameras:
        return list_cameras(args)
    if args.offline_self_test:
        return offline_self_test(args)
    if args.self_test:
        return self_test(args)
    return run_loop(args)


if __name__ == "__main__":
    sys.exit(main())
