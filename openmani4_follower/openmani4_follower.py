#!/usr/bin/env python
"""lerobot `Robot` implementation for our 4-DoF OpenMANIPULATOR-X + gripper.

Differences vs. the `omx_follower` template that this is derived from
(`lerobot/src/lerobot/robots/omx_follower/omx_follower.py`):

1.  5 motors (`joint1..joint4`, `gripper`), all `xm430-w350`, IDs 11/22/23/24/25.
2.  **Units are explicit.** The template feeds `MotorNormMode.RANGE_M100_100`
    (gripper: `RANGE_0_100`) to the bus, so its "positions" are neither ticks nor
    radians. Our training data is raw ticks from our own h5 recordings, so this class
    always reads/writes the bus with `normalize=False` (raw ticks on the wire) and does
    the unit conversion itself, according to `config.unit` (`raw_ticks` | `radians`).
    The active unit is logged on connect and, when `config.dataset_root` is given,
    checked against the unit the converter recorded in `openmani_meta.json`
    (`check_dataset_units()`): a dataset/robot unit mismatch is a hard error, not a
    warning, because it is invisible until the arm moves.
3.  **Safety.** Per-motor tick limits + per-step rate limit, both counted, with the clip
    rate reported. Reference point from the only real-robot log we have, `vla/log.md`:
    178 executed steps, 168 of them (94.4%) rate-clipped at max_step_ticks=25 -- the
    motion in that rollout was the limiter's, not the policy's. `dry_run=True` by
    default: nothing is ever written to the motors.
4.  **No automatic calibration write.** The template writes factory defaults into the
    motors on connect; we only read, unless `config.allow_calibration_write=True`.
5.  The `DynamixelMotorsBus` is created on `connect()` instead of `__init__()`, so this
    class can be imported / instantiated / unit-converted on machines that do not have
    `dynamixel-sdk` installed (the robot lives on another PC). Everything the
    `Robot` interface exposes before connecting (`observation_features`,
    `action_features`) is derived from `self.motors`, not from the bus.

The class is a plain object: it does not depend on any lerobot runner, so it can also be
driven from our own HTTP policy client (see `action_from_array()` / `state_to_array()`).
"""

import json
import logging
import math
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_openmani4_follower import (
    CENTER_TICK,
    DATASET_META_FILENAME,
    GRIPPER_NAME,
    GRIPPER_TICK_MAX,
    GRIPPER_TICK_MIN,
    MOTOR_IDS,
    MOTOR_MODEL,
    MOTOR_NAMES,
    TICK_MAX,
    TICK_MIN,
    TICKS_PER_REV,
    UNIT_RADIANS,
    UNIT_RAW_TICKS,
    Openmani4FollowerConfig,
)

logger = logging.getLogger(__name__)

RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_REV

# --------------------------------------------------------------------------------------
# Unit conversion -- the ONLY place where ticks <-> radians <-> lerobot-norm happens.
# --------------------------------------------------------------------------------------


def ticks_to_radians(tick: float) -> float:
    """Encoder tick -> radians, 0 rad at `CENTER_TICK` (2048)."""
    return (float(tick) - CENTER_TICK) * RAD_PER_TICK


def radians_to_ticks(rad: float) -> float:
    """Radians -> encoder tick (float; caller rounds)."""
    return float(rad) / RAD_PER_TICK + CENTER_TICK


class DatasetUnitMismatch(RuntimeError):
    """The dataset was written in a different unit than this robot is configured for."""


def read_dataset_meta(dataset_root: str | Path) -> dict:
    """Load `<dataset_root>/openmani_meta.json`, the converter's unit contract.

    It is a sidecar rather than a key inside `meta/info.json` because lerobot's
    `DatasetInfo.from_dict` drops unknown info.json fields and rewrites the file from the
    dataclass on every metadata update, so a custom key there disappears silently.
    """
    path = Path(dataset_root).expanduser() / DATASET_META_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. It is written by convert_h5_to_lerobot_v3.py and records "
            "the unit the dataset is in. Re-run the converter (or drop `dataset_root` and "
            "accept that the unit contract is unchecked)."
        )
    return json.loads(path.read_text())


def check_dataset_units(dataset_root: str | Path, unit: str) -> dict:
    """Raise unless the dataset was written in `unit`. Returns the parsed metadata.

    Why this is fatal and not a warning: units are an affine change, so training,
    evaluation and every offline metric look perfectly healthy in the wrong unit. It only
    becomes visible when the arm executes a radian value as a tick and drives into its
    lower limit. The previous stack got away with a rad-valued dataset only because a
    server converted rad->tick before the wire (server_infer.py:358-367); nothing in this
    stack does that.
    """
    meta = read_dataset_meta(dataset_root)
    ds_unit = meta.get("unit")
    if ds_unit != unit:
        raise DatasetUnitMismatch(
            f"dataset {Path(dataset_root)} was written with unit={ds_unit!r} (space="
            f"{meta.get('space')!r}) but this robot is configured for unit={unit!r}. "
            "Either set config.unit to the dataset's unit or re-convert with the matching "
            "--joint-units. Refusing to command the arm in the wrong unit."
        )
    if meta.get("space") != "joint":
        raise DatasetUnitMismatch(
            f"dataset {Path(dataset_root)} has space={meta.get('space')!r}, which this robot "
            f"cannot execute: {meta.get('deployable_reason')}"
        )
    if meta.get("motor_order") and list(meta["motor_order"]) != [
        f"id{MOTOR_IDS[name]}" for name in MOTOR_NAMES
    ]:
        raise DatasetUnitMismatch(
            f"dataset motor order {meta['motor_order']} != this robot's "
            f"{[f'id{MOTOR_IDS[n]}' for n in MOTOR_NAMES]}. Nothing reorders action "
            "channels for SmolVLA, so this would command the wrong joints."
        )
    return meta


class Openmani4Follower(Robot):
    """OpenMANIPULATOR-X derived 4-DoF arm + gripper (5x XM430-W350)."""

    config_class = Openmani4FollowerConfig
    name = "openmani4_follower"

    def __init__(self, config: Openmani4FollowerConfig):
        super().__init__(config)
        self.config = config

        # Canonical channel order: dict insertion order == dataset state/action order.
        # ids 11, 22, 23, 24, 25 -- see config_openmani4_follower.MOTOR_IDS.
        norm_mode_body = MotorNormMode.RANGE_M100_100
        self.motors: dict[str, Motor] = {
            name: Motor(
                MOTOR_IDS[name],
                MOTOR_MODEL,
                MotorNormMode.RANGE_0_100 if name == GRIPPER_NAME else norm_mode_body,
            )
            for name in MOTOR_NAMES
        }
        # These norm modes are only used when `unit == "lerobot_norm"`; the bus itself is
        # always addressed with `normalize=False`.

        self.bus: Any = None  # DynamixelMotorsBus, created in connect()
        self.cameras = make_cameras_from_configs(config.cameras)

        self._joint_limits: dict[str, tuple[int, int]] = dict(config.joint_limits)
        self._max_step: dict[str, float | None] = self._resolve_max_step(config.max_step_ticks)
        self._prev_cmd_ticks: dict[str, int] | None = None
        # The 4 arm joints still sitting on the full encoder range means nobody has
        # confirmed their real travel (bench/fk_open_manipulator.py, which carries the URDF
        # limits, is not on this machine). Warn once, at connect.
        self._arm_limits_unconfirmed = [
            name for name in MOTOR_NAMES
            if name != GRIPPER_NAME and self._joint_limits[name] == (TICK_MIN, TICK_MAX)
        ]
        self._flip_h: dict[str, bool] = {key: config.flip_h_for(key) for key in config.cameras}
        self.dataset_meta: dict | None = None

        # Clip accounting.
        self._n_steps = 0
        self._n_rate_clipped_steps = 0
        self._n_limit_clipped_steps = 0
        self._rate_clip_counts: dict[str, int] = dict.fromkeys(MOTOR_NAMES, 0)
        self._limit_clip_counts: dict[str, int] = dict.fromkeys(MOTOR_NAMES, 0)

    @staticmethod
    def _resolve_max_step(spec: float | dict[str, float] | None) -> dict[str, float | None]:
        if spec is None:
            return dict.fromkeys(MOTOR_NAMES, None)
        if isinstance(spec, dict):
            return {name: float(spec[name]) for name in MOTOR_NAMES}
        return dict.fromkeys(MOTOR_NAMES, float(spec))

    # ---------------------------------------------------------------------------------
    # Features
    # ---------------------------------------------------------------------------------
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            key: (cfg.height, cfg.width, 3) for key, cfg in self.config.cameras.items()
        }  # -> dataset keys `observation.images.<key>`

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def motor_names(self) -> tuple[str, ...]:
        """Canonical order of the state/action vector."""
        return tuple(self.motors)

    @property
    def unit(self) -> str:
        return self.config.unit

    # ---------------------------------------------------------------------------------
    # Unit conversion bound to this robot's motors / calibration
    # ---------------------------------------------------------------------------------
    def ticks_to_unit(self, motor: str, tick: float) -> float:
        unit = self.config.unit
        if unit == UNIT_RAW_TICKS:
            return float(tick)
        if unit == UNIT_RADIANS:
            return ticks_to_radians(tick)
        raise ValueError(f"unknown unit {unit!r}")

    def unit_to_ticks(self, motor: str, value: float) -> float:
        unit = self.config.unit
        if unit == UNIT_RAW_TICKS:
            return float(value)
        if unit == UNIT_RADIANS:
            return radians_to_ticks(value)
        raise ValueError(f"unknown unit {unit!r}")

    # ---------------------------------------------------------------------------------
    # Helpers for the "policy only, our own HTTP server" path
    # ---------------------------------------------------------------------------------
    def action_from_array(self, values) -> RobotAction:
        """(5,) array in `config.unit`, in MOTOR_NAMES order -> RobotAction dict."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != len(self.motors):
            raise ValueError(f"expected {len(self.motors)} values in order {self.motor_names}, got {arr.shape[0]}")
        return {f"{name}.pos": float(v) for name, v in zip(self.motor_names, arr, strict=True)}

    def state_to_array(self, observation: RobotObservation) -> np.ndarray:
        """RobotObservation -> (5,) float32 state vector in MOTOR_NAMES order."""
        return np.array([float(observation[f"{name}.pos"]) for name in self.motor_names], dtype=np.float32)

    # ---------------------------------------------------------------------------------
    # Connection
    # ---------------------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        if self.bus is None or not self.bus.is_connected:
            return False
        return all(cam.is_connected for cam in self.cameras.values())

    def _make_bus(self):
        from lerobot.motors.dynamixel import DynamixelMotorsBus

        return DynamixelMotorsBus(
            port=self.config.port,
            motors=self.motors,
            calibration=self.calibration,
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Unit contract first: it costs nothing and it is pointless to open the bus if the
        # dataset we are about to replay was recorded in another unit.
        if self.config.dataset_root:
            self.dataset_meta = check_dataset_units(self.config.dataset_root, self.config.unit)
            logger.info(
                "%s dataset unit check PASSED: %s declares unit=%s space=%s motor_order=%s "
                "(fps=%s, %s)",
                self,
                Path(self.config.dataset_root) / DATASET_META_FILENAME,
                self.dataset_meta.get("unit"),
                self.dataset_meta.get("space"),
                self.dataset_meta.get("motor_order"),
                self.dataset_meta.get("fps"),
                self.dataset_meta.get("fps_source"),
            )
        else:
            logger.warning(
                "%s config.dataset_root is not set: the unit of the training data is "
                "UNCHECKED. Running with unit=%s on a dataset written in the other unit "
                "produces plausible-looking commands that are physically wrong. Point "
                "dataset_root at the dataset root (the directory holding %s).",
                self, self.config.unit, DATASET_META_FILENAME,
            )

        if self.bus is None:
            # (already-set bus is reused: lets tests inject a fake bus)
            self.bus = self._make_bus()

        # `DynamixelMotorsBus.default_baudrate` is 1_000_000, which is also ours, so the
        # handshake (pings every expected motor) runs normally. For any other baudrate the
        # port must be re-configured before the motors can answer, hence handshake=False.
        handshake = self.config.baudrate == self.bus.default_baudrate
        self.bus.connect(handshake=handshake)
        if not handshake:
            self.bus.set_baudrate(self.config.baudrate)
            logger.warning(
                "%s baudrate %d != bus default %d: skipped the motor-presence handshake; a wrong ID "
                "will only show up as a read failure below.",
                self,
                self.config.baudrate,
                self.bus.default_baudrate,
            )

        logger.info(
            "%s bus open on %s @ %d baud | unit=%s | dry_run=%s | motor order (== dataset "
            "state/action order): %s",
            self,
            self.config.port,
            self.config.baudrate,
            self.config.unit,
            self.config.dry_run,
            ", ".join(f"{i}:{name}(id={m.id},{m.model})" for i, (name, m) in enumerate(self.motors.items())),
        )
        logger.info(
            "%s safety: joint_limits(ticks)=%s max_step_ticks=%s",
            self,
            self._joint_limits,
            self._max_step,
        )
        logger.warning(
            "%s max_step_ticks=%s is DERIVED FROM A LOG, NOT FROM A HARDWARE SPEC: it is the "
            "p90 of the per-step motion requested in vla/log.md (178 steps; 168 of them, "
            "94.4%%, were rate-clipped at 25; requested |plan-actual| median 66 / p90 150 / "
            "max 412 ticks). CONFIRM IT ON THE REAL ARM before running with dry_run=False.",
            self, self._max_step,
        )
        if self._arm_limits_unconfirmed:
            logger.warning(
                "%s joint_limits for %s are the full encoder range (0, %d) = UNCONFIRMED. The "
                "URDF travel limits live in bench/fk_open_manipulator.py, which is not on this "
                "machine, so nothing here knows where this arm actually hits itself. Measure "
                "them before dry_run=False. (The gripper is bounded at (%d, %d) from "
                "server_infer.py:352-355.)",
                self, self._arm_limits_unconfirmed, TICK_MAX, GRIPPER_TICK_MIN, GRIPPER_TICK_MAX,
            )

        if calibrate and not self.is_calibrated:
            logger.info("%s calibration in motors differs from the calibration file (or none found).", self)
            self.calibrate()
        else:
            # Read-only refresh so that `lerobot_norm` uses the real motor ranges.
            self.calibration = self.bus.read_calibration()

        for key, cam in self.cameras.items():
            cam.connect()
            cam_cfg = self.config.cameras[key]
            # Log what each slot actually resolved to. The device index is hardware
            # dependent (UVC cameras usually take the even /dev/video* nodes, hence the
            # 0,2,4 default) and a slot swap is a silent, plausible-looking failure.
            shape = None
            try:
                frame = cam.read_latest()
                shape = tuple(np.asarray(frame).shape)
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                logger.warning("%s camera '%s': first read failed (%s)", self, key, exc)
            logger.info(
                "%s camera '%s' -> observation.images.%s | device=%s requested_hxw=%sx%s "
                "first_frame_shape=%s flip_h=%s",
                self, key, key,
                getattr(cam_cfg, "index_or_path", "?"),
                getattr(cam_cfg, "height", "?"), getattr(cam_cfg, "width", "?"),
                shape, self._flip_h.get(key, False),
            )
        logger.warning(
            "%s CAMERA SLOT IDENTITY IS UNVERIFIED: nothing here checks that camera1/2/3 are "
            "the same physical cameras, in the same order, as in the training data (pixel "
            "statistics did not separate the three). Confirm the mapping with the operator. "
            "flip_h=%s -- the old server needed FLIP_CAM_H=1 to stop the arm from chasing the "
            "object's mirror image (server_infer.py:591-595); if the live view is mirrored "
            "w.r.t. training, set camera_flip_h.",
            self, self._flip_h,
        )

        n_expected = len(self.config.cameras)
        n_open = sum(1 for cam in self.cameras.values() if cam.is_connected)
        if n_open != n_expected:
            raise RuntimeError(
                f"{self}: {n_open}/{n_expected} cameras connected "
                f"({sorted(self.cameras)}). Refusing to run with missing cameras."
            )

        self.configure()
        self._ramp_up()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return self.bus is not None and self.bus.is_calibrated

    def calibrate(self) -> None:
        """Read calibration from the motors. Writes only if `allow_calibration_write`.

        The omx template (omx_follower.py:118-131) unconditionally writes factory
        defaults (homing_offset=0, range 0..4095) into the motors and saves them to disk.
        That silently changes the hardware, so it is opt-in here.
        """
        if self.bus is None:
            raise RuntimeError(f"{self}: calibrate() requires a connected bus.")

        if not self.config.allow_calibration_write:
            self.calibration = self.bus.read_calibration()
            logger.warning(
                "%s calibration READ ONLY (allow_calibration_write=False): nothing written to the "
                "motors nor to %s. Read values: %s",
                self,
                self.calibration_fpath,
                {k: (v.homing_offset, v.range_min, v.range_max) for k, v in self.calibration.items()},
            )
            return

        if self.config.dry_run:
            raise RuntimeError(
                f"{self}: allow_calibration_write=True conflicts with dry_run=True. "
                "Set dry_run=False explicitly to write calibration to the motors."
            )

        from lerobot.motors.dynamixel import DriveMode

        logger.warning("%s writing FACTORY DEFAULT calibration to the motors.", self)
        self.bus.disable_torque()
        for motor in self.motors:
            self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)

        self.calibration = {
            name: MotorCalibration(id=m.id, drive_mode=0, homing_offset=0, range_min=TICK_MIN, range_max=TICK_MAX)
            for name, m in self.motors.items()
        }
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info("%s calibration saved to %s", self, self.calibration_fpath)

    def configure(self) -> None:
        """Optional motor configuration. No-op unless `configure_motors_on_connect`."""
        if self.config.dry_run or not self.config.configure_motors_on_connect:
            logger.info(
                "%s configure(): skipped (dry_run=%s, configure_motors_on_connect=%s). Operating "
                "modes / Return_Delay_Time left as they are on the motors.",
                self,
                self.config.dry_run,
                self.config.configure_motors_on_connect,
            )
            return

        from lerobot.motors.dynamixel import OperatingMode

        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.motors:
                if motor != GRIPPER_NAME:
                    # POSITION (not EXTENDED_POSITION): this arm is single-turn and all our
                    # data lives in 0..4095, so let the motor enforce its own limits too.
                    self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # Current-based position control for the gripper so it can grasp without
            # forcing (same rationale as the omx/koch templates).
            self.bus.write("Operating_Mode", GRIPPER_NAME, OperatingMode.CURRENT_POSITION.value)
        logger.info("%s configure(): operating modes written, torque re-enabled by torque_disabled().", self)

    def _ramp_up(self) -> None:
        """Seed the rate limiter from the present position; Goal := present before torque.

        Prevents the classic power-on snap: a stale Goal_Position in the motor would be
        executed at full speed the instant torque is enabled.
        """
        present = self.bus.sync_read("Present_Position", normalize=False)
        present_ticks = {name: int(present[name]) for name in self.motors}

        clipped = {}
        for name, tick in present_ticks.items():
            lo, hi = self._joint_limits[name]
            clipped[name] = int(min(hi, max(lo, tick)))
        if clipped != present_ticks:
            logger.warning(
                "%s present position %s is OUTSIDE joint_limits; seeding the rate limiter at the "
                "clipped %s (the first commands will drag the arm back into range).",
                self,
                present_ticks,
                clipped,
            )
        self._prev_cmd_ticks = clipped

        if self.config.dry_run:
            logger.info(
                "%s dry_run=True: READ ONLY. present(ticks)=%s, torque untouched, no Goal_Position written.",
                self,
                present_ticks,
            )
            return

        if self.config.ramp_up_on_connect:
            self.bus.sync_write("Goal_Position", present_ticks, normalize=False)
            self.bus.enable_torque()
            logger.info("%s ramp-up: Goal_Position := present %s, torque ON.", self, present_ticks)
        else:
            self.bus.enable_torque()
            logger.warning("%s ramp_up_on_connect=False: torque enabled with the motors' stale goal.", self)

    # ---------------------------------------------------------------------------------
    # I/O
    # ---------------------------------------------------------------------------------
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        raw = self.bus.sync_read("Present_Position", normalize=False)  # ticks, never normalized
        obs_dict: dict[str, Any] = {
            f"{name}.pos": self.ticks_to_unit(name, raw[name]) for name in self.motors
        }
        logger.debug("%s read state: %.1fms", self, (time.perf_counter() - start) * 1e3)

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            frame = cam.read_latest()
            if self._flip_h.get(cam_key, False):
                # Horizontal mirror, same correction as the old server's FLIP_CAM_H
                # (server_infer.py:591-595). ascontiguousarray: the negative-stride view a
                # slice produces breaks downstream code that expects a contiguous buffer.
                frame = np.ascontiguousarray(np.asarray(frame)[:, ::-1])
            obs_dict[cam_key] = frame
            logger.debug("%s read %s: %.1fms", self, cam_key, (time.perf_counter() - start) * 1e3)

        return obs_dict

    def _apply_safety(self, goal_ticks: dict[str, float]) -> tuple[dict[str, int], bool, bool]:
        """Clamp to joint_limits, then rate-limit against the previous command.

        Returns (safe ticks, any-limit-clipped, any-rate-clipped) and updates the
        per-motor counters. The rate limit is measured against the previous *command*
        (seeded from the present position at connect), like our existing deployment loop.
        """
        assert self._prev_cmd_ticks is not None, "connect() seeds the rate limiter"
        safe: dict[str, int] = {}
        hit_limit = False
        hit_rate = False
        for name in self.motors:
            target = float(goal_ticks[name])
            lo, hi = self._joint_limits[name]
            bounded = min(float(hi), max(float(lo), target))
            if abs(bounded - target) > 1e-6:
                hit_limit = True
                self._limit_clip_counts[name] += 1

            prev = float(self._prev_cmd_ticks[name])
            step = self._max_step[name]
            if step is not None:
                delta = bounded - prev
                if delta > step:
                    bounded = prev + step
                    hit_rate = True
                    self._rate_clip_counts[name] += 1
                elif delta < -step:
                    bounded = prev - step
                    hit_rate = True
                    self._rate_clip_counts[name] += 1
            safe[name] = int(round(bounded))
        return safe, hit_limit, hit_rate

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command absolute joint targets, expressed in `config.unit`.

        Args:
            action: `{"joint1.pos": v, ..., "gripper.pos": v}` -- every motor is
                required; a partial dict raises instead of silently holding a joint.

        Returns:
            The action actually sent (after limit + rate clipping), in `config.unit`.
            With `dry_run=True` this is what *would* have been sent; no bus write happens.
        """
        goal_unit = {key.removesuffix(".pos"): float(val) for key, val in action.items() if key.endswith(".pos")}
        if set(goal_unit) != set(self.motors):
            raise ValueError(
                f"{self}: action keys {sorted(action)} do not match the expected "
                f"{[f'{m}.pos' for m in self.motor_names]} (order-independent, but all 5 are required)."
            )

        goal_ticks = {name: self.unit_to_ticks(name, goal_unit[name]) for name in self.motors}
        safe_ticks, hit_limit, hit_rate = self._apply_safety(goal_ticks)

        self._n_steps += 1
        self._n_limit_clipped_steps += int(hit_limit)
        self._n_rate_clipped_steps += int(hit_rate)

        if self.config.dry_run:
            # First few steps at INFO so a dry run is visible, then DEBUG to avoid
            # flooding the log at control rate.
            level = logging.INFO if self._n_steps <= 5 else logging.DEBUG
            logger.log(
                level,
                "%s [dry_run] step %d: goal(%s)=%s -> ticks=%s clip=%s/%s (NOT sent)",
                self,
                self._n_steps,
                self.config.unit,
                {k: round(v, 4) for k, v in goal_unit.items()},
                safe_ticks,
                "LIM" if hit_limit else "---",
                "RATE" if hit_rate else "----",
            )
        else:
            self.bus.sync_write("Goal_Position", safe_ticks, normalize=False)

        self._prev_cmd_ticks = dict(safe_ticks)

        if self.config.clip_report_every and self._n_steps % self.config.clip_report_every == 0:
            self.log_clip_stats()

        return {f"{name}.pos": self.ticks_to_unit(name, tick) for name, tick in safe_ticks.items()}

    # ---------------------------------------------------------------------------------
    # Clip statistics
    # ---------------------------------------------------------------------------------
    @property
    def clip_stats(self) -> dict[str, Any]:
        n = max(self._n_steps, 1)
        return {
            "steps": self._n_steps,
            "rate_clipped_steps": self._n_rate_clipped_steps,
            "rate_clipped_frac": self._n_rate_clipped_steps / n,
            "limit_clipped_steps": self._n_limit_clipped_steps,
            "limit_clipped_frac": self._n_limit_clipped_steps / n,
            "rate_clipped_per_motor": dict(self._rate_clip_counts),
            "limit_clipped_per_motor": dict(self._limit_clip_counts),
        }

    def reset_clip_stats(self) -> None:
        self._n_steps = 0
        self._n_rate_clipped_steps = 0
        self._n_limit_clipped_steps = 0
        self._rate_clip_counts = dict.fromkeys(MOTOR_NAMES, 0)
        self._limit_clip_counts = dict.fromkeys(MOTOR_NAMES, 0)

    def log_clip_stats(self) -> None:
        s = self.clip_stats
        if s["steps"] == 0:
            logger.info("%s clip stats: no action sent.", self)
            return
        logger.info(
            "%s clip stats: steps=%d rate-clipped=%d (%.1f%%) limit-clipped=%d (%.1f%%) | "
            "per-motor rate=%s limit=%s",
            self,
            s["steps"],
            s["rate_clipped_steps"],
            100 * s["rate_clipped_frac"],
            s["limit_clipped_steps"],
            100 * s["limit_clipped_frac"],
            s["rate_clipped_per_motor"],
            s["limit_clipped_per_motor"],
        )
        if s["rate_clipped_frac"] > 0.5:
            logger.warning(
                "%s >50%% of the steps were rate-limited (max_step_ticks=%s). The motion you see is "
                "the limiter's, not the policy's -- raise max_step_ticks or slow the policy down "
                "before drawing any conclusion from this rollout.",
                self,
                self._max_step,
            )

    # ---------------------------------------------------------------------------------
    @check_if_not_connected
    def disconnect(self):
        self.log_clip_stats()
        self.bus.disconnect(self.config.disable_torque_on_disconnect and not self.config.dry_run)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)

    def setup_motors(self) -> None:
        """Assign IDs/baudrate one motor at a time (used by `lerobot-setup-motors`)."""
        for motor in reversed(list(self.motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.motors[motor].id}")
