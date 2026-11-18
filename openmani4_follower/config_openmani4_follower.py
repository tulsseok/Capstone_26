#!/usr/bin/env python
"""Config for the OpenMANIPULATOR-X based 4-DoF + gripper follower arm.

Hardware (fixed, confirmed by the user):
  * 4-DoF arm + 1 gripper  -> 5 Dynamixel XM430-W350 motors
  * IDs 11, 22, 23, 24, 25 -- **this order is the dataset state/action order**
  * port /dev/ttyUSB0 @ 1 Mbaud
  * 3 fixed third-person RGB cameras, 240x320

This config is registered with draccus as `openmani4_follower`, so
`lerobot.robots.utils.make_robot_from_config()` can build the robot through its
generic `make_device_from_device_class()` fallback -- **without patching the
lerobot repo**. The only requirement is that this package has been imported
(see `openmani4_follower/__init__.py`).
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig

# --------------------------------------------------------------------------------------
# Motor layout -- SINGLE SOURCE OF TRUTH for the channel order.
# The dict order below == the order of `observation.state` / `action` vectors in our
# dataset and in our HTTP policy server. Do not reorder, do not sort.
# (SmolVLA does not expose `action_feature_names`, so lerobot's rollout re-ordering in
#  `rollout/context.py::_resolve_action_key_order` is a no-op for us -- the order is
#  ours to guarantee.)
# --------------------------------------------------------------------------------------
MOTOR_IDS: dict[str, int] = {
    "joint1": 11,
    "joint2": 22,
    "joint3": 23,
    "joint4": 24,
    "gripper": 25,
}
MOTOR_MODEL = "xm430-w350"  # present in lerobot/motors/dynamixel/tables.py (model number 1020)
MOTOR_NAMES: tuple[str, ...] = tuple(MOTOR_IDS)
GRIPPER_NAME = "gripper"

# Encoder constants (XM430-W350, 12-bit single turn).
TICKS_PER_REV = 4096
CENTER_TICK = 2048
TICK_MIN = 0
TICK_MAX = 4095

# Gripper travel used by the previous deployment: `grip_norm_to_tick()` maps [0,1] onto
# [1000, 2700] and clips (dex_ref/server_infer.py:352-355). Confirmed against the h5:
# qpos[:,5] * 1700 + 1000 is an exact integer in every frame of the 120 files sampled.
GRIPPER_TICK_MIN = 1000
GRIPPER_TICK_MAX = 2700

# Unit of everything crossing get_observation() / send_action().
# NOTE: "lerobot_norm" was removed. Our converter cannot produce such a dataset (it writes
# raw ticks or radians), so the option only offered a way to silently mis-scale every
# command. It is still named here to give a specific error instead of "unknown unit".
UNIT_RAW_TICKS = "raw_ticks"  # 0..4095 encoder ticks (what our h5 / joints.csv store)
UNIT_RADIANS = "radians"  # (tick - CENTER_TICK) * 2*pi / TICKS_PER_REV
UNIT_REMOVED = {
    "lerobot_norm": (
        "unit='lerobot_norm' is removed: convert_h5_to_lerobot_v3.py can only write "
        "'raw_ticks' (--joint-units tick, the default) or 'radians' (--joint-units rad), "
        "so no dataset in this project is in lerobot's [-100,100]/[0,100] convention. "
        "Use unit='raw_ticks'."
    )
}
VALID_UNITS = (UNIT_RAW_TICKS, UNIT_RADIANS)

# Sidecar written by convert_h5_to_lerobot_v3.py next to the dataset (NOT in meta/:
# lerobot rewrites meta/info.json from a dataclass and drops unknown keys).
DATASET_META_FILENAME = "openmani_meta.json"

DEFAULT_CAMERA_KEYS = ("camera1", "camera2", "camera3")
# USB UVC cameras usually enumerate on the EVEN /dev/video* indices (each device claims a
# capture + a metadata node), which is why the existing client defaults to 0,2,4
# (gr00t_client_local.py:815-817). Hardware dependent -- the resolved index and the first
# frame's shape are logged per slot on connect, and the slot->physical-camera identity is
# NOT verified by anything (HANDOFF_PROMPT.md:258-260).
DEFAULT_CAMERA_INDICES = [0, 2, 4]


@RobotConfig.register_subclass("openmani4_follower")
@dataclass
class Openmani4FollowerConfig(RobotConfig):
    # ---- bus -------------------------------------------------------------------------
    port: str = "/dev/ttyUSB0"
    baudrate: int = 1_000_000

    # ---- units -----------------------------------------------------------------------
    # Our training data comes from our own h5 recordings (raw ticks), NOT from a lerobot
    # recording session. If we let the DynamixelMotorsBus normalize (`use_degrees=False`
    # in the omx template -> MotorNormMode.RANGE_M100_100 / RANGE_0_100), deployment
    # units would silently differ from training units. So the robot always talks to the
    # bus with `normalize=False` (raw ticks on the wire) and converts to `unit` itself.
    unit: str = UNIT_RAW_TICKS
    # Dataset the policy was trained on. When set, connect() reads
    # <dataset_root>/openmani_meta.json and REFUSES to run if its unit differs from
    # `unit` above (see openmani4_follower.check_dataset_units). Left None the contract
    # is simply unchecked, and connect() says so.
    dataset_root: str | None = None

    # ---- safety ----------------------------------------------------------------------
    # No writes to the motors at all (torque, config, goals) when True.
    dry_run: bool = True
    # Max |delta| in TICKS per send_action() call. Scalar or per-motor. None disables.
    #
    # 150 is derived from the only real-robot log we have, `vla/log.md`: 178 executed
    # steps, of which 168 (94.4%) were rate-clipped at max_step_ticks=25. The move each
    # step actually asked for (|plan - actual|, taken as the max over the 5 motors) has
    # median 66 ticks, p90 150, max 412 -- so 25 throttled almost everything and even 60
    # would still clip more than half of the steps. 150 = that p90.
    # SAFETY-RELEVANT: this is a limit derived from a log, not from a hardware spec.
    # Confirm it on the real arm (slowly, dry_run=False only when watching) before
    # trusting it; the value is echoed at connect() and the clip rate is counted.
    max_step_ticks: float | dict[str, float] | None = 150.0
    # Absolute per-motor tick bounds. None -> arm (TICK_MIN, TICK_MAX), gripper
    # (GRIPPER_TICK_MIN, GRIPPER_TICK_MAX). The gripper bounds come from the previous
    # deployment; the ARM bounds are NOT confirmed (bench/fk_open_manipulator.py, which
    # holds the URDF joint ranges, is not on this machine) -- connect() warns about it.
    joint_limits: dict[str, tuple[int, int]] | None = None
    # Read present position and set Goal_Position := present BEFORE enabling torque.
    ramp_up_on_connect: bool = True
    disable_torque_on_disconnect: bool = True
    # Log the clip statistics every N send_action() calls (0 = only on disconnect).
    clip_report_every: int = 100

    # ---- calibration / motor configuration -------------------------------------------
    # The omx template writes factory-default calibration to the motors on connect
    # (omx_follower.py:118-131). We do NOT: calibration is read-only unless this is set.
    allow_calibration_write: bool = False
    # Write Return_Delay_Time=0 and operating modes (arm: POSITION, gripper:
    # CURRENT_POSITION) on connect. Off by default: our data was recorded with whatever
    # modes the motors already have, and these are EEPROM writes.
    configure_motors_on_connect: bool = False

    # ---- cameras ---------------------------------------------------------------------
    # Keys must be camera1/camera2/camera3: the dataset/policy keys become
    # `observation.images.camera1` ... which is what the smolvla_base checkpoint expects.
    # A mismatch here is dropped silently downstream.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    camera_indices: list[int | str] = field(default_factory=lambda: list(DEFAULT_CAMERA_INDICES))
    camera_width: int = 320
    camera_height: int = 240
    camera_fps: int = 30
    # Horizontal mirror per camera, applied to the frame right after it is read.
    # bool -> all cameras, dict -> per camera key. Default off.
    # The old server had this as env FLIP_CAM_H (server_infer.py:591-595) and the handoff
    # notes list it as a MUST-PORT hardware correction: if the live cameras are mirrored
    # w.r.t. the training data the robot chases the object's mirror image
    # (HANDOFF_PROMPT.md:253-255). There is no equivalent anywhere else in the new stack.
    camera_flip_h: bool | dict[str, bool] = False

    def __post_init__(self):
        if self.unit in UNIT_REMOVED:
            raise ValueError(UNIT_REMOVED[self.unit])
        if self.unit not in VALID_UNITS:
            raise ValueError(f"unit must be one of {VALID_UNITS}, got {self.unit!r}")

        if self.joint_limits is None:
            self.joint_limits = {
                name: (GRIPPER_TICK_MIN, GRIPPER_TICK_MAX) if name == GRIPPER_NAME
                else (TICK_MIN, TICK_MAX)
                for name in MOTOR_NAMES
            }
        else:
            unknown = set(self.joint_limits) - set(MOTOR_NAMES)
            if unknown:
                raise ValueError(f"joint_limits has unknown motors {sorted(unknown)}; expected {MOTOR_NAMES}")
            for name in MOTOR_NAMES:
                lo, hi = self.joint_limits.get(name, (TICK_MIN, TICK_MAX))
                if lo >= hi:
                    raise ValueError(f"joint_limits[{name}] = ({lo}, {hi}) has lo >= hi")
                self.joint_limits[name] = (int(lo), int(hi))

        if isinstance(self.max_step_ticks, dict):
            missing = set(MOTOR_NAMES) - set(self.max_step_ticks)
            if missing:
                raise ValueError(f"max_step_ticks is missing motors {sorted(missing)}")
        elif self.max_step_ticks is not None and self.max_step_ticks <= 0:
            raise ValueError(f"max_step_ticks must be > 0 or None, got {self.max_step_ticks}")

        if not self.cameras and self.camera_indices:
            if len(self.camera_indices) > len(DEFAULT_CAMERA_KEYS):
                raise ValueError(
                    f"camera_indices has {len(self.camera_indices)} entries but only "
                    f"{DEFAULT_CAMERA_KEYS} keys are defined for this robot."
                )
            self.cameras = {
                key: OpenCVCameraConfig(
                    index_or_path=idx,
                    fps=self.camera_fps,
                    width=self.camera_width,
                    height=self.camera_height,
                )
                for key, idx in zip(DEFAULT_CAMERA_KEYS, self.camera_indices, strict=False)
            }

        if isinstance(self.camera_flip_h, dict):
            unknown = set(self.camera_flip_h) - set(self.cameras)
            if unknown:
                raise ValueError(
                    f"camera_flip_h has unknown camera keys {sorted(unknown)}; "
                    f"expected a subset of {sorted(self.cameras)}"
                )

        super().__post_init__()

    def flip_h_for(self, camera_key: str) -> bool:
        """Resolve `camera_flip_h` (bool or per-key dict) for one camera."""
        if isinstance(self.camera_flip_h, dict):
            return bool(self.camera_flip_h.get(camera_key, False))
        return bool(self.camera_flip_h)
