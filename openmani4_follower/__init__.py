#!/usr/bin/env python
"""`openmani4_follower` -- lerobot Robot for our 4-DoF OpenMANIPULATOR-X + gripper.

Registration (no lerobot repo modification):
    Importing this package runs the `@RobotConfig.register_subclass("openmani4_follower")`
    decorator in `config_openmani4_follower.py`. After that,
    `lerobot.robots.utils.make_robot_from_config(cfg)` falls through its if/elif chain
    (utils.py:33-36 only knows the built-ins) to
    `lerobot.utils.import_utils.make_device_from_device_class(cfg)`, which resolves
    `Openmani4FollowerConfig` -> class `Openmani4Follower` in the parent module
    (this file re-exports it). So:

        import openmani4_follower                      # registers the "openmani4_follower" type
        from lerobot.robots.utils import make_robot_from_config
        robot = make_robot_from_config(Openmani4FollowerConfig(port="/dev/ttyUSB0"))

    For the draccus CLI entry points (`lerobot-record`, `lerobot-rollout`, ...) the type
    string must be registered *before* the arguments are parsed. Those scripts only
    auto-import distributions whose *distribution name* starts with `lerobot_robot_`
    (`register_third_party_plugins()`), which this source directory is not. Use a
    one-line wrapper instead of patching the repo, e.g.

        PYTHONPATH=/home/vcl/Desktop/minjun/vla python -c \
          "import openmani4_follower; from lerobot.scripts.lerobot_record import main; main()" \
          --robot.type=openmani4_follower ...

    (Packaging this directory as a `lerobot_robot_openmani4` distribution would make the
    auto-discovery work too, but that requires an install step; not done.)
"""

from .config_openmani4_follower import (
    CENTER_TICK,
    DATASET_META_FILENAME,
    DEFAULT_CAMERA_INDICES,
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
    VALID_UNITS,
    Openmani4FollowerConfig,
)
from .openmani4_follower import (
    DatasetUnitMismatch,
    Openmani4Follower,
    check_dataset_units,
    radians_to_ticks,
    read_dataset_meta,
    ticks_to_radians,
)

__all__ = [
    "CENTER_TICK",
    "DATASET_META_FILENAME",
    "DEFAULT_CAMERA_INDICES",
    "GRIPPER_TICK_MAX",
    "GRIPPER_TICK_MIN",
    "MOTOR_IDS",
    "MOTOR_MODEL",
    "MOTOR_NAMES",
    "TICKS_PER_REV",
    "TICK_MAX",
    "TICK_MIN",
    "UNIT_RADIANS",
    "UNIT_RAW_TICKS",
    "VALID_UNITS",
    "DatasetUnitMismatch",
    "Openmani4Follower",
    "Openmani4FollowerConfig",
    "check_dataset_units",
    "radians_to_ticks",
    "read_dataset_meta",
    "ticks_to_radians",
]
