#!/usr/bin/env python3
"""
DexVLA h5 episodes -> LeRobot **v3.0** dataset (SmolVLA-ready).

Why a new script instead of patching convert_h5_to_groot_lerobot.py:
  * lerobot 0.6.2 pins CODEBASE_VERSION = "v3.0"
    (lerobot/src/lerobot/datasets/dataset_metadata.py:60) and rejects v2.x with
    BackwardCompatibilityError. The official v2.1->v3.0 migration script does not
    accept the v2.0 layout our GR00T converter emits, so migration is not an option.
  * v3.0 concatenates several episodes into one mp4 + per-episode `from_timestamp`
    offsets and writes chunked parquet with a live ParquetWriter. Hand-writing those
    files is not reproducible -> we go through the public write API only:
        LeRobotDataset.create() -> add_frame() -> save_episode() -> finalize()
    Skipping finalize() leaves the parquet footer unwritten and the dataset invalid.

Camera key naming is NOT free: the smolvla_base checkpoint ships populated
`input_features`, and lerobot only fills them from the dataset when they are empty
(policies/factory.py:311 `if not cfg.input_features:`). A key the checkpoint does not
know is silently dropped. The checkpoint declares:
    observation.state              (6,)
    observation.images.camera1/2/3 (3, 256, 256)
    action                         (6,)
so the h5 keys image_0/1/2 are renamed to camera1/2/3 here. The *shapes* it declares are
only metadata for SmolVLA (state is padded to `max_state_dim=32` and the normalizer
buffers come from the dataset stats, not from `input_features`), so a state width other
than 6 -- 5 in joint space, 7 with the default sin/cos yaw encoding -- is expected to be
fine; the *key names* are what must match.

Two action spaces:
  --space ee     (default) 6-D EE state/action straight out of the h5, yaw encoded as
                 sin/cos by default (7-D state). **OFFLINE COMPARISON ONLY -- NOT
                 DEPLOYABLE** on the new stack: openmani4_follower commands absolute
                 joint ticks and the new stack has neither an IK solver nor a
                 [0,1] -> gripper-tick map. Anything trained from it can be evaluated
                 offline but not executed.
  --space joint  (code path ready, NOT runnable yet) 5-D absolute Dynamixel ticks; needs
                 the original joints.csv files, which we have not received. Without
                 --joints-csv-root pointing at real files the script exits with an error;
                 it never fabricates joint values. This is the deployable space.

Units on the wire are RAW TICKS (--joint-units tick, the default): client_live.py:394-399
writes the int32 ticks it received straight to the motors, and the only rad->tick
conversion in the old stack lives *inside* server_infer.py:358-367, which the SmolVLA
path does not have. The unit actually used is written to `<dataset>/openmani_meta.json`
and `openmani4_follower` refuses to run against a dataset whose unit differs from its
own (a custom key inside meta/info.json would NOT survive: lerobot's
`DatasetInfo.from_dict` (datasets/utils.py:223) drops unknown fields and every later
`write_info()` rewrites the file from the dataclass).

Two input formats (--input-format, default h5 = unchanged behaviour):
  h5    the DexVLA export described above (--h5-dir).
  raw   the ORIGINAL recorder output (--input <root>), laid out as
            <person>/<task>/<timestamp>/{joints.csv, metadata.csv, image_0/1/2/NNNN.png}
        This is the deployable path and needs no FK: joints.csv already holds the 5 raw
        Dynamixel ticks. --input-format raw therefore implies --space joint and REFUSES
        --space ee (that would need pinocchio FK, a dependency we do not add here).
        Episodes are collected with a 3-level scan and ordered deterministically by
        (person, task, timestamp) lexicographically; that order is written to the report.

Optional termination signal (--termination-channels, OFF by default; raw/joint only):
  appends [progress_episode, progress_task, done_flag] to the ACTION vector, after the 5
  motor channels (action 5-D -> 8-D). SmolVLA zero-pads actions to max_action_dim=32, so
  the extra channels reuse padding: no re-initialised layer, no lerobot change, and the
  flow-matching MSE regresses them like any other channel. WITH THE FLAG OFF THE OUTPUT IS
  BYTE-FOR-BYTE UNCHANGED. The deploy server must keep sending only action[:5] to the
  robot -- openmani_meta.json records `n_motor_channels` for exactly that.

Typical use:
    python convert_h5_to_lerobot_v3.py --out /path/ds --fps 10 --dry-run
    python convert_h5_to_lerobot_v3.py --out /path/ds --fps 10
    python convert_h5_to_lerobot_v3.py --input-format raw --input /path/dataset_new \
        --out /path/ds --fps 10 --joint-action-mode absolute_next \
        --reasoning-json /path/dataset_new/capstone_reasoning_labels_new.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

logger = logging.getLogger("h5_to_lerobot_v3")


# ---------------------------------------------------------------------------
# Fixed schema constants (measured on data/h5py_train_user_ee6, 924 files)
# ---------------------------------------------------------------------------
H5_STATE_KEY = "observations/qpos"          # (T, 6) EE space, pinocchio FK output
H5_ACTION_KEY = "action"                    # (T, 6) delta pose + ABSOLUTE gripper
H5_IMAGE_GROUP = "observations/images"
H5_IMAGE_KEYS = ["image_0", "image_1", "image_2"]     # (T, 240, 320, 3) uint8
H5_LANG_KEY = "language_raw"                # (3,) |S<n>, n per file -> L1 / L2 / L3

# Must match the smolvla_base checkpoint input_features exactly, index-aligned with
# H5_IMAGE_KEYS above.
CAMERA_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]

EE_STATE_NAMES = ["x", "y", "z", "pitch", "yaw", "grip_norm"]
# Default: yaw -> (sin, cos). Raw yaw spans nearly the full circle (measured on the first
# 120 h5 files: [-3.137, +3.137] rad with 47 |dyaw|>pi frame-to-frame jumps), so its
# MEAN_STD normalization -- what smolvla_base uses for STATE/ACTION, see its config.json
# "normalization_mapping" -- would be computed over a bimodal +-pi distribution and is
# meaningless. The old deployment avoided this the same way
# (server_infer.py:6-11, 297-300).
EE_STATE_NAMES_SINCOS = ["x", "y", "z", "pitch", "sin_yaw", "cos_yaw", "grip_norm"]
# The ACTION side stays raw on purpose: `qpos_to_action` (user5dof_to_h5py_ee6.py:109-130)
# already builds dyaw with the shortest-arc `_angle_diff` and then clips pitch/yaw deltas
# to +-PITCH_YAW_DELTA_CLAMP = 0.5 rad, so it has no wrap discontinuity (measured on the
# first 120 files: max |dyaw| = 0.5000 exactly). Encoding a bounded delta as sin/cos would
# make it ambiguous, not safer.
EE_ACTION_NAMES = ["dx", "dy", "dz", "dpitch", "dyaw", "grip_abs"]
EE_YAW_INDEX = 4

# joints.csv header is `timestep,t_seconds,id11,id22,id23,id24,id25`.
# lerobot does NOT reorder state channels for SmolVLA (SmolVLAConfig has no
# `action_feature_names`), so this order is the contract between conversion and
# deployment. Do not sort, do not rename.
JOINT_CSV_META_COLS = ["timestep", "t_seconds"]
JOINT_MOTOR_ORDER = ["id11", "id22", "id23", "id24", "id25"]

# --- raw recorder layout: <root>/<person>/<task>/<timestamp>/ ----------------
RAW_JOINTS_CSV = "joints.csv"
RAW_METADATA_CSV = "metadata.csv"
# Index-aligned with CAMERA_KEYS above: image_0 -> camera1, image_1 -> camera2,
# image_2 -> camera3. Same rule as the h5 path (H5_IMAGE_KEYS), so a policy trained on
# either conversion sees the same physical camera under the same key.
RAW_IMAGE_DIRS = ["image_0", "image_1", "image_2"]
RAW_META_PERSON = "person_name"
RAW_META_TASK = "task_name"
RAW_META_STARTED = "started_at"
RAW_META_FPS = "record_fps"

# Dynamixel 12-bit Goal_Position range. A tick outside it cannot be commanded, so a
# state/action value outside it means the CSV is not what we think it is.
DXL_TICK_MIN = 0
DXL_TICK_MAX = 4095

# metadata.csv `task_name` is NOT trusted for the language cross-check. MEASURED on
# data/dataset_new (972 episodes, 2026-08-14): all 243 episodes under moon/push/ carry
# task_name=pick_place, i.e. the recorder kept a stale field while writing into the push
# folder. The FOLDER name is the one corroborated by the joint data itself -- median
# id25 (gripper) travel over an episode is 639 / 1 / 701 / 1 ticks for
# han/pick_and_place, han/push, moon/pick_place, moon/push: the "push" folders never
# actuate the gripper, the "place" folders always do. So folder task -> expected verb:
RAW_TASK_TO_VERB = {
    "pick_and_place": "place",
    "pick_place": "place",
    "push": "push",
}

# Tick <-> radian map. Kept identical to openmani_deploy_adapter.py
# (ASSUMED_CENTER_TICK / ASSUMED_TICKS_PER_REV) — if these ever diverge every commanded
# joint is silently wrong.
DXL_CENTER_TICK = 2048
DXL_TICKS_PER_REV = 4096

# Gripper tick span used by the old deployment to turn grip_norm into a tick
# (server_infer.py:352-355, `grip_norm_to_tick`). Inverting it gives the only
# FK-free cross-check we have between the h5 and joints.csv:
#     qpos[t, 5] (grip_norm) == clip((id25[t] - 1000) / 1700, 0, 1)
# Verified exact on the first 120 h5 files / 3564 frames: grip_norm * 1700 + 1000 is an
# integer to the last bit in every frame, and grip_norm stays inside [0.0000, 0.9959].
GRIP_MIN_TICK = 1000
GRIP_MAX_TICK = 2700
GRIP_TICK_SPAN = GRIP_MAX_TICK - GRIP_MIN_TICK  # 1700

# Sidecar file written next to the dataset. NOT inside meta/: lerobot's DatasetInfo
# drops unknown info.json keys on load (datasets/utils.py:223) and rewrites the file from
# the dataclass on every metadata update (datasets/io_utils.py:120), so a custom field
# there would disappear silently. openmani4_follower reads this file and refuses to run
# on a unit mismatch.
OPENMANI_META_FILENAME = "openmani_meta.json"
OPENMANI_META_VERSION = 1


class EpisodeSkip(Exception):
    """Raised per-episode; the file is recorded in the report and conversion continues."""


class FatalConversionError(Exception):
    """Raised for conditions that must NOT be swallowed as a per-episode skip.

    A skip is invisible until the 5 % skip budget runs out and even then only shows up as
    a reason string. Anything that means "this input is a different dataset than the one
    this script was written for" belongs here so it aborts on the first file.
    """


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def natural_key_name(name: str):
    """episode_2 < episode_10, 9.png < 10.png (lexicographic sort gets both wrong)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def natural_key(path: Path):
    return natural_key_name(path.name)


class RssGuard:
    """Hard RAM ceiling. Counts this process + its children (ffmpeg / encoder pool)."""

    def __init__(self, max_gb: float):
        self.max_gb = max_gb
        self.proc = psutil.Process()
        self.peak_gb = 0.0

    def current_gb(self) -> float:
        rss = self.proc.memory_info().rss
        for child in self.proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return rss / 1024**3

    def check(self, where: str) -> float:
        cur = self.current_gb()
        self.peak_gb = max(self.peak_gb, cur)
        if cur > self.max_gb:
            raise RuntimeError(
                f"RSS ceiling exceeded at {where}: {cur:.2f} GB > --max-rss-gb {self.max_gb} GB. "
                "Aborting instead of pushing the machine into swap."
            )
        return cur


def decode_language(raw) -> tuple[list[str], list[int]]:
    """-> (decoded strings, raw byte length of each entry before decoding)."""
    out, byte_lens = [], []
    for item in raw:
        if isinstance(item, bytes):
            byte_lens.append(len(item))
            out.append(item.decode("utf-8", errors="replace").strip())
        else:
            text = str(item)
            byte_lens.append(len(text.encode("utf-8")))
            out.append(text.strip())
    return out, byte_lens


def parse_image_hw(spec: str | None) -> tuple[int, int] | None:
    if spec is None:
        return None
    m = re.fullmatch(r"(\d+)x(\d+)", spec.strip())
    if not m:
        raise ValueError(f"--image-hw must look like 256x256, got {spec!r}")
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# joint-space (blocked until joints.csv arrives)
# ---------------------------------------------------------------------------
def resolve_joints_csv(h5_attrs: dict, csv_root: Path) -> Path:
    """Locate the joints.csv matching one h5 episode.

    The h5 keeps its provenance in attrs (`src_ep_dir`, `src_person`, `src_sub_task`,
    `src_time`), so we try the layouts those imply. We never fall back to "any csv".
    """
    person = str(h5_attrs.get("src_person", ""))
    sub_task = str(h5_attrs.get("src_sub_task", ""))
    stamp = str(h5_attrs.get("src_time", ""))
    src_ep_dir = str(h5_attrs.get("src_ep_dir", ""))

    candidates = [
        csv_root / person / sub_task / stamp / "joints.csv",
        csv_root / person / sub_task / f"{stamp}.csv",
        csv_root / stamp / "joints.csv",
        csv_root / f"{stamp}.csv",
    ]
    if src_ep_dir:
        candidates.append(csv_root / Path(src_ep_dir).name / "joints.csv")

    for cand in candidates:
        if cand.is_file():
            return cand
    raise EpisodeSkip(
        "joints.csv not found; tried: " + " | ".join(str(c) for c in candidates)
    )


def read_joints_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (ticks (N, 5) float32 in JOINT_MOTOR_ORDER, t_seconds (N,) float64)."""
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        expected = JOINT_CSV_META_COLS + JOINT_MOTOR_ORDER
        if [h.strip() for h in header] != expected:
            raise EpisodeSkip(f"joints.csv header {header} != expected {expected} ({path})")
        ticks, secs = [], []
        for row in reader:
            ticks.append([float(row[c]) for c in JOINT_MOTOR_ORDER])
            secs.append(float(row["t_seconds"]))
    if not ticks:
        raise EpisodeSkip(f"joints.csv is empty ({path})")
    return np.asarray(ticks, dtype=np.float32), np.asarray(secs, dtype=np.float64)


def ticks_to_units(ticks: np.ndarray, units: str) -> np.ndarray:
    if units == "tick":
        return ticks.astype(np.float32)
    if units == "rad":
        return (
            (ticks.astype(np.float64) - DXL_CENTER_TICK) * (2.0 * np.pi / DXL_TICKS_PER_REV)
        ).astype(np.float32)
    raise ValueError(f"unknown joint units {units!r}")


def joint_state_action(ticks_u: np.ndarray, mode: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice a (N, 5) joint trajectory into aligned (T, 5) state / action arrays.

    Only `absolute_next` is allowed. The two modes that used to be here are refused:

    * `absolute_current` made `action[t]` the very same row of the very same CSV as
      `state[t]`, i.e. bitwise identical tensors. The training loss goes to zero and the
      policy learns "stay". Removed, not merely discouraged.
    * `delta_next` needs an integrator at deployment. The new stack has none:
      `openmani4_follower.send_action()` writes `Goal_Position` from the absolute target
      it is given (openmani4_follower.py, `_apply_safety` + `sync_write`), so a delta
      would be interpreted as an absolute tick near zero and slam the arm into its lower
      limit.
    """
    if mode == "delta_next":
        raise FatalConversionError(
            "--joint-action-mode delta_next is refused: the new deployment stack takes "
            "ABSOLUTE targets only. openmani4_follower.send_action() writes the value it "
            "receives straight into Goal_Position (after clipping) and never integrates a "
            "delta, so a delta-trained policy would command ~0 ticks. Use absolute_next."
        )
    if mode == "absolute_current":
        raise FatalConversionError(
            "--joint-action-mode absolute_current is removed: it makes action[t] and "
            "state[t] the same CSV row (bitwise identical), so the loss collapses to 0 "
            "and the policy learns to stand still. Use absolute_next."
        )
    if mode != "absolute_next":
        raise ValueError(f"unknown --joint-action-mode {mode!r}")

    n = len(ticks_u)
    if n == n_frames + 1:
        state = ticks_u[:n_frames]
        nxt = ticks_u[1:]
    elif n == n_frames:
        state = ticks_u
        nxt = np.concatenate([ticks_u[1:], ticks_u[-1:]], axis=0)  # hold last target
    else:
        raise EpisodeSkip(f"joints.csv has {n} rows, h5 has {n_frames} frames (need T or T+1)")

    action = nxt
    return state.astype(np.float32), action.astype(np.float32)


def check_grip_alignment(
    ticks: np.ndarray,
    grip_norm: np.ndarray,
    tol: float,
    shift_margin: float,
    csv_path: Path,
) -> dict:
    """FK-free row-alignment check between joints.csv and the h5, via the gripper.

    `joint_state_action` only compares row COUNTS, so a joints.csv that is one row out of
    phase with the h5 passes silently -- and a one-step shift is exactly what fabricates
    the "action == state" pathology we just removed. The gripper gives a free handle:
    qpos[t, 5] is an affine function of id25[t] (see GRIP_MIN_TICK/GRIP_MAX_TICK), with no
    FK involved, so the two files can be compared frame by frame.

    Compared against the CLIPPED affine value: grip_norm is provably inside [0, 1] in the
    h5 (measured min 0.0 exactly), so out-of-band ticks must have been clipped somewhere
    upstream and an unclipped comparison would produce false skips.

    Raises EpisodeSkip on a mismatch. Returns a diagnostics dict (also used to flag the
    episodes where the gripper never moves, for which this check proves nothing).
    """
    n = min(len(ticks), len(grip_norm))
    id25 = np.asarray(ticks[:n, JOINT_MOTOR_ORDER.index("id25")], dtype=np.float64)
    gn = np.asarray(grip_norm[:n], dtype=np.float64)
    expected = np.clip((id25 - GRIP_MIN_TICK) / GRIP_TICK_SPAN, 0.0, 1.0)

    err = np.abs(gn - expected)
    max_err = float(err.max()) if n else float("nan")
    mean_err = float(err.mean()) if n else float("nan")

    # How far off is a shifted alignment? Cheap and it is the only thing that can name
    # an off-by-one instead of just failing.
    shift_errs: dict[int, float] = {}
    for s in (-2, -1, 0, 1, 2):
        lo = max(0, s)
        hi = min(n, n + s)
        if hi - lo < 3:
            continue
        shift_errs[s] = float(np.abs(gn[lo:hi] - expected[lo - s : hi - s]).mean())
    best_shift = min(shift_errs, key=lambda k: shift_errs[k]) if shift_errs else 0

    tick_span = float(id25.max() - id25.min()) if n else 0.0
    info = {
        "n": int(n),
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "id25_tick_span": tick_span,
        "shift_mean_abs_err": {str(k): v for k, v in shift_errs.items()},
        "best_shift": int(best_shift),
        # < 10 ticks of gripper travel (ARBITRARY threshold): the gripper is essentially
        # static, so ANY shift looks equally good and this episode is unchecked in
        # practice. Counted in the report instead of being silently trusted.
        "informative": tick_span >= 10.0,
    }

    if max_err > tol:
        raise EpisodeSkip(
            f"joints.csv/h5 gripper mismatch: max|grip_norm - (id25-{GRIP_MIN_TICK})/"
            f"{GRIP_TICK_SPAN}| = {max_err:.4f} > --grip-check-tol {tol} "
            f"(mean {mean_err:.4f}, best row shift {best_shift}, {csv_path}). Either the "
            "csv belongs to another episode or the two files are not row-aligned."
        )
    if best_shift != 0 and (shift_errs.get(0, 0.0) - shift_errs[best_shift]) > shift_margin:
        raise EpisodeSkip(
            f"joints.csv/h5 look off by {best_shift} row(s): mean gripper error "
            f"{shift_errs.get(0, float('nan')):.4f} at shift 0 vs "
            f"{shift_errs[best_shift]:.4f} at shift {best_shift} "
            f"(> --grip-shift-margin {shift_margin}) ({csv_path})"
        )
    return info


# ---------------------------------------------------------------------------
# termination signals -- OPT-IN EXTRA ACTION CHANNELS (--termination-channels)
# ---------------------------------------------------------------------------
# Appended AFTER the 5 motor channels, never in front of them: the deploy server hands
# action[:5] to openmani4_follower and drops the rest, so both the WIDTH and the OFFSET
# of the motor block must stay what they are. The consumer reads `n_motor_channels` out
# of openmani_meta.json rather than assuming 5.
#
# Why extra action channels and not a new head: SmolVLA zero-pads the action vector to
# max_action_dim=32, so channels 5..7 reuse padding that is already there -- no layer is
# re-initialised, no lerobot code changes, and the flow-matching loss
# (F.mse_loss(u_t, v_t)) regresses them like any other channel.
#
# Why not a bare done flag: it is 1 on a small minority of frames, so "always predict 0"
# is already near-optimal under MSE. The two progress channels are uniform on [0, 1] by
# construction and carry the same information without that imbalance (the same reason
# Learning-to-Fold regresses t/T with an MSE completion head).
TERMINATION_CHANNEL_NAMES = ["progress_episode", "progress_task", "done_flag"]
N_MOTOR_CHANNELS = len(JOINT_MOTOR_ORDER)       # deployable prefix of `action`
GRIPPER_MOTOR_INDEX = JOINT_MOTOR_ORDER.index("id25")
ARM_MOTOR_INDICES = [i for i in range(len(JOINT_MOTOR_ORDER)) if i != GRIPPER_MOTOR_INDEX]

# MEASURED on data/dataset_new0822 (972 episodes, 2026-08-24), and it contradicts the
# "low tick = closed" reading that was handed to this script, so the evidence is written
# down here instead of in a commit message:
#   * In every place episode the arm STOPS (median |d arm| over the 4 arm joints drops
#     from ~28 ticks/frame to ~1) exactly while id25 RISES, then moves again while id25
#     stays high, then stops again while id25 FALLS. Actuating a gripper while the arm
#     dwells is a grasp and a release; the high plateau in between is the transport.
#   * The push episodes hold id25 at ~2632-2696 ticks (GRIPPER_TICK_MAX is 2700) for the
#     whole recording: you push with the fingers shut, not with them wide open.
#   * Place episodes end at ~1000-1300 ticks, i.e. the object has been let go.
# => HIGH tick = CLOSED. A close->open crossing is the RELEASE (the task being finished);
# a low->high crossing is the grasp, which happens near the MIDDLE of the episode
# (median 0.52 of the episode vs 0.93 for the release). --gripper-closed-side flips this
# back if the hardware is ever re-cabled.
GRIPPER_CLOSED_SIDE_DEFAULT = "high"


def _gripper_open_threshold(g: np.ndarray, rule: str) -> float:
    """Tick level that separates the open gripper from the closed one, per episode.

    `midpoint` ((min+max)/2, the default) instead of the median because the gripper sits
    at ONE of the two levels for most of an episode, which drags the median onto that
    level: with `median` the split degenerates (every frame ends up on one side) in 125
    of the 486 place episodes vs 84 with `midpoint`, and the detected release lands at a
    frame where the arm is moving far more often (arm speed p90 at t_done: 36 ticks/frame
    for `median` vs 13 for `midpoint`; a real gripper actuation happens while the arm
    dwells). Both are exposed because neither is a calibration, just a split rule.
    """
    if rule == "median":
        return float(np.percentile(g, 50))
    if rule == "midpoint":
        return float((float(g.min()) + float(g.max())) / 2.0)
    raise ValueError(f"unknown --place-gripper-threshold {rule!r}")


def find_t_done(
    ticks: np.ndarray,
    folder_task: str,
    *,
    push_return_threshold: float,
    gripper_closed_side: str = GRIPPER_CLOSED_SIDE_DEFAULT,
    place_gripper_threshold: str = "midpoint",
) -> dict:
    """First frame at which the TASK is done (not the frame the recording stops at).

    `ticks` is the (T, 5) raw joints.csv block for ONE episode, in JOINT_MOTOR_ORDER.
    The task kind comes from the FOLDER name, never from metadata.csv `task_name`: that
    field is stale for a quarter of the tree (RAW_TASK_TO_VERB) while the folder name is
    corroborated by the joint data itself (the push folders never actuate the gripper).

    place -> the release. First close->open crossing of the per-episode gripper threshold
             after T//3 frames. The T//3 guard drops the opening that happens during the
             approach when the previous episode left the gripper shut.
    push  -> the start of the return to rest, the only observable end of a push:
             d[t] = ||arm[t] - arm[T-1]|| over the 4 non-gripper joints, t_done = the LAST
             frame with d >= push_return_threshold * max(d).

    Returns a dict; `found` is False when the rule fired nothing and `t_done` fell back to
    the last frame. The caller MUST report the fallbacks -- 84 of 486 place episodes hit
    it (measured), and in 83 of those the gripper is still shut on the last frame, i.e.
    the recording stopped before/while the operator opened it. "done only at the last
    frame" is the honest reading of those, not a detection we can fake.
    """
    t = np.asarray(ticks, dtype=np.float64)
    n_frames = int(len(t))
    verb = RAW_TASK_TO_VERB.get(folder_task)
    if verb is None:
        raise FatalConversionError(
            f"--termination-channels: folder task {folder_task!r} is not one of "
            f"{sorted(RAW_TASK_TO_VERB)}. t_done is defined per task kind (gripper release "
            "for place, return-to-rest for push); guessing one for an unknown task would "
            "put a fabricated signal in the action vector."
        )
    out = {
        "task": folder_task,
        "verb": verb,
        "n_frames": n_frames,
        "found": False,
        "t_done": max(n_frames - 1, 0),
        "method": "fallback_last_frame",
    }
    if n_frames < 2:
        out["fallback_reason"] = "episode has fewer than 2 frames"
        return out

    if verb == "place":
        g = t[:, GRIPPER_MOTOR_INDEX]
        thr = _gripper_open_threshold(g, place_gripper_threshold)
        closed = g >= thr if gripper_closed_side == "high" else g <= thr
        guard = n_frames // 3
        cross = [k for k in range(1, n_frames) if closed[k - 1] and not closed[k] and k > guard]
        out.update({
            "gripper_threshold_tick": thr,
            "gripper_tick_span": float(g.max() - g.min()),
            "guard_frames": int(guard),
            "n_close_to_open_crossings_after_guard": len(cross),
        })
        if cross:
            out.update({"found": True, "t_done": int(cross[0]),
                        "method": f"gripper_release ({place_gripper_threshold} split, "
                                  f"closed={gripper_closed_side})"})
        else:
            out["fallback_reason"] = (
                "no close->open gripper crossing after T//3"
                + ("; the gripper is still closed on the last frame (recording ended "
                   "before the release)" if closed[-1] else "")
            )
        return out

    arm = t[:, ARM_MOTOR_INDICES]
    d = np.linalg.norm(arm - arm[-1], axis=1)
    d_max = float(d.max())
    out.update({"arm_max_dist_to_final_tick": d_max,
                "push_return_threshold": float(push_return_threshold)})
    if d_max <= 0.0:
        out["fallback_reason"] = "the arm never leaves its final pose (max distance is 0)"
        return out
    idx = np.nonzero(d >= push_return_threshold * d_max)[0]
    out.update({"found": True, "t_done": int(idx[-1]),
                "method": f"push_return_start (threshold {push_return_threshold:g} of max)"})
    return out


def termination_channel_array(n_frames: int, t_done: int, done_flag_lead: int) -> np.ndarray:
    """(T, 3) float32 [progress_episode, progress_task, done_flag], rows aligned to frames.

    Time base: row t describes FRAME t, i.e. the same instant as observation.state[t] --
    it is deliberately NOT shifted the way the motor channels are (action[t] = state[t+1]).
    A 1-frame shift is 0.1 s here, and a progress channel that starts at 1/(T-1) instead
    of 0 is harder to check than one that is exactly t/(T-1).

    done_flag_lead > 0 turns the flag on that many frames BEFORE t_done (so a policy can
    signal "about to finish" early); progress_task is left on the raw t_done either way.
    """
    t = np.arange(n_frames, dtype=np.float64)
    progress_episode = t / max(n_frames - 1, 1)
    progress_task = np.minimum(t / max(int(t_done), 1), 1.0)
    done_from = max(int(t_done) - int(done_flag_lead), 0)
    done_flag = (t >= done_from).astype(np.float64)
    return np.stack([progress_episode, progress_task, done_flag], axis=1).astype(np.float32)


def _new_termination_accumulator(args) -> dict:
    return {
        "settings": {
            "push_return_threshold": float(args.push_return_threshold),
            "done_flag_lead": int(args.done_flag_lead),
            "gripper_closed_side": args.gripper_closed_side,
            "place_gripper_threshold": args.place_gripper_threshold,
        },
        "n_episodes": 0,
        "n_frames": 0,
        "n_found": 0,
        "fallbacks": [],
        "t_done_frac": {},          # verb -> [frac, ...]
        "t_done_frames": {},        # verb -> [t_done, ...]
        "methods": {},
        "channel_min": None,
        "channel_max": None,
        "channel_sum": np.zeros(len(TERMINATION_CHANNEL_NAMES), dtype=np.float64),
        "verb_mismatches": [],
    }


def _accumulate_termination(acc: dict, term_info: dict, term: np.ndarray, ep: dict) -> None:
    """Fold one episode into the accumulator and SHOUT about every fallback.

    A missing t_done is logged per episode on purpose: it is a real gap in the label
    (17 % of place episodes, measured), and a count buried in the report at the end is
    exactly the kind of thing that gets read as "it worked".
    """
    verb = term_info["verb"]
    n = int(term_info["n_frames"])
    acc["n_episodes"] += 1
    acc["n_frames"] += n
    acc["methods"][term_info["method"]] = acc["methods"].get(term_info["method"], 0) + 1
    acc["t_done_frac"].setdefault(verb, []).append(term_info["t_done"] / max(n - 1, 1))
    acc["t_done_frames"].setdefault(verb, []).append(int(term_info["t_done"]))
    if term_info["found"]:
        acc["n_found"] += 1
    else:
        acc["fallbacks"].append({
            "dir": ep["rel"], "folder_task": ep["task"], "frames": n,
            "reason": term_info.get("fallback_reason", "rule found nothing"),
            "t_done_used": int(term_info["t_done"]),
        })
        logger.warning(
            "t_done NOT FOUND for %s (%s, T=%d): %s -> falling back to t_done = T-1 = %d "
            "(done_flag fires only on the last frame, progress_task == progress_episode)",
            ep["rel"], ep["task"], n, term_info.get("fallback_reason", "rule found nothing"),
            term_info["t_done"],
        )

    # Folder task decides t_done; the language verb is only a cross-check (it is the
    # SAME evidence the language binding already uses, so a disagreement here means the
    # instruction and the termination label describe different tasks).
    language = str(ep.get("language") or "")
    # skip the folder-task fallback (--reasoning-json absent): the "instruction" is then
    # the folder name itself, so comparing it to the verb derived from that same folder
    # name would only ever report the pick_place/place spelling difference.
    actual_verb = (language.split()[0].lower()
                   if language.split() and language != ep["task"] else "")
    if actual_verb and actual_verb != verb:
        acc["verb_mismatches"].append({"dir": ep["rel"], "folder_task": ep["task"],
                                       "expected_verb": verb, "language": language})
        logger.warning("termination cross-check: %s is under folder task %r (verb %r) but "
                       "its instruction reads %r", ep["rel"], ep["task"], verb, language)

    ch_min = term.min(axis=0).astype(np.float64)
    ch_max = term.max(axis=0).astype(np.float64)
    acc["channel_min"] = ch_min if acc["channel_min"] is None else np.minimum(acc["channel_min"], ch_min)
    acc["channel_max"] = ch_max if acc["channel_max"] is None else np.maximum(acc["channel_max"], ch_max)
    acc["channel_sum"] += term.sum(axis=0, dtype=np.float64)


def _pctl(values: list) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    p = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    return {"n": int(a.size), "min": round(float(p[0]), 4), "p5": round(float(p[1]), 4),
            "p25": round(float(p[2]), 4), "p50": round(float(p[3]), 4),
            "p75": round(float(p[4]), 4), "p95": round(float(p[5]), 4),
            "max": round(float(p[6]), 4), "mean": round(float(a.mean()), 4)}


def _summarize_termination(acc: dict, args) -> dict:
    n_frames = max(acc["n_frames"], 1)
    mean = acc["channel_sum"] / n_frames
    per_channel = {}
    for i, name in enumerate(TERMINATION_CHANNEL_NAMES):
        per_channel[name] = {
            "index": N_MOTOR_CHANNELS + i,
            "min": round(float(acc["channel_min"][i]), 6) if acc["channel_min"] is not None else None,
            "max": round(float(acc["channel_max"][i]), 6) if acc["channel_max"] is not None else None,
            "mean": round(float(mean[i]), 6),
        }
    all_frac = [v for vals in acc["t_done_frac"].values() for v in vals]
    return {
        "enabled": True,
        **acc["settings"],
        "channel_names": list(TERMINATION_CHANNEL_NAMES),
        "n_motor_channels": N_MOTOR_CHANNELS,
        "action_dim": N_MOTOR_CHANNELS + len(TERMINATION_CHANNEL_NAMES),
        "time_base": ("row t = frame t (same instant as observation.state[t]); the motor "
                      "channels are the shifted ones (action[t] = state[t+1])"),
        "definitions": {
            "progress_episode": "t / (T-1); 0 on the first frame, 1 on the last",
            "progress_task": "min(t / t_done, 1)",
            "done_flag": f"1 from frame (t_done - {int(args.done_flag_lead)}) on, else 0",
        },
        "t_done_rule": {
            "place": ("first close->open crossing of the per-episode gripper split after "
                      f"T//3 ({args.place_gripper_threshold} split, closed side = "
                      f"{args.gripper_closed_side} ticks)"),
            "push": ("last frame with ||arm[t]-arm[T-1]|| >= "
                     f"{args.push_return_threshold:g} * max, i.e. the start of the return"),
            "task_kind_from": "folder name (metadata.csv task_name is stale for part of the tree)",
        },
        "n_episodes": acc["n_episodes"],
        "n_t_done_found": acc["n_found"],
        "n_t_done_fallback": len(acc["fallbacks"]),
        "fallback_policy": "t_done = T-1 (done only on the last frame). NOT a detection.",
        "fallback_episodes": acc["fallbacks"],
        "methods": acc["methods"],
        "t_done_frac_overall": _pctl(all_frac),
        "t_done_frac_by_verb": {k: _pctl(v) for k, v in sorted(acc["t_done_frac"].items())},
        "t_done_frames_by_verb": {k: _pctl(v) for k, v in sorted(acc["t_done_frames"].items())},
        "channel_stats_written": per_channel,
        "done_flag_one_frac": round(float(mean[TERMINATION_CHANNEL_NAMES.index("done_flag")]), 6),
        "language_verb_mismatches": len(acc["verb_mismatches"]),
        "language_verb_mismatch_examples": acc["verb_mismatches"][:5],
        "language_verb_check_note": (
            "vacuous when --reasoning-json is absent: the instruction is then the folder "
            "task name itself, which is what t_done already keys off."
        ),
    }


def _log_termination_summary(summary: dict) -> None:
    logger.info("termination channels: action is %d-D = %d motor + %s",
                summary["action_dim"], summary["n_motor_channels"], summary["channel_names"])
    logger.info("t_done: found %d/%d, fallback (T-1) %d; per-verb t_done/(T-1): %s",
                summary["n_t_done_found"], summary["n_episodes"],
                summary["n_t_done_fallback"],
                {k: (v and v["p50"]) for k, v in summary["t_done_frac_by_verb"].items()})
    logger.info("written channel stats (over all frames): %s",
                json.dumps(summary["channel_stats_written"]))
    logger.info("done_flag is 1 on %.2f%% of frames", 100 * summary["done_flag_one_frac"])
    if summary["language_verb_mismatches"]:
        logger.warning("%d episode(s) whose instruction verb disagrees with the folder task "
                       "used for t_done: %s", summary["language_verb_mismatches"],
                       summary["language_verb_mismatch_examples"])


# ---------------------------------------------------------------------------
# raw recorder folders (<root>/<person>/<task>/<timestamp>/)
# ---------------------------------------------------------------------------
def png_header_hw(path: Path) -> tuple[int, int]:
    """(H, W) from the PNG IHDR chunk -- 24 bytes read, no decode.

    Used to check EVERY frame's resolution before a single frame is written; decoding
    ~300 PNGs per episode just to learn the size would cost more than the conversion.
    """
    with path.open("rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise EpisodeSkip(f"{path.name} is not a valid PNG (bad signature or IHDR)")
    w, h = struct.unpack(">II", head[16:24])
    return int(h), int(w)


def load_png_rgb(path: Path) -> np.ndarray:
    """(H, W, 3) uint8 RGB.

    PIL + .convert("RGB") on purpose: that is exactly what the h5 exporter used
    (dex_ref/user5dof_to_h5py_ee6.py:171-190, `load_image_seq`), so the raw path and the
    h5 path agree on channel order. cv2.imread would hand back BGR and silently swap the
    red and blue channels relative to every dataset produced so far.
    """
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_metadata_csv(path: Path) -> dict:
    """metadata.csv is a two-column key,value sheet (not a table with a header row)."""
    out: dict[str, str] = {}
    with path.open("r", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            key = row[0].strip()
            if key == "key":  # header line
                continue
            out[key] = row[1].strip()
    return out


def discover_raw_episodes(root: Path) -> tuple[list[dict], list[dict]]:
    """3-level scan -> ([episode dicts in a FIXED order], [rejected dirs]).

    Order is `sorted(person) -> sorted(task) -> sorted(timestamp)`, lexicographic, which
    is deterministic across machines and filesystems (os.walk / iterdir order is inode
    order and is NOT). It also happens to be the order the previous h5 exporter used --
    see `verify_language_mapping`.
    """
    episodes, rejected = [], []
    for person_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        for task_dir in sorted((p for p in person_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            for ep_dir in sorted((p for p in task_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                if not (ep_dir / RAW_JOINTS_CSV).is_file():
                    rejected.append({"dir": str(ep_dir.relative_to(root)),
                                     "reason": f"no {RAW_JOINTS_CSV}"})
                    continue
                episodes.append({
                    "dir": ep_dir,
                    "rel": str(ep_dir.relative_to(root)),
                    "person": person_dir.name,
                    "task": task_dir.name,
                    "stamp": ep_dir.name,
                    "language": None,
                })
    return episodes, rejected


def check_raw_fps(t_seconds: np.ndarray, fps: int, tol: float, where: str) -> dict:
    """Confirm --fps against the recorder's own t_seconds. Mismatch = FATAL.

    In the h5 path --fps can only be warned about (no timestamps exist there). Here the
    period is measured, so a disagreement is not a warning: lerobot synthesizes every
    `timestamp` column from --fps, and a wrong --fps silently re-times the whole dataset
    (and with it the action chunk horizon the policy learns).
    """
    if t_seconds.size < 2:
        raise EpisodeSkip(f"joints.csv has {t_seconds.size} row(s); cannot verify the period")
    dt = np.diff(t_seconds)
    if np.any(dt <= 0):
        raise EpisodeSkip(f"t_seconds is not strictly increasing ({where})")
    target = 1.0 / fps
    median_dt = float(np.median(dt))
    if abs(median_dt - target) > tol * target:
        raise FatalConversionError(
            f"--fps {fps} (period {target:.4f}s) disagrees with {where}: median dt "
            f"{median_dt:.4f}s => {1.0 / median_dt:.3f} Hz, outside --fps-tol {tol:.1%}. "
            "Timestamps in the dataset come from --fps, so this would re-time everything."
        )
    worst = float(np.max(np.abs(dt - target)))
    if worst > tol * target:
        raise EpisodeSkip(
            f"non-uniform sampling in {where}: worst |dt - {target:.4f}s| = {worst:.4f}s "
            f"exceeds --fps-tol {tol:.1%}. lerobot timestamps are a fixed 1/fps grid, so "
            "a jittery episode would be mislabelled in time."
        )
    return {"median_dt": median_dt, "worst_dt_dev": worst}


def open_raw_episode(ep: dict, args) -> dict:
    """Validate + load ONE episode folder. Images are only listed, never loaded here."""
    ep_dir: Path = ep["dir"]

    meta_path = ep_dir / RAW_METADATA_CSV
    meta = read_metadata_csv(meta_path) if meta_path.is_file() else {}
    if not meta_path.is_file():
        raise EpisodeSkip(f"{RAW_METADATA_CSV} missing")
    # person / timestamp must agree with the folder. `task_name` is deliberately NOT
    # enforced: it is provably stale for moon/push (see RAW_TASK_TO_VERB) and rejecting
    # those 243 episodes would throw away a quarter of the dataset over a metadata typo.
    for key, folder_val, label in (
        (RAW_META_PERSON, ep["person"], "person"),
        (RAW_META_STARTED, ep["stamp"], "timestamp"),
    ):
        val = meta.get(key)
        if val is None:
            raise EpisodeSkip(f"{RAW_METADATA_CSV} has no {key!r}")
        if val != folder_val:
            raise EpisodeSkip(f"{RAW_METADATA_CSV} {key}={val!r} != folder {label} {folder_val!r}")
    meta_task_mismatch = meta.get(RAW_META_TASK) != ep["task"]

    rec_fps = meta.get(RAW_META_FPS)
    if rec_fps is not None and rec_fps.strip():
        try:
            rec_fps_f = float(rec_fps)
        except ValueError:
            rec_fps_f = None
        if rec_fps_f is not None and abs(rec_fps_f - args.fps) > 1e-6:
            raise FatalConversionError(
                f"{meta_path} says {RAW_META_FPS}={rec_fps_f:g} but --fps is {args.fps}. "
                "Refusing to re-time the recording."
            )

    ticks, t_seconds = read_joints_csv(ep_dir / RAW_JOINTS_CSV)
    n_rows = int(len(ticks))
    fps_info = check_raw_fps(t_seconds, args.fps, args.fps_tol, f"{ep['rel']}/{RAW_JOINTS_CSV}")

    if not np.isfinite(ticks).all():
        raise EpisodeSkip("joints.csv contains NaN/Inf")
    if np.any(ticks != np.round(ticks)):
        raise EpisodeSkip("joints.csv holds non-integer ticks (expected raw Dynamixel counts)")
    lo, hi = float(ticks.min()), float(ticks.max())
    if lo < DXL_TICK_MIN or hi > DXL_TICK_MAX:
        raise EpisodeSkip(
            f"tick range [{lo:.0f}, {hi:.0f}] leaves the Dynamixel 12-bit range "
            f"[{DXL_TICK_MIN}, {DXL_TICK_MAX}]"
        )

    img_files: list[list[Path]] = []
    img_hw: tuple[int, int] | None = None
    for cam_dir_name in RAW_IMAGE_DIRS:
        cam_dir = ep_dir / cam_dir_name
        if not cam_dir.is_dir():
            raise EpisodeSkip(f"image directory {cam_dir_name}/ missing")
        pngs = [p for p in cam_dir.iterdir() if p.suffix.lower() == ".png"]
        # Every frame must be NNNN.png. Enforced before sorting: natural_key_name would
        # otherwise compare an int against a str and blow up with a TypeError that says
        # nothing about which file is wrong. (Measured: 0 offenders in data/dataset_new.)
        odd = [p.name for p in pngs if not re.fullmatch(r"\d+\.png", p.name, re.IGNORECASE)]
        if odd:
            raise EpisodeSkip(f"{cam_dir_name}/ has non-numeric PNG name(s): {odd[:5]}")
        files = sorted(pngs, key=lambda p: natural_key_name(p.name))   # 9.png < 10.png
        if len(files) != n_rows:
            raise EpisodeSkip(
                f"{cam_dir_name}/ has {len(files)} PNG(s) but joints.csv has {n_rows} data "
                "row(s); frames and joint targets would be mis-paired"
            )
        for p in files:
            hw = png_header_hw(p)
            if img_hw is None:
                img_hw = hw
            elif hw != img_hw:
                raise EpisodeSkip(
                    f"resolution changes inside the episode: {cam_dir_name}/{p.name} is "
                    f"{hw[0]}x{hw[1]}, expected {img_hw[0]}x{img_hw[1]}"
                )
        img_files.append(files)

    return {
        "n_rows": n_rows,
        "ticks": ticks,
        "t_seconds": t_seconds,
        "img_files": img_files,
        "img_hw": img_hw,
        "meta": meta,
        "meta_task_mismatch": meta_task_mismatch,
        "fps_info": fps_info,
    }


# --- language --------------------------------------------------------------
def load_reasoning_labels(path: Path) -> list[dict]:
    """capstone_reasoning_labels_new.json -> entries ordered by the N in episode_N.hdf5."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise FatalConversionError(f"{path} is not a JSON list")
    indexed = []
    for entry in data:
        m = re.search(r"(\d+)", str(entry.get("filename", "")))
        if m is None:
            raise FatalConversionError(f"{path}: entry without a numeric filename: {entry!r}")
        indexed.append((int(m.group(1)), entry))
    indexed.sort(key=lambda kv: kv[0])
    numbers = [n for n, _ in indexed]
    if numbers != list(range(len(numbers))):
        raise FatalConversionError(
            f"{path}: episode numbers are not 0..{len(numbers) - 1} without gaps "
            f"(min {min(numbers)}, max {max(numbers)}, n {len(numbers)}); the index cannot "
            "be used as a position."
        )
    return [entry for _, entry in indexed]


def verify_language_mapping(episodes: list[dict], labels: list[dict], min_match: float) -> dict:
    """Bind labels[i] to episodes[i] and PROVE the binding before using it.

    `episode_N.hdf5` is the row index of an index spreadsheet
    (dex_ref/user5dof_to_h5py_ee6.py:384-415: `episode_{i}` where i enumerates the CSV
    rows), which we do not have for this recording. So the ordering rule is established
    empirically instead:

    * On the PREVIOUS export -- 924 h5 files that each carry their own provenance in
      attrs (src_person / src_sub_task / src_time) AND their own language -- the index
      order equals sorted(person, task, timestamp) EXACTLY, at all 924 positions
      (measured 2026-08-14). Persons came out ascending (cha < han < minjun < seok) and
      tasks ascending inside each person. That is the ordering used here.
    * The binding is then re-checked against THIS recording: the verb of each
      language_raw must match the folder task (RAW_TASK_TO_VERB). That check is what
      `match_rate` below reports, and the conversion refuses to attach language when it
      comes out below --lang-min-match.

    What this does NOT prove: the check is verb-level, so it constrains the mapping at
    the block level, not per episode. The per-episode part rests on the ordering rule
    above plus the fact that the language runs (9 consecutive episodes share a phrasing)
    line up with the long inter-episode gaps where the operator swapped the object
    (permutation test over 20000 draws: median boundary gap 34-45 s vs 16-20 s for random
    positions, p < 5e-5 in each of the 4 person/task blocks).
    """
    n_match = n_checkable = 0
    unknown_tasks: dict[str, int] = {}
    examples: list[dict] = []
    for ep, entry in zip(episodes, labels, strict=True):
        language = str(entry.get("language_raw", "")).strip()
        expected_verb = RAW_TASK_TO_VERB.get(ep["task"])
        if expected_verb is None:
            unknown_tasks[ep["task"]] = unknown_tasks.get(ep["task"], 0) + 1
            continue
        n_checkable += 1
        actual_verb = language.split()[0].lower() if language else ""
        if actual_verb == expected_verb:
            n_match += 1
        elif len(examples) < 5:
            examples.append({"episode": ep["rel"], "task": ep["task"],
                             "expected_verb": expected_verb, "language": language})
    rate = n_match / n_checkable if n_checkable else 0.0
    return {
        "hypothesis": "episode_N -> episodes sorted by (person, task, timestamp), N as position",
        "n_episodes": len(episodes),
        "n_checkable": n_checkable,
        "n_verb_match": n_match,
        "verb_match_rate": rate,
        "min_match_required": min_match,
        "accepted": rate >= min_match,
        "unknown_tasks": unknown_tasks,
        "mismatch_examples": examples,
        "check": "first word of language_raw vs folder task via RAW_TASK_TO_VERB "
                 "(folder task, NOT metadata.csv task_name -- that field is stale for "
                 "moon/push and is contradicted by the gripper travel in joints.csv)",
        # MEASURED on data/dataset_new 2026-08-14. The verb check alone does NOT single
        # out one ordering, so this records how the tie was broken.
        "alternatives_measured": {
            "(a) person, task, timestamp [ADOPTED]": "verb match 972/972 = 100%",
            "(b) task, person, timestamp": "verb match 486/972 = 50% -> rejected",
            "(c) timestamp, global": "verb match 100%, but assigns the IDENTICAL string to "
                                     "all 972 episodes as (a) -- same output",
            "(d) os.walk / inode order": "verb match 100% at block level, but disagrees with "
                                         "(a) on 934/972 episodes. Rejected: (i) the previous "
                                         "exporter's index was exactly sorted(person, task, "
                                         "time) at all 924 positions, which inode order is "
                                         "not; (ii) under (d) the language runs do not line "
                                         "up with the operator's object-swap gaps "
                                         "(permutation p = 1.00/0.73/1.00/1.00 vs p < 2e-4 "
                                         "for (a))",
            "(e) person DESC (moon first)": "verb match 100%; assigns the IDENTICAL string to "
                                            "all 972 episodes as (a), because the two 'place' "
                                            "blocks are elementwise equal and so are the two "
                                            "'push' blocks -- the han/moon order is therefore "
                                            "unobservable in the output",
        },
        "metadata_task_name_rate_note": (
            "the same check against metadata.csv task_name instead of the folder task gives "
            "729/972 = 75% for (a); the 243 disagreements are exactly moon/push, whose "
            "task_name field says pick_place while the joints.csv gripper never moves."
        ),
    }


def attach_raw_language(episodes: list[dict], args) -> dict:
    """Fill ep['language'] and return the report block describing where it came from."""
    if args.reasoning_json is None:
        for ep in episodes:
            ep["language"] = ep["task"]
        return {
            "language_source": "task_name only (no --reasoning-json given)",
            "language_note": "the `task` string is the FOLDER task name, e.g. 'push'.",
        }

    labels = load_reasoning_labels(Path(args.reasoning_json))
    if len(labels) != len(episodes):
        raise FatalConversionError(
            f"--reasoning-json has {len(labels)} entries but {len(episodes)} episode folders "
            "were found. episode_N is a POSITION in this ordering, so a length mismatch "
            "means the file describes a different export; refusing to guess an alignment."
        )
    mapping = verify_language_mapping(episodes, labels, args.lang_min_match)
    if not mapping["accepted"]:
        raise FatalConversionError(
            f"--reasoning-json mapping rejected: verb match {mapping['verb_match_rate']:.2%} "
            f"({mapping['n_verb_match']}/{mapping['n_checkable']}) is below --lang-min-match "
            f"{args.lang_min_match:.2%}. Examples: {mapping['mismatch_examples']}. Re-run "
            "without --reasoning-json to fall back to the folder task name."
        )
    n_reasoning = 0
    for ep, entry in zip(episodes, labels, strict=True):
        ep["language"] = str(entry.get("language_raw", "")).strip()
        if entry.get("reasoning_labels"):
            n_reasoning += 1
        if not ep["language"]:
            raise FatalConversionError(f"empty language_raw for {ep['rel']}")
    return {
        "language_source": (
            f"reasoning json ({Path(args.reasoning_json).name}), mapping VERIFIED: "
            f"verb match {mapping['verb_match_rate']:.2%}"
        ),
        "language_mapping": mapping,
        # Recorded, deliberately unused: SmolVLA conditions on ONE string per frame
        # (`task`), so a multi-step narration has nowhere to go in this schema. Left for a
        # separate experiment.
        "reasoning_labels_present": n_reasoning,
        "reasoning_labels_used": False,
        "reasoning_labels_note": (
            "`reasoning_labels` (step_0 = a multi-step narration) exists for "
            f"{n_reasoning}/{len(labels)} entries and is NOT written to the dataset: "
            "SmolVLA takes a single instruction string per frame."
        ),
    }


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def encode_ee_state(state6: np.ndarray, encoding: str) -> np.ndarray:
    """(T, 6) EE state -> (T, 6) raw or (T, 7) with yaw split into sin/cos."""
    if encoding == "raw":
        return state6.astype(np.float32)
    if encoding != "sincos":
        raise ValueError(f"unknown --ee-yaw-encoding {encoding!r}")
    yaw = state6[:, EE_YAW_INDEX].astype(np.float64)
    return np.concatenate(
        [
            state6[:, :EE_YAW_INDEX],
            np.sin(yaw)[:, None].astype(np.float32),
            np.cos(yaw)[:, None].astype(np.float32),
            state6[:, EE_YAW_INDEX + 1 :],
        ],
        axis=1,
    ).astype(np.float32)


def build_features(space: str, img_h: int, img_w: int, ee_yaw_encoding: str = "sincos",
                   termination_channels: bool = False) -> dict:
    if space == "ee":
        state_names = EE_STATE_NAMES_SINCOS if ee_yaw_encoding == "sincos" else EE_STATE_NAMES
        action_names = list(EE_ACTION_NAMES)
    else:
        state_names = list(JOINT_MOTOR_ORDER)
        action_names = list(JOINT_MOTOR_ORDER)
    if termination_channels:
        # ACTION only. observation.state stays 5-D on purpose: at deployment the state is
        # read off the motors, and there is nothing on the robot that could fill in
        # "progress" -- putting it in the observation would make the policy depend on an
        # input it cannot have.
        if space != "joint":
            raise ValueError("--termination-channels is only defined for --space joint")
        action_names = action_names + list(TERMINATION_CHANNEL_NAMES)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
    }
    for cam_key in CAMERA_KEYS:
        features[cam_key] = {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


# ---------------------------------------------------------------------------
# h5 reading (one episode at a time, images streamed frame by frame)
# ---------------------------------------------------------------------------
def open_episode(path: Path):
    """Open read-only and validate the schema. Returns (h5file, T, meta)."""
    try:
        f = h5py.File(path, "r")  # read-only, never 'a'/'r+'
    except Exception as exc:  # corrupted / truncated file
        raise EpisodeSkip(f"cannot open h5: {type(exc).__name__}: {exc}") from exc

    try:
        # `densify_factor` > 1 means qpos/action were linearly upsampled in tick space
        # while only the real image frames were stored (user5dof_to_h5py_ee6.py:203-225).
        # T(qpos) != T(images) then, which would surface here as a plain "T mismatch"
        # EpisodeSkip on every file and abort on the skip budget with no visible cause.
        # None of the 924 current files carries the attr; if one ever does, say why.
        densify = f.attrs.get("densify_factor")
        if densify is not None and int(densify) > 1:
            raise FatalConversionError(
                f"{path.name} has attrs['densify_factor'] = {int(densify)}: qpos/action are "
                f"interpolated to a dense rate while only the real image frames are stored, "
                f"so state and images are NOT frame-aligned. This converter writes one frame "
                f"per image and would mis-pair them. Re-export the h5 with densify_factor=1 "
                f"(native rate) or add an explicit de-densify step here first."
            )

        for key in [H5_STATE_KEY, H5_ACTION_KEY, H5_LANG_KEY]:
            if key not in f:
                raise EpisodeSkip(f"missing dataset {key!r}")
        for cam in H5_IMAGE_KEYS:
            if f"{H5_IMAGE_GROUP}/{cam}" not in f:
                raise EpisodeSkip(f"missing dataset {H5_IMAGE_GROUP}/{cam!r}")

        state_shape = f[H5_STATE_KEY].shape
        action_shape = f[H5_ACTION_KEY].shape
        if len(state_shape) != 2 or len(action_shape) != 2:
            raise EpisodeSkip(f"qpos/action must be 2-D, got {state_shape} / {action_shape}")
        n_frames = int(state_shape[0])
        if action_shape[0] != n_frames:
            raise EpisodeSkip(f"T mismatch qpos {state_shape[0]} vs action {action_shape[0]}")
        if state_shape[1] != 6 or action_shape[1] != 6:
            raise EpisodeSkip(f"expected 6-D qpos/action, got {state_shape[1]} / {action_shape[1]}")

        img_hw = None
        for cam in H5_IMAGE_KEYS:
            shp = f[f"{H5_IMAGE_GROUP}/{cam}"].shape
            if len(shp) != 4 or shp[3] != 3:
                raise EpisodeSkip(f"{cam} shape {shp} is not (T, H, W, 3)")
            if shp[0] != n_frames:
                raise EpisodeSkip(f"T mismatch {cam} {shp[0]} vs qpos {n_frames}")
            if img_hw is None:
                img_hw = (int(shp[1]), int(shp[2]))
            elif img_hw != (int(shp[1]), int(shp[2])):
                raise EpisodeSkip(f"cameras disagree on resolution: {img_hw} vs {shp[1:3]}")

        lang_ds = f[H5_LANG_KEY]
        languages, lang_byte_lens = decode_language(lang_ds[:])
        meta = {
            "n_frames": n_frames,
            "img_hw": img_hw,
            "attrs": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in f.attrs.items()},
            "languages": languages,
            # `language_raw` is a fixed-width |S<n> array whose width differs per file.
            # An entry that fills the full width may have been clipped on write, so we
            # compare against the real itemsize instead of a hardcoded constant.
            "lang_byte_lens": lang_byte_lens,
            "lang_itemsize": int(lang_ds.dtype.itemsize) if lang_ds.dtype.kind == "S" else None,
        }
        return f, n_frames, meta
    except (EpisodeSkip, FatalConversionError):
        f.close()
        raise
    except Exception as exc:
        f.close()
        raise EpisodeSkip(f"schema check failed: {type(exc).__name__}: {exc}") from exc


def record_language_width(report: dict, meta: dict, lang_idx: int) -> None:
    """Log the fixed-width |S<n> slot size so a global truncation cap is visible.

    `language_raw` is stored as (3,) |S<n> where n is sized per file, so "the string
    fills its slot" is NOT evidence of truncation (L1 is usually the longest of the
    three and therefore always fills it). What *would* be evidence is every file
    sharing one and the same n: that means a hard cap was applied on write. So we
    report the histogram of n and the longest instruction seen instead of guessing.
    """
    itemsize = meta.get("lang_itemsize")
    byte_lens = meta.get("lang_byte_lens") or []
    if itemsize is not None:
        hist = report.setdefault("language_itemsize_histogram", {})
        hist[str(itemsize)] = hist.get(str(itemsize), 0) + 1
    if lang_idx < len(byte_lens):
        report["max_language_bytes"] = max(report.get("max_language_bytes", 0), byte_lens[lang_idx])


def pick_language(languages: list[str], lang_level: str, rng: np.random.Generator) -> tuple[str, int]:
    """Choose the instruction fed to the policy as `task`.

    Default is L1 (the most specific phrasing, naming the object) because at deployment
    the operator types an L1-style sentence; training on L1 keeps the train and the
    deploy language distributions aligned. L2/L3 are progressively more generic and
    would leave the policy under-conditioned on the object at test time.
    """
    if lang_level == "random":
        idx = int(rng.integers(0, len(languages)))
    else:
        idx = int(lang_level) - 1
    if idx >= len(languages) or not languages[idx]:
        raise EpisodeSkip(f"language level {idx + 1} missing/empty in {languages!r}")
    return languages[idx], idx


def maybe_resize(img: np.ndarray, target_hw: tuple[int, int] | None) -> np.ndarray:
    if target_hw is None:
        return img
    import cv2  # local import: only needed when --image-hw is given

    h, w = target_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------
def dry_run(files: list[Path], args, rng) -> dict:
    report = {
        "mode": "dry-run",
        "n_input_files": len(files),
        "n_ok": 0,
        "total_frames": 0,
        "skipped": [],
        "resolutions": {},
        "length_min": None,
        "length_max": None,
        "language_itemsize_histogram": {},
        "max_language_bytes": 0,
    }
    lengths = []
    for path in files:
        try:
            f, n_frames, meta = open_episode(path)
        except EpisodeSkip as exc:
            report["skipped"].append({"file": path.name, "reason": str(exc)})
            continue
        try:
            task, lang_idx = pick_language(meta["languages"], args.lang_level, rng)
            record_language_width(report, meta, lang_idx)
            if args.space == "joint":
                if args.joints_csv_root is None:
                    raise EpisodeSkip("--space joint requires --joints-csv-root")
                csv_path = resolve_joints_csv(meta["attrs"], Path(args.joints_csv_root))
                ticks, _ = read_joints_csv(csv_path)
                joint_state_action(ticks, args.joint_action_mode, n_frames)
                grip_info = check_grip_alignment(
                    ticks, f[H5_STATE_KEY][:, 5], args.grip_check_tol,
                    args.grip_shift_margin, csv_path,
                )
                if not grip_info["informative"]:
                    report["grip_check_uninformative"] = report.get("grip_check_uninformative", 0) + 1
            report["n_ok"] += 1
            report["total_frames"] += n_frames
            lengths.append(n_frames)
            key = f"{meta['img_hw'][0]}x{meta['img_hw'][1]}"
            report["resolutions"][key] = report["resolutions"].get(key, 0) + 1
        except EpisodeSkip as exc:
            report["skipped"].append({"file": path.name, "reason": str(exc)})
        finally:
            f.close()
    if lengths:
        report["length_min"] = int(min(lengths))
        report["length_max"] = int(max(lengths))
    return report


# ---------------------------------------------------------------------------
# verification (runs inside the converter, on the freshly written dataset)
# ---------------------------------------------------------------------------
def verify_dataset(out_dir: Path, repo_id: str, expected_eps: int, expected_frames: int,
                   features: dict, *, raw_joint_checks: bool = False,
                   tick_range: tuple[int, int] | None = None,
                   termination_channels: bool = False,
                   done_flag_lead: int = 0) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    result = {"passed": False, "checks": {}}

    ds = LeRobotDataset(repo_id, root=out_dir)
    checks = result["checks"]

    checks["codebase_version"] = ds.meta.info.get("codebase_version")
    checks["num_episodes"] = ds.meta.total_episodes
    checks["num_frames"] = ds.meta.total_frames
    checks["len_dataset"] = len(ds)
    checks["fps"] = ds.meta.fps

    problems = []
    if ds.meta.total_episodes != expected_eps:
        problems.append(f"episode count {ds.meta.total_episodes} != converted {expected_eps}")
    if ds.meta.total_frames != expected_frames:
        problems.append(f"frame count {ds.meta.total_frames} != converted {expected_frames}")
    if len(ds) != expected_frames:
        problems.append(f"len(dataset) {len(ds)} != converted {expected_frames}")

    # one decoded sample: all three camera keys must survive the round trip
    sample = ds[0]
    sample_info = {}
    for cam in CAMERA_KEYS:
        if cam not in sample:
            problems.append(f"camera key {cam} missing from sample")
            continue
        val = sample[cam]
        sample_info[cam] = {"shape": list(val.shape), "dtype": str(val.dtype)}
        if val.ndim != 3 or val.shape[0] != 3:
            problems.append(f"{cam} sample shape {tuple(val.shape)} is not (3, H, W)")
    for key in ("observation.state", "action"):
        if key not in sample:
            problems.append(f"{key} missing from sample")
            continue
        val = sample[key]
        sample_info[key] = {"shape": list(val.shape), "dtype": str(val.dtype)}
        expected_dim = features[key]["shape"][0]
        if tuple(val.shape) != (expected_dim,):
            problems.append(f"{key} sample shape {tuple(val.shape)} != ({expected_dim},)")
    if "task" not in sample:
        problems.append("sample has no 'task' (language) field")
    else:
        sample_info["task"] = sample["task"]
    checks["sample"] = sample_info

    # stats.json: the GR00T pipeline once produced an empty stats file silently, so the
    # channel counts are checked explicitly rather than just the file's existence.
    stats_path = out_dir / "meta" / "stats.json"
    checks["stats_json_exists"] = stats_path.is_file()
    if not stats_path.is_file():
        problems.append("meta/stats.json is missing")
    else:
        stats = json.loads(stats_path.read_text())
        checks["stats_keys"] = sorted(stats.keys())
        for key in ("observation.state", "action"):
            expected_dim = features[key]["shape"][0]
            if key not in stats:
                problems.append(f"stats.json has no entry for {key}")
                continue
            mean = np.asarray(stats[key]["mean"]).reshape(-1)
            std = np.asarray(stats[key]["std"]).reshape(-1)
            checks[f"stats_{key}_dim"] = int(mean.size)
            if mean.size != expected_dim or std.size != expected_dim:
                problems.append(
                    f"stats.json {key} has {mean.size} channels, expected {expected_dim}"
                )
        for cam in CAMERA_KEYS:
            if cam not in stats:
                problems.append(f"stats.json has no entry for {cam}")

    # action must not be a copy of state. Read straight from the parquet shards so this
    # costs nothing (no video decode) even on the full dataset.
    checks["action_vs_state"] = _check_action_differs_from_state(out_dir, problems)

    if raw_joint_checks:
        checks["raw_joint_invariants"] = _check_raw_joint_invariants(
            out_dir, problems, tick_range,
            # the tick/absolute_next invariants only hold on the MOTOR prefix; the
            # termination channels are [0,1] floats and would trip every one of them
            n_action_motor_channels=N_MOTOR_CHANNELS if termination_channels else None,
        )
    if termination_channels:
        checks["termination_channels"] = _check_termination_channels(
            out_dir, problems, N_MOTOR_CHANNELS, done_flag_lead
        )

    result["problems"] = problems
    result["passed"] = not problems
    return result


def _check_raw_joint_invariants(out_dir: Path, problems: list[str],
                                tick_range: tuple[int, int] | None,
                                n_action_motor_channels: int | None = None) -> dict:
    """Raw-path-only: are state/action integral ticks in range, and is action[t] == state[t+1]?

    `_check_action_differs_from_state` only proves action != state. It cannot tell
    "shifted by one row" (what we want) from "shifted by seven rows" or "the next
    episode's rows" (silent data corruption that still looks non-identical). This
    reconstructs the intended relation per episode instead of trusting the loop that
    wrote it.
    """
    import pyarrow.parquet as pq

    files = sorted((out_dir / "data").rglob("*.parquet"))
    out: dict = {"parquet_files": len(files)}
    if not files:
        problems.append("no data/*.parquet found; cannot check the joint invariants")
        return out

    states, actions, ep_idx, fr_idx = [], [], [], []
    n_rows = 0
    for path in files:
        table = pq.read_table(
            path, columns=["observation.state", "action", "episode_index", "frame_index"]
        )
        states.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64))
        actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float64))
        ep_idx.append(np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64))
        fr_idx.append(np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64))
        n_rows += len(states[-1])
        if n_rows >= 200_000:  # ARBITRARY cap, same as _check_action_differs_from_state
            break
    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    episode = np.concatenate(ep_idx, axis=0)
    frame = np.concatenate(fr_idx, axis=0)
    if n_action_motor_channels is not None:
        out["action_motor_channels_checked"] = int(n_action_motor_channels)
        out["action_extra_channels_ignored_here"] = int(
            action.shape[1] - n_action_motor_channels)
        action = action[:, :n_action_motor_channels]
    out["rows_checked"] = int(len(state))
    out["episodes_checked"] = int(len(np.unique(episode)))

    both = np.concatenate([state, action], axis=0)
    out["value_min"] = float(both.min())
    out["value_max"] = float(both.max())
    n_non_integer = int(np.count_nonzero(both != np.round(both)))
    out["n_non_integer_values"] = n_non_integer
    if tick_range is not None:
        lo, hi = tick_range
        n_oob = int(np.count_nonzero((both < lo) | (both > hi)))
        out["tick_range"] = [lo, hi]
        out["n_out_of_range_values"] = n_oob
        if n_oob:
            problems.append(
                f"{n_oob} state/action value(s) outside the Dynamixel range [{lo}, {hi}] "
                f"(observed [{out['value_min']:.1f}, {out['value_max']:.1f}])"
            )
        if n_non_integer:
            problems.append(f"{n_non_integer} state/action value(s) are not whole ticks")

    # action[t] == state[t+1] inside each episode, action[T-1] == state[T-1] (hold last)
    n_bad_next = n_bad_hold = n_pairs = 0
    worst = 0.0
    for ep in np.unique(episode):
        m = episode == ep
        order = np.argsort(frame[m], kind="stable")
        s = state[m][order]
        a = action[m][order]
        if len(s) < 2:
            continue
        d_next = np.abs(a[:-1] - s[1:])
        n_pairs += int(d_next.shape[0])
        n_bad_next += int(np.count_nonzero(d_next.max(axis=1) > 0.0))
        worst = max(worst, float(d_next.max()))
        if float(np.abs(a[-1] - s[-1]).max()) > 0.0:
            n_bad_hold += 1
    out["n_pairs_checked"] = n_pairs
    out["n_violations_action_eq_next_state"] = n_bad_next
    out["n_violations_hold_last"] = n_bad_hold
    out["worst_abs_action_minus_next_state"] = worst
    if n_bad_next or n_bad_hold:
        problems.append(
            f"absolute_next relation broken: {n_bad_next}/{n_pairs} frames where "
            f"action[t] != state[t+1] (worst {worst}), {n_bad_hold} episode(s) whose last "
            "action is not the held last state"
        )
    return out


def _check_termination_channels(out_dir: Path, problems: list[str], n_motor: int,
                                done_flag_lead: int) -> dict:
    """Re-derive the termination channels FROM THE PARQUET, not from the writer.

    Checks the properties the channels are supposed to have by construction, so a bug in
    the concatenation order (e.g. the block landing before the motors, or a per-episode
    array getting reused) shows up as a violated invariant instead of as three plausible
    looking columns.
    """
    import pyarrow.parquet as pq

    files = sorted((out_dir / "data").rglob("*.parquet"))
    out: dict = {"parquet_files": len(files), "channel_names": list(TERMINATION_CHANNEL_NAMES)}
    if not files:
        problems.append("no data/*.parquet found; cannot check the termination channels")
        return out

    actions, ep_idx, fr_idx = [], [], []
    n_rows = 0
    truncated = False
    for path in files:
        table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
        actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float64))
        ep_idx.append(np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64))
        fr_idx.append(np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64))
        n_rows += len(actions[-1])
        if n_rows >= 200_000:      # ARBITRARY cap, same as the other parquet checks
            truncated = True
            break
    action = np.concatenate(actions, axis=0)
    episode = np.concatenate(ep_idx, axis=0)
    frame = np.concatenate(fr_idx, axis=0)
    if truncated:
        # the last episode in the truncated block may be cut in half, and a half episode
        # legitimately fails "progress ends at 1"
        keep = episode != episode[-1]
        action, episode, frame = action[keep], episode[keep], frame[keep]
        out["truncated_at_rows"] = int(n_rows)
    out["rows_checked"] = int(len(action))
    out["action_width"] = int(action.shape[1])

    expected_width = n_motor + len(TERMINATION_CHANNEL_NAMES)
    if action.shape[1] != expected_width:
        problems.append(f"action has {action.shape[1]} channels, expected {expected_width} "
                        f"({n_motor} motor + {len(TERMINATION_CHANNEL_NAMES)} termination)")
        return out
    term = action[:, n_motor:]

    stats = {}
    for i, name in enumerate(TERMINATION_CHANNEL_NAMES):
        col = term[:, i]
        stats[name] = {"index": n_motor + i, "min": float(col.min()), "max": float(col.max()),
                       "mean": float(col.mean()), "std": float(col.std())}
    out["per_channel"] = stats
    done = term[:, TERMINATION_CHANNEL_NAMES.index("done_flag")]
    out["done_flag_one_frac"] = float((done == 1.0).mean())

    if not np.all((term >= -1e-6) & (term <= 1 + 1e-6)):
        problems.append("a termination channel leaves [0, 1]")
    if not np.all((done == 0.0) | (done == 1.0)):
        problems.append("done_flag is not binary")

    i_pe = TERMINATION_CHANNEL_NAMES.index("progress_episode")
    i_pt = TERMINATION_CHANNEL_NAMES.index("progress_task")
    bad_start = bad_end = bad_mono = bad_done_mono = bad_done_last = 0
    t_done_frac = []
    for ep in np.unique(episode):
        m = episode == ep
        order = np.argsort(frame[m], kind="stable")
        e = term[m][order]
        if len(e) < 2:
            continue
        if abs(e[0, i_pe]) > 1e-6:
            bad_start += 1
        if abs(e[-1, i_pe] - 1.0) > 1e-6:
            bad_end += 1
        if np.any(np.diff(e[:, i_pe]) < -1e-9) or np.any(np.diff(e[:, i_pt]) < -1e-9):
            bad_mono += 1
        d = e[:, TERMINATION_CHANNEL_NAMES.index("done_flag")]
        if np.any(np.diff(d) < 0):
            bad_done_mono += 1
        if d[-1] != 1.0:
            bad_done_last += 1
        on = np.nonzero(d == 1.0)[0]
        if on.size:
            t_done_frac.append(float((on[0] + done_flag_lead) / (len(e) - 1)))
    out.update({
        "episodes_checked": int(len(np.unique(episode))),
        "n_progress_episode_not_0_at_start": bad_start,
        "n_progress_episode_not_1_at_end": bad_end,
        "n_progress_not_monotone": bad_mono,
        "n_done_flag_not_monotone": bad_done_mono,
        "n_done_flag_not_1_on_last_frame": bad_done_last,
        "t_done_frac_recovered": _pctl(t_done_frac),
    })
    for count, msg in (
        (bad_start, "progress_episode does not start at 0"),
        (bad_end, "progress_episode does not end at 1"),
        (bad_mono, "a progress channel is not non-decreasing"),
        (bad_done_mono, "done_flag switches back to 0"),
        (bad_done_last, "done_flag is not 1 on the last frame"),
    ):
        if count:
            problems.append(f"{count} episode(s) where {msg}")
    return out


def _check_action_differs_from_state(out_dir: Path, problems: list[str]) -> dict:
    """mean |action - state| over the whole dataset; 0 means the policy is a copycat.

    This is the check that was missing when `--joint-action-mode absolute_current` existed:
    it made action[t] and state[t] the same CSV row, the loss went to 0, and every
    structural check above still passed. It also catches a joints.csv that is aligned to
    the wrong frame in a way that happens to duplicate rows.

    Only meaningful when state and action live in the same space (joint space: both are
    absolute ticks). In EE space action is a delta, so the number is large by
    construction -- it is still reported, but the failure it guards against is the
    identical-tensor one.
    """
    import pyarrow.parquet as pq

    files = sorted((out_dir / "data").rglob("*.parquet"))
    out: dict = {"parquet_files": len(files)}
    if not files:
        problems.append("no data/*.parquet found; cannot compare action against state")
        return out

    states, actions = [], []
    n_rows = 0
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action"])
        states.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64))
        actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float64))
        n_rows += len(states[-1])
        if n_rows >= 200_000:  # ARBITRARY cap: enough rows to detect a constant relation
            break
    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    out["rows_compared"] = int(len(state))

    if state.shape[1] != action.shape[1]:
        out["note"] = (
            f"state is {state.shape[1]}-D and action is {action.shape[1]}-D; "
            "compared on the shared leading channels only"
        )
    k = min(state.shape[1], action.shape[1])
    diff = np.abs(action[:, :k] - state[:, :k])
    out["mean_abs_action_minus_state"] = float(diff.mean())
    out["per_channel_mean_abs"] = [float(v) for v in diff.mean(axis=0)]
    identical = np.all(diff == 0.0, axis=1)
    out["identical_frame_frac"] = float(identical.mean())

    if out["mean_abs_action_minus_state"] == 0.0 or out["identical_frame_frac"] == 1.0:
        problems.append(
            "action is bitwise identical to observation.state on "
            f"{100 * out['identical_frame_frac']:.1f}% of frames "
            f"(mean|action-state| = {out['mean_abs_action_minus_state']}). A policy trained "
            "on this converges to 'hold the current pose'. Refusing to call the dataset good."
        )
    return out


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def _make_dataset(args, features: dict, out_dir: Path):
    """LeRobotDataset.create() with the codec overrides applied (shared by both paths)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rgb_encoder = None
    if args.video_codec is not None or args.video_crf is not None:
        from lerobot.configs.video import RGBEncoderConfig

        kwargs = {}
        if args.video_codec is not None:
            kwargs["vcodec"] = args.video_codec
        if args.video_crf is not None:
            kwargs["crf"] = args.video_crf
        rgb_encoder = RGBEncoderConfig(**kwargs)
        logger.info("RGB encoder overridden: %s", kwargs)
    else:
        logger.info("RGB encoder: lerobot defaults (libsvtav1 / crf 30). NOTE: torchcodec "
                    "is not installed here, so training will decode AV1 through PyAV; "
                    "pass --video-codec h264 if the dataloader turns out to be the "
                    "bottleneck.")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=out_dir,
        robot_type=args.robot_type,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        streaming_encoding=args.streaming_encoding,
        rgb_encoder=rgb_encoder,
    )
    logger.info("created v3.0 dataset at %s (repo_id=%s)", out_dir, args.repo_id)
    return dataset


def convert(files: list[Path], args, rng) -> dict:
    out_dir = Path(args.out).expanduser().resolve()
    target_hw = parse_image_hw(args.image_hw)
    guard = RssGuard(args.max_rss_gb)

    # --- probe the first usable file to pin the video resolution -------------
    probe_hw = None
    for path in files:
        try:
            f, _, meta = open_episode(path)
        except EpisodeSkip:
            continue
        probe_hw = meta["img_hw"]
        f.close()
        break
    if probe_hw is None:
        raise RuntimeError("no readable h5 file found; nothing to convert")
    img_h, img_w = target_hw if target_hw is not None else probe_hw
    logger.info("video resolution: %dx%d (source %dx%d, resize=%s)",
                img_h, img_w, probe_hw[0], probe_hw[1],
                "yes" if target_hw is not None else "no")

    features = build_features(args.space, img_h, img_w, args.ee_yaw_encoding)
    logger.info("features: %s", {k: (v["dtype"], tuple(v["shape"])) for k, v in features.items()})

    dataset = _make_dataset(args, features, out_dir)

    report = {
        "mode": "convert",
        "out": str(out_dir),
        "repo_id": args.repo_id,
        "space": args.space,
        "fps": args.fps,
        "lang_level": args.lang_level,
        "image_hw": [img_h, img_w],
        "resized": target_hw is not None,
        "video_codec": args.video_codec,
        "video_crf": args.video_crf,
        "n_input_files": len(files),
        "n_episodes": 0,
        "n_frames": 0,
        "skipped": [],
        "language_itemsize_histogram": {},
        "max_language_bytes": 0,
        "episodes": [],
        "peak_rss_gb": 0.0,
        "joint_units": args.joint_units if args.space == "joint" else None,
        "joint_action_mode": args.joint_action_mode if args.space == "joint" else None,
        "joint_motor_order": JOINT_MOTOR_ORDER if args.space == "joint" else None,
        "ee_yaw_encoding": args.ee_yaw_encoding if args.space == "ee" else None,
        "state_names": list(features["observation.state"]["names"]),
        "action_names": list(features["action"]["names"]),
        # fps provenance. The h5 files carry no timestamps; only joints.csv has t_seconds,
        # so EE conversions cannot check --fps at all. Background: the recording period is
        # still unconfirmed (user5dof_to_h5py_ee6.py:139-142 interpolates "existing ~2 Hz"
        # data, while the deployment loop paces at 10 Hz -- two different things).
        "fps_source": (
            "user-supplied, cross-checked against joints.csv t_seconds"
            if args.space == "joint"
            else "user-supplied, UNVERIFIED (no timestamps in the h5)"
        ),
        # See pick_language(): the old DexVLA dataloader drew L1/L2/L3 uniformly at random
        # per sample; this converter freezes one level per episode.
        "language_augmentation": (
            "REMOVED vs. the previous DexVLA dataloader: it sampled L1/L2/L3 uniformly at "
            "random per __getitem__, here every frame of an episode carries one fixed "
            f"instruction (--lang-level {args.lang_level}). The policy therefore sees ~1/3 "
            "of the phrasings it used to. Deliberate (deployment types L1-style sentences), "
            "but it is a real reduction in language variety."
        ),
        **_deployability(args.space),
    }
    if args.space == "joint":
        report["grip_check"] = {
            "rule": f"qpos[t,5] == clip((id25[t] - {GRIP_MIN_TICK}) / {GRIP_TICK_SPAN}, 0, 1)",
            "tol": args.grip_check_tol,
            "shift_margin": args.grip_shift_margin,
            "n_checked": 0,
            "n_uninformative": 0,
            "worst_max_abs_err": 0.0,
        }

    t0 = time.time()
    attempted = 0
    try:
        for path in files:
            attempted += 1
            try:
                f, n_frames, meta = open_episode(path)
            except EpisodeSkip as exc:
                report["skipped"].append({"file": path.name, "reason": str(exc)})
                _abort_if_too_many_skips(report, attempted, args)
                continue

            try:
                if n_frames < args.min_frames:
                    raise EpisodeSkip(f"T={n_frames} < --min-frames {args.min_frames}")

                task, lang_idx = pick_language(meta["languages"], args.lang_level, rng)
                record_language_width(report, meta, lang_idx)

                if args.space == "ee":
                    state = encode_ee_state(
                        np.asarray(f[H5_STATE_KEY][:], dtype=np.float32), args.ee_yaw_encoding
                    )
                    action = np.asarray(f[H5_ACTION_KEY][:], dtype=np.float32)
                else:
                    csv_path = resolve_joints_csv(meta["attrs"], Path(args.joints_csv_root))
                    ticks, t_seconds = read_joints_csv(csv_path)
                    _warn_fps_mismatch(t_seconds, args.fps, csv_path)
                    grip_info = check_grip_alignment(
                        ticks, f[H5_STATE_KEY][:, 5], args.grip_check_tol,
                        args.grip_shift_margin, csv_path,
                    )
                    gc_rep = report["grip_check"]
                    gc_rep["n_checked"] += 1
                    gc_rep["n_uninformative"] += int(not grip_info["informative"])
                    gc_rep["worst_max_abs_err"] = max(
                        gc_rep["worst_max_abs_err"], grip_info["max_abs_err"]
                    )
                    state, action = joint_state_action(
                        ticks_to_units(ticks, args.joint_units), args.joint_action_mode, n_frames
                    )

                if not (np.isfinite(state).all() and np.isfinite(action).all()):
                    raise EpisodeSkip("state/action contain NaN or Inf")

                img_ds = [f[f"{H5_IMAGE_GROUP}/{cam}"] for cam in H5_IMAGE_KEYS]
                for t in range(n_frames):
                    frame = {
                        "observation.state": state[t],
                        "action": action[t],
                        "task": task,
                    }
                    for cam_key, ds_h5 in zip(CAMERA_KEYS, img_ds, strict=True):
                        img = np.asarray(ds_h5[t], dtype=np.uint8)
                        frame[cam_key] = maybe_resize(img, target_hw)
                    dataset.add_frame(frame)
                    del frame

                dataset.save_episode(parallel_encoding=not args.no_parallel_encoding)

                report["n_episodes"] += 1
                report["n_frames"] += n_frames
                report["episodes"].append({
                    "file": path.name,
                    "frames": n_frames,
                    "task": task,
                    "lang_index": lang_idx,
                    "src_person": meta["attrs"].get("src_person"),
                    "src_sub_task": meta["attrs"].get("src_sub_task"),
                    "src_time": meta["attrs"].get("src_time"),
                })
                del state, action, img_ds
            except EpisodeSkip as exc:
                report["skipped"].append({"file": path.name, "reason": str(exc)})
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                _abort_if_too_many_skips(report, attempted, args)
            finally:
                f.close()
                gc.collect()

            rss = guard.check(f"after {path.name}")
            if report["n_episodes"] % args.rss_log_every == 0 or attempted == len(files):
                logger.info(
                    "[%d/%d] episodes=%d frames=%d skipped=%d rss=%.2fGB peak=%.2fGB %.1fs",
                    attempted, len(files), report["n_episodes"], report["n_frames"],
                    len(report["skipped"]), rss, guard.peak_gb, time.time() - t0,
                )
    finally:
        # finalize() must run even on abort, otherwise the parquet footer is never
        # written and even the partial dataset is unreadable.
        logger.info("finalizing dataset (flushing parquet writers / video encoders)...")
        dataset.finalize()
        report["peak_rss_gb"] = round(guard.peak_gb, 3)

    report["elapsed_s"] = round(time.time() - t0, 1)
    if report["n_episodes"] > 0:
        report["openmani_meta"] = str(write_openmani_meta(out_dir, args, features, report))
    logger.info("conversion done: %d episodes / %d frames in %.1fs (peak RSS %.2f GB)",
                report["n_episodes"], report["n_frames"], report["elapsed_s"], guard.peak_gb)
    return report


def convert_raw(episodes: list[dict], args, language_report: dict) -> dict:
    """Raw recorder folders -> LeRobot v3.0, joint space only.

    One episode is resident at a time and its images are read frame by frame straight
    into `add_frame`, so peak RAM is one frame x 3 cameras plus whatever lerobot buffers
    -- the 30 GB source is never held in memory.
    """
    out_dir = Path(args.out).expanduser().resolve()
    target_hw = parse_image_hw(args.image_hw)
    guard = RssGuard(args.max_rss_gb)

    # --- probe the first usable episode to pin the video resolution ----------
    probe_hw = None
    for ep in episodes:
        try:
            probe_hw = open_raw_episode(ep, args)["img_hw"]
        except EpisodeSkip:
            continue
        break
    if probe_hw is None:
        raise RuntimeError("no readable episode folder found; nothing to convert")
    img_h, img_w = target_hw if target_hw is not None else probe_hw
    logger.info("video resolution: %dx%d (source %dx%d, resize=%s)",
                img_h, img_w, probe_hw[0], probe_hw[1],
                "yes" if target_hw is not None else "no")

    features = build_features("joint", img_h, img_w,
                              termination_channels=args.termination_channels)
    logger.info("features: %s", {k: (v["dtype"], tuple(v["shape"])) for k, v in features.items()})

    dataset = _make_dataset(args, features, out_dir)

    report = {
        "mode": "convert",
        "input_format": "raw",
        "raw_root": str(Path(args.input).expanduser().resolve()),
        "raw_episode_order": (
            "sorted(person) -> sorted(task) -> sorted(timestamp), lexicographic; "
            "deterministic, independent of filesystem/inode order"
        ),
        "out": str(out_dir),
        "repo_id": args.repo_id,
        "space": "joint",
        "fps": args.fps,
        "lang_level": "raw",
        "image_hw": [img_h, img_w],
        "resized": target_hw is not None,
        "video_codec": args.video_codec,
        "video_crf": args.video_crf,
        "n_input_files": len(episodes),
        "n_episodes": 0,
        "n_frames": 0,
        "skipped": [],
        "episodes": [],
        "peak_rss_gb": 0.0,
        "joint_units": args.joint_units,
        "joint_action_mode": args.joint_action_mode,
        "joint_motor_order": list(JOINT_MOTOR_ORDER),
        "state_names": list(features["observation.state"]["names"]),
        "action_names": list(features["action"]["names"]),
        "state_definition": "state[t] = joints.csv row t, raw Dynamixel ticks, order "
                            f"{JOINT_MOTOR_ORDER}",
        "action_definition": (
            "action[t] = state[t+1] (absolute_next); the last frame holds the last target. "
            "Absolute because openmani4_follower.send_action() writes Goal_Position "
            "directly (client_live.py:394-399) and no integrator exists in the new stack."
        ),
        "fps_source": "verified from joints.csv t_seconds",
        "fps_tol": args.fps_tol,
        "tick_range_checked": [DXL_TICK_MIN, DXL_TICK_MAX],
        "metadata_task_name_mismatches": 0,
        "metadata_task_name_note": (
            "metadata.csv task_name is NOT used and NOT enforced: it is stale for "
            "moon/push (says pick_place). The folder task is used instead, corroborated "
            "by gripper travel in joints.csv."
        ),
        **language_report,
        **_deployability("joint"),
    }
    term_acc = _new_termination_accumulator(args) if args.termination_channels else None

    t0 = time.time()
    attempted = 0
    try:
        for ep in episodes:
            attempted += 1
            try:
                info = open_raw_episode(ep, args)
                n_frames = info["n_rows"]
                if n_frames < args.min_frames:
                    raise EpisodeSkip(f"T={n_frames} < --min-frames {args.min_frames}")
                if info["img_hw"] != probe_hw:
                    raise EpisodeSkip(
                        f"episode resolution {info['img_hw'][0]}x{info['img_hw'][1]} differs "
                        f"from the dataset resolution {probe_hw[0]}x{probe_hw[1]}"
                    )
                report["metadata_task_name_mismatches"] += int(info["meta_task_mismatch"])

                state, action = joint_state_action(
                    ticks_to_units(info["ticks"], args.joint_units),
                    args.joint_action_mode,
                    n_frames,
                )
                term_info = None
                if args.termination_channels:
                    term_info = find_t_done(
                        info["ticks"][:n_frames], ep["task"],
                        push_return_threshold=args.push_return_threshold,
                        gripper_closed_side=args.gripper_closed_side,
                        place_gripper_threshold=args.place_gripper_threshold,
                    )
                    term = termination_channel_array(
                        n_frames, term_info["t_done"], args.done_flag_lead
                    )
                    action = np.concatenate([action, term], axis=1)
                    _accumulate_termination(term_acc, term_info, term, ep)
                if not (np.isfinite(state).all() and np.isfinite(action).all()):
                    raise EpisodeSkip("state/action contain NaN or Inf")

                task = ep["language"]
                for t in range(n_frames):
                    frame = {
                        "observation.state": state[t],
                        "action": action[t],
                        "task": task,
                    }
                    for cam_key, files in zip(CAMERA_KEYS, info["img_files"], strict=True):
                        img = load_png_rgb(files[t])
                        if img.shape[:2] != tuple(probe_hw):
                            raise EpisodeSkip(
                                f"{files[t].name} decoded to {img.shape[:2]}, expected {probe_hw}"
                            )
                        frame[cam_key] = maybe_resize(img, target_hw)
                    dataset.add_frame(frame)
                    del frame

                dataset.save_episode(parallel_encoding=not args.no_parallel_encoding)

                report["n_episodes"] += 1
                report["n_frames"] += n_frames
                episode_entry = {
                    "raw_index": attempted - 1,
                    "dir": ep["rel"],
                    "frames": n_frames,
                    "task": task,
                    "person": ep["person"],
                    "folder_task": ep["task"],
                    "stamp": ep["stamp"],
                }
                if term_info is not None:
                    episode_entry["t_done"] = int(term_info["t_done"])
                    episode_entry["t_done_frac"] = round(
                        term_info["t_done"] / max(n_frames - 1, 1), 4)
                    episode_entry["t_done_method"] = term_info["method"]
                report["episodes"].append(episode_entry)
                del state, action, info
            except EpisodeSkip as exc:
                report["skipped"].append({"file": ep["rel"], "reason": str(exc)})
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                _abort_if_too_many_skips(report, attempted, args)
            finally:
                gc.collect()

            rss = guard.check(f"after {ep['rel']}")
            if report["n_episodes"] % args.rss_log_every == 0 or attempted == len(episodes):
                logger.info(
                    "[%d/%d] episodes=%d frames=%d skipped=%d rss=%.2fGB peak=%.2fGB %.1fs",
                    attempted, len(episodes), report["n_episodes"], report["n_frames"],
                    len(report["skipped"]), rss, guard.peak_gb, time.time() - t0,
                )
    finally:
        logger.info("finalizing dataset (flushing parquet writers / video encoders)...")
        dataset.finalize()
        report["peak_rss_gb"] = round(guard.peak_gb, 3)

    report["elapsed_s"] = round(time.time() - t0, 1)
    if term_acc is not None:
        report["termination_channels"] = _summarize_termination(term_acc, args)
        _log_termination_summary(report["termination_channels"])
    if report["n_episodes"] > 0:
        report["openmani_meta"] = str(write_openmani_meta(out_dir, args, features, report))
    logger.info("conversion done: %d episodes / %d frames in %.1fs (peak RSS %.2f GB)",
                report["n_episodes"], report["n_frames"], report["elapsed_s"], guard.peak_gb)
    return report


def dry_run_raw(episodes: list[dict], args, language_report: dict) -> dict:
    """Scan the episode folders and report; write nothing."""
    report = {
        "mode": "dry-run",
        "input_format": "raw",
        "n_input_files": len(episodes),
        "n_ok": 0,
        "total_frames": 0,
        "skipped": [],
        "resolutions": {},
        "length_min": None,
        "length_max": None,
        "metadata_task_name_mismatches": 0,
        **language_report,
    }
    term_acc = _new_termination_accumulator(args) if args.termination_channels else None
    lengths = []
    for ep in episodes:
        try:
            info = open_raw_episode(ep, args)
        except EpisodeSkip as exc:
            report["skipped"].append({"file": ep["rel"], "reason": str(exc)})
            continue
        joint_state_action(
            ticks_to_units(info["ticks"], args.joint_units), args.joint_action_mode,
            info["n_rows"],
        )
        if term_acc is not None:
            ti = find_t_done(
                info["ticks"][: info["n_rows"]], ep["task"],
                push_return_threshold=args.push_return_threshold,
                gripper_closed_side=args.gripper_closed_side,
                place_gripper_threshold=args.place_gripper_threshold,
            )
            _accumulate_termination(
                term_acc, ti,
                termination_channel_array(info["n_rows"], ti["t_done"], args.done_flag_lead),
                ep,
            )
        report["metadata_task_name_mismatches"] += int(info["meta_task_mismatch"])
        report["n_ok"] += 1
        report["total_frames"] += info["n_rows"]
        lengths.append(info["n_rows"])
        key = f"{info['img_hw'][0]}x{info['img_hw'][1]}"
        report["resolutions"][key] = report["resolutions"].get(key, 0) + 1
    if lengths:
        report["length_min"] = int(min(lengths))
        report["length_max"] = int(max(lengths))
    if term_acc is not None and term_acc["n_episodes"]:
        report["termination_channels"] = _summarize_termination(term_acc, args)
        _log_termination_summary(report["termination_channels"])
    return report


def _abort_if_too_many_skips(report: dict, attempted: int, args) -> None:
    n_skipped = len(report["skipped"])
    if attempted < args.skip_check_after:
        return
    frac = n_skipped / attempted
    if frac > args.max_skip_frac:
        raise RuntimeError(
            f"skip rate {frac:.1%} ({n_skipped}/{attempted}) exceeds --max-skip-frac "
            f"{args.max_skip_frac:.1%}. Last reasons: "
            + " | ".join(s["reason"] for s in report["skipped"][-5:])
        )


def _deployability(space: str) -> dict:
    """Can a policy trained on this dataset actually be executed by our new stack?"""
    if space == "joint":
        return {
            "deployable": True,
            "deployable_reason": (
                "5-D absolute Dynamixel targets in the openmani4_follower motor order "
                f"{JOINT_MOTOR_ORDER}; send_action() writes them to Goal_Position directly."
            ),
        }
    return {
        "deployable": False,
        "deployable_reason": (
            "EE space (6-D pose delta + absolute grip_norm) cannot be executed by "
            "openmani4_follower: that robot takes 5 ABSOLUTE joint values, and the new "
            "stack has neither an IK solver (the old one lived in server_infer.py, "
            "rollout_chunk_to_joint_ticks/IK, which is not ported) nor a "
            "grip_norm[0,1] -> gripper-tick map. OFFLINE COMPARISON ONLY."
        ),
    }


def write_openmani_meta(out_dir: Path, args, features: dict, report: dict) -> Path:
    """Write the unit/space contract next to the dataset (see OPENMANI_META_FILENAME)."""
    meta = {
        "openmani_meta_version": OPENMANI_META_VERSION,
        "produced_by": Path(__file__).name,
        "repo_id": args.repo_id,
        "space": args.space,
        # THE field openmani4_follower checks. "raw_ticks" / "radians" mirror
        # config_openmani4_follower.UNIT_*; EE datasets declare their own unit string so a
        # mismatch is loud rather than assumed.
        "unit": (
            {"tick": "raw_ticks", "rad": "radians"}[args.joint_units]
            if args.space == "joint"
            else "ee6_meters_radians"
        ),
        "motor_order": JOINT_MOTOR_ORDER if args.space == "joint" else None,
        "joint_action_mode": args.joint_action_mode if args.space == "joint" else None,
        "tick_center": DXL_CENTER_TICK,
        "ticks_per_rev": DXL_TICKS_PER_REV,
        "state_names": list(features["observation.state"]["names"]),
        "action_names": list(features["action"]["names"]),
        "ee_yaw_encoding": args.ee_yaw_encoding if args.space == "ee" else None,
        "fps": args.fps,
        "fps_source": report.get("fps_source"),
        "deployable": report.get("deployable"),
        "deployable_reason": report.get("deployable_reason"),
    }
    if getattr(args, "termination_channels", False):
        # THE field the deploy server must read: action_names is now longer than the motor
        # list, and sending channels 5..7 to a motor would be a command, not a no-op.
        term = report.get("termination_channels", {})
        meta["n_motor_channels"] = N_MOTOR_CHANNELS
        meta["motor_channel_slice"] = [0, N_MOTOR_CHANNELS]
        meta["termination_channels"] = {
            "names": list(TERMINATION_CHANNEL_NAMES),
            "slice": [N_MOTOR_CHANNELS, N_MOTOR_CHANNELS + len(TERMINATION_CHANNEL_NAMES)],
            "definitions": term.get("definitions"),
            "t_done_rule": term.get("t_done_rule"),
            "time_base": term.get("time_base"),
            "done_flag_lead": int(args.done_flag_lead),
            "push_return_threshold": float(args.push_return_threshold),
            "gripper_closed_side": args.gripper_closed_side,
            "place_gripper_threshold": args.place_gripper_threshold,
            "n_t_done_fallback": term.get("n_t_done_fallback"),
            "deploy_note": (
                "SEND action[:n_motor_channels] TO THE ROBOT AND NOTHING ELSE. The "
                "termination channels are predictions about the episode, not motor "
                "targets; done_flag is normalized like every other action channel, so "
                "un-normalize before thresholding it."
            ),
        }
    path = out_dir / OPENMANI_META_FILENAME
    path.write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s: unit=%s space=%s deployable=%s",
                path, meta["unit"], meta["space"], meta["deployable"])
    return path


def _warn_fps_mismatch(t_seconds: np.ndarray, fps: int, csv_path: Path) -> None:
    if len(t_seconds) < 3:
        return
    dt = np.diff(t_seconds)
    dt = dt[dt > 0]
    if dt.size == 0:
        return
    measured = 1.0 / float(np.median(dt))
    if abs(measured - fps) / fps > 0.10:
        logger.warning(
            "fps mismatch: --fps %d but %s implies %.2f Hz (median dt %.4f s). "
            "Timestamps written to the dataset use --fps, not t_seconds.",
            fps, csv_path.name, measured, float(np.median(dt)),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-format", choices=["h5", "raw"], default="h5",
                   help="h5 (default, unchanged): DexVLA *.hdf5 episodes from --h5-dir. "
                        "raw: the original recorder tree given by --input, laid out as "
                        "<person>/<task>/<timestamp>/{joints.csv, metadata.csv, "
                        "image_0/1/2/NNNN.png}. raw implies --space joint.")
    p.add_argument("--input", type=Path, default=None,
                   help="--input-format raw only: root of the recorder tree "
                        "(e.g. .../data/dataset_new). Read-only; never modified.")
    p.add_argument("--h5-dir", type=Path,
                   default=Path("/home/vcl/Desktop/minjun/vla/data/h5py_train_user_ee6"),
                   help="--input-format h5 only: directory of *.hdf5 episodes (read-only)")
    p.add_argument("--out", type=Path, required=True,
                   help="output dataset root (must not exist unless --force)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="LeRobot repo_id, metadata only. Default: local/<out dir name>")
    p.add_argument("--robot-type", type=str, default=None,
                   help="robot_type string stored in info.json (metadata only)")

    p.add_argument("--fps", type=int, required=True,
                   help="REQUIRED, no default. The h5 files carry no timestamps, so the "
                        "control rate cannot be inferred; guessing it would silently "
                        "distort every timestamp. Pass the real rate "
                        "(joints.csv t_seconds will confirm it). With --input-format raw "
                        "it is VERIFIED against t_seconds and a mismatch is a hard error.")
    p.add_argument("--fps-tol", type=float, default=0.02,
                   help="--input-format raw only: relative tolerance when checking --fps "
                        "against joints.csv t_seconds, and when checking that sampling is "
                        "uniform. Default 0.02 (= 2 ms at 10 Hz) is ARBITRARY but far "
                        "wider than the measured jitter: every one of the 97844 intervals "
                        "in data/dataset_new is exactly 0.100 s.")

    p.add_argument("--space", choices=["ee", "joint"], default=None,
                   help="ee (default for --input-format h5): 6-D EE state/action straight "
                        "from the h5. **OFFLINE COMPARISON ONLY -- a policy trained on it "
                        "CANNOT be run by openmani4_follower** (no IK, no grip_norm->tick "
                        "map in the new stack). joint: 5-D absolute Dynamixel targets, the "
                        "deployable space; from h5 it requires --joints-csv-root. With "
                        "--input-format raw the joint values are already in the tree, so "
                        "joint is the default and the ONLY allowed value (ee would need "
                        "pinocchio FK, which this script does not depend on).")
    p.add_argument("--ee-yaw-encoding", choices=["sincos", "raw"], default="sincos",
                   help="--space ee only. sincos (default): yaw becomes sin(yaw),cos(yaw), "
                        "so observation.state is 7-D. Raw yaw covers nearly the whole "
                        "circle (measured [-3.137,+3.137] rad with 47 wrap jumps in the "
                        "first 120 files) and SmolVLA normalizes STATE with MEAN_STD, "
                        "which is meaningless on a bimodal +-pi variable; the old server "
                        "did the same expansion (server_infer.py:297-300). raw: keep the "
                        "6-D state (for a like-for-like comparison against the old data). "
                        "The ACTION is left untouched either way -- dyaw is already a "
                        "shortest-arc delta clipped to +-0.5 rad.")
    p.add_argument("--joints-csv-root", type=Path, default=None,
                   help="root directory holding the original joints.csv files "
                        "(header: timestep,t_seconds,id11,id22,id23,id24,id25). "
                        "Required by --space joint.")
    p.add_argument("--joint-action-mode",
                   choices=["absolute_next", "delta_next"], default=None,
                   help="how joints.csv rows become actions. Required by --space joint, no "
                        "default on purpose. absolute_next: action[t] = joints[t+1], the "
                        "only mode the new deployment stack can execute. delta_next is "
                        "REFUSED (no integrator in openmani4_follower) and the former "
                        "absolute_current is GONE (it made action == state bitwise).")
    p.add_argument("--joint-units", choices=["tick", "rad"], default="tick",
                   help="unit written to disk for --space joint. Default 'tick': raw ticks "
                        "are what actually travels on the wire (client_live.py:394-399 "
                        "writes the int32 ticks it got straight to the motors; the only "
                        "rad->tick step in the old stack was INSIDE server_infer.py:358-367, "
                        "and the SmolVLA path has no such server). Training is invariant to "
                        "this affine change, DEPLOYMENT IS NOT -- the unit is recorded in "
                        f"{OPENMANI_META_FILENAME} and openmani4_follower refuses to run on "
                        "a mismatch.")
    p.add_argument("--grip-check-tol", type=float, default=0.01,
                   help="--space joint only. Max allowed |qpos[t,5] - clip((id25[t]-1000)/"
                        "1700,0,1)| before the episode is skipped. This is the FK-free "
                        "joints.csv<->h5 row-alignment check. Default 0.01 grip_norm "
                        "(~17 of the 1700 gripper ticks) is ARBITRARY; the relation is "
                        "exact to the last bit in the h5 files measured so far.")
    p.add_argument("--grip-shift-margin", type=float, default=0.005,
                   help="--space joint only. Skip the episode when shifting joints.csv by "
                        "+-1..2 rows lowers the mean gripper error by more than this "
                        "(= the two files are out of phase). Default 0.005 is ARBITRARY. "
                        "Note the blind spot: episodes where the gripper never moves "
                        "cannot be checked at all; they are counted as "
                        "grip_check.n_uninformative in the report.")

    p.add_argument("--termination-channels", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="OFF BY DEFAULT (the output is then byte-for-byte what it was "
                        "before this flag existed). ON: append 3 extra channels to the "
                        f"ACTION vector -- {TERMINATION_CHANNEL_NAMES} -- after the 5 "
                        "motor channels, so `action` becomes 8-D. Free in SmolVLA (it "
                        "zero-pads actions to max_action_dim=32, so no layer changes and "
                        "no lerobot code changes; the flow-matching MSE just regresses "
                        "them). --input-format raw / --space joint only. THE ROBOT MUST "
                        "STILL ONLY BE SENT action[:5] -- openmani_meta.json records "
                        "n_motor_channels for that. NOTE this is a pilot: RT-1/RT-2 "
                        "predict termination as a DISCRETE class; regressing it with the "
                        "action head has no published precedent we found.")
    p.add_argument("--push-return-threshold", type=float, default=0.5,
                   help="--termination-channels only. push episodes have no gripper event, "
                        "so t_done is the start of the return to rest: the LAST frame with "
                        "||arm[t]-arm[T-1]|| >= THIS FRACTION of that episode's maximum. "
                        "Default 0.5 is ARBITRARY -- there is no independent signal to "
                        "calibrate it against. MEASURED sensitivity on the 486 push "
                        "episodes of data/dataset_new0822 (median t_done/(T-1)): "
                        "0.3 -> 0.89, 0.4 -> 0.84, 0.5 -> 0.77, 0.6 -> 0.71, 0.7 -> 0.63.")
    p.add_argument("--done-flag-lead", type=int, default=0,
                   help="--termination-channels only. Turn done_flag on this many frames "
                        "BEFORE t_done (default 0 = exactly at t_done). >0 gives the "
                        "controller warning time at the cost of a label that fires before "
                        "the task is done. progress_task keeps using the raw t_done.")
    p.add_argument("--gripper-closed-side", choices=["high", "low"], default="high",
                   help="--termination-channels only. Which side of the per-episode "
                        "gripper split means CLOSED, i.e. which crossing is the release "
                        "that ends a place episode. Default 'high' is MEASURED on "
                        "data/dataset_new0822: the arm dwells while id25 RISES (grasp) and "
                        "dwells again while it FALLS (release), push episodes hold id25 at "
                        "~2650 of a 2700 maximum, and place episodes end at ~1000-1300. "
                        "Pass 'low' to get the opposite convention -- with 'low' the "
                        "detected frame is the GRASP (median 0.52 of the episode) rather "
                        "than the release (median 0.93).")
    p.add_argument("--place-gripper-threshold", choices=["midpoint", "median"],
                   default="midpoint",
                   help="--termination-channels only. How the open/closed split is placed "
                        "inside one episode: 'midpoint' = (min+max)/2 (default), 'median' "
                        "= percentile 50. The gripper rests at one of the two levels most "
                        "of the time, which pulls the median onto that level: MEASURED on "
                        "the 486 place episodes, 'median' finds no release in 125 of them "
                        "vs 84 for 'midpoint', and its hits land on a moving arm far more "
                        "often (arm-speed p90 at t_done 36 vs 13 ticks/frame).")
    p.add_argument("--lang-level", choices=["1", "2", "3", "random"], default="1",
                   help="which language_raw entry becomes the `task`. Default 1 (most "
                        "specific, names the object) because the operator types an "
                        "L1-style sentence at deployment; this keeps the training "
                        "language distribution matched to the deployment one.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed, ONLY used by --lang-level random (value is ARBITRARY)")

    p.add_argument("--reasoning-json", type=Path, default=None,
                   help="--input-format raw only: capstone_reasoning_labels_new.json. Its "
                        "`episode_N.hdf5` index is treated as a POSITION in the "
                        "(person, task, timestamp) episode order (the ordering the "
                        "previous exporter used -- verified to hold at all 924 positions "
                        "of the earlier export, which carries its provenance in h5 attrs). "
                        "The binding is re-checked here: the verb of every language_raw "
                        "must match the folder task, and the conversion ABORTS below "
                        "--lang-min-match. Omit this flag to use the folder task name as "
                        "the instruction instead. `reasoning_labels` (the multi-step "
                        "narration) is never written: SmolVLA takes one string per frame.")
    p.add_argument("--lang-min-match", type=float, default=0.99,
                   help="--reasoning-json only: minimum verb<->task agreement before the "
                        "language mapping is accepted (default 0.99). MEASURED on "
                        "data/dataset_new: 972/972 = 100%%.")

    p.add_argument("--episodes", type=int, default=None,
                   help="convert only the first N episodes in input order -- natural sort "
                        "for h5, (person, task, timestamp) for raw (partial conversion / "
                        "smoke test). With --reasoning-json the language is bound to the "
                        "FULL ordering first, so truncating does not shift instructions.")
    p.add_argument("--min-frames", type=int, default=1,
                   help="skip episodes shorter than this (default 1 = keep everything)")
    p.add_argument("--image-hw", type=str, default=None,
                   help="optional resize as HxW, e.g. 256x256 (cv2 INTER_AREA). "
                        "Default: none, frames are stored at the native 240x320.")

    p.add_argument("--max-rss-gb", type=float, default=12.0,
                   help="hard RAM ceiling for this process + children; abort above it "
                        "(default 12 GB on a 31 GB shared machine — ARBITRARY but "
                        "deliberately conservative)")
    p.add_argument("--rss-log-every", type=int, default=25,
                   help="log RSS every N converted episodes (ARBITRARY, cosmetic)")
    p.add_argument("--max-skip-frac", type=float, default=0.05,
                   help="abort if the skip rate exceeds this (default 5%%)")
    p.add_argument("--skip-check-after", type=int, default=20,
                   help="only enforce --max-skip-frac after this many attempted files "
                        "(ARBITRARY, avoids aborting on a single early bad file)")

    p.add_argument("--image-writer-threads", type=int, default=0,
                   help="async PNG writer threads (0 = synchronous, lowest RAM)")
    p.add_argument("--streaming-encoding", action="store_true",
                   help="encode video while adding frames instead of staging PNGs")
    p.add_argument("--no-parallel-encoding", action="store_true",
                   help="encode the 3 cameras sequentially instead of in a process pool")
    p.add_argument("--video-codec", type=str, default=None,
                   help="override the RGB video codec (lerobot default: libsvtav1). "
                        "Valid names come from lerobot VALID_VIDEO_CODECS (h264, hevc, "
                        "libsvtav1, ...); h264 decodes far cheaper than AV1 when "
                        "torchcodec is absent. "
                        "Default None = leave lerobot's default untouched.")
    p.add_argument("--video-crf", type=int, default=None,
                   help="override the RGB video CRF (lerobot default: 30). Lower = better "
                        "quality, bigger files. Default None = leave lerobot's default.")

    p.add_argument("--force", action="store_true",
                   help="delete an existing --out directory (destructive, off by default)")
    p.add_argument("--dry-run", action="store_true",
                   help="scan the h5 files and report, write no dataset")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the post-conversion reload check (not recommended)")
    p.add_argument("--log-file", type=Path, default=None,
                   help="also write the log to this file")
    return p


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_file)

    out_dir = Path(args.out).expanduser().resolve()
    if args.repo_id is None:
        args.repo_id = f"local/{out_dir.name}"

    raw_mode = args.input_format == "raw"

    # --space has no argparse default so that "explicitly ee" can be told apart from
    # "not given"; the h5 path keeps its historical default of ee.
    if raw_mode:
        if args.space is None:
            args.space = "joint"
        elif args.space != "joint":
            logger.error(
                "--input-format raw only supports --space joint. The recorder tree stores "
                "5 Dynamixel ticks per frame and nothing else; producing an EE pose would "
                "require pinocchio FK, a dependency this script deliberately does not add "
                "(the h5 path already gives you EE data). Nothing was written."
            )
            return 2
    elif args.space is None:
        args.space = "ee"

    if args.termination_channels:
        if not raw_mode or args.space != "joint":
            logger.error(
                "--termination-channels needs --input-format raw (which is --space joint): "
                "t_done is read off the raw joints.csv (gripper release for place, "
                "return-to-rest for push) and off the FOLDER task name, neither of which "
                "exists on the h5 path. Nothing was written."
            )
            return 2
        if args.done_flag_lead < 0:
            logger.error("--done-flag-lead must be >= 0 (got %d). Nothing was written.",
                         args.done_flag_lead)
            return 2
        if not 0.0 < args.push_return_threshold <= 1.0:
            logger.error("--push-return-threshold must be in (0, 1] (got %g). Nothing was "
                         "written.", args.push_return_threshold)
            return 2

    files: list[Path] = []
    episodes: list[dict] = []
    language_report: dict = {}

    if raw_mode:
        if args.input is None:
            logger.error("--input-format raw requires --input <recorder root>. "
                         "Nothing was written.")
            return 2
        raw_root = Path(args.input).expanduser().resolve()
        if not raw_root.is_dir():
            logger.error("--input %s is not a directory. Nothing was written.", raw_root)
            return 2
        if args.joints_csv_root is not None:
            logger.error("--joints-csv-root is meaningless with --input-format raw: every "
                         "episode folder already contains its own %s. Nothing was written.",
                         RAW_JOINTS_CSV)
            return 2
        if args.joint_action_mode is None:
            args.joint_action_mode = "absolute_next"
            logger.warning(
                "--joint-action-mode not given; using absolute_next (the only mode the new "
                "stack can execute: openmani4_follower.send_action writes Goal_Position "
                "directly and never integrates a delta)."
            )
        episodes, rejected = discover_raw_episodes(raw_root)
        if not episodes:
            logger.error("no <person>/<task>/<timestamp>/%s found under %s",
                         RAW_JOINTS_CSV, raw_root)
            return 2
        logger.info("input: %d raw episode folder(s) from %s", len(episodes), raw_root)
        logger.info("episode order (DETERMINISTIC): sorted(person) -> sorted(task) -> "
                    "sorted(timestamp), lexicographic. First: %s | Last: %s",
                    episodes[0]["rel"], episodes[-1]["rel"])
        if rejected:
            logger.warning("%d directory/ies at episode depth have no %s and were ignored: %s",
                           len(rejected), RAW_JOINTS_CSV, [r["dir"] for r in rejected[:5]])
        try:
            language_report = attach_raw_language(episodes, args)
        except FatalConversionError as exc:
            logger.error("ABORTED (language mapping): %s", exc)
            return 2
        logger.info("language source: %s", language_report["language_source"])
        if "language_mapping" in language_report:
            logger.info("language mapping check: %s", json.dumps(
                {k: v for k, v in language_report["language_mapping"].items()
                 if k != "check"}, default=str))
        # --episodes truncates AFTER the language binding: episode_N is a position in the
        # full ordering, so binding a truncated list would shift every instruction.
        if args.episodes is not None:
            episodes = episodes[: args.episodes]
            logger.info("--episodes %d: converting the first %d episode(s) of that order",
                        args.episodes, len(episodes))
    else:
        if args.input is not None:
            logger.error("--input is only used by --input-format raw; the h5 path takes "
                         "--h5-dir. Nothing was written.")
            return 2
        if args.reasoning_json is not None:
            logger.error("--reasoning-json is only used by --input-format raw (the h5 files "
                         "carry their own language_raw). Nothing was written.")
            return 2
        if args.space == "joint":
            missing = []
            if args.joints_csv_root is None:
                missing.append("--joints-csv-root")
            if args.joint_action_mode is None:
                missing.append("--joint-action-mode")
            if missing:
                logger.error(
                    "--space joint requires %s. The original joints.csv files have not been "
                    "received yet, so joint-space conversion cannot run. Nothing was written. "
                    "Joint values are NEVER inferred from the EE data.",
                    " and ".join(missing),
                )
                return 2
            root = Path(args.joints_csv_root).expanduser()
            if not root.is_dir():
                logger.error("--joints-csv-root %s does not exist or is not a directory. "
                             "Nothing was written.", root)
                return 2

        if not args.h5_dir.is_dir():
            logger.error("--h5-dir %s is not a directory", args.h5_dir)
            return 2
        files = sorted(args.h5_dir.glob("*.hdf5"), key=natural_key)
        if not files:
            logger.error("no *.hdf5 found under %s", args.h5_dir)
            return 2
        if args.episodes is not None:
            files = files[: args.episodes]
        logger.info("input: %d h5 files from %s (natural sort order)", len(files), args.h5_dir)

    rng = np.random.default_rng(args.seed)

    if raw_mode:
        logger.info("space=joint fps=%d (VERIFIED against joints.csv t_seconds, tol %.1f%%)",
                    args.fps, 100 * args.fps_tol)
    else:
        logger.info("space=%s fps=%d (user-supplied, not inferred) lang-level=%s%s",
                    args.space, args.fps, args.lang_level,
                    "  [seed=%d, ARBITRARY default]" % args.seed if args.lang_level == "random" else "")
        logger.info("lang-level rationale: L1 names the object and matches what the operator "
                    "types at deployment, so train/deploy language distributions agree. NOTE: "
                    "the old DexVLA dataloader picked L1/L2/L3 uniformly at random per sample, "
                    "so this conversion REMOVES that language augmentation (recorded in the "
                    "report as `language_augmentation`).")
    if args.space == "ee":
        for line in (
            "=" * 78,
            "--space ee: THIS DATASET IS NOT DEPLOYABLE.",
            "  state/action are a 6-D EE pose (+ delta) with an absolute grip_norm.",
            "  openmani4_follower commands 5 ABSOLUTE Dynamixel targets; the new stack has",
            "  NO IK solver and NO grip_norm[0,1] -> gripper-tick map (both lived in the old",
            "  server_infer.py, which is not ported). A policy trained on this can be scored",
            "  offline and compared against the DexVLA runs -- it cannot move the robot.",
            "  The deployable path is --space joint, which needs joints.csv.",
            "=" * 78,
        ):
            logger.warning(line)
        logger.warning("fps=%d is UNVERIFIED in EE mode: the h5 has no timestamps, so nothing "
                       "here can confirm it (only joints.csv carries t_seconds). Recording "
                       "period is still unconfirmed -- user5dof_to_h5py_ee6.py:139-142 "
                       "interpolates 'existing ~2 Hz' data while the deploy loop paces at "
                       "10 Hz. Wrong fps distorts every timestamp in the dataset.",
                       args.fps)
        logger.info("ee yaw encoding=%s (state is %d-D)", args.ee_yaw_encoding,
                    7 if args.ee_yaw_encoding == "sincos" else 6)
    if args.space == "joint":
        logger.info("joint motor order (FIXED CONTRACT, no auto-reorder in lerobot/SmolVLA): %s",
                    JOINT_MOTOR_ORDER)
        logger.info("joint units=%s (tick<->rad map: center=%d, ticks/rev=%d). Recorded in "
                    "%s; openmani4_follower raises on a mismatch with its own config.unit.",
                    args.joint_units, DXL_CENTER_TICK, DXL_TICKS_PER_REV, OPENMANI_META_FILENAME)
        if args.joint_units == "rad":
            logger.warning("--joint-units rad: the wire unit is raw ticks and the new stack has "
                           "no rad->tick server in between. openmani4_follower must then be "
                           "configured with unit='radians' or it will command nonsense.")
        logger.info("joint action mode=%s", args.joint_action_mode)
        if args.termination_channels:
            logger.warning(
                "--termination-channels: action becomes %d-D = %s + %s. PILOT SIGNAL -- "
                "the deploy server must send only action[:%d] to the robot "
                "(n_motor_channels in %s).",
                N_MOTOR_CHANNELS + len(TERMINATION_CHANNEL_NAMES), JOINT_MOTOR_ORDER,
                TERMINATION_CHANNEL_NAMES, N_MOTOR_CHANNELS, OPENMANI_META_FILENAME)
            logger.info(
                "t_done rules: place = first close->open gripper crossing after T//3 "
                "(%s split, closed side = %s ticks); push = last frame at >= %g of the max "
                "distance to the final arm pose (--push-return-threshold, ARBITRARY). "
                "done_flag lead = %d frame(s). Episodes where the rule finds nothing fall "
                "back to t_done = T-1 and are logged one by one.",
                args.place_gripper_threshold, args.gripper_closed_side,
                args.push_return_threshold, args.done_flag_lead)
        if raw_mode:
            logger.info("state[t] = joints.csv row t (raw ticks); action[t] = state[t+1], "
                        "last frame holds the last target. joints.csv/image alignment is "
                        "checked by count (PNG count per camera == data rows) -- there is no "
                        "second file to be out of phase with here.")
        else:
            logger.info("joints.csv<->h5 alignment is checked per episode through the gripper: "
                        "qpos[t,5] == clip((id25[t]-%d)/%d, 0, 1), tol=%s, shift-margin=%s "
                        "(both ARBITRARY defaults)",
                        GRIP_MIN_TICK, GRIP_TICK_SPAN, args.grip_check_tol, args.grip_shift_margin)

    if args.dry_run:
        try:
            report = (dry_run_raw(episodes, args, language_report) if raw_mode
                      else dry_run(files, args, rng))
        except FatalConversionError as exc:
            logger.error("ABORTED (unsupported input): %s", exc)
            return 2
        report.update({
            "space": args.space,
            "fps": args.fps,
            "lang_level": "raw" if raw_mode else args.lang_level,
            "fps_source": ("verified from joints.csv t_seconds" if raw_mode
                           else "user-supplied, cross-checked against joints.csv t_seconds"
                           if args.space == "joint"
                           else "user-supplied, UNVERIFIED (no timestamps in the h5)"),
            **_deployability(args.space),
        })
        report_path = Path(str(out_dir) + ".dryrun_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        printable = dict(report)
        report_path.write_text(json.dumps(report, indent=2, default=str))
        printable["skipped"] = printable["skipped"][:10]
        logger.info("DRY RUN summary:\n%s", json.dumps(printable, indent=2, default=str))
        logger.info("dry-run report written to %s", report_path)
        n_skip = len(report["skipped"])
        if report["n_input_files"] and n_skip / report["n_input_files"] > args.max_skip_frac:
            logger.error("dry-run skip rate %.1f%% exceeds --max-skip-frac %.1f%%",
                         100 * n_skip / report["n_input_files"], 100 * args.max_skip_frac)
            return 1
        return 0

    # --- output directory safety ------------------------------------------
    if out_dir.exists():
        if not args.force:
            logger.error("output directory %s already exists. Refusing to overwrite; "
                         "pass --force to delete it first.", out_dir)
            return 2
        looks_like_ds = (out_dir / "meta" / "info.json").is_file()
        if not looks_like_ds and any(out_dir.iterdir()):
            logger.error("--force refused: %s is not empty and does not look like a "
                         "LeRobot dataset (no meta/info.json). Delete it yourself if "
                         "that is really what you want.", out_dir)
            return 2
        logger.warning("--force: deleting existing %s", out_dir)
        shutil.rmtree(out_dir)

    exit_code = 0
    report = None
    try:
        report = (convert_raw(episodes, args, language_report) if raw_mode
                  else convert(files, args, rng))
    except FatalConversionError as exc:
        logger.error("CONVERSION ABORTED (unsupported input): %s", exc)
        exit_code = 2
    except Exception as exc:
        logger.exception("conversion aborted: %s", exc)
        exit_code = 1

    if report is None:
        return exit_code

    if report["n_episodes"] == 0:
        # Reaching the end with nothing written used to exit 0 whenever the file count was
        # below --skip-check-after: an empty dataset that reported success.
        logger.error("no episode was written (%d/%d input files skipped). The output is an "
                     "empty dataset, not a usable one.", len(report["skipped"]), len(files))
        exit_code = 1

    if not args.no_verify and report["n_episodes"] > 0:
        logger.info("verifying: reloading the dataset through LeRobotDataset ...")
        try:
            features = build_features(args.space, *report["image_hw"], args.ee_yaw_encoding,
                                      termination_channels=args.termination_channels)
            verification = verify_dataset(
                out_dir, args.repo_id, report["n_episodes"], report["n_frames"], features,
                raw_joint_checks=raw_mode,
                # the [0, 4095] range only means anything while the values ARE ticks
                tick_range=((DXL_TICK_MIN, DXL_TICK_MAX)
                            if raw_mode and args.joint_units == "tick" else None),
                termination_channels=args.termination_channels,
                done_flag_lead=args.done_flag_lead,
            )
        except Exception as exc:
            logger.exception("verification crashed: %s", exc)
            verification = {"passed": False, "problems": [f"verification crashed: {exc}"]}
        report["verification"] = verification
        if verification["passed"]:
            logger.info("VERIFICATION PASSED:\n%s",
                        json.dumps(verification["checks"], indent=2, default=str))
        else:
            logger.error("VERIFICATION FAILED: %s", verification.get("problems"))
            logger.error("checks:\n%s",
                         json.dumps(verification.get("checks", {}), indent=2, default=str))
            exit_code = 1

    report_path = out_dir / "conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("report written to %s", report_path)
    logger.info(
        "SUMMARY: input_files=%d episodes=%d frames=%d skipped=%d fps=%d lang=%s space=%s "
        "peak_rss=%.2fGB verified=%s",
        report["n_input_files"], report["n_episodes"], report["n_frames"],
        len(report["skipped"]), report["fps"], report["lang_level"], report["space"],
        report["peak_rss_gb"],
        report.get("verification", {}).get("passed", "skipped"),
    )
    if report["skipped"]:
        logger.warning("skipped %d file(s); first reasons: %s", len(report["skipped"]),
                       [s["reason"] for s in report["skipped"][:5]])
    if report.get("grip_check"):
        gc_rep = report["grip_check"]
        logger.info("gripper alignment check: %d episode(s) checked, worst max err %.5f, "
                    "%d episode(s) UNINFORMATIVE (gripper never moved -> a row shift would "
                    "not have been detected there)",
                    gc_rep["n_checked"], gc_rep["worst_max_abs_err"], gc_rep["n_uninformative"])
    if not report.get("deployable", False):
        logger.warning("REMINDER: deployable=false -- %s", report.get("deployable_reason"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
