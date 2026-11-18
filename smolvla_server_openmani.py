#!/usr/bin/env python3
"""
SmolVLA joint-space inference server for the OpenMANIPULATOR-X 4-DoF + gripper arm.

Same HTTP contract as `gr00t_server_openmani.py` (which itself is byte-compatible with the
original DexVLA client `dex_ref/client_live.py`), so the existing clients run unchanged:

  POST /init    {lang: str, reasoning_cache_n: int}          -> {ok, lang, reasoning_cache_n}
  POST /predict {images: [b64 JPEG x3], qpos: [5 DXL ticks]} -> {joint_ticks_b64, chunk_size,
                                                                 joint_dim, reasoning?, done?,
                                                                 preprocess_ms, inference_ms,
                                                                 postprocess_ms, + meta}
  GET  /health                                               -> {ok, ...}

  joint_ticks_b64 = base64(np.int32 array, shape [chunk_size, joint_dim=5], C-order, tobytes())
  -- identical to gr00t_server_openmani.py:486-490 and decoded by gr00t_client_local.py:184-192
     / dex_ref/client_live.py:124-126 as `np.frombuffer(..., int32).reshape(chunk_size, joint_dim)`.

Pipeline per /predict
---------------------
  1. base64 JPEG x3 -> RGB uint8 (H,W,3)
  2. hardware image corrections SWAP_CAM01 / FLIP_CAM_H / BLACK_IMAGE (env; preserved from
     dex_ref/server_infer.py -- physical camera fixes, model-agnostic). Their state is printed
     at startup so a stale export cannot silently change the input distribution.
  3. client image slot i -> `observation.images.camera{i+1}` (the converter's own mapping:
     convert_h5_to_lerobot_v3.py:132-135 "image_0 -> camera1, image_1 -> camera2,
     image_2 -> camera3"). Overridable with --cam-indices.
  4. 5-D DXL ticks -> `observation.state` **verbatim**. NO unit conversion: the dataset
     sidecar says unit=raw_ticks (data/lerobot_v3_d2_joint/openmani_meta.json), so the policy
     was trained on raw ticks and expects raw ticks. This is the single biggest difference
     from gr00t_server_openmani.py, which converted ticks->radians because ITS converter
     wrote radians. The sidecar gate below makes the assumption non-silent.
  5. lerobot's own deployment path, unchanged:
       prepare_observation_for_inference (policies/utils.py:97) -> preprocessor
       -> policy.predict_action_chunk -> postprocessor
     Normalization is therefore the checkpoint's own MEAN_STD stats, applied exactly once.
  5b. --n-avg N (default 4): N independent flow-matching draws are averaged (see below).
  6. denormalized absolute tick targets -> round -> per-motor clamp -> int32 [n,5] -> base64.

Sample averaging (--n-avg, default 4)
-------------------------------------
SmolVLA's head is a flow-matching sampler: the SAME observation gives a DIFFERENT chunk on every
call, because `sample_actions()` starts from a fresh `noise` draw. `diag/sampling_noise.py`
measured that spread on this checkpoint (runs/d2_unfrozen/checkpoints/004000) and found it is a
large fraction of the residual motion the policy is asking for; averaging N draws shrinks it by
~1/sqrt(N) and ALSO reduced the error against ground truth (48.4 -> 41.3 ticks at N=4 on train
frames). The N draws are run as ONE batched forward pass -- the observation row is expanded to N
and the noise tensor carries N independent rows -- which costs far less than N sequential calls
(measured here: 89.8 ms at N=1 vs 112.2 ms at N=4, vs 356.9 ms for 4 sequential calls).

  * WHERE the average is taken: in TICK space, i.e. AFTER the postprocessor, the same space
    `diag/sampling_noise.py` averaged in (`batched_draw_chunks` returns `postprocessor(...)`
    output and `frame_stats` means over that). Every unnormalization mode in
    `normalize_processor.py:_apply_transform` is affine (`tensor * std + mean` and friends, no
    clipping), so this is mathematically the same as averaging in normalized space; matching the
    diagnostic keeps the two paths comparable rather than merely equivalent-in-theory. The
    --self-test asserts the two agree, so a future non-affine postprocessor step fails loudly.
  * Rounding, the per-motor clamp, and every motion/clamp/rate-limit measurement are applied
    AFTER the average, to the chunk that is actually sent.
  * --n-avg 1 keeps the original single-draw call byte-for-byte (no batch expansion, no explicit
    noise tensor, so even the RNG consumption is unchanged); --self-test checks that against an
    inline re-implementation of the old three lines.
  * Averaging is NOT free: if the policy is genuinely multi-modal (go left / go right) the mean
    of the modes is a third thing that can be worse than either. The response carries
    `spread_ticks` (mean pairwise |draw_i - draw_j| over the full predicted chunk) so that
    multi-modality is visible in the log instead of being averaged away silently.

Real-Time Chunking (--rtc, OFF by default)
------------------------------------------
Independent draws are not the only source of jitter: consecutive REQUESTS disagree too. The
command the arm feels at a re-plan boundary, |chunk(t)[emit-1] - chunk(t+emit)[0]| (the "seam",
diag/input_ablation.py:693-701), was measured at 57-65 ticks on these checkpoints while one real
control step moves 9.34 ticks -- a ~6.7x kick every 0.8 s, in the middle of a motion.

--rtc turns the overlap into an inpainting constraint instead of leaving it free. The UNEXECUTED
TAIL of the previous chunk (steps emit_steps..chunk_size, in NORMALIZED action space) is handed
to lerobot's `RTCProcessor` (policies/rtc/modeling_rtc.py) as `prev_chunk_left_over`; during
every Euler step the sampled velocity is nudged so the new chunk agrees with that tail over a
`--rtc-execution-horizon` window, with weight 1.0 on the first `--rtc-inference-delay` steps and
decaying to 0 by the end of the window (`--rtc-schedule`). No retraining, no weight change: the
same checkpoint, a different sampling path.

  * Alignment. The client executes exactly `emit_steps` rows of each chunk and then re-queries
    (smolvla_client_local.py --exec-steps, default 8 == this server's default emit_steps), so the
    new chunk's step 0 lands on the previous chunk's step `emit_steps`. That is precisely where
    the tail is sliced. If the client executes a DIFFERENT number of rows the two are misaligned
    and RTC pulls the plan toward the wrong timestep -- keep --exec-steps == --emit-steps.
  * Space. The tail is sliced from `_chunk_norm`, the normalized twin of the chunk that was
    actually SERVED (the mean over the --n-avg draws), not from ticks: RTCProcessor lives in the
    sampler's own normalized, action-padded space. Its channel width (5) is smaller than the
    sampler's max_action_dim (32); RTCProcessor.denoise_step zero-pads the tail to 32, which is
    the same zero padding SmolVLA's `pad_vector` applied to the training targets, so channels
    5..31 are pulled toward the value they were trained to hold. Rounding and the per-motor clamp
    are NOT reflected in the tail, and cannot be: the tail is the part of the chunk that was
    never emitted, so no rounded/clamped version of it exists.
  * --n-avg. All N draws of a request share ONE prev tail (expanded to N rows). There is only one
    previous plan -- the one the arm is executing -- so anchoring different draws to different
    prefixes would be anchoring to something that never happened.
  * State, and when it is dropped. The tail is cross-request state, the only such state in this
    server. It is dropped on /init, on an instruction change, and when the previous request is
    older than --rtc-stale-reset-s, so a plan for the old task cannot drag the new one. After a
    drop the next request runs completely unguided, exactly like the first request of a rollout.

`seam_ticks_mean` is reported on EVERY response (RTC on or off, same definition as
diag/input_ablation.py) so the effect is visible in the deployment log, not only in the self-test.

What this server does NOT do
----------------------------
* No rate limiting. The per-step move limit belongs to exactly one place, and that place is
  the robot/client (`openmani4_follower.Openmani4Follower._apply_safety`, max_step_ticks).
  Applying it here as well would silently double-limit and, worse, hide from the robot's clip
  counters how big the policy's real command was. Instead this server MEASURES the motion it
  is about to hand over (max |dtick| between consecutive chunk steps, and the jump from the
  present position to step 0) and reports it in the log and in the response meta, so you know
  before moving how much of the chunk the limiter will eat.
  Reference point, `vla/log.md`: the only real-robot rollout we have executed 178 steps of
  which 168 (94.4%) were rate-clipped at max_step_ticks=25 -- the motion in that run was the
  limiter's, not the policy's. This measurement exists so that cannot happen unnoticed again.
* No IK, no FK, no EE space. Actions are absolute joint targets (joint_action_mode
  =absolute_next in the sidecar), written straight to Goal_Position.

Gates that refuse to start (all before/right after the model load, never mid-rollout)
-------------------------------------------------------------------------------------
* sidecar unit/space/motor-order gate -- `openmani4_follower.check_dataset_units()`, i.e. the
  SAME check the robot class runs on connect. space must be "joint", unit "raw_ticks", and the
  motor order must equal the robot's [id11,id22,id23,id24,id25]. Bypass: --allow-mismatch.
  Why fatal: SmolVLA has no `action_feature_names`, so lerobot's rollout re-ordering
  (rollout/context.py::_resolve_action_key_order) is a no-op here -- nothing downstream can
  repair a wrong channel order, and a wrong unit is invisible until the arm drives into a stop.
* normalization gate -- `normalize_processor.py:_apply_transform` returns the tensor UNCHANGED
  when `key not in self._tensor_stats` (no warning, no error). A checkpoint saved without
  state/action stats would therefore run happily and emit garbage ticks. Startup asserts the
  stats exist, are finite, have std>0 and the right width, and then EMPIRICALLY pushes a known
  vector through both pipelines to confirm the transform is really applied. No bypass flag.

Env: conda `lerobot` (py3.12, torch 2.9.1+cu128), sdpa attention (no flash-attn on sm_120).
Run:  CUDA_VISIBLE_DEVICES=1 python smolvla_server_openmani.py --port 8000
Test: CUDA_VISIBLE_DEVICES=1 python smolvla_server_openmani.py --self-test
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Single source of truth for motor order / tick limits / the unit contract. Importing this
# package does NOT need dynamixel-sdk (the bus is built in connect(), see
# openmani4_follower.py:24-28), so the server runs on this machine where the robot is absent.
from openmani4_follower import (  # noqa: E402
    GRIPPER_TICK_MAX,
    GRIPPER_TICK_MIN,
    MOTOR_IDS,
    MOTOR_NAMES,
    TICK_MAX,
    TICK_MIN,
    DatasetUnitMismatch,
    Openmani4FollowerConfig,
    check_dataset_units,
    read_dataset_meta,
)
from openmani4_follower.config_openmani4_follower import GRIPPER_NAME  # noqa: E402

JOINT_DIM = len(MOTOR_NAMES)  # 5: [id11, id22, id23, id24, id25]
DEFAULT_CLIENT_CAMS = 3       # dex_ref/client_live.py always sends image_0/1/2

DEFAULT_MODEL_PATH = "/home/vcl/Desktop/minjun/vla/runs/d2_unfrozen/checkpoints/004000/pretrained_model"
DEFAULT_DATASET_ROOT = "/home/vcl/Desktop/minjun/vla/data/lerobot_v3_d2_joint"

# Per-motor absolute tick clamp. Mirrors Openmani4FollowerConfig.__post_init__ exactly:
# gripper -> [1000, 2700] (dex_ref/server_infer.py:352-355, confirmed against the h5),
# arm     -> [0, 4095] (full 12-bit encoder range; the real joint travel is NOT confirmed --
#            the URDF limits live in bench/fk_open_manipulator.py which is not on this machine).
TICK_LIMITS: dict[str, tuple[int, int]] = {
    name: (GRIPPER_TICK_MIN, GRIPPER_TICK_MAX) if name == GRIPPER_NAME else (TICK_MIN, TICK_MAX)
    for name in MOTOR_NAMES
}
LIMIT_LO = np.array([TICK_LIMITS[n][0] for n in MOTOR_NAMES], dtype=np.int64)
LIMIT_HI = np.array([TICK_LIMITS[n][1] for n in MOTOR_NAMES], dtype=np.int64)

# Reference rate limit used ONLY to predict how much the robot-side limiter will clip.
# Not applied here. Taken from the robot config default so the two cannot drift apart.
DEFAULT_RATE_LIMIT_TICKS = float(getattr(Openmani4FollowerConfig, "max_step_ticks", 150.0) or 150.0)

EXPECTED_FPS = 10  # the control rate the client must pace at; checked against the sidecar

# Flow-matching draws averaged per /predict. 4 = the value measured in diag/sampling_noise.py
# (spread ~1/sqrt(N), gt error 48.4 -> 41.3 ticks on TRAIN frames, one batched forward:
# 89.8 -> 112.2 ms). The client's /predict call is BLOCKING, so this latency is dead time ADDED
# in front of the 0.8 s it then spends executing 8 steps -- not time hidden inside it. Measured
# cycle 2026-08-16: 800 + ~170 ms RTT ~= 970 ms; n_avg=4 accounts for 22 ms of that.
DEFAULT_N_AVG = 4
# Termination thresholds. Both are ARBITRARY starting points, not measured on a deployed
# policy: 0.5 is the midpoint of the {0, 1} training target, and 3 consecutive requests is
# 3 x emit_steps x step_dt = 2.4 s at the default 8 steps / 100 ms. The offline done probe
# needed hysteresis (tau=0.9, K=5 frames) to keep premature fires under 10%, so treat these
# as placeholders until the same measurement is run on a real 6-D checkpoint.
DEFAULT_DONE_THRESHOLD = 0.5
DEFAULT_DONE_CONSECUTIVE = 3

# ---- Real-Time Chunking defaults (see the module docstring). OFF unless --rtc. ----------
# execution_horizon has NO constant here on purpose: its default is `emit_steps`, i.e. the number
# of steps the client actually executes, which is only known after the checkpoint is loaded.
# RTCConfig's own default is 10, which would silently disagree with this server's 8.
DEFAULT_RTC_INFERENCE_DELAY = 1
DEFAULT_RTC_SCHEDULE = "LINEAR"
DEFAULT_RTC_MAX_GUIDANCE_WEIGHT = 10.0
# [ARBITRARY] Seconds of silence after which the stored previous chunk is dropped. Not measured,
# not tuned: one control cycle is ~0.97 s (0.8 s of execution + ~0.17 s RTT, measured 2026-08-16),
# so 2.0 s is "about two cycles" -- long enough that a slow request still gets its guidance, short
# enough that a paused/restarted client never inherits a plan from before the pause. Change it
# with --rtc-stale-reset-s if the client's cycle changes.
RTC_STALE_RESET_S = 2.0
# [ARBITRARY] Number of (frame t, frame t+emit) pairs --self-test uses for the seam comparison.
# 8 pairs x 2 conditions x 2 requests ~= 4 s of GPU time; large enough that the paired difference
# is not one lucky frame, small enough to keep the self-test interactive.
RTC_SEAM_TEST_PAIRS = 8

# Debug-only keys stripped from every response before it reaches the wire (they hold numpy /
# torch objects that are not JSON-serializable). Listed once so a new one cannot be forgotten.
DEBUG_KEYS = ("_ticks_chunk", "_chunk_norm", "_batch", "_ticks_draws", "_chunk_norm_draws")


# ==========================================================================
# Gate 1 -- unit / space / motor order, from the converter's sidecar.
# ==========================================================================

def resolve_sidecar_root(dataset_root: str | None, model_path: str) -> str | None:
    """Find the directory holding `openmani_meta.json`.

    Order: explicit --dataset-root, then a copy next to the checkpoint (the checkpoint dir
    itself and its parents up to the run root), so a checkpoint that was moved to the robot
    PC can carry its own unit contract.
    """
    cands: list[Path] = []
    if dataset_root:
        cands.append(Path(dataset_root).expanduser())
    mp = Path(model_path).expanduser()
    cands.extend([mp, *list(mp.parents)[:4]])
    for c in cands:
        if (c / "openmani_meta.json").is_file():
            return str(c)
    return None


def assert_units_and_order(sidecar_root: str | None, allow_mismatch: bool) -> dict | None:
    """Print the unit contract and REFUSE to start on a mismatch."""
    if sidecar_root is None:
        msg = ("no openmani_meta.json found (looked at --dataset-root and next to the "
               "checkpoint). That file is the ONLY record of whether the policy speaks raw "
               "ticks or radians and in which motor order; without it nothing can tell a "
               "correct command from a wrong one.")
        if allow_mismatch:
            print(f"[units][WARN] {msg} --allow-mismatch given: continuing UNCHECKED.")
            return None
        raise SystemExit(f"refusing to start: {msg} Pass --dataset-root <ds> (or "
                         "--allow-mismatch if you accept an unchecked unit contract).")

    meta = read_dataset_meta(sidecar_root)
    print(f"[units] sidecar        : {Path(sidecar_root) / 'openmani_meta.json'}")
    print(f"[units] space          : {meta.get('space')!r}")
    print(f"[units] unit           : {meta.get('unit')!r}")
    print(f"[units] motor_order    : {meta.get('motor_order')}")
    print(f"[units] joint_action   : {meta.get('joint_action_mode')!r}")
    print(f"[units] fps            : {meta.get('fps')} ({meta.get('fps_source')})")
    print(f"[units] deployable     : {meta.get('deployable')}")
    print(f"[units] robot expects  : unit='raw_ticks' space='joint' order="
          f"{[f'id{MOTOR_IDS[n]}' for n in MOTOR_NAMES]}")

    try:
        # Same function the robot class runs on connect(): checks unit, space AND that the
        # sidecar motor_order equals this robot's [id11,id22,id23,id24,id25].
        check_dataset_units(sidecar_root, "raw_ticks")
    except DatasetUnitMismatch as e:
        bar = "!" * 78
        print(f"\n{bar}\n!! DATASET/ROBOT CONTRACT MISMATCH -- REFUSING TO SERVE A REAL ROBOT\n{bar}\n"
              f"   {e}\n{bar}\n")
        if allow_mismatch:
            print("[units][WARN] --allow-mismatch given: continuing anyway. The ticks this "
                  "server emits are NOT guaranteed to mean what the motors think they mean.")
            return meta
        raise SystemExit(f"refusing to start: {e}")

    if meta.get("joint_action_mode") != "absolute_next":
        print(f"[units][WARN] joint_action_mode={meta.get('joint_action_mode')!r}, not "
              "'absolute_next'. This server writes the policy output straight to "
              "Goal_Position; a delta-action dataset would need integration first.")
    fps = meta.get("fps")
    if fps != EXPECTED_FPS:
        print(f"[units][WARN] dataset fps={fps} but the deploy loop is documented at "
              f"{EXPECTED_FPS} Hz. The SERVER CANNOT ENFORCE THIS -- the client's "
              f"--step_dt_ms sets the real rate. Set it to {1000.0 / float(fps or 1):.0f} ms.")
    print("[units] OK: unit/space/motor order match the robot.")
    return meta


# ==========================================================================
# Gate 2 -- normalization really loaded AND really applied.
# ==========================================================================

def _normalizer_steps(pipeline):
    """Every step in a processor pipeline that carries loaded (un)normalization stats."""
    return [s for s in getattr(pipeline, "steps", ()) if hasattr(s, "_tensor_stats")]


def audit_normalization(preprocessor, postprocessor, action_dim: int, state_dim: int) -> dict:
    """Fail loudly if state/action normalization stats are missing or degenerate.

    Why this is a hard error and not a warning: NormalizerProcessorStep._apply_transform
    (lerobot/processor/normalize_processor.py) begins with

        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

    -- a missing stats key returns the tensor UNCHANGED, with no warning. The policy would
    then see raw ticks (~2000) where it was trained to see ~0, and emit normalized-scale
    numbers (~1) that get written to Goal_Position as tick 1. Silent, and destructive.
    """
    out: dict = {}
    pre_steps, post_steps = _normalizer_steps(preprocessor), _normalizer_steps(postprocessor)
    if not pre_steps:
        raise SystemExit("refusing to start: the preprocessor pipeline has no normalizer step "
                         "with loaded stats. The checkpoint is missing its "
                         "policy_preprocessor_step_*_normalizer_processor.safetensors.")
    if not post_steps:
        raise SystemExit("refusing to start: the postprocessor pipeline has no unnormalizer "
                         "step with loaded stats -- predicted actions would stay in normalized "
                         "units and be written to the motors as ticks ~1.")

    def _stats(steps, key, where):
        for s in steps:
            st = getattr(s, "_tensor_stats", {}) or {}
            if key in st:
                return s, st[key]
        raise SystemExit(
            f"refusing to start: '{key}' has NO normalization stats in the {where} pipeline. "
            f"normalize_processor returns the tensor unchanged for an unknown key (no warning), "
            f"so this would run and silently emit garbage. Keys present: "
            f"{sorted({k for s in steps for k in (getattr(s, '_tensor_stats', {}) or {})})}")

    for key, dim, steps, where in (
        ("observation.state", state_dim, pre_steps, "preprocessor"),
        ("action", action_dim, pre_steps, "preprocessor"),
        ("action", action_dim, post_steps, "postprocessor"),
    ):
        _, st = _stats(steps, key, where)
        mean, std = st.get("mean"), st.get("std")
        if mean is None or std is None:
            raise SystemExit(f"refusing to start: {where}['{key}'] has stats but no mean/std "
                             f"(MEAN_STD normalization needs both). Present: {sorted(st)}")
        m = mean.detach().float().cpu().numpy().reshape(-1)
        s = std.detach().float().cpu().numpy().reshape(-1)
        if m.size != dim or s.size != dim:
            raise SystemExit(f"refusing to start: {where}['{key}'] stats have width {m.size}/"
                             f"{s.size} but the policy feature is {dim}-D. A width mismatch "
                             f"broadcasts silently onto the wrong channels.")
        if not (np.isfinite(m).all() and np.isfinite(s).all()):
            raise SystemExit(f"refusing to start: {where}['{key}'] stats contain NaN/Inf.")
        if float(s.min()) <= 0.0:
            raise SystemExit(f"refusing to start: {where}['{key}'] std has a non-positive entry "
                             f"{s.tolist()} -- that channel would be divided by eps=1e-8.")
        out[f"{where}.{key}"] = {"mean": m.tolist(), "std": s.tolist()}
        print(f"[norm] {where:<13} {key:<18} mean={np.round(m, 2).tolist()} "
              f"std={np.round(s, 2).tolist()}")
    return out


def verify_normalization_applied(preprocessor, postprocessor, stats: dict, device: str,
                                 image_keys: list[str], build_obs) -> None:
    """EMPIRICAL check: push a known vector through both pipelines and verify the numbers move.

    Key presence (above) proves the stats were loaded; it does not prove the pipeline actually
    uses them for the keys this server sends. So: feed state = mean + std and require the
    preprocessed state to be ~1.0 on every channel, and feed a normalized action of all-ones
    and require the postprocessed action to be ~mean + std.
    """
    import torch

    sm = np.asarray(stats["preprocessor.observation.state"]["mean"], dtype=np.float64)
    ss = np.asarray(stats["preprocessor.observation.state"]["std"], dtype=np.float64)
    am = np.asarray(stats["postprocessor.action"]["mean"], dtype=np.float64)
    asd = np.asarray(stats["postprocessor.action"]["std"], dtype=np.float64)

    probe_state = (sm + ss).astype(np.float32)
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in image_keys]
    obs = build_obs(frames, probe_state, "normalization self-check")
    preprocessor.reset()
    with torch.no_grad():
        batch = preprocessor(obs)
    got = batch["observation.state"].detach().float().cpu().numpy().reshape(-1)
    if got.size != sm.size or not np.allclose(got, 1.0, atol=1e-3):
        raise SystemExit(
            "refusing to start: STATE NORMALIZATION IS NOT BEING APPLIED. Fed state = "
            f"mean+std {np.round(probe_state, 2).tolist()}, expected all ~1.0 out of the "
            f"preprocessor, got {np.round(got, 4).tolist()}. (normalize_processor skips any "
            "key that is missing from its stats, without warning.)")
    print(f"[norm] verified: state mean+std -> {np.round(got, 4).tolist()} (expected ~1.0)")

    postprocessor.reset()
    ones = torch.ones(1, 2, am.size, dtype=torch.float32, device=device)
    with torch.no_grad():
        back = postprocessor(ones)
    back_np = back.detach().float().cpu().numpy().reshape(2, am.size)[0]
    if not np.allclose(back_np, am + asd, rtol=1e-3, atol=1e-2):
        raise SystemExit(
            "refusing to start: ACTION UNNORMALIZATION IS NOT BEING APPLIED. Fed a normalized "
            f"chunk of ones, expected mean+std {np.round(am + asd, 2).tolist()} ticks, got "
            f"{np.round(back_np, 2).tolist()}. Writing that to Goal_Position would command "
            "tick ~1 on every motor.")
    print(f"[norm] verified: normalized ones -> {np.round(back_np, 2).tolist()} ticks "
          f"(expected mean+std {np.round(am + asd, 2).tolist()})")


# ==========================================================================
# Sampling spread -- the diagnostic that says whether averaging was needed.
# ==========================================================================

def pairwise_spread_ticks(draws: np.ndarray) -> tuple[float, list[float]]:
    """(mean over all draws, per-motor) mean pairwise |draw_i - draw_j| in ticks.

    SAME definition as `diag/sampling_noise.py::_pairwise_spread` / `_pairwise_spread_per_ch`
    (mean over unordered pairs of the mean absolute difference), so the number printed here is
    directly comparable to the numbers that motivated --n-avg. Re-implemented rather than
    imported because importing `diag.sampling_noise` at server start would drag in
    `diag.input_ablation` + `diag.smolvla_introspect` (a second policy loader) into the
    deployment process; the definition is three lines and is checked against the diag helper in
    --self-test.

    draws: (n, T, D) float. n < 2 -> nan.
    """
    n = int(draws.shape[0])
    if n < 2:
        return float("nan"), [float("nan")] * int(draws.shape[-1])
    diffs = [np.abs(draws[i] - draws[j]) for i in range(n) for j in range(i + 1, n)]
    stack = np.stack(diffs)                      # (n*(n-1)/2, T, D)
    per_ch = stack.mean(axis=(0, 1))             # mean over pairs and horizon steps
    return float(stack.mean()), [float(v) for v in per_ch]


# ==========================================================================
# Training-distribution language, printed at startup.
# ==========================================================================

def load_dataset_tasks(dataset_root: str | None) -> list[str]:
    """Unique task sentences the policy was actually trained on.

    v3 datasets store them in meta/tasks.parquet (confirmed for
    data/lerobot_v3_d2_joint); v2 used meta/tasks.jsonl. Both are read.
    """
    if not dataset_root:
        return []
    root = Path(dataset_root).expanduser()
    pq = root / "meta" / "tasks.parquet"
    if pq.is_file():
        try:
            import pandas as pd

            df = pd.read_parquet(pq)
            if df.index.name == "task":
                return [str(t) for t in df.index.tolist()]
            for col in ("task", "task_name"):
                if col in df.columns:
                    return [str(t) for t in df[col].tolist()]
        except Exception as e:  # noqa: BLE001 - never block startup on a diagnostic
            print(f"[tasks][warn] could not read {pq}: {type(e).__name__}: {e}")
    jl = root / "meta" / "tasks.jsonl"
    if jl.is_file():
        out = []
        for line in jl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(str(json.loads(line)["task"]))
            except Exception:  # noqa: BLE001
                continue
        return out
    return []


# ==========================================================================
# Server
# ==========================================================================

class SmolvlaOpenmaniServer:
    """Loads the SmolVLA checkpoint + its processors and runs ticks -> chunk -> ticks.

    Plain object (not FastAPI-bound) so --self-test drives predict_from_request() directly.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        dataset_root: str | None = DEFAULT_DATASET_ROOT,
        device: str = "cuda",
        cam_indices: list[int] | None = None,
        client_cams: int = DEFAULT_CLIENT_CAMS,
        emit_steps: int = 0,
        allow_mismatch: bool = False,
        dry_run: bool = False,
        n_avg: int = DEFAULT_N_AVG,
        rate_limit_ticks: float = DEFAULT_RATE_LIMIT_TICKS,
        diag_out: str | None = None,
        diag_every: int = 10,
        diag_expert_layers: tuple[int, ...] = (0, 1, 8, 15),
        attn_out: str | None = None,
        attn_every: int = 50,
        rtc: bool = False,
        rtc_execution_horizon: int = 0,
        rtc_inference_delay: int = DEFAULT_RTC_INFERENCE_DELAY,
        rtc_schedule: str = DEFAULT_RTC_SCHEDULE,
        rtc_max_guidance_weight: float = DEFAULT_RTC_MAX_GUIDANCE_WEIGHT,
        rtc_stale_reset_s: float = RTC_STALE_RESET_S,
        coda_rtc: str = "auto",
        rtc_weight_explicit: bool = False,
        done_threshold: float = DEFAULT_DONE_THRESHOLD,
        done_consecutive: int = DEFAULT_DONE_CONSECUTIVE,
    ):
        import torch

        self.torch = torch
        self.lang: str | None = None
        self.model_path = str(model_path)
        self.device = str(device)
        self.client_cams = int(client_cams)
        self.dry_run = bool(dry_run)
        # termination channel. A checkpoint trained on a 6-D action carries done_flag as
        # channel 5; a 5-D one has no such channel and `done` stays None, exactly as before.
        self.done_threshold = float(done_threshold)
        self.done_consecutive = int(done_consecutive)
        self.has_done = False          # set from the checkpoint width in _load()
        self._done_streak = 0
        # language snapping (configured from main(); the attributes exist even when the flag is
        # off so /health and the summary never have to guard on hasattr)
        self.lang_snap = False
        self.lang_snap_mode = "rules"
        self.lang_snap_pool = "last"
        self.lang_snap_min_sim = 0.0
        self.lang_snap_min_margin = 0.0
        self._lang_vecs = None
        self.lang_raw = None
        self.lang_snap_info = None
        # Round-trip accounting. The client measures it; the server's own clock stops at the
        # response, so JPEG encode and the network are invisible here without being told.
        self._pending_client_timing: dict = {}
        self._rtt_samples: list = []
        self._slack_samples: list = []
        self._late_requests = 0
        self.rate_limit_ticks = float(rate_limit_ticks)
        self._req_count = 0
        # ---- cross-request plan memory (RTC prefix + seam measurement) -------------------
        # `_prev_plan_state` is the ONLY state this server carries between requests. It is set
        # up before anything else so rtc_reset() is callable from any point below.
        self.rtc = None                       # RTCConfig once --rtc is on; None == pre-RTC path
        self.rtc_execution_horizon = 0
        self.rtc_inference_delay = int(rtc_inference_delay)
        self.rtc_stale_reset_s = float(rtc_stale_reset_s)
        self.coda_rtc = "off"                 # set by configure_coda_rtc() once the policy is up
        self.coda_rtc_requested = str(coda_rtc)
        self._rtc_weight_explicit = bool(rtc_weight_explicit)
        self._prev_plan_state: dict | None = None
        self._rtc_reset_reason: str | None = "startup"
        self.n_avg = int(n_avg)
        if self.n_avg < 1:
            raise SystemExit(f"--n-avg must be >= 1, got {n_avg}")

        # ---- gate 1: units / space / motor order (before the model load) ---------------
        self.sidecar_root = resolve_sidecar_root(dataset_root, model_path)
        self.sidecar = assert_units_and_order(self.sidecar_root, allow_mismatch)
        self.fps = int((self.sidecar or {}).get("fps") or EXPECTED_FPS)
        self.motor_order = list((self.sidecar or {}).get("motor_order")
                                or [f"id{MOTOR_IDS[n]}" for n in MOTOR_NAMES])

        # ---- policy ---------------------------------------------------------------------
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        print(f"[load] SmolVLAPolicy model_path={model_path} device={device}")
        # SmolVLAPolicy.from_pretrained(path, chunk_size=...) SILENTLY IGNORES the kwarg
        # (measured); the config object is the only thing that actually changes it. So load
        # the config, touch only `device`, and hand the object back.
        cfg = SmolVLAConfig.from_pretrained(model_path)
        cfg.device = self.device
        if os.path.isfile(os.path.join(str(model_path), "smolvla_ext.json")):
            # PVI / Coda / ROCKET checkpoint: stock load + extension attached per the sidecar,
            # extension tensors loaded strictly from model.safetensors (smolvla_ext.load_policy).
            from smolvla_ext import load_policy as _ext_load_policy
            self.policy = _ext_load_policy(model_path, self.device, config=cfg)
        else:
            self.policy = SmolVLAPolicy.from_pretrained(model_path, config=cfg)
        self.policy.to(self.device)
        self.policy.eval()

        pc = self.policy.config
        self.chunk_size = int(pc.chunk_size)
        self.n_action_steps = int(pc.n_action_steps)
        self.action_dim = int(pc.action_feature.shape[0])
        # width of the flow-matching noise tensor (SmolVLA pads actions to max_action_dim=32 and
        # slices back to action_dim after sampling, modeling_smolvla.py:216-218)
        self.max_action_dim = int(pc.max_action_dim)
        self.image_keys = list(pc.image_features)
        print(f"[load] APPLIED chunk_size={self.chunk_size} n_action_steps={self.n_action_steps} "
              f"num_steps(denoise)={pc.num_steps} resize={pc.resize_imgs_with_padding}")
        print(f"[load] input_features   = "
              f"{ {k: tuple(v.shape) for k, v in pc.input_features.items()} }")
        print(f"[load] output action    = {tuple(pc.action_feature.shape)}  "
              f"max_state_dim={pc.max_state_dim} max_action_dim={pc.max_action_dim}")
        print(f"[load] image_features   = {self.image_keys} (empty_cameras={pc.empty_cameras})")
        # input_features['observation.state'] is inherited from smolvla_base and is NOT
        # trustworthy (lerobot's factory only fills input_features when empty, so a pretrained
        # checkpoint keeps the dim it was pretrained with). The dataset/stats width is the real
        # one; both are printed so the discrepancy is visible rather than assumed away.
        declared_state = None
        try:
            declared_state = int(pc.input_features["observation.state"].shape[0])
        except (KeyError, AttributeError, TypeError):
            pass
        # JOINT_DIM (5) = motors only. JOINT_DIM + 1 (6) = motors + done_flag, produced by
        # append_progress_channel_v3.py --channel done_flag. Anything else is the wrong robot.
        # The extra channel is NEVER sent to the arm: only action[:JOINT_DIM] becomes ticks.
        if self.action_dim == JOINT_DIM:
            self.has_done = False
        elif self.action_dim == JOINT_DIM + 1:
            self.has_done = True
            print(f"[load] action is {self.action_dim}-D: {MOTOR_NAMES} + done_flag. "
                  f"Only the first {JOINT_DIM} channels are sent to the motors; channel "
                  f"{JOINT_DIM} is reported as `done_score` and thresholded into `done` "
                  f"(--done-threshold {self.done_threshold}, "
                  f"--done-consecutive {self.done_consecutive}).")
        else:
            raise SystemExit(f"refusing to start: the checkpoint's action feature is "
                             f"{self.action_dim}-D but this arm has {JOINT_DIM} motors "
                             f"{MOTOR_NAMES} (6-D = motors + done_flag is also accepted). "
                             f"Wrong checkpoint for this robot.")

        # ---- cameras (before the processors: the normalization self-check below builds a
        #      full observation, which needs the slot -> camera key mapping) ----------------
        n = len(self.image_keys)
        if cam_indices is None:
            # convert_h5_to_lerobot_v3.py:132-135 -- image_0 -> camera1, image_1 -> camera2,
            # image_2 -> camera3. Which physical camera produced image_0 is NOT verified by
            # anything in this stack (HANDOFF_PROMPT.md:258-260); keep it fixed across runs.
            cam_indices = list(range(n))
        if len(cam_indices) != n:
            raise SystemExit(f"--cam-indices has {len(cam_indices)} entries but the checkpoint "
                             f"needs {n} camera keys {self.image_keys}.")
        bad = [s for s in cam_indices if s < 0 or s >= self.client_cams]
        if bad:
            raise SystemExit(f"--cam-indices {cam_indices} references client slot(s) {bad}; "
                             f"--client-cams={self.client_cams} means valid slots are "
                             f"0..{self.client_cams - 1}.")
        if self.client_cams < n:
            raise SystemExit(f"CAMERA COUNT MISMATCH: the checkpoint needs {n} cameras "
                             f"{self.image_keys} but the client sends only {self.client_cams}. "
                             f"prepare_images() would silently drop the missing key "
                             f"(empty_cameras={pc.empty_cameras}).")
        self.cam_indices = list(cam_indices)
        self.dropped_slots = [s for s in range(self.client_cams) if s not in self.cam_indices]
        print("[load] cam mapping: " + ", ".join(
            f"{k}<-client_slot{self.cam_indices[i]}" for i, k in enumerate(self.image_keys)))
        if self.dropped_slots:
            bar = "=" * 78
            print(f"\n{bar}\n== CAMERA COUNT MISMATCH (not fatal, but read this)\n{bar}\n"
                  f"   client sends : {self.client_cams} images\n"
                  f"   checkpoint   : {n} keys {self.image_keys}\n"
                  f"   DROPPED      : client slot(s) {self.dropped_slots} never reach the model\n"
                  f"{bar}\n")
        for var in ("SWAP_CAM01", "FLIP_CAM_H", "BLACK_IMAGE"):
            if os.environ.get(var) == "1":
                print(f"[load][WARN] {var}=1 is set: the images are being altered before the "
                      f"policy sees them. Unset it unless you know the rig needs it.")

        # ---- processors -----------------------------------------------------------------
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        print(f"[load] preprocessor  steps = {[type(s).__name__ for s in self.preprocessor.steps]}")
        print(f"[load] postprocessor steps = {[type(s).__name__ for s in self.postprocessor.steps]}")

        # ---- gate 2: normalization -------------------------------------------------------
        self.norm_stats = audit_normalization(
            self.preprocessor, self.postprocessor, self.action_dim, JOINT_DIM)
        if declared_state is not None and declared_state != JOINT_DIM:
            print(f"[load][note] config.input_features['observation.state'] says {declared_state}-D "
                  f"but the loaded stats and the dataset are {JOINT_DIM}-D. The stale config value "
                  f"is inherited from smolvla_base and is unused at runtime (SmolVLA pads state to "
                  f"{pc.max_state_dim}); the {JOINT_DIM}-D stats above are what is applied.")
        verify_normalization_applied(
            self.preprocessor, self.postprocessor, self.norm_stats, self.device,
            self.image_keys, self._build_observation)
        st = self.norm_stats["preprocessor.observation.state"]
        self.state_mean = np.asarray(st["mean"], dtype=np.float64)
        self.state_std = np.asarray(st["std"], dtype=np.float64)
        act = self.norm_stats["postprocessor.action"]
        self.action_mean = np.asarray(act["mean"], dtype=np.float64)
        self.action_std = np.asarray(act["std"], dtype=np.float64)
        # Training-set state range, used only to flag an out-of-distribution present position.
        self.state_lo, self.state_hi = self._training_state_range()

        # ---- emitted chunk length ---------------------------------------------------------
        if emit_steps <= 0:
            emit_steps = self.n_action_steps
        if emit_steps > self.chunk_size:
            raise SystemExit(f"--emit-steps {emit_steps} > chunk_size {self.chunk_size}: the "
                             f"policy does not predict that many steps.")
        self.emit_steps = int(emit_steps)
        if self.emit_steps > self.n_action_steps:
            print(f"[load][WARN] emitting {self.emit_steps} steps but the checkpoint's "
                  f"n_action_steps is {self.n_action_steps}: steps beyond {self.n_action_steps} "
                  f"are further from the observation than the policy was configured to execute "
                  f"open-loop. Keep the client's exec-steps <= {self.n_action_steps}.")
        print(f"[load] emitting {self.emit_steps} of {self.chunk_size} predicted steps per "
              f"/predict (n_action_steps={self.n_action_steps}); the client must execute at "
              f"most that many before re-querying.")

        # ---- Coda + RTC: resolve --coda-rtc auto and the guidance weight -------------------
        coda_rtc, rtc_max_guidance_weight = self._resolve_coda_rtc(
            coda_rtc, rtc_max_guidance_weight, bool(rtc) )

        # ---- real-time chunking (OFF unless --rtc) -----------------------------------------
        # Deliberately NOT touched when disabled: leaving config.rtc_config as the checkpoint
        # wrote it (None) keeps predict_action_chunk on the exact call it made before this
        # feature existed. --self-test proves the two are byte-identical.
        if rtc:
            self.configure_rtc(
                enabled=True,
                execution_horizon=rtc_execution_horizon,
                inference_delay=rtc_inference_delay,
                schedule=rtc_schedule,
                max_guidance_weight=rtc_max_guidance_weight,
                horizon_explicit=int(rtc_execution_horizon) > 0,
            )
        else:
            print("[rtc] OFF (default). Every /predict starts its chunk from free noise; the "
                  "re-plan seam is whatever the sampler produces. Turn it on with --rtc.")
        self.configure_coda_rtc(coda_rtc)

        # ---- language distribution ---------------------------------------------------------
        self.tasks = load_dataset_tasks(self.sidecar_root)
        if self.tasks:
            print(f"[lang] the policy was trained on {len(self.tasks)} unique instructions "
                  f"(from {Path(self.sidecar_root) / 'meta'}):")
            for t in self.tasks:
                print(f"[lang]   {t!r}")
            print("[lang] anything else is OUT of the training distribution; /init logs a "
                  "warning when the instruction is not on this list.")
        else:
            print("[lang][warn] could not read the dataset's task list; the instruction sent to "
                  "/init cannot be checked against the training distribution.")

        # ---- diagnostics ---------------------------------------------------------------------
        self.probe = None
        self.diag_every = max(1, int(diag_every))
        if diag_out:
            from diag.probe import Probe, ProbeConfig

            names = list((self.sidecar or {}).get("action_names")
                         or [f"id{MOTOR_IDS[n]}" for n in MOTOR_NAMES])
            self.probe = Probe(self.policy, ProbeConfig(
                out_path=diag_out,
                every=1,                 # gating is done by the server (_req_count % diag_every)
                tag="deploy",
                mode="joint",            # absolute targets -> step magnitude = |a[t+1]-a[t]|
                gripper_index=-1,
                expert_layers=tuple(diag_expert_layers),
                # physical units: the probe denormalizes with these, so every magnitude it
                # reports is in TICKS and directly comparable to max_step_ticks.
                action_mean=self.action_mean.tolist(),
                action_std=self.action_std.tolist(),
                inference=True,
                extra={"action_names": names},
            ))
            hooked = self.probe.attach()   # forward hooks only; NO optimizer/grad hooks
            print(f"[diag] probe attached to {len(hooked)} modules -> {diag_out} "
                  f"(every {self.diag_every} requests, no_grad, no optimizer hooks)")

        # ---- attention maps (optional, OFF by default: one EXTRA forward pass per record) ----
        self.attn = None
        if attn_out:
            from diag.attention_map import AttentionRecorder
            from diag.smolvla_introspect import LoadedPolicy

            self.attn = AttentionRecorder(
                LoadedPolicy(policy=self.policy, preprocessor=self.preprocessor,
                             postprocessor=self.postprocessor, device=self.device,
                             model_path=self.model_path, action_mean=self.action_mean,
                             action_std=self.action_std, tasks=self.tasks),
                out_dir=attn_out, every=max(1, int(attn_every)))
            print(f"[attn] attention maps every {max(1, int(attn_every))} requests -> {attn_out}.\n"
                  f"[attn] COST: the record runs INLINE, before the response is sent -- measured "
                  f"~0.5-1.0 s added\n[attn]       to the RTT of a recorded request, which the "
                  f"client will see as a stalled control step.\n[attn]       Use a large "
                  f"--attn-every during real rollouts, or record offline with\n"
                  f"[attn]       `python -m diag.attention_map --frame-index ...`.\n"
                  f"[attn] The served chunk is untouched; because flow matching is stochastic the "
                  f"recorded\n[attn]       attention belongs to a different noise draw "
                  f"(separate_forward_pass=true in the JSONL)"
                  + (f" -- and with --n-avg {self.n_avg} the\n[attn]       served chunk is the "
                     f"mean of {self.n_avg} draws, so the map is ONE draw, not the answer."
                     if self.n_avg > 1 else "."))

        # ---- sample averaging ----------------------------------------------------------------
        if self.n_avg > 1:
            print(f"[avg] n_avg={self.n_avg}: every /predict runs {self.n_avg} flow-matching "
                  f"draws in ONE batched forward pass and serves their mean (averaged in TICK "
                  f"space, after the postprocessor -- the same space diag/sampling_noise.py "
                  f"measured in). Draw-to-draw spread is reported per request as spread_ticks; "
                  f"a large spread means the policy is multi-modal here and the mean may be a "
                  f"mode nobody predicted. --n-avg 1 restores the single-draw behaviour.")
        else:
            print("[avg] n_avg=1: single flow-matching draw per /predict (pre-averaging "
                  "behaviour). The chunk carries the full sampling noise measured in "
                  "diag/sampling_noise.py.")
        print(f"[load] ready: joint_dim={JOINT_DIM} motor_order={self.motor_order} "
              f"tick limits={ {n: TICK_LIMITS[n] for n in MOTOR_NAMES} }")

    # ---- helpers -----------------------------------------------------------
    def _training_state_range(self) -> tuple[np.ndarray, np.ndarray]:
        """min/max of observation.state over the training set, if the stats carry them."""
        lo = np.full(JOINT_DIM, -np.inf)
        hi = np.full(JOINT_DIM, np.inf)
        for s in _normalizer_steps(self.preprocessor):
            st = (getattr(s, "_tensor_stats", {}) or {}).get("observation.state")
            if not st:
                continue
            if "min" in st and "max" in st:
                lo = st["min"].detach().float().cpu().numpy().reshape(-1).astype(np.float64)
                hi = st["max"].detach().float().cpu().numpy().reshape(-1).astype(np.float64)
            break
        return lo, hi

    # ---- real-time chunking --------------------------------------------------
    def configure_rtc(self, enabled: bool = True, execution_horizon: int = 0,
                      inference_delay: int = DEFAULT_RTC_INFERENCE_DELAY,
                      schedule: str = DEFAULT_RTC_SCHEDULE,
                      max_guidance_weight: float = DEFAULT_RTC_MAX_GUIDANCE_WEIGHT,
                      horizon_explicit: bool = False) -> None:
        """Turn RTC guidance on/off on the ALREADY LOADED policy.

        A method rather than inline __init__ code because --self-test needs to flip the same
        server between the two paths and compare them on one loaded checkpoint.

        `SmolVLAPolicy.init_rtc_processor()` reads `self.config.rtc_config` and pushes the
        processor onto `self.model` when the model already exists (modeling_smolvla.py:176-190),
        and `VLAFlowMatching` holds the SAME config object, so writing the config and calling
        that one method is the whole switch. It is verified rather than assumed below: an
        rtc_config that never reaches the sampler would leave a server that reports RTC on and
        samples exactly as before.
        """
        from lerobot.configs import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        if not enabled:
            self.policy.config.rtc_config = None
            self.policy.init_rtc_processor()          # clears policy.rtc_processor
            self.policy.model.rtc_processor = None    # init_rtc_processor only sets, never clears
            self.rtc = None
            self.rtc_execution_horizon = 0
            self.rtc_reset("rtc disabled")
            print("[rtc] disabled: back on the pre-RTC sampling path.")
            return

        try:
            sched = RTCAttentionSchedule(str(schedule).upper())
        except ValueError:
            raise SystemExit(f"--rtc-schedule {schedule!r} is not one of "
                             f"{[e.value for e in RTCAttentionSchedule]}") from None

        horizon = int(execution_horizon) if int(execution_horizon) > 0 else int(self.emit_steps)
        if horizon < 1:
            raise SystemExit(f"--rtc-execution-horizon must be >= 1, got {execution_horizon}")
        delay = int(inference_delay)
        if delay < 0:
            raise SystemExit(f"--rtc-inference-delay must be >= 0, got {inference_delay}")
        overlap = int(self.chunk_size) - int(self.emit_steps)

        if horizon_explicit and horizon != self.emit_steps:
            print(f"[rtc][WARN] --rtc-execution-horizon {horizon} != --emit-steps "
                  f"{self.emit_steps}. The guidance window is the part of the OLD plan the "
                  f"client is executing; those two are the same number unless the client's "
                  f"--exec-steps differs from --emit-steps. Check smolvla_client_local.py "
                  f"--exec-steps before trusting this.")
        if overlap <= 0:
            print(f"[rtc][WARN] emit_steps {self.emit_steps} == chunk_size {self.chunk_size}: a "
                  f"chunk leaves NO unexecuted tail, so there is nothing to inpaint against and "
                  f"RTC can never engage -- every request will report rtc_used_prev_chunk=false. "
                  f"Lower --emit-steps (or use a checkpoint with a longer chunk) if you want "
                  f"RTC to do anything.")
        if horizon > overlap:
            print(f"[rtc][WARN] --rtc-execution-horizon {horizon} > the {overlap} unexecuted "
                  f"steps a chunk leaves over (chunk_size {self.chunk_size} - emit_steps "
                  f"{self.emit_steps}). RTCProcessor clamps the window to what the prefix "
                  f"actually contains (modeling_rtc.py:181-183), so the effective horizon will "
                  f"be {overlap}.")
        if delay > horizon:
            print(f"[rtc][WARN] --rtc-inference-delay {delay} >= --rtc-execution-horizon "
                  f"{horizon}: get_prefix_weights() does start=min(start,end), so every step of "
                  f"the window is pinned hard to the old plan and the new observation only "
                  f"reaches the steps beyond it.")

        try:
            cfg = RTCConfig(enabled=True, prefix_attention_schedule=sched,
                            max_guidance_weight=float(max_guidance_weight),
                            execution_horizon=horizon)
        except ValueError as e:   # RTCConfig.__post_init__ validates max_guidance_weight
            raise SystemExit(f"bad RTC settings: {e}") from None
        self.policy.config.rtc_config = cfg
        self.policy.init_rtc_processor()
        # Verify the switch actually reached the sampler instead of trusting the assignment.
        if not self.policy._rtc_enabled() or self.policy.model.rtc_processor is None \
                or not self.policy.model._rtc_enabled():
            raise SystemExit("refusing to start: rtc_config was set but the sampler did not "
                             "pick it up (policy._rtc_enabled="
                             f"{self.policy._rtc_enabled()}, model.rtc_processor="
                             f"{self.policy.model.rtc_processor}). lerobot's RTC wiring changed; "
                             "run without --rtc until this is re-checked.")
        self.rtc = cfg
        self.rtc_execution_horizon = horizon
        self.rtc_inference_delay = delay
        self.rtc_reset("rtc configured")
        w = self.policy.rtc_processor.get_prefix_weights(delay, horizon, self.chunk_size)
        print(f"[rtc] ON: schedule={sched.value} execution_horizon={horizon} "
              f"inference_delay={delay} max_guidance_weight={float(max_guidance_weight):g} "
              f"stale_reset={self.rtc_stale_reset_s:g}s")
        print(f"[rtc] prefix weights over the {self.chunk_size}-step chunk = "
              f"{[round(float(v), 3) for v in w.tolist()]}")
        print(f"[rtc] each /predict is guided by the previous chunk's unexecuted tail "
              f"(steps {self.emit_steps}..{self.chunk_size} -> {overlap} steps), aligned so that "
              f"the tail's step 0 IS this chunk's step 0. That alignment assumes the client "
              f"executes exactly {self.emit_steps} rows per chunk (--exec-steps).")

    # MEASURED 2026-10-17 on ext_PC_L_c5@006000, d9_0912, 24 in-sample pairs x 2 frame offsets
    # (results/rtc_coda/, EXPERIMENTS.md). seam, paired against the same RTC-off observations:
    #   RTC off 20.93 / 21.42 -> main 13.55 / 13.11, correction 12.27 / 12.67, both 12.27 / 12.66
    # "correction" is the best point estimate in BOTH replicates, is flat in the guidance weight
    # (12.1-12.7 over w = 10 / 5 / 2.5) and costs 35 ms less than "both", which measures the same.
    # The gap to "main" (0.4-1.3 ticks) is INSIDE the paired SE (2.1-3.4) and inside the Coda
    # displacement D (3.3-3.6 ticks), so this is a preference, not a significant win.
    CODA_RTC_AUTO = "correction"

    def _resolve_coda_rtc(self, mode: str, weight: float, rtc_on: bool):
        """--coda-rtc auto -> the measured default; and pick the guidance weight automatically.

        Returns (mode, max_guidance_weight). Called once, before configure_rtc, so the server
        starts with the right weight instead of being reconfigured.

        The weight rule: the pull applied to the pinned head of the window is
        max_guidance_weight / num_steps per Euler step; above 1 the first step overshoots the old
        plan (x -> 2*prev - x) and pushes x_t off the narrow band the correction AE was trained
        on. The 5-step Coda checkpoints run pull 2.0 at the server default of 10, and the sweep
        shows no seam cost in clamping it to 1.0, so a Coda checkpoint gets
        max_guidance_weight = (guided Euler steps) unless --rtc-max-guidance-weight was given.
        Checkpoints without Coda are left exactly as they were.
        """
        from smolvla_ext.config import CODA_RTC_MODES

        mode = str(mode)
        if mode not in CODA_RTC_MODES + ("auto",):
            raise SystemExit(f"--coda-rtc must be auto|{'|'.join(CODA_RTC_MODES)}, got {mode!r}")
        ext = getattr(self.policy, "_smolvla_ext", None)
        has_coda = ext is not None and bool(ext.coda)
        if mode == "auto":
            mode = self.CODA_RTC_AUTO if (has_coda and rtc_on) else "off"
            if has_coda and rtc_on:
                print(f"[coda-rtc] auto -> {mode} (measured default, 2026-10-17; "
                      f"results/rtc_coda). Pass --coda-rtc main|both|off to override.")
        if not (has_coda and rtc_on and mode != "off") or self._rtc_weight_explicit:
            return mode, weight
        guided_steps = (int(self.policy.config.num_steps) if mode == "main"
                        else int(ext.coda_num_steps))
        if float(weight) / guided_steps > 1.0:
            auto_w = float(guided_steps)
            print(f"[coda-rtc] max_guidance_weight {float(weight):g} -> {auto_w:g} automatically: "
                  f"the guided stage runs {guided_steps} Euler steps, so {float(weight):g} would "
                  f"pull {float(weight) / guided_steps:.2f} per step and overshoot the old plan. "
                  f"Pass --rtc-max-guidance-weight to keep your own value.")
            return mode, auto_w
        return mode, weight

    def configure_coda_rtc(self, mode: str = "off") -> None:
        """WHERE the RTC guidance acts on a Coda checkpoint (ExtConfig.coda_rtc, inference only).

        A method, like configure_rtc, so --self-test can flip it on one loaded checkpoint. The
        value saved in the checkpoint's sidecar is never trusted: smolvla_ext.load_policy forces
        it to "off" and this call sets whatever --coda-rtc asked for.

          off         today's behaviour -- Coda + RTC refuses at predict time
          main        guidance on the main AE only; the correction AE then moves the overlap freely
          correction  guidance on the correction AE, i.e. on the chunk the robot executes
          both        both stages guided

        Guidance strength is per Euler step: pull = max_guidance_weight / num_steps on the pinned
        head of the window (modeling_rtc.py:219-226 linearised). pull > 1 overshoots the old plan
        (x -> 2*prev - x), which is why it is printed here rather than buried in a script.
        """
        from smolvla_ext.config import CODA_RTC_MODES

        mode = str(mode)
        if mode not in CODA_RTC_MODES + ("auto",):
            raise SystemExit(f"--coda-rtc must be auto|{'|'.join(CODA_RTC_MODES)}, got {mode!r}")
        ext = getattr(self.policy, "_smolvla_ext", None)
        has_coda = ext is not None and bool(ext.coda)
        if mode == "auto":
            mode = self.CODA_RTC_AUTO if (has_coda and self.rtc is not None) else "off"
        if mode != "off" and not has_coda:
            raise SystemExit(f"--coda-rtc {mode} but this checkpoint has no Coda correction "
                             f"expert (smolvla_ext.json coda=false). Drop the flag.")
        if has_coda:
            ext.coda_rtc = mode
            if getattr(self.policy.model, "_ext", None) is not ext \
                    or self.policy.model._ext.coda_rtc != mode:
                raise SystemExit("refusing to start: coda_rtc did not reach the sampler "
                                 "(policy._smolvla_ext and model._ext are not the same object)")
        self.coda_rtc = mode
        if has_coda and self.rtc is not None and mode == "off":
            raise SystemExit("refusing to start: this is a Coda checkpoint and --rtc is on, but "
                             "--coda-rtc is off, so every /predict would raise. Pick "
                             "main|correction|both, or run without --rtc.")
        if not has_coda:
            return
        main_steps = int(self.policy.config.num_steps)
        corr_steps = int(ext.coda_num_steps)
        if mode == "off":
            print(f"[coda-rtc] off: the correction AE runs unguided ({main_steps}+{corr_steps} "
                  f"Euler steps) and --rtc is off.")
            return
        w = float(self.rtc.max_guidance_weight) if self.rtc is not None else float("nan")
        guided = {"main": [("main AE", main_steps)], "correction": [("correction AE", corr_steps)],
                  "both": [("main AE", main_steps), ("correction AE", corr_steps)]}[mode]
        print(f"[coda-rtc] {mode}: guidance on " + " + ".join(n for n, _ in guided)
              + f" (main AE {main_steps} steps, correction AE {corr_steps} steps)")
        for name, n in guided:
            pull = w / n
            note = "  <-- > 1: the first pinned step overshoots the old plan" if pull > 1 else ""
            print(f"[coda-rtc]   {name}: |dt|={1.0 / n:.3f} max_guidance_weight={w:g} "
                  f"pull/step={pull:.2f}{note}")
        print("[coda-rtc] the pull acts on the pinned head of the window only "
              f"(--rtc-inference-delay {self.rtc_inference_delay} steps at weight 1.0); "
              "measurements: results/rtc_coda/, EXPERIMENTS.md")

    def rtc_reset(self, reason: str) -> None:
        """Drop the previous chunk. The next request then runs unguided, like a first request."""
        had = self._prev_plan_state is not None
        self._prev_plan_state = None
        self._rtc_reset_reason = reason
        if had and self.rtc is not None:
            print(f"[rtc] previous-chunk state dropped ({reason}); the next /predict runs "
                  f"unguided.")

    def _prev_plan_take(self, task: str):
        """(previous plan | None, reason it was dropped | None) for an incoming request.

        Kept OUTSIDE the `self.rtc is not None` test on purpose: the seam measurement uses the
        same memory and must be reported whether or not RTC is on, otherwise the number that
        justifies RTC would only exist when RTC is already running.
        """
        st = self._prev_plan_state
        if st is None:
            return None, self._rtc_reset_reason
        if st["task"] != task:
            self.rtc_reset("task_changed")
            return None, "task_changed"
        age = time.monotonic() - st["t"]
        if age > self.rtc_stale_reset_s:
            self.rtc_reset(f"stale ({age:.2f}s > {self.rtc_stale_reset_s:g}s)")
            return None, "stale"
        return st, None

    def rtc_status(self) -> dict:
        """RTC state as exposed by /health (a superset of what /predict carries)."""
        st = self._prev_plan_state
        sched = getattr(self.rtc, "prefix_attention_schedule", None) if self.rtc else None
        return {
            "rtc": self.rtc is not None,
            "rtc_execution_horizon": self.rtc_execution_horizon if self.rtc else None,
            "rtc_inference_delay": self.rtc_inference_delay if self.rtc else None,
            "rtc_schedule": getattr(sched, "value", None if sched is None else str(sched)),
            "rtc_max_guidance_weight": (float(self.rtc.max_guidance_weight)
                                        if self.rtc else None),
            "rtc_stale_reset_s": self.rtc_stale_reset_s,
            "coda_rtc": self.coda_rtc,
            "rtc_prefix_from_previous_chunk_steps": (int(self.chunk_size) - int(self.emit_steps)
                                                     if self.rtc else None),
            "rtc_has_prev_chunk": st is not None,
            "rtc_prev_chunk_task": (st["task"] if st is not None else None),
            "rtc_prev_chunk_age_s": (round(time.monotonic() - st["t"], 3)
                                     if st is not None else None),
        }

    @staticmethod
    def _decode_frames(images_b64: list[str]) -> list[np.ndarray]:
        from PIL import Image

        frames = []
        for b64 in images_b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            frames.append(np.asarray(img, dtype=np.uint8))  # (H, W, 3) RGB uint8
        return frames

    @staticmethod
    def _apply_camera_corrections(frames: list[np.ndarray]) -> list[np.ndarray]:
        """SWAP_CAM01 / FLIP_CAM_H / BLACK_IMAGE, preserved from dex_ref/server_infer.py.

        Physical-camera corrections (enumeration order, mirroring, diagnostic blackout);
        backbone-independent, so they carry over unchanged.
        """
        from PIL import Image

        frames = list(frames)
        if os.environ.get("SWAP_CAM01") == "1" and len(frames) >= 2:
            frames[0], frames[1] = frames[1], frames[0]
        if os.environ.get("FLIP_CAM_H") == "1":
            frames = [np.asarray(Image.fromarray(f).transpose(Image.FLIP_LEFT_RIGHT),
                                 dtype=np.uint8) for f in frames]
        if os.environ.get("BLACK_IMAGE") == "1":
            frames = [np.zeros_like(f) for f in frames]
        return frames

    def _build_observation(self, frames: list[np.ndarray], state_ticks: np.ndarray,
                           task: str) -> dict:
        """Raw observation in lerobot's deployment format (policies/utils.py:97 consumes it).

        state : (5,) float32 RAW TICKS -- no unit conversion, see the module docstring.
        images: (H,W,3) uint8; prepare_observation_for_inference does /255 + HWC->CHW, and
                SmolVLA's prepare_images does the resize-with-pad and the [0,1]->[-1,1] shift.
        """
        from lerobot.policies.utils import prepare_observation_for_inference

        obs: dict = {"observation.state": np.asarray(state_ticks, dtype=np.float32).reshape(-1)}
        for i, key in enumerate(self.image_keys):
            slot = self.cam_indices[i]
            if slot >= len(frames):
                raise IndexError(f"cam-index {slot} for '{key}' but only {len(frames)} images sent")
            f = frames[slot]
            obs[key] = f if f.dtype == np.uint8 else f.astype(np.uint8)
        return prepare_observation_for_inference(
            obs, self.torch.device(self.device), task, "openmani4_follower")

    # ---- sampling ------------------------------------------------------------
    def _draw_chunk_norm(self, batch: dict, n_avg: int, noise=None, rtc_kwargs: dict | None = None):
        """(n_avg, chunk_size, action_dim) normalized draws from ONE forward pass.

        n_avg == 1, noise is None and no rtc_kwargs is the ORIGINAL call, untouched: `predict_action_chunk`
        samples its own noise internally (modeling_smolvla.py:740-742), so the RNG is consumed
        exactly as before and the result is bit-identical to the pre-averaging server.

        For n_avg > 1 the expansion is the one validated in
        `diag/sampling_noise.py::batched_draw_chunks`: expand every batch-dim-1 tensor to n rows
        (a stride-0 view, so the images are not copied) and hand `predict_action_chunk` a noise
        tensor with n independent rows. That helper is not imported directly because it builds
        its own observation from raw frames and returns ticks only -- this server needs the
        NORMALIZED chunk (for the probe) and must keep its own camera-correction / cam-index /
        probe-ordering path. The noise comes from the policy's OWN `sample_noise` (the same
        function the n_avg == 1 path calls) rather than `diag.smolvla_introspect.make_noise`,
        whose fixed CPU seed would freeze the noise realization at every request of a rollout
        instead of letting the draw-to-draw error average out over time; `noise` is still
        accepted so --self-test can pin it.
        """
        torch = self.torch
        if n_avg == 1 and noise is None and not rtc_kwargs:
            return self.policy.predict_action_chunk(batch)      # unchanged single-draw path
        big = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.shape[0] == 1:
                big[k] = v.expand(n_avg, *v.shape[1:])
            elif isinstance(v, list) and len(v) == 1:
                big[k] = v * n_avg
            else:
                big[k] = v
        if noise is None:
            noise = self.policy.model.sample_noise(
                (n_avg, self.chunk_size, self.max_action_dim), self.device)
        if int(noise.shape[0]) != n_avg:
            raise ValueError(f"noise has {int(noise.shape[0])} rows but n_avg={n_avg}")
        # rtc_kwargs = {prev_chunk_left_over, inference_delay, execution_horizon}. They ride the
        # ActionSelectKwargs channel predict_action_chunk -> _get_action_chunk -> sample_actions
        # -> euler_integrate (modeling_smolvla.py:768-771); an empty dict is the un-guided call.
        # Note the noise above is sampled from the policy's OWN sample_noise even at n_avg == 1
        # when RTC is on, i.e. from the same RNG in the same order as the internal draw, so an
        # RTC-on and an RTC-off request under the same seed differ ONLY by the guidance.
        out = self.policy.predict_action_chunk(big, noise=noise, **(rtc_kwargs or {}))
        if int(out.shape[0]) != n_avg:
            raise RuntimeError(f"batched draw returned {int(out.shape[0])} rows, expected "
                               f"{n_avg}: the batch expansion did not reach the sampler")
        return out

    # ---- core predict -------------------------------------------------------
    def predict_from_frames(self, frames: list[np.ndarray], qpos, task: str | None = None,
                            n_avg: int | None = None, noise=None) -> dict:
        torch = self.torch
        n_avg = int(self.n_avg if n_avg is None else n_avg)
        if n_avg < 1:
            raise ValueError(f"n_avg must be >= 1, got {n_avg}")
        ticks_in = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if ticks_in.size != JOINT_DIM:
            raise ValueError(f"expected {JOINT_DIM} DXL ticks, got {ticks_in.size}")
        if not np.isfinite(ticks_in).all():
            raise ValueError(f"qpos contains NaN/Inf: {ticks_in.tolist()}")
        instruction = task if task is not None else (self.lang or "")

        # --- previous plan: the RTC prefix and the seam reference. Taken BEFORE anything is
        #     overwritten, and self-dropping on /init, an instruction change or staleness.
        prev_plan, prev_dropped = self._prev_plan_take(instruction)
        rtc_kwargs = None
        rtc_prev_steps = 0
        if self.rtc is not None and prev_plan is not None:
            # The tail the client has NOT executed: previous chunk steps [emit .. chunk_size),
            # in the sampler's normalized action space (see the module docstring). Sliced with
            # the PREVIOUS request's emit count, which is how many rows that client was given.
            left = prev_plan["norm"][:, int(prev_plan["emit"]):, :]
            if int(left.shape[1]) > 0:
                # One prev, N draws: every draw is anchored to the SAME plan, the one the arm is
                # executing. expand() is a stride-0 view, so this copies nothing.
                if n_avg > 1:
                    left = left.expand(n_avg, -1, -1)
                rtc_prev_steps = int(left.shape[1])
                rtc_kwargs = {
                    "prev_chunk_left_over": left,
                    "inference_delay": self.rtc_inference_delay,
                    "execution_horizon": self.rtc_execution_horizon,
                }

        t0 = time.perf_counter()
        frames = self._apply_camera_corrections(frames)
        obs = self._build_observation(frames, ticks_in.astype(np.float32), instruction)

        # No cross-request state: SmolVLA keeps an n_obs_steps queue and an action queue.
        # Each request carries the arm's real present position, so every request must be a
        # fresh episode step as far as the policy is concerned.
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

        record = self.probe is not None and (self._req_count % self.diag_every == 0)
        with torch.no_grad():
            batch = self.preprocessor(obs)
            t1 = time.perf_counter()
            if record:
                # begin_step also re-reads the REAL state width off the live tensor and
                # asserts pad_vector's tail (slots 5..32) is exactly zero.
                self.probe.begin_step(self._req_count, batch)
            # (n_avg, chunk_size, 5) normalized -- n_avg == 1 is the original single call
            chunk_norm_draws = self._draw_chunk_norm(batch, n_avg, noise=noise,
                                                     rtc_kwargs=rtc_kwargs)
            if self.rtc is not None:
                # RTCProcessor.denoise_step runs an autograd pass inside torch.enable_grad()
                # (modeling_rtc.py:211-217), so the chunk can come back carrying a graph even
                # though this block is under no_grad. Detach now: the normalized chunk is STORED
                # for the next request, and storing a live graph would pin one denoising pass'
                # activations in GPU memory for the whole rollout.
                chunk_norm_draws = chunk_norm_draws.detach()
            t2 = time.perf_counter()
            chunk = self.postprocessor(chunk_norm_draws)               # denormalized -> ticks, cpu

        draws_np = chunk.detach().float().cpu().numpy()
        if draws_np.ndim == 2:
            draws_np = draws_np[None]
        if draws_np.shape[0] != n_avg:
            raise RuntimeError(f"expected {n_avg} draws after the postprocessor, got "
                               f"{draws_np.shape[0]} (shape {tuple(draws_np.shape)})")
        if draws_np.shape[-1] != self.action_dim:
            raise RuntimeError(f"policy returned {draws_np.shape[-1]} channels, expected "
                               f"{self.action_dim}")
        # Peel the termination channel off HERE, so every line below this point still works on
        # a 5-D motor chunk and needed no change. done_flag is unnormalized by the same
        # MEAN_STD step as the motors, so it comes back on the training scale where 0 = not
        # done and 1 = done; it is a flow-matching DRAW, not a probability, which is why the
        # draws are averaged before thresholding (n_avg draws -> an estimate of the mean).
        done_draws = None
        if self.has_done:
            done_draws = draws_np[..., JOINT_DIM].astype(np.float64)   # (n_avg, chunk_size)
            draws_np = draws_np[..., :JOINT_DIM]
        # --- the average, in TICK space (see the module docstring) ---------------------------
        if n_avg == 1:
            chunk_np = draws_np[0]                       # untouched: no averaging arithmetic
            spread_ticks, spread_per_motor = None, None
            chunk_norm = chunk_norm_draws
        else:
            d64 = draws_np.astype(np.float64)            # float64 like diag/sampling_noise.py
            chunk_np = d64.mean(axis=0)
            spread_ticks, spread_per_motor = pairwise_spread_ticks(d64)
            # The normalized twin of the served chunk, for the probe. Exact because every
            # unnormalization mode is affine; --self-test asserts the two agree in ticks.
            chunk_norm = chunk_norm_draws.mean(dim=0, keepdim=True)
        if chunk_np.shape[-1] != JOINT_DIM:
            raise RuntimeError(f"policy returned {chunk_np.shape[-1]} channels, expected {JOINT_DIM}")
        full_steps = int(chunk_np.shape[0])
        emit = min(self.emit_steps, full_steps)
        raw = chunk_np[:emit]

        # round -> per-motor clamp. Clamping is reported, never silent: a clamped value means
        # the policy asked for a position outside the motor's allowed range.
        rounded = np.rint(raw).astype(np.int64)
        clamped = np.clip(rounded, LIMIT_LO, LIMIT_HI)
        n_clamped = int((clamped != rounded).sum())
        clamped_per_motor = (clamped != rounded).sum(axis=0).astype(int).tolist()
        ticks_chunk = clamped.astype(np.int32)

        # --- motion measurement (NOT a limit; see the module docstring) -------------------
        if emit >= 2:
            step_delta = np.abs(np.diff(ticks_chunk.astype(np.int64), axis=0))
            max_step = int(step_delta.max())
            max_step_per_motor = step_delta.max(axis=0).astype(int).tolist()
            n_over = int((step_delta.max(axis=1) > self.rate_limit_ticks).sum())
        else:
            max_step, max_step_per_motor, n_over = 0, [0] * JOINT_DIM, 0
        first_jump_per_motor = np.abs(ticks_chunk[0].astype(np.int64)
                                      - np.rint(ticks_in).astype(np.int64))
        first_jump = int(first_jump_per_motor.max())

        # --- re-plan seam: |plan(t)[emit-1] - plan(t+emit)[0]|, the two consecutive commands
        #     that straddle the boundary between the previous chunk and this one. SAME definition
        #     as diag/input_ablation.py:693-701 (mean of the per-motor absolute difference), so
        #     the deployment number and the offline number are the same number. Reported whether
        #     or not RTC is on -- it is the metric RTC is supposed to move.
        if prev_plan is not None and prev_plan.get("last_ticks") is not None:
            seam_per_motor = np.abs(ticks_chunk[0].astype(np.int64) - prev_plan["last_ticks"])
            seam_mean = float(seam_per_motor.mean())
            seam_max = int(seam_per_motor.max())
            seam_list = seam_per_motor.astype(int).tolist()
        else:
            seam_mean, seam_max, seam_list = None, None, None

        # --- become the previous plan for the NEXT request -----------------------------------
        self._prev_plan_state = {
            "norm": chunk_norm.detach(),            # (1, chunk_size, 5), the SERVED chunk
            "emit": int(emit),                      # rows handed out == rows the client executes
            "task": instruction,
            "t": time.monotonic(),
            "last_ticks": ticks_chunk[-1].astype(np.int64).copy(),   # == row emit-1
        }
        self._rtc_reset_reason = None

        # --- termination -------------------------------------------------------------------
        # Scored over the rows the client is about to EXECUTE, not the whole predicted chunk:
        # the later rows of the chunk are further into the future and get replaced by the next
        # request anyway. A streak is required because one chunk above the threshold is one
        # sample of a noisy draw; the offline probe needed hysteresis for the same reason.
        done_out = None
        done_score_list = None
        done_mean = None
        if done_draws is not None:
            done_emit = done_draws.mean(axis=0)[:emit]        # average the draws, then cut
            done_mean = float(done_emit.mean())
            done_score_list = [float(x) for x in done_emit]
            if done_mean >= self.done_threshold:
                self._done_streak += 1
            else:
                self._done_streak = 0
            done_out = bool(self._done_streak >= self.done_consecutive)

        oob = [i for i in range(JOINT_DIM)
               if ticks_in[i] < self.state_lo[i] or ticks_in[i] > self.state_hi[i]]
        t3 = time.perf_counter()

        out = {
            # --- client contract (identical keys/types to gr00t_server_openmani.py) ---
            "joint_ticks_b64": base64.b64encode(ticks_chunk.tobytes()).decode("ascii"),
            "chunk_size": int(ticks_chunk.shape[0]),
            "joint_dim": JOINT_DIM,
            "reasoning": None,   # SmolVLA has no chain-of-thought output
            "done": done_out,           # None on a 5-D checkpoint (no termination channel)
            "done_score": done_score_list,
            "done_mean": done_mean,
            "done_threshold": self.done_threshold if self.has_done else None,
            "done_streak": self._done_streak if self.has_done else None,
            "done_consecutive": self.done_consecutive if self.has_done else None,
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
            # --- extra meta (additive; old clients ignore unknown keys) ---
            "execute": not self.dry_run,
            "dry_run": self.dry_run,
            "n_action_steps": self.n_action_steps,
            "policy_chunk_size": full_steps,
            # --- sample averaging (additive; n_avg=1 keeps the pre-averaging behaviour) ---
            "n_avg": n_avg,
            "spread_ticks": spread_ticks,
            "spread_ticks_per_motor": spread_per_motor,
            "max_step_ticks_in_chunk": max_step,
            "max_step_ticks_per_motor": max_step_per_motor,
            "first_step_jump_ticks": first_jump,
            "first_step_jump_per_motor": first_jump_per_motor.astype(int).tolist(),
            "rate_limit_reference_ticks": self.rate_limit_ticks,
            "n_steps_over_rate_limit": n_over,
            "n_clamped_values": n_clamped,
            "n_clamped_per_motor": clamped_per_motor,
            "state_out_of_training_range": [self.motor_order[i] for i in oob],
            # --- real-time chunking (additive; rtc=false is the pre-RTC behaviour) ---
            "rtc": self.rtc is not None,
            "rtc_used_prev_chunk": rtc_kwargs is not None,
            "rtc_prev_chunk_steps": rtc_prev_steps,
            "rtc_execution_horizon": self.rtc_execution_horizon if self.rtc else None,
            "rtc_inference_delay": self.rtc_inference_delay if self.rtc else None,
            "rtc_prev_dropped_reason": prev_dropped,
            # --- re-plan seam vs the previous request (None on the first request) ---
            "seam_ticks_mean": seam_mean,
            "seam_ticks_max": seam_max,
            "seam_ticks_per_motor": seam_list,
            "_ticks_chunk": ticks_chunk,      # debug only, stripped before the wire (DEBUG_KEYS)
            "_chunk_norm": chunk_norm,        # the SERVED chunk, normalized (mean over draws)
            "_chunk_norm_draws": chunk_norm_draws,
            "_ticks_draws": draws_np,
            "_batch": batch,
        }

        if record:
            self.probe.end_step(extra={
                "request": self._req_count,
                "qpos_ticks": ticks_in.astype(int).tolist(),
                "instruction": instruction,
                "max_step_ticks_in_chunk": max_step,
                "first_step_jump_ticks": first_jump,
                "n_clamped_values": n_clamped,
                "inference_ms": round(out["inference_ms"], 2),
                # The forward pass carried n_avg rows of the SAME observation, so every
                # per-module tensor statistic in this record (rms, dead_frac, ...) is pooled
                # over that many draws. Stated rather than left to be misread as one draw.
                "n_avg": n_avg,
                "module_stats_pooled_over_n_draws": n_avg,
                "spread_ticks": spread_ticks,
            })
            # Score the chunk we ACTUALLY answered with -- Probe.prediction_check() would
            # re-run the denoiser and score a DIFFERENT sample (flow matching is stochastic).
            # Note the "pred" record covers the full predicted chunk (chunk_size=16) in
            # normalized space, which the probe denormalizes to ticks; the response carries
            # only the first `emit_steps` of it.
            self.probe.record_prediction(chunk_norm, batch, step=self._req_count)
        if self.attn is not None:
            # frames arrive in CLIENT SLOT order; the recorder wants them in image_keys order.
            self.attn.maybe_record([frames[s] for s in self.cam_indices],
                                   ticks_in, instruction, self._req_count)
        self._req_count += 1

        # budget = how long the client will spend executing the rows it is about to receive.
        # slack = that budget minus the round trip it just paid. Negative slack means the arm ran
        # out of plan before the next one arrived, i.e. --emit-steps is too small for this
        # latency, and it is the number to look at before lowering it further.
        rtt = self._pending_client_timing.get("prev_rtt_ms")
        step_dt = self._pending_client_timing.get("step_dt_ms")
        budget = slack = None
        if isinstance(rtt, (int, float)) and rtt > 0:
            self._rtt_samples.append(float(rtt))
            if isinstance(step_dt, (int, float)) and step_dt > 0:
                budget = float(emit) * float(step_dt)
                slack = budget - float(rtt)
                self._slack_samples.append(slack)
                if slack < 0:
                    self._late_requests += 1
        out["client_prev_rtt_ms"] = rtt
        out["exec_budget_ms"] = budget
        out["slack_ms"] = slack
        out["transport_ms"] = (None if not isinstance(rtt, (int, float)) else
                               round(float(rtt) - out["inference_ms"] - out["preprocess_ms"]
                                     - out["postprocess_ms"], 2))
        self._pending_client_timing = {}

        flags = []
        if slack is not None and slack < 0:
            flags.append(f"LATE: round trip {rtt:.0f}ms > {budget:.0f}ms of plan "
                         f"({emit} steps x {step_dt:.0f}ms)")
        if n_clamped:
            flags.append(f"CLAMPED {n_clamped} value(s) {clamped_per_motor}")
        if n_over:
            flags.append(f"{n_over}/{max(emit - 1, 0)} steps exceed the robot's "
                         f"{self.rate_limit_ticks:g}-tick rate limit")
        if first_jump > self.rate_limit_ticks:
            flags.append(f"first step jumps {first_jump} ticks from present")
        if oob:
            flags.append(f"present state outside training range on "
                         f"{[self.motor_order[i] for i in oob]}")
        if self.dry_run:
            flags.append("DRY-RUN: response says execute=false")
        avg = "" if n_avg == 1 else f" n_avg={n_avg} spread={spread_ticks:.1f}t"
        if self.rtc is None:
            rtc_s = ""
        elif rtc_kwargs is not None:
            rtc_s = f" rtc=on(prev {rtc_prev_steps}st)"
        else:
            rtc_s = f" rtc=on(no prev: {prev_dropped})"
        seam_s = "" if seam_mean is None else f" seam={seam_mean:.1f}t"
        print(f"[predict #{self._req_count - 1}] qpos={ticks_in.astype(int).tolist()} -> "
              f"chunk{tuple(ticks_chunk.shape)} step0={ticks_chunk[0].tolist()} "
              f"max|dtick|={max_step} jump0={first_jump} "
              f"infer={out['inference_ms']:.0f}ms"
              + ("" if slack is None else f" rtt={rtt:.0f}ms slack={slack:+.0f}ms")
              + f"{avg}{rtc_s}{seam_s}"
              + (" | " + "; ".join(flags) if flags else ""))
        return out

    # ======================================================================
    # Language snapping: map a free-form sentence onto one of the 54 the policy was trained on
    # ======================================================================

    # ---- rule-based slot filling (default) --------------------------------------------------
    # The instruction space is a closed grammar -- 2 verbs x 9 objects x 6 directions -- so a
    # dictionary lookup beats any similarity search and, unlike one, can say WHICH slot it
    # failed to fill. Measured: embedding retrieval with this policy's own text model got 58%
    # of paraphrases right (diag/lang_snap_eval.py), mostly confusing place with push.
    _VERB_WORDS = {
        "place": ["place", "put", "set", "lay", "drop", "position"],
        "push": ["push", "shove", "slide", "nudge", "scoot", "roll"],
    }
    # Longer phrases first: "in front of me" must win over "front", "to the left" over "left".
    _DIR_WORDS = [
        ("in front of me", ["in front of me", "right in front of me", "in front", "over here",
                            "closer to me", "close to me", "near me"]),
        ("toward me", ["toward me", "towards me", "back this way", "my direction", "this way",
                       "back to me"]),
        ("on the left", ["on the left", "at the left", "on my left", "left side", "leftmost"]),
        ("on the right", ["on the right", "at the right", "on my right", "right side",
                          "rightmost"]),
        ("to the left", ["to the left", "towards the left", "toward the left", "leftwards",
                         "off to the left", "left"]),
        ("to the right", ["to the right", "towards the right", "toward the right", "rightwards",
                          "off to the right", "right"]),
    ]

    def _build_rule_tables(self) -> None:
        """Objects come from the training sentences themselves, so they cannot drift."""
        import re as _re
        objs, pairs = set(), set()
        for t in self.tasks:
            m = _re.match(r"(\w+) the (.+?) (to the \w+|toward me|on the \w+|in front of me)$", t)
            if m:
                objs.add(m.group(2))
                pairs.add((m.group(1), m.group(3)))
        self._rule_objects = sorted(objs, key=lambda o: -len(o))   # longest match first
        self._rule_pairs = pairs
        # which directions each verb actually appears with, so "place ... to the left" (never
        # recorded) is repaired to "place ... on the left" instead of being refused
        self._verb_dirs = {}
        for v, d in pairs:
            self._verb_dirs.setdefault(v, []).append(d)
        self._dir_alias = {}
        for canon, words in self._DIR_WORDS:
            self._dir_alias[canon] = canon
        for a, b in (("on the left", "to the left"), ("to the left", "on the left"),
                     ("on the right", "to the right"), ("to the right", "on the right"),
                     ("in front of me", "toward me"), ("toward me", "in front of me")):
            self._dir_alias.setdefault(a, a)

    def snap_lang_rules(self, sentence: str) -> dict:
        """Fill verb / object / direction from keywords; report exactly which slot failed."""
        if not hasattr(self, "_rule_objects"):
            self._build_rule_tables()
        low = " " + sentence.lower().strip().rstrip(".!?") + " "
        obj = next((o for o in self._rule_objects if f" {o} " in low or f" {o}" in low), None)
        verb = next((v for v, ws in self._VERB_WORDS.items()
                     if any(f" {w} " in low or f" {w}" in low for w in ws)), None)
        direction = next((canon for canon, ws in self._DIR_WORDS
                          if any(w in low for w in ws)), None)
        missing = [n for n, v in (("object", obj), ("verb", verb), ("direction", direction))
                   if v is None]
        if missing:
            return {"matched": None, "accepted": False, "exact": False, "missing": missing,
                    "slots": {"verb": verb, "object": obj, "direction": direction},
                    "detail": f"could not find the {' and '.join(missing)} in {sentence!r}"}
        repaired = None
        if (verb, direction) not in self._rule_pairs:
            # e.g. "push the pen in front of me": the verb is real, the direction is real, but
            # that combination was never recorded. Snap to the same-side direction this verb
            # does use rather than inventing one.
            side = ("left" if "left" in direction else
                    "right" if "right" in direction else "me")
            cand = [d for d in self._verb_dirs.get(verb, [])
                    if (side in d) or (side == "me" and ("me" in d))]
            if not cand:
                return {"matched": None, "accepted": False, "exact": False,
                        "missing": ["direction"],
                        "slots": {"verb": verb, "object": obj, "direction": direction},
                        "detail": f"{verb!r} was never recorded with a direction like "
                                  f"{direction!r}"}
            repaired = (direction, cand[0])
            direction = cand[0]
        matched = f"{verb} the {obj} {direction}"
        ok = matched in self.tasks
        return {"matched": matched if ok else None, "accepted": ok, "exact": matched == sentence,
                "missing": [] if ok else ["combination"],
                "slots": {"verb": verb, "object": obj, "direction": direction},
                "repaired_direction": repaired,
                "detail": None if ok else f"{matched!r} is not one of the training sentences"}

    def _embed_sentences(self, sentences: list[str]):
        """One vector per sentence, from THIS policy's own text model. No extra model is loaded.

        Every sentence is embedded through the full prefix path with the SAME constant grey
        images, so the only thing that differs between two vectors is the language. Using the
        policy's own encoder is the point: the retrieval then lives in the space the action
        expert actually reads, rather than in some third model's opinion of sentence similarity.
        """
        import numpy as np
        import torch as _t
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import (OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS)

        grey = [np.full((240, 320, 3), 128, dtype=np.uint8) for _ in self.image_keys]
        state = np.zeros(JOINT_DIM, dtype=np.float32)
        out = []
        for sent in sentences:
            obs = self._build_observation(grey, state, sent)
            self.policy.reset(); self.preprocessor.reset(); self.postprocessor.reset()
            with _t.no_grad():
                batch = self.preprocessor(obs)
                images, img_masks = self.policy.prepare_images(batch)
                st = self.policy.prepare_state(batch)
                lang = batch[OBS_LANGUAGE_TOKENS]
                lmask = batch[OBS_LANGUAGE_ATTENTION_MASK]
                pe, ppad, patt = self.policy.model.embed_prefix(
                    images, img_masks, lang, lmask, state=st)
                att2d = make_att_2d_masks(ppad, patt)
                pos = _t.cumsum(ppad, dim=1) - 1
                hs, _ = self.policy.model.vlm_with_expert.forward(
                    attention_mask=att2d, position_ids=pos, past_key_values=None,
                    inputs_embeds=[pe, None], use_cache=True)
                h = hs[0].float()
                n_lang = int(lang.shape[1])
                lang_start = h.shape[1] - 1 - n_lang        # [images | language | state]
                seg = h[:, lang_start:lang_start + n_lang]
                w = lmask.float().unsqueeze(-1)
                if self.lang_snap_pool == "last":
                    # Causal attention means only the LAST real token has seen the whole
                    # sentence; every earlier position is a prefix summary. Mean pooling over a
                    # causal decoder therefore dilutes the sentence with its own prefixes.
                    idx = lmask.long().sum(1).clamp(min=1) - 1
                    v = seg[_t.arange(seg.shape[0], device=seg.device), idx]
                else:
                    v = (seg * w).sum(1) / w.sum(1).clamp(min=1)
            out.append(v[0].cpu().numpy())
        return np.stack(out)

    def build_lang_index(self) -> None:
        if not self.tasks:
            print("[snap] no training sentences available; --lang-snap disabled")
            self.lang_snap = False
            return
        t0 = time.perf_counter()
        self._lang_vecs = self._embed_sentences(self.tasks)
        print(f"[snap] indexed {len(self.tasks)} training sentences in "
              f"{(time.perf_counter() - t0) * 1000:.0f} ms "
              f"(dim {self._lang_vecs.shape[1]}, this policy's own text model)")

    def snap_lang(self, sentence: str) -> dict:
        if self.lang_snap_mode == "rules":
            r = self.snap_lang_rules(sentence)
            r.setdefault("sim", None); r.setdefault("margin", None); r.setdefault("top", [])
            return r
        return self.snap_lang_embed(sentence)

    def snap_lang_embed(self, sentence: str) -> dict:
        """-> {'matched', 'sim', 'margin', 'top', 'accepted'}.

        `margin` is best minus runner-up. A high similarity with a low margin means the sentence
        sits between two instructions -- "push the pen" with no direction, say -- and picking the
        winner would be a coin flip, so it is refused rather than silently resolved.
        """
        if sentence in self.tasks:
            return {"matched": sentence, "sim": 1.0, "margin": 1.0, "accepted": True,
                    "top": [(sentence, 1.0)], "exact": True}
        if self._lang_vecs is None:
            self.build_lang_index()
        import numpy as np
        v = self._embed_sentences([sentence])[0]
        sims = np.array([_cos(v, u) for u in self._lang_vecs])
        order = np.argsort(-sims)
        best, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        margin = float(sims[best] - sims[second])
        ok = bool(sims[best] >= self.lang_snap_min_sim and margin >= self.lang_snap_min_margin)
        return {"matched": self.tasks[best], "sim": float(sims[best]), "margin": margin,
                "accepted": ok, "exact": False,
                "top": [(self.tasks[int(i)], float(sims[int(i)])) for i in order[:3]]}

    def predict_from_request(self, images_b64: list[str], qpos: list[float]) -> dict:
        if len(images_b64) != self.client_cams:
            raise ValueError(
                f"expected exactly {self.client_cams} images (--client-cams) but got "
                f"{len(images_b64)}; the slot->camera mapping "
                f"{dict(zip(self.image_keys, self.cam_indices))} is defined against a "
                f"{self.client_cams}-camera client.")
        return self.predict_from_frames(self._decode_frames(images_b64), qpos)

    def close(self):
        if self._rtt_samples:
            import statistics as _st
            r = sorted(self._rtt_samples)
            print(f"[timing] {len(r)} requests: round trip median {_st.median(r):.0f} ms, "
                  f"p90 {r[int(0.9 * (len(r) - 1))]:.0f} ms, max {r[-1]:.0f} ms")
            if self._slack_samples:
                sl = sorted(self._slack_samples)
                print(f"[timing] slack (plan budget - round trip) median {_st.median(sl):+.0f} ms, "
                      f"worst {sl[0]:+.0f} ms; {self._late_requests}/{len(sl)} requests arrived "
                      f"after the arm ran out of plan")
                spare = _st.median(sl)
                if self._late_requests == 0 and spare > 0:
                    print(f"[timing] room to lower --emit-steps: the median request finished "
                          f"{spare:.0f} ms before the plan ran out")
        if self.probe is not None:
            self.probe.close()
            self.probe = None
        if self.attn is not None:
            self.attn.close()
            self.attn = None


# ==========================================================================
# HTTP layer -- backend-agnostic handlers + FastAPI / stdlib transports,
# copied structurally from gr00t_server_openmani.py so the two servers cannot drift.
# ==========================================================================

def _cos(a, b):
    import numpy as _np
    return float(_np.dot(a, b) / (_np.linalg.norm(a) * _np.linalg.norm(b) + 1e-12))


def handle_init(server: SmolvlaOpenmaniServer, body: dict) -> tuple[int, dict]:
    lang = body.get("lang")
    if not isinstance(lang, str):
        return 400, {"detail": "field 'lang' (str) is required"}
    reasoning_cache_n = int(body.get("reasoning_cache_n", 0) or 0)
    server._done_streak = 0          # a new instruction starts a new episode
    server.lang = lang
    # /init means "new rollout". Anything remembered from the previous one -- the RTC prefix and
    # the seam reference -- would otherwise pull the first chunk of the new rollout toward a plan
    # made for a different episode, and possibly a different instruction.
    server.rtc_reset("/init")
    if reasoning_cache_n:
        print(f"[init] reasoning_cache_n={reasoning_cache_n} ignored (SmolVLA has no reasoning)")
    known = lang in server.tasks if server.tasks else None
    print(f"[init] lang={lang!r}" + ("" if known is None
                                     else f" (in training distribution: {known})"))
    snap = None
    if server.lang_snap and known is False:
        snap = server.snap_lang(lang)
        if snap.get("sim") is None:
            print(f"[snap/rules] {lang!r} -> {snap['matched']!r}  slots={snap.get('slots')}"
                  + (f"  repaired {snap['repaired_direction']}" if snap.get("repaired_direction")
                     else ""))
        else:
            top = ", ".join(f"{t!r} {v:.3f}" for t, v in snap["top"])
            print(f"[snap/embed] {lang!r} -> {snap['matched']!r} "
                  f"(sim {snap['sim']:.3f}, margin {snap['margin']:.3f}) | top3: {top}")
        if snap["accepted"]:
            server.lang_raw = lang
            server.lang = snap["matched"]
            lang = snap["matched"]
            known = True
        else:
            # Refusing beats guessing: a wrong snap makes the arm confidently do the wrong task,
            # and the operator has no way to tell that from a policy failure.
            print(f"[snap][REJECT] {snap.get('detail') or 'below thresholds'}")
            return 400, {"detail": "instruction could not be matched to a training sentence",
                         "lang": lang, "snap": snap}
    elif known is False:
        print(f"[init][WARN] {lang!r} is NOT one of the {len(server.tasks)} sentences this "
              f"policy was trained on. Closest by prefix: "
              f"{[t for t in server.tasks if t.split()[0] == lang.split()[0]][:3] or server.tasks[:3]}"
              f"  (pass --lang-snap to map it onto the nearest one instead)")
    server.lang_snap_info = snap
    return 200, {"ok": True, "lang": lang, "reasoning_cache_n": reasoning_cache_n,
                 "in_training_distribution": known, "n_training_tasks": len(server.tasks),
                 "lang_raw": server.lang_raw, "snap": snap}


def handle_predict(server: SmolvlaOpenmaniServer, body: dict) -> tuple[int, dict]:
    images = body.get("images")
    qpos = body.get("qpos")
    if not isinstance(images, list) or not all(isinstance(s, str) for s in images):
        return 400, {"detail": "field 'images' must be a list of base64 JPEG strings"}
    if not isinstance(qpos, list):
        return 400, {"detail": "field 'qpos' must be a list of DXL ticks"}
    if server.lang is None:
        return 400, {"detail": "call /init first"}
    if len(images) != server.client_cams:
        return 400, {"detail": (
            f"expected exactly {server.client_cams} images (--client-cams) for camera keys "
            f"{server.image_keys} via slots {server.cam_indices}, got {len(images)}.")}
    if len(qpos) != JOINT_DIM:
        return 400, {"detail": f"expected {JOINT_DIM} DXL ticks, got {len(qpos)}"}
    server._pending_client_timing = {
        "prev_rtt_ms": body.get("prev_rtt_ms"),
        "step_dt_ms": body.get("step_dt_ms"),
    }
    try:
        out = server.predict_from_request(images, [float(x) for x in qpos])
    except Exception as e:  # surface the real message instead of a bare 500
        import traceback

        traceback.print_exc()
        return 500, {"detail": f"{type(e).__name__}: {e}"}
    for k in DEBUG_KEYS:
        out.pop(k, None)
    return 200, out


def handle_health(server: SmolvlaOpenmaniServer) -> tuple[int, dict]:
    return 200, {
        "ok": True,
        "policy": "smolvla",
        "lang_set": server.lang is not None,
        "model_loaded": True,
        "joint_dim": JOINT_DIM,
        "action_dim": server.action_dim,
        "has_done": server.has_done,
        "done_threshold": server.done_threshold if server.has_done else None,
        "done_consecutive": server.done_consecutive if server.has_done else None,
        "chunk_size": server.emit_steps,
        "policy_chunk_size": server.chunk_size,
        "n_action_steps": server.n_action_steps,
        "n_avg": server.n_avg,
        "n_avg_space": "ticks (after the postprocessor)" if server.n_avg > 1 else None,
        "camera_keys": server.image_keys,
        "cam_indices": server.cam_indices,
        "client_cams": server.client_cams,
        "dropped_client_slots": server.dropped_slots,
        "model_path": server.model_path,
        "dataset_root": server.sidecar_root,
        "space": (server.sidecar or {}).get("space"),
        "unit": (server.sidecar or {}).get("unit"),
        "motor_order": server.motor_order,
        "fps": server.fps,
        "tick_limits": {n: list(TICK_LIMITS[n]) for n in MOTOR_NAMES},
        "rate_limit_reference_ticks": server.rate_limit_ticks,
        "rate_limiting_applied_here": False,
        "dry_run": server.dry_run,
        "n_training_tasks": len(server.tasks),
        **server.rtc_status(),
    }


def build_app(server: SmolvlaOpenmaniServer):
    """FastAPI app (preferred backend). Thin wrapper over the shared handlers."""
    from typing import Any, List, Optional

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI()

    class InitRequest(BaseModel):
        lang: str
        reasoning_cache_n: int = 0  # accepted for client compat; SmolVLA has no reasoning cache

    class PredictRequest(BaseModel):
        images: List[str]
        qpos: List[float]

    class PredictResponse(BaseModel):
        # contract
        joint_ticks_b64: str
        chunk_size: int
        joint_dim: int
        reasoning: Optional[str] = None
        done: Optional[bool] = None
        done_score: Optional[List[float]] = None
        done_mean: Optional[float] = None
        done_threshold: Optional[float] = None
        done_streak: Optional[int] = None
        done_consecutive: Optional[int] = None
        preprocess_ms: float
        inference_ms: float
        postprocess_ms: float
        # additive meta -- declared so FastAPI does not strip it from the response
        execute: bool = True
        dry_run: bool = False
        n_action_steps: Optional[int] = None
        policy_chunk_size: Optional[int] = None
        n_avg: Optional[int] = None
        spread_ticks: Optional[float] = None
        spread_ticks_per_motor: Optional[List[float]] = None
        max_step_ticks_in_chunk: Optional[int] = None
        max_step_ticks_per_motor: Optional[List[int]] = None
        first_step_jump_ticks: Optional[int] = None
        first_step_jump_per_motor: Optional[List[int]] = None
        rate_limit_reference_ticks: Optional[float] = None
        n_steps_over_rate_limit: Optional[int] = None
        n_clamped_values: Optional[int] = None
        n_clamped_per_motor: Optional[List[int]] = None
        state_out_of_training_range: Optional[List[Any]] = None
        # RTC + seam (declared, or FastAPI strips them from the response)
        rtc: bool = False
        rtc_used_prev_chunk: bool = False
        rtc_prev_chunk_steps: Optional[int] = None
        rtc_execution_horizon: Optional[int] = None
        rtc_inference_delay: Optional[int] = None
        rtc_prev_dropped_reason: Optional[str] = None
        seam_ticks_mean: Optional[float] = None
        seam_ticks_max: Optional[int] = None
        seam_ticks_per_motor: Optional[List[int]] = None

    @app.post("/init")
    def init(req: InitRequest):
        code, payload = handle_init(server, req.model_dump())
        if code != 200:
            raise HTTPException(status_code=code, detail=payload["detail"])
        return payload

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        code, payload = handle_predict(server, req.model_dump())
        if code != 200:
            raise HTTPException(status_code=code, detail=payload["detail"])
        return PredictResponse(**payload)

    @app.get("/health")
    def health():
        code, payload = handle_health(server)
        return JSONResponse(status_code=code, content=payload)

    return app


def make_stdlib_handler(server: SmolvlaOpenmaniServer):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Without this, Nagle holds the small header+body writes for an ACK and every /predict
        # pays ~50 ms of pure transport -- measured 2026-08-16 on the robot LAN: client RTT
        # 148 ms vs 90 ms server-side, gap 47 ms; with NODELAY the gap falls to 1.2 ms.
        # That is more than twice what --n-avg 4 costs (22 ms), so it is not optional here.
        disable_nagle_algorithm = True

        def _send(self, code: int, payload: dict):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return None
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                self._send(*handle_health(server))
            else:
                self._send(404, {"detail": f"no route {self.path}"})

        def do_POST(self):
            route = self.path.rstrip("/")
            try:
                body = self._read_json()
            except Exception as e:
                self._send(400, {"detail": f"bad JSON body: {type(e).__name__}: {e}"})
                return
            if not isinstance(body, dict):
                self._send(400, {"detail": "body must be a JSON object"})
                return
            if route == "/init":
                self._send(*handle_init(server, body))
            elif route == "/predict":
                self._send(*handle_predict(server, body))
            else:
                self._send(404, {"detail": f"no route {self.path}"})

        def log_message(self, fmt, *a):  # quiet: we print our own per-request lines
            pass

    return Handler


def serve_stdlib(server: SmolvlaOpenmaniServer, host: str, port: int):
    """Dependency-free HTTP transport (same routes/JSON as the FastAPI app).

    Single-threaded on purpose: one GPU model, a strictly request/response client at 10 Hz.
    """
    from http.server import HTTPServer

    httpd = HTTPServer((host, port), make_stdlib_handler(server))
    print(f"[http] stdlib backend serving on http://{host}:{port} "
          f"(routes: POST /init, POST /predict, GET /health)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[http] shutting down")
    finally:
        httpd.server_close()


# ==========================================================================
# Self-test: no robot, no cameras, no dynamixel_sdk. Log-based pass/fail.
# ==========================================================================

def _jpeg_b64(arr: np.ndarray, quality: int = 90) -> str:
    """Encode exactly like the real client does (client_live.py:62-66, quality 90)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _random_jpeg_b64(h: int, w: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    return _jpeg_b64(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


_DATASET_CACHE: dict = {}


def _dataset_open(dataset_root: str, chunk_size: int):
    """LeRobotDataset, cached per (root, chunk_size).

    Building one indexes the whole video set; the self-test opens the same dataset for section 2
    and again for every seam pair of section 5, so without the cache the RTC section would spend
    most of its time re-opening the dataset.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_root).expanduser()
    key = (str(root), int(chunk_size))
    if key not in _DATASET_CACHE:
        meta = json.loads((root / "meta" / "info.json").read_text())
        fps = int(meta.get("fps", EXPECTED_FPS))
        _DATASET_CACHE[key] = LeRobotDataset(
            f"local/{root.name}", root=str(root),
            delta_timestamps={"action": [i / fps for i in range(chunk_size)]},
        )
    return _DATASET_CACHE[key]


def _dataset_item(ds, index: int) -> dict:
    item = ds[index % len(ds)]
    frames = []
    for c in (1, 2, 3):
        img = item[f"observation.images.camera{c}"]         # (3,H,W) float32 in [0,1]
        frames.append((img.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8))
    return {
        "n": len(ds),
        "index": index % len(ds),
        "task": item["task"],
        "episode_index": int(item["episode_index"]),
        "state": item["observation.state"].numpy().astype(np.float64),
        # motors only: on a 6-D done_flag dataset the action row is 5 ticks + a flag, and
        # everything downstream compares this against the MOTOR chunk the server emits
        "gt_chunk": item["action"].numpy().astype(np.float64)[..., :JOINT_DIM],
        "is_pad": item["action_is_pad"].numpy(),
        "frames": frames,
    }


def _dataset_frame(dataset_root: str, index: int, chunk_size: int):
    """One real frame + its ground-truth action chunk, straight from the LeRobot dataset."""
    return _dataset_item(_dataset_open(dataset_root, chunk_size), index)


def _dataset_seam_pairs(dataset_root: str, chunk_size: int, gap: int, n_pairs: int,
                        start: int = 0) -> list:
    """(frame t, frame t+gap) pairs from INSIDE one episode -- the re-plan boundary of 5(e).

    A pair straddling an episode boundary would compare two plans for unrelated scenes, so
    episode_index has to match; the task string alone cannot tell two episodes of the same task
    apart (same test as diag/input_ablation.py:678-683).
    """
    ds = _dataset_open(dataset_root, chunk_size)
    n = len(ds)
    stride = max(int(gap) + 1, n // max(int(n_pairs), 1))
    out, i, tried = [], int(start) % n, 0
    while len(out) < n_pairs and tried < n_pairs * 8:
        tried += 1
        a, b = i % n, (i % n) + int(gap)
        i += stride
        if b >= n:
            continue
        fa, fb = _dataset_item(ds, a), _dataset_item(ds, b)
        if fa["episode_index"] != fb["episode_index"] or fa["task"] != fb["task"]:
            continue
        out.append((fa, fb))
    return out


def self_test(args) -> int:
    print("== smolvla_server_openmani self-test ==")
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))
        ok = ok and bool(cond)

    try:
        server = SmolvlaOpenmaniServer(
            model_path=args.model_path, dataset_root=args.dataset_root, device=args.device,
            cam_indices=args.cam_indices, client_cams=args.client_cams,
            emit_steps=args.emit_steps, allow_mismatch=args.allow_mismatch,
            dry_run=args.dry_run, n_avg=args.n_avg, rate_limit_ticks=args.rate_limit_ticks,
            diag_out=args.diag_out, diag_every=1,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"  [FAIL] server construction raised: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        print("== self-test FAILED (load) ==")
        return 1

    # ---------------------------------------------------------------- 1. synthetic /predict
    print("\n-- 1. synthetic request (random JPEGs + dummy ticks) ------------------------")
    images = [_random_jpeg_b64(240, 320, seed=i) for i in range(args.client_cams)]
    rng = np.random.default_rng(0)
    qpos = (2048 + rng.integers(-400, 400, size=JOINT_DIM)).astype(float).tolist()
    print(f"  {args.client_cams} JPEGs (240x320), qpos={[int(v) for v in qpos]}")
    server.lang = args.instruction
    try:
        out = server.predict_from_request(images, qpos)
    except Exception as e:
        print(f"  [FAIL] predict raised: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        print("== self-test FAILED (predict) ==")
        return 1

    tc = out["_ticks_chunk"]
    check("output dtype int32", tc.dtype == np.int32, str(tc.dtype))
    check(f"output shape ({server.emit_steps}, {JOINT_DIM})",
          tc.shape == (server.emit_steps, JOINT_DIM), str(tc.shape))
    per_motor_ok = bool((tc >= LIMIT_LO).all() and (tc <= LIMIT_HI).all())
    check("every tick inside its motor's limits", per_motor_ok,
          f"min={tc.min(axis=0).tolist()} max={tc.max(axis=0).tolist()} "
          f"limits={ {n: TICK_LIMITS[n] for n in MOTOR_NAMES} }")
    check("gripper inside [1000,2700]",
          int(tc[:, -1].min()) >= GRIPPER_TICK_MIN and int(tc[:, -1].max()) <= GRIPPER_TICK_MAX,
          f"[{int(tc[:, -1].min())},{int(tc[:, -1].max())}]")
    check("chunk_size field matches the array", out["chunk_size"] == tc.shape[0])
    check("joint_dim == 5", out["joint_dim"] == JOINT_DIM)
    check("reasoning is None (SmolVLA has no chain-of-thought output)",
          out["reasoning"] is None)
    if server.has_done:
        ds = out["done_score"]
        check("done is a bool on a 6-D checkpoint", isinstance(out["done"], bool), str(out["done"]))
        check(f"done_score has one entry per emitted step ({server.emit_steps})",
              isinstance(ds, list) and len(ds) == tc.shape[0], str(ds))
        check("done_mean is the mean of done_score",
              ds and abs(out["done_mean"] - float(np.mean(ds))) < 1e-6)
        check("done_score is on the training scale (roughly [0, 1])",
              ds and all(-1.0 <= v <= 2.0 for v in ds), str(np.round(ds, 3).tolist()))
        check("the emitted chunk is motors only, not 6 wide", tc.shape[1] == JOINT_DIM,
              str(tc.shape))
    else:
        check("done is None on a 5-D checkpoint (no termination channel)",
              out["done"] is None)

    raw = base64.b64decode(out["joint_ticks_b64"])
    expected_bytes = out["chunk_size"] * out["joint_dim"] * 4
    check(f"payload is {expected_bytes} bytes (chunk_size*joint_dim*4)",
          len(raw) == expected_bytes, f"{len(raw)} bytes")
    decoded = np.frombuffer(raw, dtype=np.int32).reshape(out["chunk_size"], out["joint_dim"])
    check("base64 round-trip is identical (C-order int32)", bool(np.array_equal(decoded, tc)))
    print(f"  chunk:\n{tc}")
    print(f"  max|dtick| between steps = {out['max_step_ticks_in_chunk']} "
          f"(per motor {out['max_step_ticks_per_motor']}), "
          f"jump from present = {out['first_step_jump_ticks']}")
    print(f"  clamped values = {out['n_clamped_values']} {out['n_clamped_per_motor']}; "
          f"steps over the {out['rate_limit_reference_ticks']:g}-tick robot limit = "
          f"{out['n_steps_over_rate_limit']}/{max(out['chunk_size'] - 1, 0)}")
    print(f"  timings ms: pre={out['preprocess_ms']:.1f} infer={out['inference_ms']:.1f} "
          f"post={out['postprocess_ms']:.1f}")

    # ------------------------------------------------------- 2. real dataset frame vs GT
    print("\n-- 2. real dataset frame vs. its ground-truth action chunk ------------------")
    fr = None   # reused by section 4 (averaging) when a real frame is available
    if not server.sidecar_root:
        print("  SKIPPED: no dataset root resolved.")
    else:
        try:
            fr = _dataset_frame(server.sidecar_root, args.frame_index, server.chunk_size)
        except Exception as e:
            print(f"  [FAIL] could not read a dataset frame: {type(e).__name__}: {e}")
            check("dataset frame loaded", False)
            fr = None
        if fr is not None:
            print(f"  dataset frame {fr['index']}/{fr['n']}  task={fr['task']!r}")
            print(f"  state (ticks) = {fr['state'].astype(int).tolist()}")
            server.lang = fr["task"]
            # Through the SAME path deployment uses, JPEG q=90 included, so this measures the
            # deployed pipeline and not an idealized one.
            imgs = [_jpeg_b64(f) for f in fr["frames"]]
            out2 = server.predict_from_request(imgs, fr["state"].tolist())
            pred = out2["_ticks_chunk"].astype(np.float64)
            gt = fr["gt_chunk"][: pred.shape[0]]
            valid = ~fr["is_pad"][: pred.shape[0]]
            err = pred - gt
            names = server.motor_order
            print(f"  {'step':>4} | {'predicted (ticks)':^34} | {'ground truth':^34} | max|err|")
            for t in range(pred.shape[0]):
                p = " ".join(f"{int(v):5d}" for v in pred[t])
                g = " ".join(f"{int(v):5d}" for v in gt[t])
                mark = "" if valid[t] else "  (pad)"
                print(f"  {t:>4} | {p} | {g} | {np.abs(err[t]).max():7.1f}{mark}")
            print(f"  per-channel MAE (ticks, valid steps only): "
                  + ", ".join(f"{names[i]}={np.abs(err[valid, i]).mean():.1f}"
                              for i in range(JOINT_DIM)))
            print(f"  per-channel training std  (ticks)        : "
                  + ", ".join(f"{names[i]}={server.action_std[i]:.1f}" for i in range(JOINT_DIM)))
            print(f"  step-0 error = {np.abs(err[0]).astype(int).tolist()} ticks "
                  f"(step 0 should be closest to the present state)")
            gt_step = np.abs(np.diff(gt[valid], axis=0)).max() if valid.sum() > 1 else 0.0
            print(f"  motion: predicted max|dtick|={out2['max_step_ticks_in_chunk']} vs "
                  f"ground-truth max|dtick|={gt_step:.0f} over the same window")
            check("prediction is not a constant chunk (policy is doing something)",
                  out2["max_step_ticks_in_chunk"] > 0,
                  f"max|dtick|={out2['max_step_ticks_in_chunk']}")
            check("step-0 prediction is within 3x the action std of ground truth",
                  bool((np.abs(err[0]) < 3 * server.action_std[:JOINT_DIM]).all()),
                  f"|err0|={np.abs(err[0]).astype(int).tolist()} 3*std="
                  f"{(3 * server.action_std[:JOINT_DIM]).astype(int).tolist()}")

    # ------------------------------------------------------------ 3. real HTTP round-trip
    print("\n-- 3. real HTTP round-trip over the stdlib backend ---------------------------")
    import threading
    import urllib.error
    import urllib.request
    from http.server import HTTPServer

    httpd = HTTPServer(("127.0.0.1", 0), make_stdlib_handler(server))
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"

    def post(path, payload):
        req = urllib.request.Request(f"{base}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode())

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
            health = json.loads(r.read().decode())
        check("GET /health -> ok", health.get("ok") is True)
        check("health reports joint_dim==5", health.get("joint_dim") == JOINT_DIM)
        check("health reports unit=raw_ticks / space=joint",
              health.get("unit") == "raw_ticks" and health.get("space") == "joint",
              f"unit={health.get('unit')} space={health.get('space')}")
        check("health says the server does NOT rate limit",
              health.get("rate_limiting_applied_here") is False)
        print(f"  /health: chunk_size={health['chunk_size']} "
              f"policy_chunk_size={health['policy_chunk_size']} "
              f"n_action_steps={health['n_action_steps']} motor_order={health['motor_order']} "
              f"fps={health['fps']}")

        code, jinit = post("/init", {"lang": args.instruction, "reasoning_cache_n": 0})
        check("POST /init -> 200", code == 200, str(jinit))
        code, jp = post("/predict", {"images": images, "qpos": qpos})
        check("POST /predict -> 200", code == 200)
        # Decode exactly the way the real clients do:
        #   dex_ref/client_live.py:124-126 / gr00t_client_local.py:184-192
        t = np.frombuffer(base64.b64decode(jp["joint_ticks_b64"]), dtype=np.int32)
        t = t.reshape(jp["chunk_size"], jp["joint_dim"])
        check("client-style decode gives the right shape",
              t.shape == (server.emit_steps, JOINT_DIM), str(t.shape))
        check("all contract keys present in the JSON response",
              all(k in jp for k in ("joint_ticks_b64", "chunk_size", "joint_dim",
                                    "preprocess_ms", "inference_ms", "postprocess_ms")),
              str(sorted(jp)))
        check("dry-run flag surfaces as execute=<not dry_run>",
              jp.get("execute") is (not args.dry_run) and jp.get("dry_run") is args.dry_run,
              f"execute={jp.get('execute')} dry_run={jp.get('dry_run')}")
        check("health exposes n_avg", health.get("n_avg") == server.n_avg,
              f"health n_avg={health.get('n_avg')} server n_avg={server.n_avg}")
        check("health exposes the RTC state (off at this point)",
              health.get("rtc") is False and "rtc_execution_horizon" in health
              and "rtc_inference_delay" in health and health.get("rtc_has_prev_chunk") in
              (True, False),
              f"{ {k: v for k, v in health.items() if k.startswith('rtc')} }")
        check("/predict response exposes the RTC state and the re-plan seam",
              jp.get("rtc") is False and "rtc_used_prev_chunk" in jp and "seam_ticks_mean" in jp,
              f"rtc={jp.get('rtc')} used_prev={jp.get('rtc_used_prev_chunk')} "
              f"seam={jp.get('seam_ticks_mean')}")
        check("/predict response exposes the n_avg it used",
              jp.get("n_avg") == server.n_avg, f"n_avg={jp.get('n_avg')}")
        check("/predict response carries spread_ticks when n_avg>1",
              (jp.get("spread_ticks") is None) if server.n_avg == 1
              else (isinstance(jp.get("spread_ticks"), float)
                    and np.isfinite(jp["spread_ticks"])),
              f"spread_ticks={jp.get('spread_ticks')} per_motor={jp.get('spread_ticks_per_motor')}")
        check("the JSON response carries no debug keys",
              not any(k in jp for k in DEBUG_KEYS), str([k for k in DEBUG_KEYS if k in jp]))
        print(f"  HTTP chunk row0={t[0].tolist()} row-1={t[-1].tolist()}")

        # negative cases must be rejected, not silently coerced
        for name, payload in (
            ("wrong image count", {"images": images[:1], "qpos": qpos}),
            ("wrong qpos length", {"images": images, "qpos": qpos[:3]}),
        ):
            try:
                post("/predict", payload)
                check(f"/predict rejects {name}", False)
            except urllib.error.HTTPError as e:
                check(f"/predict rejects {name}", e.code == 400, f"HTTP {e.code}")
    finally:
        httpd.shutdown()
        httpd.server_close()

    # ------------------------------------------------------------ 4. sample averaging
    print("\n-- 4. sample averaging (--n-avg) ---------------------------------------------")
    torch = server.torch
    if fr is not None:
        avg_frames, avg_qpos, avg_task = fr["frames"], fr["state"].tolist(), fr["task"]
        print(f"  using real dataset frame {fr['index']} (task={fr['task']!r})")
    else:
        avg_frames, avg_qpos, avg_task = server._decode_frames(images), qpos, args.instruction
        print("  using the synthetic random-JPEG request (no dataset frame available)")
    server.lang = avg_task

    def _seed_rng(s: int = 20260816):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _legacy_ticks():
        """The pre-averaging predict path, re-implemented verbatim from the single-draw server:
        preprocessor -> predict_action_chunk(batch) -> postprocessor -> rint -> per-motor clamp.
        If --n-avg 1 ever stops matching this, the regression is here and not in a comment."""
        f = server._apply_camera_corrections(list(avg_frames))
        obs = server._build_observation(
            f, np.asarray(avg_qpos, dtype=np.float64).reshape(-1).astype(np.float32), avg_task)
        server.policy.reset()
        server.preprocessor.reset()
        server.postprocessor.reset()
        with torch.no_grad():
            batch_l = server.preprocessor(obs)
            cn = server.policy.predict_action_chunk(batch_l)
            ck = server.postprocessor(cn)
        a = ck.detach().float().cpu().numpy()
        if a.ndim == 3:
            a = a[0]
        # motors only: LIMIT_LO/HI are per-motor, and predict_from_frames peels the
        # termination channel off before rounding and clamping
        r = a[:server.emit_steps, :JOINT_DIM]
        return np.clip(np.rint(r).astype(np.int64), LIMIT_LO, LIMIT_HI).astype(np.int32)

    # (a) --n-avg 1 must be the ORIGINAL path, bit for bit (same RNG consumption included)
    _seed_rng()
    out_n1 = server.predict_from_frames(list(avg_frames), avg_qpos, task=avg_task, n_avg=1)
    _seed_rng()
    legacy = _legacy_ticks()
    same = bool(np.array_equal(out_n1["_ticks_chunk"], legacy))
    check("n_avg=1 reproduces the pre-averaging path exactly (same seed, same int32 chunk)",
          same, f"max|d|={int(np.abs(out_n1['_ticks_chunk'].astype(np.int64) - legacy.astype(np.int64)).max())} ticks")
    check("n_avg=1 emits identical bytes to the legacy chunk",
          base64.b64decode(out_n1["joint_ticks_b64"]) == legacy.tobytes())
    check("n_avg=1 reports no spread (a single draw has none)",
          out_n1["spread_ticks"] is None and out_n1["n_avg"] == 1)

    # (b) n_avg=4: shape / dtype / limits / fields
    out_n4 = server.predict_from_frames(list(avg_frames), avg_qpos, task=avg_task, n_avg=4)
    t4 = out_n4["_ticks_chunk"]
    check("n_avg=4 output dtype int32", t4.dtype == np.int32, str(t4.dtype))
    check(f"n_avg=4 output shape ({server.emit_steps}, {JOINT_DIM})",
          t4.shape == (server.emit_steps, JOINT_DIM), str(t4.shape))
    check("n_avg=4 stays inside the per-motor tick limits",
          bool((t4 >= LIMIT_LO).all() and (t4 <= LIMIT_HI).all()),
          f"min={t4.min(axis=0).tolist()} max={t4.max(axis=0).tolist()}")
    check("n_avg=4 ran 4 draws in one pass",
          out_n4["_ticks_draws"].shape[0] == 4 and out_n4["_chunk_norm_draws"].shape[0] == 4,
          f"ticks_draws{tuple(out_n4['_ticks_draws'].shape)} "
          f"norm_draws{tuple(out_n4['_chunk_norm_draws'].shape)}")
    check("n_avg=4 reports a finite draw-to-draw spread",
          out_n4["spread_ticks"] is not None and np.isfinite(out_n4["spread_ticks"])
          and out_n4["spread_ticks"] > 0
          and len(out_n4["spread_ticks_per_motor"]) == JOINT_DIM,
          f"spread={out_n4['spread_ticks']:.2f} ticks per motor="
          f"{[round(v, 2) for v in out_n4['spread_ticks_per_motor']]}")
    _mean_all = out_n4["_ticks_draws"].astype(np.float64).mean(axis=0)
    _expect = np.clip(np.rint(_mean_all[:server.emit_steps]).astype(np.int64),
                      LIMIT_LO, LIMIT_HI).astype(np.int32)
    check("the served chunk is clamp(round(mean of the draws)) -- averaging happens BEFORE "
          "rounding and clamping",
          bool(np.array_equal(_expect, t4)),
          f"max|d|={int(np.abs(_expect.astype(np.int64) - t4.astype(np.int64)).max())} ticks")

    # (e) averaging in TICK space == averaging in NORMALIZED space, i.e. the postprocessor really
    #     is affine. If a future step adds a clip, this fails instead of silently mis-feeding the
    #     probe, which is handed `_chunk_norm` (the normalized mean).
    server.postprocessor.reset()
    with torch.no_grad():
        back = server.postprocessor(out_n4["_chunk_norm"]).detach().float().cpu().numpy()
    if back.ndim == 3:
        back = back[0]
    # _chunk_norm keeps every channel (the probe and RTC need the full width); _ticks_draws is
    # motors only, because predict_from_frames peels the termination channel off first.
    back = back[..., :JOINT_DIM]
    mean_ticks = out_n4["_ticks_draws"].astype(np.float64).mean(axis=0)
    d_affine = float(np.abs(back - mean_ticks).max())
    check("unnormalize(mean of draws) == mean of unnormalized draws (postprocessor is affine)",
          d_affine < 0.05, f"max|d|={d_affine:.2e} ticks")

    # (c) the point of the exercise: repeated requests must disagree LESS with n_avg=4
    reps = 4
    lat = {}
    spreads = {}
    for n in (1, 4):
        chunks, ms = [], []
        for _ in range(reps):
            o = server.predict_from_frames(list(avg_frames), avg_qpos, task=avg_task, n_avg=n)
            chunks.append(o["_ticks_chunk"].astype(np.float64))
            ms.append(o["inference_ms"])
        spreads[n] = pairwise_spread_ticks(np.stack(chunks))[0]
        lat[n] = float(np.median(ms))
    print(f"  request-to-request spread over {reps} repeats of the SAME observation: "
          f"n_avg=1 {spreads[1]:.2f} ticks -> n_avg=4 {spreads[4]:.2f} ticks "
          f"(ratio {spreads[4] / spreads[1]:.2f}; 1/sqrt(4)=0.50 is the iid expectation)")
    print(f"  within-request draw spread reported at n_avg=4: {out_n4['spread_ticks']:.2f} ticks "
          f"-- NOT comparable to the figure above: spread_ticks covers all "
          f"{out_n4['policy_chunk_size']} predicted steps while the request-to-request figure "
          f"covers only the {out_n4['chunk_size']} emitted ones, and "
          f"spread grows with horizon (measured ~20 ticks at step 0 vs ~52 at step 15)")
    check("averaging 4 draws reduces the request-to-request spread",
          spreads[4] < spreads[1], f"{spreads[4]:.2f} < {spreads[1]:.2f} ticks")

    # (d) latency is measured and reported per request
    print(f"  median inference_ms over {reps} requests: n_avg=1 {lat[1]:.1f} ms -> "
          f"n_avg=4 {lat[4]:.1f} ms ({lat[4] / lat[1]:.2f}x, vs 4.00x if the draws were "
          f"sequential)")
    check("inference latency is recorded for both settings",
          np.isfinite(lat[1]) and lat[1] > 0 and np.isfinite(lat[4]) and lat[4] > 0,
          f"n_avg=1 {lat[1]:.1f} ms, n_avg=4 {lat[4]:.1f} ms")

    # (f) the spread definition must be the one diag/sampling_noise.py used
    try:
        from diag.sampling_noise import _pairwise_spread, _pairwise_spread_per_ch

        x = np.random.default_rng(0).normal(size=(5, 7, JOINT_DIM))
        s, per = pairwise_spread_ticks(x)
        check("spread definition matches diag/sampling_noise.py",
              abs(s - _pairwise_spread(x)) < 1e-9
              and float(np.abs(np.asarray(per) - _pairwise_spread_per_ch(x)).max()) < 1e-9,
              f"{s:.6f} vs {_pairwise_spread(x):.6f}")
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] could not import diag.sampling_noise to cross-check the spread "
              f"definition: {type(e).__name__}: {e}")


    # ------------------------------------------------------------ 5. real-time chunking
    print("\n-- 5. real-time chunking (--rtc) ----------------------------------------------")
    horizon_want = args.rtc_execution_horizon or server.emit_steps
    print(f"  the server under test was built with RTC OFF (regardless of the --rtc flag) and is "
          f"toggled in place, so (a) and (b)-(f) are measured on ONE loaded checkpoint.")
    print(f"  parameters under test: execution_horizon={horizon_want} "
          f"inference_delay={args.rtc_inference_delay} schedule={args.rtc_schedule} "
          f"max_guidance_weight={args.rtc_max_guidance_weight:g} "
          f"stale_reset={server.rtc_stale_reset_s:g}s")

    _ext_st = getattr(server.policy, "_smolvla_ext", None)
    _has_coda = _ext_st is not None and bool(_ext_st.coda)
    if _has_coda:
        print(f"  this is a Coda checkpoint ({_ext_st.summary()}), so RTC needs a placement: "
              f"sections (a)-(g) run with --coda-rtc "
              f"{args.coda_rtc if args.coda_rtc != 'off' else 'main'} and (h0)/(h) sweep the rest.")

    def _rtc_on(mode=None, max_w=None):
        server.configure_rtc(enabled=True, execution_horizon=args.rtc_execution_horizon,
                             inference_delay=args.rtc_inference_delay,
                             schedule=args.rtc_schedule,
                             max_guidance_weight=(args.rtc_max_guidance_weight if max_w is None
                                                  else float(max_w)),
                             horizon_explicit=int(args.rtc_execution_horizon) > 0)
        if _has_coda:
            # a Coda checkpoint raises at predict time with coda_rtc=off, so (a)-(g) need one
            if mode is None:
                mode = (args.coda_rtc if args.coda_rtc not in ("off", "auto")
                        else server.CODA_RTC_AUTO)
            server.configure_coda_rtc(mode)

    def _one(task=None, n=1, seed=None):
        if seed is not None:
            _seed_rng(seed)
        return server.predict_from_frames(list(avg_frames), avg_qpos,
                                          task=avg_task if task is None else task, n_avg=n)

    # --- (a) --rtc OFF is the pre-RTC path, byte for byte ---------------------------------
    check("(a) the server starts with RTC off and an untouched policy config",
          server.rtc is None and server.policy.config.rtc_config is None,
          f"server.rtc={server.rtc} cfg.rtc_config={server.policy.config.rtc_config}")
    server.rtc_reset("self-test (a)")
    out_off = _one(seed=20260816)
    _seed_rng(20260816)
    legacy_off = _legacy_ticks()
    d_off = int(np.abs(out_off["_ticks_chunk"].astype(np.int64)
                       - legacy_off.astype(np.int64)).max())
    check("(a) RTC OFF emits the pre-RTC chunk byte for byte (same seed, legacy re-implementation)",
          bool(np.array_equal(out_off["_ticks_chunk"], legacy_off))
          and base64.b64decode(out_off["joint_ticks_b64"]) == legacy_off.tobytes(),
          f"max|d|={d_off} ticks")
    check("(a) RTC OFF responses report rtc=false and carry no RTC parameters",
          out_off["rtc"] is False and out_off["rtc_used_prev_chunk"] is False
          and out_off["rtc_execution_horizon"] is None
          and out_off["rtc_inference_delay"] is None,
          f"rtc={out_off['rtc']} horizon={out_off['rtc_execution_horizon']}")
    off_ref = out_off["_ticks_chunk"].copy()

    # --- (b) enabling it, and the first request ------------------------------------------
    _rtc_on()
    check("(b) configure_rtc reached the sampler, not just the config object",
          server.rtc is not None and server.policy._rtc_enabled()
          and server.policy.model.rtc_processor is not None
          and server.policy.model._rtc_enabled(),
          f"policy._rtc_enabled={server.policy._rtc_enabled()} "
          f"model.rtc_processor={type(server.policy.model.rtc_processor).__name__}")
    check("(b) execution_horizon defaults to emit_steps, NOT RTCConfig's own default of 10",
          server.rtc_execution_horizon == horizon_want
          and (args.rtc_execution_horizon > 0 or server.rtc_execution_horizon == server.emit_steps),
          f"horizon={server.rtc_execution_horizon} emit_steps={server.emit_steps}")
    server.rtc_reset("self-test (b)")
    out_on1 = _one(seed=20260816)
    check("(b) RTC ON, first request: runs with no previous chunk",
          out_on1["rtc"] is True and out_on1["rtc_used_prev_chunk"] is False
          and out_on1["rtc_prev_chunk_steps"] == 0,
          f"used_prev={out_on1['rtc_used_prev_chunk']} reason={out_on1['rtc_prev_dropped_reason']}")
    d_first = int(np.abs(out_on1["_ticks_chunk"].astype(np.int64)
                         - off_ref.astype(np.int64)).max())
    check("(b) RTC ON without a prefix is identical to RTC OFF (same seed) -- guidance only "
          "appears once there is something to inpaint against",
          bool(np.array_equal(out_on1["_ticks_chunk"], off_ref)), f"max|d|={d_first} ticks")
    check("(b) RTC ON exposes its parameters on the response",
          out_on1["rtc_execution_horizon"] == server.rtc_execution_horizon
          and out_on1["rtc_inference_delay"] == server.rtc_inference_delay,
          f"horizon={out_on1['rtc_execution_horizon']} delay={out_on1['rtc_inference_delay']}")

    # --- (c) the second request really consumes the tail ----------------------------------
    out_on2 = _one(seed=20260819)
    want_tail = server.chunk_size - server.emit_steps
    check("(c) the second request consumed the previous chunk's unexecuted tail",
          out_on2["rtc_used_prev_chunk"] is True
          and out_on2["rtc_prev_chunk_steps"] == want_tail,
          f"used_prev={out_on2['rtc_used_prev_chunk']} steps={out_on2['rtc_prev_chunk_steps']} "
          f"expected {want_tail} (= chunk_size {server.chunk_size} - emit_steps "
          f"{server.emit_steps})")
    server.rtc_reset("self-test (c) control")
    out_ctl = _one(seed=20260819)          # SAME seed, same observation, no prefix
    d_guided = int(np.abs(out_on2["_ticks_chunk"].astype(np.int64)
                          - out_ctl["_ticks_chunk"].astype(np.int64)).max())
    check("(c) the prefix changed the sample (same seed + same observation, guided vs unguided)",
          d_guided > 0, f"max|d|={d_guided} ticks")
    # direction, not just difference: step 0 of the guided chunk must sit CLOSER to the step the
    # old plan wanted next. Ticks, via the postprocessor's own affine stats.
    prev_norm_np = out_on1["_chunk_norm"].detach().float().cpu().numpy()[0]
    # prev_norm_np is the SERVED normalized chunk, which is action_dim wide (6 with a
    # termination channel); the ticks compared against it are motors only.
    tail0_ticks = (prev_norm_np[server.emit_steps][:JOINT_DIM] * server.action_std[:JOINT_DIM]
                   + server.action_mean[:JOINT_DIM])
    near_on = float(np.abs(out_on2["_ticks_chunk"][0] - tail0_ticks).mean())
    near_off = float(np.abs(out_ctl["_ticks_chunk"][0] - tail0_ticks).mean())
    check("(c) guidance pulls step 0 TOWARD the old plan's next step",
          near_on < near_off,
          f"|step0 - prev_tail[0]| = {near_on:.1f} ticks guided vs {near_off:.1f} unguided")

    # --- (d) the resets --------------------------------------------------------------------
    _one(seed=1)                                            # re-establish a prefix
    o_task = _one(task=avg_task + " -- a different instruction", seed=2)
    check("(d) an instruction change drops the previous chunk",
          o_task["rtc_used_prev_chunk"] is False
          and o_task["rtc_prev_dropped_reason"] == "task_changed",
          f"used_prev={o_task['rtc_used_prev_chunk']} reason={o_task['rtc_prev_dropped_reason']}")
    _one(seed=3)                                            # back to avg_task (drops again)
    _one(seed=4)                                            # now a prefix for avg_task exists
    check("(d) a prefix exists again before the /init test", server._prev_plan_state is not None)
    code_i, _ = handle_init(server, {"lang": avg_task, "reasoning_cache_n": 0})
    check("(d) /init clears the remembered chunk",
          code_i == 200 and server._prev_plan_state is None, f"HTTP {code_i}")
    o_init = _one(seed=5)
    check("(d) the first request after /init is unguided",
          o_init["rtc_used_prev_chunk"] is False and o_init["rtc_prev_dropped_reason"] == "/init",
          f"reason={o_init['rtc_prev_dropped_reason']}")
    _keep_stale = server.rtc_stale_reset_s
    server.rtc_stale_reset_s = 0.0
    o_stale = _one(seed=6)
    server.rtc_stale_reset_s = _keep_stale
    check("(d) a stale previous chunk is dropped (--rtc-stale-reset-s)",
          o_stale["rtc_used_prev_chunk"] is False
          and o_stale["rtc_prev_dropped_reason"] == "stale",
          f"reason={o_stale['rtc_prev_dropped_reason']}")

    # --- (e) does it actually close the seam? ---------------------------------------------
    print(f"\n  (e) seam = |plan(t)[{server.emit_steps - 1}] - plan(t+{server.emit_steps})[0]| "
          f"averaged over the {JOINT_DIM} motors -- the diag/input_ablation.py:693-701 "
          f"definition, so it is comparable to the 57-65 ticks measured there.")
    pairs = []
    if server.sidecar_root:
        try:
            pairs = _dataset_seam_pairs(server.sidecar_root, server.chunk_size, server.emit_steps,
                                        int(args.rtc_seam_pairs), start=args.frame_index)
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] could not build seam pairs: {type(e).__name__}: {e}")
    if not pairs:
        print("  SKIPPED: no dataset pairs available, so the seam cannot be measured on real "
              "consecutive observations. (e) is NOT covered by this run.")
    else:
        _keep_stale = server.rtc_stale_reset_s
        server.rtc_stale_reset_s = 1e9   # a slow request must not silently drop the prefix here

        def _seam_run():
            """One pass over the pairs: predict at A, then at A+emit_steps with A's plan in
            memory. Same observations and same noise seeds in every condition, so conditions
            are compared pair by pair. Returns the per-pair arrays (e), (h0) and (h) share."""
            seams, used, ms, jump, spread, step0, mx, over, clamp = [], [], [], [], [], [], [], [], []
            for k, (fa, fb) in enumerate(pairs):
                server.rtc_reset(f"seam pair {k}")
                _seed_rng(70000 + k)
                server.predict_from_frames(fa["frames"], fa["state"].tolist(), task=fa["task"],
                                           n_avg=args.n_avg)
                _seed_rng(80000 + k)
                ob = server.predict_from_frames(fb["frames"], fb["state"].tolist(),
                                                task=fb["task"], n_avg=args.n_avg)
                seams.append(ob["seam_ticks_mean"])
                used.append(bool(ob["rtc_used_prev_chunk"]))
                ms.append(float(ob["inference_ms"]))
                jump.append(float(ob["first_step_jump_ticks"]))
                spread.append(float("nan") if ob["spread_ticks"] is None
                              else float(ob["spread_ticks"]))
                step0.append(np.asarray(ob["_ticks_chunk"][0], dtype=np.float64))
                mx.append(float(ob["max_step_ticks_in_chunk"]))
                over.append(int(ob["n_steps_over_rate_limit"]))
                clamp.append(int(ob["n_clamped_values"]))
            return {"seam": np.asarray(seams, dtype=np.float64), "used": used,
                    "ms": float(np.median(ms)), "jump": np.asarray(jump, dtype=np.float64),
                    "spread": np.asarray(spread, dtype=np.float64),
                    "step0": np.stack(step0), "max_step": np.asarray(mx, dtype=np.float64),
                    "n_over": int(sum(over)), "n_clamped": int(sum(clamp))}

        def _paired(off, on):
            """Paired reduction off-on: mean, standard error, how many pairs improved."""
            d = off - on
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            return float(d.mean()), se, int((d > 0).sum()), len(d)

        server.configure_rtc(enabled=False)
        if _has_coda:
            server.configure_coda_rtc("off")
        r_off = _seam_run()
        seam_off, used_off, ms_off, jump_off = r_off["seam"], r_off["used"], r_off["ms"], r_off["jump"]
        _rtc_on()
        # GPU memory across the guided run: RTCProcessor denoises inside torch.enable_grad(),
        # so a missing detach would park one autograd graph per request in VRAM forever.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mem0 = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        r_on = _seam_run()
        seam_on, used_on, ms_on, jump_on = r_on["seam"], r_on["used"], r_on["ms"], r_on["jump"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mem1 = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        server.rtc_stale_reset_s = _keep_stale

        gt_vals = [float(np.abs(fb["gt_chunk"][0] - fa["gt_chunk"][server.emit_steps - 1]).mean())
                   for fa, fb in pairs
                   if not bool(fa["is_pad"][server.emit_steps - 1]) and not bool(fb["is_pad"][0])]
        gt_step = float(np.mean(gt_vals)) if gt_vals else float("nan")
        diff = seam_off - seam_on
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
        print(f"  {len(pairs)} pairs, indices "
              f"{[int(fa['index']) for fa, _ in pairs]}, n_avg={args.n_avg}. PAIRED: both "
              f"conditions see the same observations AND the same noise seeds, so the only "
              f"difference is the guidance.")
        print(f"    RTC off : mean {seam_off.mean():7.2f} ticks  per pair "
              f"{np.round(seam_off, 1).tolist()}")
        print(f"    RTC on  : mean {seam_on.mean():7.2f} ticks  per pair "
              f"{np.round(seam_on, 1).tolist()}")
        print(f"    paired reduction {diff.mean():+.2f} +/- {se:.2f} ticks "
              f"({100.0 * diff.mean() / seam_off.mean():+.0f}%), better on "
              f"{int((diff > 0).sum())}/{len(diff)} pairs")
        print(f"    ground-truth one-step move across the same boundary: {gt_step:.2f} ticks "
              f"-- the seam is only pathological in excess of the motion that should happen")
        print(f"    median inference_ms: off {ms_off:.1f} -> on {ms_on:.1f} "
              f"({ms_on / ms_off:.2f}x; RTC adds an autograd pass per denoising step)")
        print(f"    SIDE EFFECT -- first_step_jump_ticks (|present qpos - chunk[0]|): "
              f"off {jump_off.mean():.1f} -> on {jump_on.mean():.1f} ticks. RTC pins step 0 to "
              f"the OLD plan's next step, so the jump from the measured present position grows "
              f"by exactly as much as the old plan was wrong. In THIS test the 'arm' is the "
              f"dataset, which never executed the policy's plan, so that error is the full "
              f"open-loop prediction error; in a closed loop the arm is at the old plan's step "
              f"{server.emit_steps - 1} and the two collapse onto each other. Watch "
              f"first_step_jump_ticks / n_steps_over_rate_limit on the first real rollout.")
        print(f"    GPU memory across the {len(pairs) * 2} guided requests: "
              f"{mem0 / 2**20:.1f} -> {mem1 / 2**20:.1f} MiB "
              f"({(mem1 - mem0) / 2**20:+.1f} MiB)")
        check("(e) the guided path retains no autograd graph between requests "
              "(<32 MiB growth over the run)",
              (mem1 - mem0) < 32 * 2**20, f"{(mem1 - mem0) / 2**20:+.1f} MiB")
        check("(e) RTC reduces the re-plan seam on the same observations",
              float(seam_on.mean()) < float(seam_off.mean()),
              f"{seam_on.mean():.2f} < {seam_off.mean():.2f} ticks")
        check("(e) every RTC-on pair really ran guided", all(used_on), str(used_on))
        check("(e) every RTC-off pair really ran unguided", not any(used_off), str(used_off))

        # --- (h0) how far does the Coda correction AE move step 0? (no RTC involved) ------
        # This is the ceiling on what any --coda-rtc placement can change: with RTC on the main
        # stage, step 0 of stage 1 is pinned to the old plan to ~1% (pull per Euler step =
        # max_guidance_weight / num_steps, linearised in configure_coda_rtc's docstring), so the
        # seam that survives is exactly the correction AE's own displacement D.
        rows = []
        r_coda_on = None
        if _has_coda:
            print(f"\n  (h0) Coda displacement D, RTC OFF: how far the correction AE moves "
                  f"step 0 of the served chunk. Same pairs, same seeds, ext.coda toggled in "
                  f"place -- the ceiling on what any placement can do.")
            server.configure_rtc(enabled=False)
            server.configure_coda_rtc("off")
            r_coda_on = _seam_run()
            _keep_coda = server.policy.model._ext.coda
            try:
                server.policy.model._ext.coda = False
                r_coda_off = _seam_run()
            finally:
                server.policy.model._ext.coda = _keep_coda
            D = float(np.abs(r_coda_on["step0"] - r_coda_off["step0"]).mean())
            d_seam, se_seam, better, n = _paired(r_coda_off["seam"], r_coda_on["seam"])
            d_spr, se_spr, better_s, _ = _paired(r_coda_off["spread"], r_coda_on["spread"])
            print(f"    D = {D:.2f} ticks (mean over {n} pairs x {JOINT_DIM} motors)")
            print(f"    seam   : coda off {r_coda_off['seam'].mean():7.2f} -> on "
                  f"{r_coda_on['seam'].mean():7.2f} ticks "
                  f"(paired {d_seam:+.2f} +/- {se_seam:.2f}, better on {better}/{n})")
            print(f"    spread : coda off {r_coda_off['spread'].mean():7.2f} -> on "
                  f"{r_coda_on['spread'].mean():7.2f} ticks "
                  f"(paired {d_spr:+.2f} +/- {se_spr:.2f}, better on {better_s}/{n})")
            print(f"    ms     : coda off {r_coda_off['ms']:.1f} -> on {r_coda_on['ms']:.1f}")
            print(f"    READ THIS FIRST: if D is small next to the draw-noise floor, no "
                  f"placement below can be separated from another with {n} pairs.")
            rows += [{"arm": "coda_off_rtc_off", "coda_rtc": None, "coda": False,
                      "max_guidance_weight": None,
                      "seam_mean": float(r_coda_off["seam"].mean()),
                      "seam_per_pair": r_coda_off["seam"].round(3).tolist(),
                      "spread_mean": float(np.nanmean(r_coda_off["spread"])),
                      "jump_mean": float(r_coda_off["jump"].mean()),
                      "max_step_mean": float(r_coda_off["max_step"].mean()),
                      "n_over_rate_limit": r_coda_off["n_over"],
                      "n_clamped": r_coda_off["n_clamped"], "median_ms": r_coda_off["ms"]},
                     {"arm": "coda_on_rtc_off", "coda_rtc": "off", "coda": True,
                      "max_guidance_weight": None,
                      "seam_mean": float(r_coda_on["seam"].mean()),
                      "seam_per_pair": r_coda_on["seam"].round(3).tolist(),
                      "spread_mean": float(np.nanmean(r_coda_on["spread"])),
                      "jump_mean": float(r_coda_on["jump"].mean()),
                      "max_step_mean": float(r_coda_on["max_step"].mean()),
                      "n_over_rate_limit": r_coda_on["n_over"],
                      "n_clamped": r_coda_on["n_clamped"], "median_ms": r_coda_on["ms"],
                      "coda_displacement_D_ticks": D}]

        # --- (h) placement x guidance weight ---------------------------------------------
        modes = [m for m in str(args.self_test_coda_rtc_modes).split(",") if m.strip()]
        weights = [float(w) for w in str(args.self_test_rtc_weights).split(",") if w.strip()]
        if _has_coda and modes and weights:
            print(f"\n  (h) --coda-rtc placement x --rtc-max-guidance-weight, against the SAME "
                  f"RTC-off baseline as (e) (mean {seam_off.mean():.2f} ticks). Measurements are "
                  f"RECORDED, not asserted: with {len(pairs)} pairs a gate would be flaky. "
                  f"modes={modes} weights={weights}")
            print(f"    {'placement':<11}{'w':>6}{'pull/step':>11}{'seam':>9}{'paired':>18}"
                  f"{'better':>8}{'spread':>9}{'jump':>8}{'ms':>8}")
            mem_h0 = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
            for w in weights:
                for mode in modes:
                    _rtc_on(mode=mode, max_w=w)
                    r = _seam_run()
                    d, se_d, better, n = _paired(seam_off, r["seam"])
                    n_steps = (int(server.policy.config.num_steps) if mode == "main"
                               else int(server.policy.model._ext.coda_num_steps))
                    pull = w / n_steps
                    print(f"    {mode:<11}{w:>6g}{pull:>11.2f}{r['seam'].mean():>9.2f}"
                          f"{f'{d:+.2f}+/-{se_d:.2f}':>18}{f'{better}/{n}':>8}"
                          f"{np.nanmean(r['spread']):>9.2f}{r['jump'].mean():>8.1f}"
                          f"{r['ms']:>8.1f}")
                    check(f"(h) {mode} w={w:g}: every pair ran guided", all(r["used"]),
                          str(r["used"]))
                    # Clipping is NOT an RTC property: the open-loop RTC-off arms clip too
                    # (the dataset's own next observation is not where the plan pointed), so it
                    # is reported against those baselines instead of gated. MEASURED 2026-10-17:
                    # f500 coda-off 19 / coda-on 34 clamped values before any guidance.
                    check(f"(h) {mode} w={w:g}: finite output", bool(np.isfinite(r["seam"]).all()))
                    print(f"      clamped={r['n_clamped']} over_rate_limit={r['n_over']} "
                          f"max_step={r['max_step'].mean():.1f} ticks "
                          f"(RTC-off baseline: "
                          f"clamped={r_coda_on['n_clamped'] if r_coda_on else 'n/a'} "
                          f"over_rate_limit={r_coda_on['n_over'] if r_coda_on else 'n/a'})")
                    rows.append({"arm": f"coda_rtc_{mode}", "coda_rtc": mode, "coda": True,
                                 "max_guidance_weight": w, "pull_per_step": pull,
                                 "guided_steps": n_steps,
                                 "seam_mean": float(r["seam"].mean()),
                                 "seam_per_pair": r["seam"].round(3).tolist(),
                                 "paired_reduction_vs_rtc_off": d, "paired_se": se_d,
                                 "better_pairs": better, "n_pairs": n,
                                 "spread_mean": float(np.nanmean(r["spread"])),
                                 "jump_mean": float(r["jump"].mean()),
                                 "max_step_mean": float(r["max_step"].mean()),
                                 "n_over_rate_limit": r["n_over"],
                                 "n_clamped": r["n_clamped"], "median_ms": r["ms"]})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            mem_h1 = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
            check("(h) no autograd graph retained across the placement sweep (<32 MiB)",
                  (mem_h1 - mem_h0) < 32 * 2**20, f"{(mem_h1 - mem_h0) / 2**20:+.1f} MiB")
            print(f"    seam reference scales: RTC-off {seam_off.mean():.2f}, demonstrator's own "
                  f"one-step move {gt_step:.2f} ticks. A change inside the draw-noise band is "
                  f"not an improvement (see results/ext_eval/<tag>.json seam_noise_floor_ticks).")
            print(f"    NOTE the pairs are IN-SAMPLE and their locations depend on "
                  f"--rtc-seam-pairs / --frame-index, so absolute ticks are NOT comparable to "
                  f"results/ext_eval (18 held-out episodes, 36 pairs); only the paired "
                  f"differences within this run are.")
        if args.self_test_out and rows:
            import json as _json
            meta = {
                "checkpoint": server.model_path, "dataset_root": server.sidecar_root,
                "ext": _ext_st.summary() if _ext_st is not None else None,
                "num_steps": int(server.policy.config.num_steps),
                "coda_num_steps": (int(_ext_st.coda_num_steps) if _has_coda else None),
                "emit_steps": server.emit_steps, "chunk_size": server.chunk_size,
                "n_avg": args.n_avg, "rtc_execution_horizon": horizon_want,
                "rtc_inference_delay": args.rtc_inference_delay,
                "rtc_schedule": args.rtc_schedule,
                "rtc_seam_pairs": int(args.rtc_seam_pairs),
                "frame_index": args.frame_index,
                "pair_indices": [int(fa["index"]) for fa, _ in pairs],
                "gt_one_step_ticks": gt_step,
                "seam_rtc_off_mean": float(seam_off.mean()),
                "seam_rtc_off_per_pair": seam_off.round(3).tolist(),
                "rtc_on_default_placement_seam_mean": float(seam_on.mean()),
                "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": ("paired per-pair differences only; pairs are in-sample and move with "
                         "--rtc-seam-pairs/--frame-index. The RTC-on arms also drag the 26 "
                         "unused action channels toward 0 (modeling_rtc.py zero-pads the 6-wide "
                         "previous plan to 32) -- known limitation, same for every RTC run."),
                "arms": rows,
            }
            Path(args.self_test_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.self_test_out).write_text(_json.dumps(meta, indent=1))
            print(f"\n  wrote {len(rows)} arms -> {args.self_test_out}")

    # --- (g) the wire, with RTC on: the new fields must survive JSON --------------------
    _rtc_on()
    server.rtc_reset("self-test (g)")
    httpd2 = HTTPServer(("127.0.0.1", 0), make_stdlib_handler(server))
    base2 = f"http://127.0.0.1:{httpd2.server_address[1]}"
    th2 = threading.Thread(target=httpd2.serve_forever, daemon=True)
    th2.start()

    def post2(path, payload):
        req = urllib.request.Request(f"{base2}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode())

    try:
        with urllib.request.urlopen(f"{base2}/health", timeout=10) as r:
            h2 = json.loads(r.read().decode())
        check("(g) /health reports RTC on with its parameters",
              h2.get("rtc") is True and h2.get("rtc_execution_horizon") == horizon_want
              and h2.get("rtc_inference_delay") == args.rtc_inference_delay
              and h2.get("rtc_schedule") == args.rtc_schedule,
              f"{ {k: v for k, v in h2.items() if k.startswith('rtc')} }")
        code_g, _ = post2("/init", {"lang": args.instruction, "reasoning_cache_n": 0})
        check("(g) POST /init -> 200 with RTC on", code_g == 200)
        _, g1 = post2("/predict", {"images": images, "qpos": qpos})
        _, g2 = post2("/predict", {"images": images, "qpos": qpos})
        check("(g) over HTTP: first request unguided, second guided",
              g1.get("rtc_used_prev_chunk") is False and g2.get("rtc_used_prev_chunk") is True
              and g2.get("rtc_prev_chunk_steps") == want_tail,
              f"1st={g1.get('rtc_used_prev_chunk')} 2nd={g2.get('rtc_used_prev_chunk')} "
              f"steps={g2.get('rtc_prev_chunk_steps')}")
        check("(g) the RTC/seam fields survive JSON and the seam is a real number",
              isinstance(g2.get("seam_ticks_mean"), float)
              and isinstance(g2.get("seam_ticks_per_motor"), list)
              and len(g2["seam_ticks_per_motor"]) == JOINT_DIM
              and g2.get("rtc") is True,
              f"seam={g2.get('seam_ticks_mean')} per_motor={g2.get('seam_ticks_per_motor')}")
        gt = np.frombuffer(base64.b64decode(g2["joint_ticks_b64"]), dtype=np.int32)
        gt = gt.reshape(g2["chunk_size"], g2["joint_dim"])
        check("(g) the wire format is unchanged (int32 C-order, chunk_size x joint_dim)",
              gt.shape == (server.emit_steps, JOINT_DIM) and gt.dtype == np.int32
              and len(base64.b64decode(g2["joint_ticks_b64"]))
              == g2["chunk_size"] * g2["joint_dim"] * 4,
              f"{gt.shape} {gt.dtype}")
        check("(g) no debug keys leak with RTC on",
              not any(k in g2 for k in DEBUG_KEYS), str([k for k in DEBUG_KEYS if k in g2]))
    finally:
        httpd2.shutdown()
        httpd2.server_close()

    # --- (f) turning it back off restores the original path --------------------------------
    server.configure_rtc(enabled=False)
    server.rtc_reset("self-test (f)")
    out_back = _one(seed=20260816)
    check("(f) turning RTC back off restores the pre-RTC chunk exactly",
          bool(np.array_equal(out_back["_ticks_chunk"], off_ref)),
          f"max|d|={int(np.abs(out_back['_ticks_chunk'].astype(np.int64) - off_ref.astype(np.int64)).max())} ticks")
    st = server.rtc_status()
    check("(f) /health-style status reports RTC off again",
          st["rtc"] is False and st["rtc_execution_horizon"] is None, str(st))

    if args.diag_out:
        n = sum(1 for _ in open(args.diag_out)) if Path(args.diag_out).exists() else 0
        print(f"\n  probe wrote {n} JSONL record(s) to {args.diag_out}")
        check("probe produced records", n > 0)
    server.close()

    print(f"\n== self-test {'PASSED' if ok else 'FAILED'} ==")
    print("   (judge by these lines, not by the exit code)")
    return 0 if ok else 1


# ==========================================================================
# Entry
# ==========================================================================

def _parse_cam_indices(s: str | None) -> list[int] | None:
    if s is None:
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                    help="lerobot SmolVLA checkpoint dir (config.json + model.safetensors + "
                         "policy_pre/postprocessor*). Default = d2_unfrozen step 4000.")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                    help="dataset the policy was trained on; its openmani_meta.json is the "
                         "unit/space/motor-order contract and its meta/tasks.* is the language "
                         "distribution. A copy next to the checkpoint is used as a fallback.")
    ap.add_argument("--device", default="cuda",
                    help="with CUDA_VISIBLE_DEVICES=1, 'cuda' == physical GPU1")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--client-cams", type=int, default=DEFAULT_CLIENT_CAMS,
                    help=f"images the client sends per /predict (default {DEFAULT_CLIENT_CAMS})")
    ap.add_argument("--cam-indices", type=str, default=None,
                    help="comma-separated client slots feeding camera1,camera2,camera3 in that "
                         "order (default 0,1,2 == the converter's image_0/1/2 mapping)")
    ap.add_argument("--emit-steps", type=int, default=0,
                    help="chunk steps to return per request. 0 (default) = the checkpoint's "
                         "n_action_steps (8), i.e. the replan horizon it was configured with. "
                         "Max = chunk_size (16). The client must not execute more than this.")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="continue even if openmani_meta.json is missing or its "
                         "space/unit/motor_order disagree with the robot. UNSAFE.")
    ap.add_argument("--dry-run", action="store_true",
                    help="still runs inference, but every response carries execute=false so a "
                         "cooperating client will not write to the motors. Default OFF: this "
                         "server never touches the bus -- the client owns the writes -- so a "
                         "default-on dry run would only give a false sense of safety while the "
                         "existing clients (which do not read the flag) moved the arm anyway. "
                         "Real safety lives in Openmani4FollowerConfig.dry_run (default True).")
    ap.add_argument("--done-threshold", type=float, default=DEFAULT_DONE_THRESHOLD,
                    help=f"6-D checkpoints only. `done` turns true once the averaged done_flag "
                         f"over the emitted rows stays >= this for --done-consecutive requests "
                         f"in a row. Default {DEFAULT_DONE_THRESHOLD} is the midpoint of the "
                         f"{{0, 1}} training target and is ARBITRARY -- nothing has measured "
                         f"the operating point on a deployed policy yet. Ignored on a 5-D "
                         f"checkpoint, where `done` stays None.")
    ap.add_argument("--done-consecutive", type=int, default=DEFAULT_DONE_CONSECUTIVE,
                    help=f"how many consecutive /predict calls must be over --done-threshold "
                         f"before `done` is reported true (default {DEFAULT_DONE_CONSECUTIVE}). "
                         f"1 = fire on the first chunk. The streak resets on /init and on any "
                         f"call that falls below the threshold. ARBITRARY.")
    ap.add_argument("--n-avg", type=int, default=DEFAULT_N_AVG,
                    help=f"flow-matching draws averaged per /predict (default {DEFAULT_N_AVG}). "
                         f"The draws run as ONE batched forward pass and are averaged in TICK "
                         f"space, before rounding/clamping. 1 = the original single-draw "
                         f"behaviour, bit-for-bit. diag/sampling_noise.py measured spread ~1/"
                         f"sqrt(N) and gt error 48.4 -> 41.3 ticks at N=4, for 89.8 -> 112.2 ms.")
    ap.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=False,
                    help="Real-Time Chunking: guide each chunk with the UNEXECUTED TAIL of the "
                         "previous one so the two agree across the re-plan boundary (lerobot "
                         "policies/rtc). Inference-time only -- no retraining, same weights. OFF "
                         "by default; without it this server samples exactly as it did before "
                         "the flag existed (--self-test proves it byte for byte). Motivation: "
                         "the measured seam |chunk(t)[7]-chunk(t+8)[0]| is 57-65 ticks vs 9.34 "
                         "ticks for one real control step.")
    ap.add_argument("--rtc-execution-horizon", type=int, default=0,
                    help="steps of the new chunk over which the old plan still has a say. "
                         "0 (default) = --emit-steps, i.e. the rows the client executes; "
                         "RTCConfig's own default of 10 would disagree with this server's 8, so "
                         "it is never used. An explicit value that differs from --emit-steps "
                         "warns at startup.")
    ap.add_argument("--rtc-inference-delay", type=int, default=DEFAULT_RTC_INFERENCE_DELAY,
                    help=f"steps at the head of the chunk that are pinned HARD (weight 1.0) to "
                         f"the old plan, i.e. how many steps the client is assumed to execute "
                         f"while this request is in flight (default "
                         f"{DEFAULT_RTC_INFERENCE_DELAY}). Our client blocks on /predict, so the "
                         f"arm actually holds and the physical delay is 0; 1 is kept because "
                         f"pinning step 0 to the old plan's next step is exactly the seam this "
                         f"is meant to close. 0 leaves even step 0 free.")
    ap.add_argument("--rtc-schedule", choices=("ZEROS", "ONES", "LINEAR", "EXP"),
                    default=DEFAULT_RTC_SCHEDULE,
                    help=f"how the old plan's weight decays across the window "
                         f"(lerobot RTCAttentionSchedule; default {DEFAULT_RTC_SCHEDULE}). "
                         f"ZEROS = only the pinned head counts, ONES = the whole window is "
                         f"pinned, LINEAR/EXP decay to 0 at the end of the window.")
    ap.add_argument("--rtc-max-guidance-weight", type=float, default=None,
                    help=f"cap on the guidance gain, which otherwise diverges as the denoiser "
                         f"approaches t=0 (default {DEFAULT_RTC_MAX_GUIDANCE_WEIGHT:g}, "
                         f"RTCConfig's own default). On a Coda checkpoint it is lowered "
                         f"automatically to the guided stage's Euler-step count so the pull per "
                         f"step stays <= 1; giving this flag keeps your value instead.")
    ap.add_argument("--rtc-stale-reset-s", type=float, default=RTC_STALE_RESET_S,
                    help=f"[ARBITRARY] drop the remembered chunk when the previous request is "
                         f"older than this (default {RTC_STALE_RESET_S:g} s ~= two 0.97 s "
                         f"control cycles). Not measured or tuned; it exists so a paused or "
                         f"restarted client cannot inherit a plan from before the pause.")
    ap.add_argument("--coda-rtc", choices=("auto", "off", "main", "correction", "both"),
                    default="auto",
                    help="Coda checkpoints only (smolvla_ext.json coda=true). WHERE the RTC "
                         "guidance acts, since such a checkpoint samples in two stages: main AE "
                         "then correction AE. auto (default) = the measured placement "
                         "(correction, 2026-10-17, results/rtc_coda) when the checkpoint has "
                         "Coda and --rtc is on, else a no-op, so ./serve.sh needs no extra flag; "
                         "off = Coda + --rtc refuses, as it did up to 2026-10-16; "
                         "main = stage 1 only, so the correction AE afterwards moves the overlap "
                         "freely; correction = the executed chunk is what gets guided; both. "
                         "Inference only -- no retraining, same weights, and the value saved in "
                         "the sidecar is ignored. Pull per Euler step = "
                         "--rtc-max-guidance-weight / num_steps, printed at startup.")
    ap.add_argument("--rtc-seam-pairs", type=int, default=RTC_SEAM_TEST_PAIRS,
                    help=f"--self-test only: how many consecutive-observation pairs section 5(e) "
                         f"measures the seam on (default {RTC_SEAM_TEST_PAIRS}). A flag and not a "
                         f"constant because it decides the statistics -- and because the pair "
                         f"LOCATIONS depend on it (stride = n_frames / n_pairs from "
                         f"--frame-index), so two runs with different values are different "
                         f"frames, not more samples of the same ones.")
    ap.add_argument("--self-test-coda-rtc-modes", default="",
                    help="--self-test only: comma-separated --coda-rtc placements to sweep in "
                         "section 5(h) on a Coda checkpoint, e.g. main,correction,both. Empty "
                         "(default) skips the sweep.")
    ap.add_argument("--self-test-rtc-weights", default="",
                    help="--self-test only: comma-separated --rtc-max-guidance-weight values to "
                         "cross with the placements in 5(h). Empty (default) skips the sweep. "
                         "Pull per Euler step = weight / num_steps; > 1 overshoots the old plan.")
    ap.add_argument("--self-test-out", default=None,
                    help="--self-test only: write the 5(h0)/(h) arms as JSON here "
                         "(e.g. results/rtc_coda/<tag>_f<frame>.json).")
    ap.add_argument("--lang-snap", action=argparse.BooleanOptionalAction, default=False,
                    help="map a free-form instruction onto the nearest sentence the policy was "
                         "trained on, using THIS policy's own text model -- no second model is "
                         "loaded. Intended for the STT front end: speech rarely reproduces a "
                         "training sentence verbatim. Off by default because it silently changes "
                         "what the robot is told; the match, its similarity and its margin are "
                         "printed on every /init.")
    ap.add_argument("--lang-snap-mode", choices=["rules", "embed"], default="rules",
                    help="rules = fill verb/object/direction from keyword tables built out of "
                         "the training sentences themselves; it can say which slot it failed on. "
                         "embed = cosine retrieval with the policy's own text model, which "
                         "measured 58%% correct on paraphrases (mostly place/push confusion) and "
                         "is kept only for comparison.")
    ap.add_argument("--lang-snap-pool", choices=["last", "mean"], default="last",
                    help="embed mode only: which language positions form the sentence vector. "
                         "last = the final real token, the standard choice for a causal decoder; "
                         "mean = average over the sentence.")
    ap.add_argument("--lang-snap-min-sim", type=float, default=0.90,
                    help="[ARBITRARY, needs measuring] reject the match below this cosine "
                         "similarity. Calibrate with diag/lang_snap_eval.py before trusting it.")
    ap.add_argument("--lang-snap-min-margin", type=float, default=0.005,
                    help="[ARBITRARY, needs measuring] reject when the best and the runner-up "
                         "are this close. A sentence sitting between two instructions (\"push "
                         "the pen\" with no direction) would otherwise resolve on a coin flip.")
    ap.add_argument("--rate-limit-ticks", type=float, default=DEFAULT_RATE_LIMIT_TICKS,
                    help=f"robot-side per-step limit used ONLY to predict clipping; nothing is "
                         f"limited here (default {DEFAULT_RATE_LIMIT_TICKS:g} = "
                         f"Openmani4FollowerConfig.max_step_ticks)")
    ap.add_argument("--diag-out", default=None,
                    help="JSONL path for diag.probe records (input sanity, per-module stats, "
                         "dead-unit fraction, zero-pad integrity, predicted-chunk magnitudes)")
    ap.add_argument("--diag-every", type=int, default=10,
                    help="record every Nth /predict request (default 10)")
    ap.add_argument("--diag-expert-layers", type=int, nargs="*", default=[0, 1, 8, 15])
    ap.add_argument("--attn-out", default=None,
                    help="directory for attention heatmaps (diag/attention_map.py). OFF by "
                         "default: every record costs a SECOND forward pass.")
    ap.add_argument("--attn-every", type=int, default=50,
                    help="record an attention map every Nth /predict request (default 50)")
    ap.add_argument("--instruction", default="place the tennis ball on the left",
                    help="instruction used by --self-test only")
    ap.add_argument("--frame-index", type=int, default=500,
                    help="dataset frame index compared against ground truth in --self-test")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic + real-frame + real-HTTP checks and exit")
    ap.add_argument("--http-backend", choices=("auto", "fastapi", "stdlib"), default="auto",
                    help="'auto' uses FastAPI+uvicorn when importable, else the stdlib "
                         "http.server (identical routes/JSON)")
    args = ap.parse_args()
    args.cam_indices = _parse_cam_indices(args.cam_indices)

    if args.rtc_max_guidance_weight is None:
        args.rtc_max_guidance_weight = DEFAULT_RTC_MAX_GUIDANCE_WEIGHT
        _rtc_weight_explicit = False
    else:
        _rtc_weight_explicit = True
    if args.self_test:
        sys.exit(self_test(args))

    # Choose the transport BEFORE the multi-GB model load, so a missing web framework fails
    # (or falls back) immediately instead of after the weights are on the GPU.
    backend = args.http_backend
    if backend in ("auto", "fastapi"):
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401

            backend = "fastapi"
        except Exception as e:
            if args.http_backend == "fastapi":
                raise SystemExit(f"--http-backend fastapi requested but not importable: {e}")
            print(f"[http] fastapi/uvicorn not importable ({type(e).__name__}); falling back to "
                  f"the stdlib http.server backend (identical routes/JSON).")
            backend = "stdlib"

    server = SmolvlaOpenmaniServer(
        model_path=args.model_path, dataset_root=args.dataset_root, device=args.device,
        cam_indices=args.cam_indices, client_cams=args.client_cams,
        emit_steps=args.emit_steps, allow_mismatch=args.allow_mismatch, dry_run=args.dry_run,
        n_avg=args.n_avg, rate_limit_ticks=args.rate_limit_ticks, diag_out=args.diag_out,
        diag_every=args.diag_every, diag_expert_layers=tuple(args.diag_expert_layers),
        attn_out=args.attn_out, attn_every=args.attn_every,
        rtc=args.rtc, rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_inference_delay=args.rtc_inference_delay, rtc_schedule=args.rtc_schedule,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_weight_explicit=_rtc_weight_explicit,
        rtc_stale_reset_s=args.rtc_stale_reset_s,
        coda_rtc=args.coda_rtc,
        done_threshold=args.done_threshold,
        done_consecutive=args.done_consecutive,
    )
    server.lang_snap = bool(args.lang_snap)
    server.lang_snap_mode = args.lang_snap_mode
    server.lang_snap_pool = args.lang_snap_pool
    server.lang_snap_min_sim = float(args.lang_snap_min_sim)
    server.lang_snap_min_margin = float(args.lang_snap_min_margin)
    if server.lang_snap and server.lang_snap_mode == "embed":
        server.build_lang_index()
    print(f"Ready. joint_dim={JOINT_DIM} chunk={server.emit_steps}/{server.chunk_size} "
          f"cameras={server.image_keys} device={args.device} http_backend={backend} "
          f"dry_run={server.dry_run} rtc={server.rtc is not None}")
    print("Reminder: this server does NOT rate limit. Keep the robot's max_step_ticks active "
          "and watch `max_step_ticks_in_chunk` in the responses.")
    try:
        if backend == "fastapi":
            import uvicorn

            uvicorn.run(build_app(server), host=args.host, port=args.port, log_level="warning")
        else:
            serve_stdlib(server, args.host, args.port)
    finally:
        server.close()


if __name__ == "__main__":
    main()
