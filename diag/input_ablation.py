#!/usr/bin/env python3
"""How much does each INPUT actually move the SmolVLA policy's output?

Offline loss cannot answer this: a policy that ignores the image still fits the training set,
because at 10 Hz the *state alone* predicts the next joint target very well.

WHAT CHANGED (2026-08-16) AND WHY
---------------------------------
The first version of this script reported `mean|pred_ablated - pred_baseline|` in raw ticks and
divided it by the sampling noise floor. Adversarial review showed that headline number is unfair
to vision in three separate ways; all three are fixed here.

1. ACTIONS ARE ABSOLUTE TICK TARGETS, SO THE STATE PASSES STRAIGHT THROUGH.
   Measured on this dataset: mean|action_t - state_t| = 9.6 ticks vs action std ~380 ticks. A
   policy that only copied its own state would already score a huge "state effect", because
   moving the state input moves the output almost one-for-one -- that is arithmetic, not
   evidence of state *reliance* in any interesting sense.
   FIX: every distance is now measured in RESIDUAL SPACE,
        R = pred - state_that_this_condition_actually_fed,
   so the identity component cancels for the state conditions and nothing changes for the image
   and language conditions (they feed the real state, so it cancels in the difference anyway).
   The raw numbers are still reported next to the residual ones, so the size of the identity
   component is visible instead of hidden.

2. ONLY THE IMAGE ABLATIONS WERE OUT-OF-DISTRIBUTION.
   `img_gray` / `img_black` feed a constant image the encoder never saw; a saturated encoder then
   *under*-reports vision. The state ablation meanwhile used the training mean, which is in
   distribution. That asymmetry alone can invert the ranking.
   FIX: the headline conditions are now SWAPS -- real images / a real state taken from a
   different episode. Same marginal distribution, wrong content. `img_gray`, `img_black` and
   `state_zero` are kept as SATURATION CONTROLS, printed separately, and are not the headline.

3. THE NOISE FLOOR IS NOT A DETECTION LIMIT.
   Every comparison here is PAIRED (the same flow-matching noise tensor goes into both sides), so
   the measurement error of a difference is exactly 0.0 -- verified in --self-test. The floor is
   the model's own sampling spread, which is a property worth reporting but is not the
   denominator that says whether an input matters.
   FIX: the primary denominator is SIGNAL = mean|pred_baseline - state|, i.e. how much commanded
   motion the policy actually asks for. The floor is kept as its own column.

There is no threshold in this script that converts a ratio into a verdict. Ratios are reported;
the interpretation is left to whoever reads them.

Conditions
  headline (in-distribution perturbations)
    state_swap    state   <- the REAL state of a frame from a different episode
    state_mean    state   <- the training-set mean (in-distribution "I don't know where I am")
    img_swap      all cams<- the REAL frames of that other frame
    img_swap_camN one camera at a time <- that other frame's camera N
    lang_swap     instruction <- a different training instruction
    lang_empty    instruction <- ""
  saturation controls (OUT of distribution; a small number here is weak evidence)
    state_zero    state <- literal 0 ticks (not a reachable pose)
    img_gray      all cams <- uint8 128. NOTE: after the [0,1]->[-1,1] shift this is
                  128/255*2-1 = +0.0039, i.e. near zero but NOT exactly 0.0.
    img_black     all cams <- 0 -> -1.0, which equals the value resize_with_pad already writes
                  into the pad band, so it is at least a "seen" value.
    img_gray_camN one camera at a time (not in the default set; use --conditions)

Motion stratification
  A grasp lives in the few frames where the arm actually moves. Results are therefore reported
  for three strata (thresholds printed, all cutoffs ARBITRARY and flagged as such):
    ALL    evenly spaced frames across the dataset (representative)
    STILL  those of ALL whose 16-step ground-truth displacement is below the dataset median
    MOVE   a separate targeted sample from the global top --move-pct of that displacement

Modes
  (default)      ablation table over --n-frames + --n-move-frames frames
  --lang-sweep   one frame, every training instruction, sorted by how far it moves the chunk
  --temporal     receding-horizon consistency: chunk(t)[k:] vs chunk(t+k)[:-k], k=--exec-steps
  --self-test    determinism + wiring checks on 2 frames, then a small ablation table

    CUDA_VISIBLE_DEVICES=0 python -m diag.input_ablation --self-test
    CUDA_VISIBLE_DEVICES=0 python -m diag.input_ablation --n-frames 128 --n-move-frames 48
    CUDA_VISIBLE_DEVICES=0 python -m diag.input_ablation --lang-sweep --frame-index 500
    CUDA_VISIBLE_DEVICES=0 python -m diag.input_ablation --temporal --n-frames 24

Results are ALWAYS written to JSON (--out, default runs/diag/<mode>_<stamp>.json; --no-out to
skip). An earlier measurement was lost because the artifact was optional.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from diag.smolvla_introspect import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH,
    LoadedPolicy,
    dataset_frame,
    load_policy,
    make_noise,
    predict_ticks,
)

MOTOR_LABELS = ["id11", "id22", "id23", "id24", "id25(grip)"]

# in-distribution perturbations -- the headline table
HEADLINE_CONDITIONS = (
    "state_swap", "state_mean",
    "img_swap", "img_swap_cam1", "img_swap_cam2", "img_swap_cam3",
    "lang_swap", "lang_empty",
)
# out-of-distribution constants -- reported, but as controls
CONTROL_CONDITIONS = ("state_zero", "img_gray", "img_black")
DEFAULT_CONDITIONS = HEADLINE_CONDITIONS + CONTROL_CONDITIONS
# everything `apply_condition` understands (per-camera gray is available but off by default)
ALL_CONDITIONS = DEFAULT_CONDITIONS + ("img_gray_cam1", "img_gray_cam2", "img_gray_cam3")

# --- ARBITRARY CHOICES (all printed at runtime) ---------------------------------------------
MOVE_PCT_DEFAULT = 5.0      # MOVE stratum = global top 5% of 16-step gt displacement
STILL_PCT_DEFAULT = 50.0    # STILL stratum = below the global median displacement
SWAP_OFFSET_FRAC = 0.5      # swap partner starts at index + n//2, then walks to a new episode


# ==========================================================================
# State reference values
# ==========================================================================

def training_state_mean(lp: LoadedPolicy) -> np.ndarray:
    """MEAN_STD mean of observation.state, in ticks. Feeding this == feeding normalized zero."""
    for s in getattr(lp.preprocessor, "steps", []):
        st = (getattr(s, "_tensor_stats", {}) or {}).get("observation.state")
        if st and "mean" in st:
            return st["mean"].detach().float().cpu().numpy().reshape(-1).astype(np.float64)
    raise RuntimeError("no observation.state MEAN_STD stats in the preprocessor; the "
                       "'state_mean' ablation has no defined value. Refusing to guess.")


# ==========================================================================
# Dataset-level motion index (read from the parquet, no video decoding)
# ==========================================================================

@dataclass
class MotionIndex:
    """Per-frame 16-step ground-truth displacement for the WHOLE dataset.

    motion[i] = mean over the 5 motors of |action[min(i+H-1, episode_end)] - action[i]| in ticks.
    Read straight from meta/data parquet, so building it costs no video decoding. The value is
    cross-checked against the real `gt_chunk` for every frame the ablation actually visits.

    `identity_gap[i] = mean_j |action[i,j] - state[i,j]|` is the same-timestep gap. It is the
    measured reason the raw state ablation over-reports: the label is an ABSOLUTE tick target
    that sits almost on top of the state input.
    """
    episode: np.ndarray
    motion: np.ndarray
    identity_gap: np.ndarray
    horizon: int
    n: int

    @staticmethod
    def build(dataset_root: str, horizon: int) -> "MotionIndex":
        import pandas as pd

        root = Path(dataset_root).expanduser()
        files = sorted((root / "data").rglob("*.parquet"))
        if not files:
            raise RuntimeError(f"no data parquet under {root / 'data'}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "index" in df.columns:
            gi = df["index"].to_numpy()
            if not np.array_equal(gi, np.arange(len(df))):
                df = df.sort_values("index", kind="stable").reset_index(drop=True)
                if not np.array_equal(df["index"].to_numpy(), np.arange(len(df))):
                    raise RuntimeError(
                        "the parquet 'index' column is not 0..n-1 even after sorting; the "
                        "row<->dataset-index identity that this motion index relies on does "
                        "not hold. Refusing to guess.")
        ep = df["episode_index"].to_numpy().astype(np.int64)
        act = np.stack(df["action"].to_numpy()).astype(np.float64)
        sta = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        n = len(df)
        # last row of each episode, per frame (episodes are contiguous blocks in v3 parquet)
        last_of_ep = np.zeros(n, dtype=np.int64)
        bounds = np.flatnonzero(np.diff(ep) != 0)
        starts = np.concatenate([[0], bounds + 1])
        ends = np.concatenate([bounds, [n - 1]])
        for s, e in zip(starts, ends):
            last_of_ep[s:e + 1] = e
        j = np.minimum(np.arange(n) + horizon - 1, last_of_ep)
        motion = np.abs(act[j] - act).mean(axis=1)
        identity_gap = np.abs(act - sta).mean(axis=1)
        return MotionIndex(episode=ep, motion=motion, identity_gap=identity_gap,
                           horizon=horizon, n=n)

    def pct(self, p: float) -> float:
        return float(np.percentile(self.motion, p))

    def top_indices(self, pct: float, k: int) -> list[int]:
        """`k` frames spread evenly (by index) through the global top `pct`% of motion."""
        thr = self.pct(100.0 - pct)
        pool = np.flatnonzero(self.motion >= thr)
        if len(pool) == 0:
            raise RuntimeError(f"no frames above the {100 - pct:.1f}th motion percentile")
        take = np.linspace(0, len(pool) - 1, num=min(k, len(pool))).round().astype(int)
        return [int(pool[t]) for t in np.unique(take)]

    def swap_partner(self, idx: int) -> int:
        """A REAL frame from a DIFFERENT episode. Deterministic: idx + n//2, then walk forward."""
        j = (idx + int(self.n * SWAP_OFFSET_FRAC)) % self.n
        for _ in range(self.n):
            if self.episode[j] != self.episode[idx]:
                return int(j)
            j = (j + 1) % self.n
        raise RuntimeError("the dataset appears to contain a single episode; no swap partner")


# ==========================================================================
# Condition builder
# ==========================================================================

def apply_condition(name: str, frames: list[np.ndarray], state: np.ndarray, task: str,
                    lp: LoadedPolicy, state_mean: np.ndarray, swap_task: str | None,
                    swap_frames: list[np.ndarray] | None = None,
                    swap_state: np.ndarray | None = None,
                    ) -> tuple[list[np.ndarray], np.ndarray, str]:
    f = [x.copy() for x in frames]
    s = state.copy()
    t = task

    def need_swap_obs():
        if swap_frames is None or swap_state is None:
            raise ValueError(f"{name}: needs a real partner frame (swap_frames/swap_state); the "
                             f"caller did not provide one.")

    if name == "baseline":
        pass
    elif name == "state_mean":
        s = state_mean.copy()
    elif name == "state_zero":
        s = np.zeros_like(state)
    elif name == "state_swap":
        need_swap_obs()
        s = np.asarray(swap_state, dtype=np.float64).copy()
    elif name == "img_swap":
        need_swap_obs()
        f = [x.copy() for x in swap_frames]
    elif name.startswith("img_swap_cam"):
        need_swap_obs()
        i = int(name[len("img_swap_cam"):]) - 1
        if not 0 <= i < len(f):
            raise ValueError(f"{name}: this rig has {len(f)} cameras")
        f[i] = swap_frames[i].copy()
    elif name == "img_gray":
        f = [np.full_like(x, 128) for x in f]
    elif name == "img_black":
        f = [np.zeros_like(x) for x in f]
    elif name.startswith("img_gray_cam"):
        i = int(name[len("img_gray_cam"):]) - 1
        if not 0 <= i < len(f):
            raise ValueError(f"{name}: this rig has {len(f)} cameras")
        f[i] = np.full_like(f[i], 128)
    elif name == "lang_empty":
        t = ""
    elif name == "lang_swap":
        if not swap_task:
            raise ValueError("lang_swap needs another training instruction (--dataset-root must "
                             "carry meta/tasks, or pass --swap-task)")
        t = swap_task
    else:
        raise ValueError(f"unknown condition {name!r}")
    return f, s, t


def condition_state(name: str, state: np.ndarray, state_mean: np.ndarray,
                    swap_state: np.ndarray | None) -> np.ndarray:
    """The state this condition ACTUALLY feeds -- the reference for its residual."""
    if name == "state_mean":
        return state_mean
    if name == "state_zero":
        return np.zeros_like(state)
    if name == "state_swap":
        if swap_state is None:
            raise ValueError("state_swap without a partner state")
        return np.asarray(swap_state, dtype=np.float64)
    return state


# ==========================================================================
# Distance metrics
# ==========================================================================

def pair_metrics(pred_a: np.ndarray, state_a: np.ndarray,
                 pred_b: np.ndarray, state_b: np.ndarray, lp: LoadedPolicy) -> dict:
    """Two [T,5] tick chunks + the state each was conditioned on -> distances.

    `resid_*` is the headline: |(pred_a - state_a) - (pred_b - state_b)|. For any condition that
    leaves the state alone this is identical to the raw difference; for the state conditions it
    removes the part of the "effect" that is just the absolute-target identity map.
    """
    ra = pred_a - np.asarray(state_a, dtype=np.float64)[None, :]
    rb = pred_b - np.asarray(state_b, dtype=np.float64)[None, :]
    dr = np.abs(ra - rb)
    dw = np.abs(pred_a - pred_b)
    return {
        "resid_mean_abs_ticks": dr.mean(axis=0).tolist(),
        "resid_max_abs_ticks": dr.max(axis=0).tolist(),
        "resid_mean_abs_ticks_all": float(dr.mean()),
        "resid_mean_abs_sigma_all": float((dr / lp.action_std[None, :]).mean()),
        "raw_mean_abs_ticks": dw.mean(axis=0).tolist(),
        "raw_mean_abs_ticks_all": float(dw.mean()),
        "raw_max_abs_ticks_all": float(dw.max()),
    }


def signal_metrics(pred: np.ndarray, state: np.ndarray, lp: LoadedPolicy) -> dict:
    """SIGNAL = |pred - state|: the commanded motion, i.e. the thing an ablation must move."""
    r = np.abs(pred - np.asarray(state, dtype=np.float64)[None, :])
    return {"signal_mean_abs_ticks": r.mean(axis=0).tolist(),
            "signal_mean_abs_ticks_all": float(r.mean()),
            "signal_max_abs_ticks_all": float(r.max())}


def _agg(records: list[dict], key: str) -> np.ndarray:
    return np.asarray([r[key] for r in records], dtype=np.float64)


# ==========================================================================
# Core: ablation table
# ==========================================================================

def run_ablation(lp: LoadedPolicy, indices: list[int], conditions: list[str], dataset_root: str,
                 n_noise: int = 8, seed: int = 0, swap_task: str | None = None,
                 mi: MotionIndex | None = None, strata: dict[str, list[int]] | None = None,
                 meta: dict | None = None, verbose: bool = True) -> dict:
    """Per-frame paired ablation. `strata` maps a stratum name -> the subset of `indices`."""
    state_mean = training_state_mean(lp)
    chunk_size = int(lp.cfg.chunk_size)
    if mi is None:
        mi = MotionIndex.build(dataset_root, chunk_size)
    per_frame: list[dict] = []
    t_start = time.time()

    for n, idx in enumerate(indices):
        d = dataset_frame(dataset_root, idx, chunk_size)
        frames, state, task = d["frames"], d["state"], d["task"]
        pidx = mi.swap_partner(d["index"])
        dp = dataset_frame(dataset_root, pidx, chunk_size)
        st = swap_task
        if st is None and lp.tasks:
            others = [t for t in lp.tasks if t != task]
            st = others[idx % len(others)] if others else None

        noise = make_noise(lp, seed + n)
        base, _ = predict_ticks(lp, frames, state, task, noise=noise)

        rec: dict = {
            "index": int(d["index"]), "task": task,
            "state_ticks": state.astype(int).tolist(),
            "swap_index": int(dp["index"]), "swap_episode": int(mi.episode[pidx]),
            "episode": int(mi.episode[d["index"]]),
            "swap_task_used": st,
            # motion recomputed from the REAL gt chunk; the parquet index is only for selection
            "motion_ticks": float(np.abs(d["gt_chunk"][-1] - d["gt_chunk"][0]).mean()),
            "motion_ticks_parquet": float(mi.motion[d["index"]]),
            "signal": signal_metrics(base, state, lp),
            "floor": [], "cond": {},
        }
        # noise floor: identical input, different flow-matching draw.
        for j in range(n_noise):
            alt, _ = predict_ticks(lp, frames, state, task,
                                   noise=make_noise(lp, 10_000 + 97 * n + j))
            rec["floor"].append(pair_metrics(base, state, alt, state, lp))
        for c in conditions:
            f2, s2, t2 = apply_condition(c, frames, state, task, lp, state_mean, st,
                                         swap_frames=dp["frames"], swap_state=dp["state"])
            other, _ = predict_ticks(lp, f2, s2, t2, noise=noise)
            sref = condition_state(c, state, state_mean, dp["state"])
            rec["cond"][c] = pair_metrics(base, state, other, sref, lp)
        per_frame.append(rec)
        if verbose:
            print(f"  [{n + 1}/{len(indices)}] frame {d['index']} ep{rec['episode']} "
                  f"motion={rec['motion_ticks']:.1f} swap<-{dp['index']} {task!r}", flush=True)

    if strata is None:
        strata = {"ALL": list(indices)}
    by_index = {r["index"]: r for r in per_frame}
    out = {
        "model_path": lp.model_path,
        "dataset_root": str(dataset_root),
        "n_frames_total": len(per_frame),
        "conditions": list(conditions),
        "headline_conditions": [c for c in conditions if c in HEADLINE_CONDITIONS],
        "control_conditions": [c for c in conditions if c in CONTROL_CONDITIONS],
        "state_mean_ticks": state_mean.round(1).tolist(),
        "action_std_ticks": lp.action_std.round(2).tolist(),
        "num_steps": int(lp.cfg.num_steps),
        "n_noise_draws_per_frame": n_noise,
        "motion_stat": ("mean over 5 motors of |gt_action[t+15] - gt_action[t]| in ticks "
                        "(clamped at the episode end)"),
        "dataset_identity_gap_ticks": float(mi.identity_gap.mean()),
        "dataset_identity_gap_over_action_std": float(
            mi.identity_gap.mean() / lp.action_std.mean()),
        "motion_parquet_vs_gt_max_abs_diff": float(max(
            abs(r["motion_ticks"] - r["motion_ticks_parquet"]) for r in per_frame)),
        "elapsed_s": round(time.time() - t_start, 1),
        "frames": per_frame,
        "strata": {},
    }
    if meta:
        out.update(meta)
    for sname, sidx in strata.items():
        recs = [by_index[i % mi.n] for i in sidx if (i % mi.n) in by_index]
        if not recs:
            continue
        out["strata"][sname] = _summarize(recs, conditions, lp)
    return out


def _summarize(recs: list[dict], conditions: list[str], lp: LoadedPolicy) -> dict:
    sig = float(np.mean([r["signal"]["signal_mean_abs_ticks_all"] for r in recs]))
    sig_pc = np.mean([r["signal"]["signal_mean_abs_ticks"] for r in recs], axis=0)
    fl_all = [f for r in recs for f in r["floor"]]
    floor = float(np.mean([f["resid_mean_abs_ticks_all"] for f in fl_all])) if fl_all else float("nan")
    floor_pc = (np.mean([f["resid_mean_abs_ticks"] for f in fl_all], axis=0).tolist()
                if fl_all else [float("nan")] * len(MOTOR_LABELS))
    out = {
        "n_frames": len(recs),
        "frame_indices": [r["index"] for r in recs],
        "motion_ticks_mean": float(np.mean([r["motion_ticks"] for r in recs])),
        "motion_ticks_min": float(np.min([r["motion_ticks"] for r in recs])),
        "motion_ticks_max": float(np.max([r["motion_ticks"] for r in recs])),
        "signal_mean_abs_ticks_all": sig,
        "signal_mean_abs_ticks": sig_pc.tolist(),
        "noise_floor": {"resid_mean_abs_ticks_all": floor,
                        "resid_mean_abs_ticks": floor_pc,
                        "n_pairs": len(fl_all),
                        "over_signal": floor / sig if sig > 0 else float("nan"),
                        "what": "same input, different flow-matching noise draw"},
        "conditions": {},
    }
    for c in conditions:
        cr = [r["cond"][c] for r in recs if c in r["cond"]]
        if not cr:
            continue
        m = _agg(cr, "resid_mean_abs_ticks_all")
        out["conditions"][c] = {
            "resid_mean_abs_ticks_all": float(m.mean()),
            "resid_std_over_frames": float(m.std()),
            "resid_mean_abs_ticks": _agg(cr, "resid_mean_abs_ticks").mean(axis=0).tolist(),
            "resid_max_abs_ticks": _agg(cr, "resid_max_abs_ticks").max(axis=0).tolist(),
            "resid_mean_abs_sigma_all": float(_agg(cr, "resid_mean_abs_sigma_all").mean()),
            "raw_mean_abs_ticks_all": float(_agg(cr, "raw_mean_abs_ticks_all").mean()),
            "over_signal": float(m.mean() / sig) if sig > 0 else float("nan"),
            "over_noise_floor": float(m.mean() / floor) if floor > 0 else float("nan"),
            "is_control": c in CONTROL_CONDITIONS,
        }
    return out


def print_ablation(res: dict) -> None:
    print()
    print(f"  model      {res['model_path']}")
    print(f"  num_steps  {res.get('num_steps')}   frames measured {res['n_frames_total']}   "
          f"noise draws/frame {res.get('n_noise_draws_per_frame')}   "
          f"elapsed {res.get('elapsed_s')}s")
    print(f"  motion stat: {res['motion_stat']}")
    print(f"  (parquet-vs-gt motion cross-check, max |diff| = "
          f"{res.get('motion_parquet_vs_gt_max_abs_diff', float('nan')):.3f} ticks)")
    if "dataset_identity_gap_ticks" in res:
        print(f"  dataset-wide mean|action_t - state_t| = "
              f"{res['dataset_identity_gap_ticks']:.2f} ticks = "
              f"{100 * res['dataset_identity_gap_over_action_std']:.1f}% of the mean action std "
              f"({np.mean(res['action_std_ticks']):.0f}) -> the label is an absolute target that "
              f"sits on the state,\n  which is why the RAW state ablation over-reports and why "
              f"the residual column exists.")
    for ch in res.get("arbitrary_choices", []):
        print(f"  [ARBITRARY] {ch}")
    for sname, s in res["strata"].items():
        _print_stratum(sname, s, res)
    print("\n  columns")
    print("    resid|d|  mean|(pred_c - state_c) - (pred_0 - state_0)| in ticks   <- headline")
    print("    raw|d|    mean|pred_c - pred_0| (what the old script reported)")
    print("    /signal   resid|d| divided by signal = mean|pred_0 - state_0| (commanded motion)")
    print("    /floor    resid|d| divided by the same-input different-noise spread")
    print("  All comparisons are PAIRED (identical noise tensor on both sides), so the "
          "measurement\n  error of each difference is 0; the floor is the model's own sampling "
          "spread, not a\n  detection limit.")


def _print_stratum(sname: str, s: dict, res: dict) -> None:
    fl = s["noise_floor"]["resid_mean_abs_ticks_all"]
    sig = s["signal_mean_abs_ticks_all"]
    desc = res.get("stratum_desc", {}).get(sname, "")
    print(f"\n  === {sname} ===  n_frames={s['n_frames']}  {desc}")
    print(f"      gt 16-step motion in this stratum: mean {s['motion_ticks_mean']:.1f} "
          f"[{s['motion_ticks_min']:.1f} .. {s['motion_ticks_max']:.1f}] ticks")
    print(f"      SIGNAL mean|pred-state| = {sig:8.2f} ticks   per-ch "
          + " ".join(f"{x:7.1f}" for x in s["signal_mean_abs_ticks"]))
    print(f"      FLOOR  same-input jitter = {fl:8.2f} ticks   per-ch "
          + " ".join(f"{x:7.1f}" for x in s["noise_floor"]["resid_mean_abs_ticks"])
          + f"   ({s['noise_floor']['n_pairs']} pairs, {s['noise_floor']['over_signal']:.2f}x signal)")
    hdr = (f"      {'condition':<15s} {'resid|d|':>9s} {'raw|d|':>9s} {'/signal':>8s} "
           f"{'/floor':>7s} {'+-fr':>7s}   per-channel resid|d| {MOTOR_LABELS}")
    for group, is_ctrl in (("headline (in-distribution perturbations)", False),
                           ("saturation controls (OUT of distribution -- weak evidence)", True)):
        rows = [(c, v) for c, v in s["conditions"].items() if v["is_control"] == is_ctrl]
        if not rows:
            continue
        print(f"    -- {group}")
        print(hdr)
        for c, v in sorted(rows, key=lambda kv: -kv[1]["resid_mean_abs_ticks_all"]):
            pc = " ".join(f"{x:7.1f}" for x in v["resid_mean_abs_ticks"])
            print(f"      {c:<15s} {v['resid_mean_abs_ticks_all']:9.2f} "
                  f"{v['raw_mean_abs_ticks_all']:9.2f} {v['over_signal']:8.2f} "
                  f"{v['over_noise_floor']:7.2f} {v['resid_std_over_frames']:7.2f}   {pc}")
    print(f"      {'(noise floor)':<15s} {fl:9.2f} {fl:9.2f} "
          f"{fl / sig if sig > 0 else float('nan'):8.2f} {1.0:7.2f}")
    if "state_zero" in s["conditions"]:
        print("      note: for state_zero the residual anchor is 0 ticks, which is not a "
              "reachable pose, so its\n            resid|d| is dominated by the anchor shift "
              "itself and is NOT interpretable. Read its raw|d|.")


# ==========================================================================
# Frame selection / strata
# ==========================================================================

def build_frame_plan(mi: MotionIndex, frame_index: int, n_frames: int, n_move: int,
                     move_pct: float, still_pct: float) -> tuple[list[int], dict, dict, list[str]]:
    n = mi.n
    if n_frames <= 1:
        allidx = [frame_index % n]
    else:
        step = max(1, n // n_frames)
        allidx = [(frame_index + i * step) % n for i in range(n_frames)]
    move_idx = mi.top_indices(move_pct, n_move) if n_move > 0 else []
    still_thr = mi.pct(still_pct)
    move_thr = mi.pct(100.0 - move_pct)
    still_idx = [i for i in allidx if mi.motion[i] <= still_thr]
    indices = list(dict.fromkeys(allidx + move_idx))
    strata = {"ALL": allidx, "STILL": still_idx, "MOVE": move_idx}
    desc = {
        "ALL": f"evenly spaced every {max(1, n // max(1, n_frames))} frames from #{frame_index}",
        "STILL": f"subset of ALL with motion <= p{still_pct:g} = {still_thr:.1f} ticks",
        "MOVE": f"targeted sample from the global top {move_pct:g}% "
                f"(motion >= p{100 - move_pct:g} = {move_thr:.1f} ticks)",
    }
    arb = [
        f"MOVE cutoff = global top {move_pct:g}% of the 16-step gt displacement "
        f"(>= {move_thr:.1f} ticks) -- ARBITRARY",
        f"STILL cutoff = below the p{still_pct:g} displacement ({still_thr:.1f} ticks) -- ARBITRARY",
        f"motion statistic uses the FIRST-to-LAST gt action of the 16-step chunk, mean over the "
        f"5 motors in raw ticks (so the large-range joints dominate) -- ARBITRARY",
        f"swap partner = index + n//2 walked forward to the first frame of a different episode "
        f"-- ARBITRARY",
    ]
    return indices, strata, desc, arb


# ==========================================================================
# Language sweep
# ==========================================================================

def run_lang_sweep(lp: LoadedPolicy, index: int, dataset_root: str, seed: int = 0,
                   n_noise: int = 8) -> dict:
    chunk_size = int(lp.cfg.chunk_size)
    d = dataset_frame(dataset_root, index, chunk_size)
    frames, state, task = d["frames"], d["state"], d["task"]
    if not lp.tasks:
        raise RuntimeError("no training task list found; --lang-sweep needs meta/tasks in "
                           "--dataset-root")
    noise = make_noise(lp, seed)
    base, _ = predict_ticks(lp, frames, state, task, noise=noise)
    sig = signal_metrics(base, state, lp)["signal_mean_abs_ticks_all"]
    floor = [pair_metrics(base, state,
                          predict_ticks(lp, frames, state, task,
                                        noise=make_noise(lp, 10_000 + j))[0], state, lp)
             for j in range(n_noise)]
    fl = float(_agg(floor, "resid_mean_abs_ticks_all").mean())
    rows = []
    for t in lp.tasks:
        pred, _ = predict_ticks(lp, frames, state, task=t, noise=noise)
        m = pair_metrics(base, state, pred, state, lp)
        rows.append({"task": t,
                     "resid_mean_abs_ticks_all": m["resid_mean_abs_ticks_all"],
                     "resid_mean_abs_ticks": m["resid_mean_abs_ticks"],
                     "over_signal": m["resid_mean_abs_ticks_all"] / sig if sig > 0 else float("nan"),
                     "over_noise_floor": m["resid_mean_abs_ticks_all"] / fl if fl > 0 else float("nan"),
                     "is_own_task": t == task})
    rows.sort(key=lambda r: -r["resid_mean_abs_ticks_all"])
    return {"frame": int(d["index"]), "own_task": task, "noise_floor_ticks": fl,
            "signal_ticks": sig, "n_tasks": len(lp.tasks), "num_steps": int(lp.cfg.num_steps),
            "rows": rows}


def print_lang_sweep(res: dict, top: int = 10) -> None:
    print(f"\n  frame {res['frame']}  own task = {res['own_task']!r}")
    print(f"  signal mean|pred-state| = {res['signal_ticks']:.2f} ticks; noise floor = "
          f"{res['noise_floor_ticks']:.2f} ticks; {res['n_tasks']} instructions swept")
    print(f"  {'resid|d|':>10s} {'/signal':>8s} {'/floor':>7s}  instruction")
    for r in res["rows"][:top]:
        mark = "  <- own" if r["is_own_task"] else ""
        print(f"  {r['resid_mean_abs_ticks_all']:10.2f} {r['over_signal']:8.2f} "
              f"{r['over_noise_floor']:7.2f}  {r['task']!r}{mark}")
    if len(res["rows"]) > top:
        print(f"  ... {len(res['rows']) - top} more (least-moving last)")
        r = res["rows"][-1]
        print(f"  {r['resid_mean_abs_ticks_all']:10.2f} {r['over_signal']:8.2f} "
              f"{r['over_noise_floor']:7.2f}  {r['task']!r}")
    allm = np.asarray([r["resid_mean_abs_ticks_all"] for r in res["rows"] if not r["is_own_task"]])
    print(f"  across instructions: mean {allm.mean():.2f}  median {np.median(allm):.2f}  "
          f"max {allm.max():.2f} ticks  ({allm.mean() / res['signal_ticks']:.2f}x signal)")


# ==========================================================================
# Receding-horizon consistency
# ==========================================================================

def run_temporal(lp: LoadedPolicy, indices: list[int], dataset_root: str, exec_steps: int = 8,
                 seed: int = 0, n_noise: int = 8, paired_noise: bool = True,
                 n_avg: int = 1, overlap_steps: int | None = None,
                 frame_getter=None) -> dict:
    """chunk(t)[k:] vs chunk(t+k)[:-k]: what the client would splice at a replan boundary.

    `paired_noise=True` feeds the SAME flow-matching draw to both calls, so the disagreement is
    caused only by the moved observation window -- the clean measurement of "did the plan
    change". `paired_noise=False` is what the robot actually experiences: the server draws fresh
    noise every replan, so the sampling spread is added on top.

    `n_avg > 1` averages that many independently-drawn chunks on each side, i.e. it measures the
    replan jitter of the sample-averaged policy (see diag/sampling_noise.py).

    Reported in RAW ticks, because the two chunks are absolute position targets for the SAME
    future timesteps -- the raw disagreement is literally what the arm feels. The residual
    version is reported next to it.

    `overlap_steps` truncates the compared overlap to a FIXED number of steps. The default
    (None) keeps the historical behaviour, chunk_size - exec_steps, which is chunk_size
    dependent and therefore NOT comparable across checkpoints trained with different
    chunk_size. `overlap_steps=0`, or a checkpoint with chunk_size == exec_steps, leaves no
    overlap at all: the overlap fields come back NaN and only the seam fields are defined.

    `seam_*` exists for EVERY chunk_size, including chunk_size == exec_steps. It is the command
    discontinuity the arm actually feels at a replan boundary -- |chunk(t)[exec-1] -
    chunk(t+exec)[0]|, two CONSECUTIVE timesteps issued by two different plans. The ground-truth
    one-step move over the same timestep pair is reported next to it, because a non-zero seam is
    only pathological in excess of the motion that was supposed to happen.

    `frame_getter(dataset_root, index, chunk_size) -> dict` overrides the frame source, so a
    caller can share one decoded-frame cache across several checkpoints (and thereby guarantee
    that different checkpoints are scored on bit-identical observations).
    """
    chunk_size = int(lp.cfg.chunk_size)
    if exec_steps < 1:
        raise ValueError(f"exec_steps {exec_steps} must be >= 1")
    if exec_steps > chunk_size:
        raise ValueError(f"--exec-steps {exec_steps} must be <= chunk_size {chunk_size}")
    max_ov = chunk_size - exec_steps
    ov_n = max_ov if overlap_steps is None else max(0, min(int(overlap_steps), max_ov))
    get = frame_getter or dataset_frame
    n_avg = max(1, int(n_avg))
    rows, floor, seam_floor = [], [], []

    def predict_avg(d, seeds):
        acc = None
        for sd in seeds:
            p, _ = predict_ticks(lp, d["frames"], d["state"], d["task"],
                                 noise=make_noise(lp, sd))
            acc = p if acc is None else acc + p
        return acc / len(seeds)

    for n, idx in enumerate(indices):
        d0 = get(dataset_root, idx, chunk_size)
        d1 = get(dataset_root, idx + exec_steps, chunk_size)
        # only comparable inside one episode; a task change means the window crossed a boundary.
        # episode_index is the exact test when the frame source provides it (dataset_frame does);
        # the task string alone cannot tell two episodes of the SAME task apart.
        same_ep = d0["task"] == d1["task"]
        if int(d0.get("episode_index", -1)) >= 0 and int(d1.get("episode_index", -1)) >= 0:
            same_ep = same_ep and int(d0["episode_index"]) == int(d1["episode_index"])
        # disjoint seed blocks so that "independent" noise is never accidentally the same draw
        seeds0 = [seed + 1_000_000 + 100 * n + a for a in range(n_avg)]
        seeds1 = seeds0 if paired_noise else [seed + 2_000_000 + 100 * n + a for a in range(n_avg)]
        c0 = predict_avg(d0, seeds0)
        c1 = predict_avg(d1, seeds1)
        row = {"index": int(d0["index"]), "same_episode_guess": bool(same_ep)}
        if ov_n > 0:
            ov0, ov1 = c0[exec_steps:exec_steps + ov_n], c1[:ov_n]
            row.update(pair_metrics(ov0, d0["state"], ov1, d1["state"], lp))
        # seam = the two consecutive commands that straddle the replan boundary. Defined even
        # when there is no overlap at all (chunk_size == exec_steps).
        sj = np.abs(c1[0] - c0[exec_steps - 1])
        row["seam_mean_abs_ticks"] = sj.tolist()
        row["seam_mean_abs_ticks_all"] = float(sj.mean())
        gt_ok = (not bool(d0["gt_is_pad"][exec_steps - 1])) and (not bool(d1["gt_is_pad"][0]))
        row["seam_gt_valid"] = bool(gt_ok)
        row["seam_gt_one_step_ticks"] = float(
            np.abs(d1["gt_chunk"][0] - d0["gt_chunk"][exec_steps - 1]).mean())
        # floor = same input, an independent draw (of the SAME n_avg estimator), so that with
        # n_avg > 1 the floor shrinks the same way the measurement does.
        for j in range(max(1, n_noise)):
            alt = predict_avg(d0, [seed + 3_000_000 + 1000 * n + 10 * j + a for a in range(n_avg)])
            if ov_n > 0:
                floor.append(pair_metrics(c0[exec_steps:exec_steps + ov_n], d0["state"],
                                          alt[exec_steps:exec_steps + ov_n], d0["state"], lp))
            seam_floor.append(float(np.abs(alt[exec_steps - 1] - c0[exec_steps - 1]).mean()))
        rows.append(row)
    keep = [r for r in rows if r["same_episode_guess"]] or rows
    nan = float("nan")
    out = {
        "exec_steps": exec_steps, "overlap_steps": ov_n,
        "overlap_steps_available": max_ov, "chunk_size": chunk_size,
        "paired_noise": bool(paired_noise), "num_steps": int(lp.cfg.num_steps),
        "n_avg": n_avg,
        "n_pairs": len(keep), "n_dropped_task_change": len(rows) - len(keep),
        "rows": rows,
    }
    if ov_n > 0:
        fl = float(_agg(floor, "raw_mean_abs_ticks_all").mean())
        mt = _agg(keep, "raw_mean_abs_ticks_all")
        out.update({
            "noise_floor_ticks": fl,
            "raw_mean_abs_ticks_all": float(mt.mean()), "std_over_pairs": float(mt.std()),
            "raw_mean_abs_ticks": _agg(keep, "raw_mean_abs_ticks").mean(axis=0).tolist(),
            "resid_mean_abs_ticks_all": float(_agg(keep, "resid_mean_abs_ticks_all").mean()),
            "max_abs_ticks_all": float(_agg(keep, "raw_max_abs_ticks_all").max()),
            "over_noise_floor": float(mt.mean() / fl) if fl > 0 else float("inf"),
        })
    else:
        out.update({
            "noise_floor_ticks": nan, "raw_mean_abs_ticks_all": nan, "std_over_pairs": nan,
            "raw_mean_abs_ticks": [nan] * len(MOTOR_LABELS),
            "resid_mean_abs_ticks_all": nan, "max_abs_ticks_all": nan,
            "over_noise_floor": nan,
            "no_overlap_reason": (f"chunk_size {chunk_size} - exec_steps {exec_steps} = 0 "
                                  f"overlapping steps: this checkpoint re-plans exactly when its "
                                  f"chunk runs out, so there is nothing to disagree about. Only "
                                  f"the seam is defined."),
        })
    sm = _agg(keep, "seam_mean_abs_ticks_all")
    sfl = float(np.mean(seam_floor)) if seam_floor else nan
    gtv = [r["seam_gt_one_step_ticks"] for r in keep if r["seam_gt_valid"]]
    out.update({
        "seam_mean_abs_ticks_all": float(sm.mean()),
        "seam_std_over_pairs": float(sm.std()),
        "seam_mean_abs_ticks": _agg(keep, "seam_mean_abs_ticks").mean(axis=0).tolist(),
        "seam_max_abs_ticks_all": float(_agg(keep, "seam_mean_abs_ticks").max()),
        "seam_noise_floor_ticks": sfl,
        "seam_over_noise_floor": float(sm.mean() / sfl) if sfl > 0 else float("inf"),
        "seam_gt_one_step_ticks": float(np.mean(gtv)) if gtv else nan,
        "n_seam_gt_valid": len(gtv),
    })
    return out


def print_temporal(res: dict) -> None:
    print(f"\n  receding horizon: execute {res['exec_steps']} steps, then re-predict; the "
          f"{res['overlap_steps']} overlapping steps should agree.")
    print(f"  noise: {'PAIRED (same draw both calls -- isolates the moved window)' if res['paired_noise'] else 'INDEPENDENT (fresh draw per replan -- what the robot gets)'}"
          f"   num_steps={res['num_steps']}  n_avg={res.get('n_avg', 1)}")
    print(f"  pairs={res['n_pairs']} (dropped {res['n_dropped_task_change']} that left the "
          f"episode)   same-input noise floor={res['noise_floor_ticks']:.2f} ticks")
    if res["overlap_steps"] > 0:
        print(f"  disagreement raw mean|d| = {res['raw_mean_abs_ticks_all']:.2f} +/- "
              f"{res['std_over_pairs']:.2f} ticks  ({res['over_noise_floor']:.2f}x floor); "
              f"worst single step {res['max_abs_ticks_all']:.1f} ticks")
        print(f"  per channel {MOTOR_LABELS}")
        print("    mean " + " ".join(f"{x:7.1f}" for x in res["raw_mean_abs_ticks"]))
    else:
        print(f"  NO OVERLAP: {res.get('no_overlap_reason', '')}")
    if "seam_mean_abs_ticks_all" in res:
        print(f"  seam |plan(t)[{res['exec_steps'] - 1}] - plan(t+{res['exec_steps']})[0]| = "
              f"{res['seam_mean_abs_ticks_all']:.2f} +/- {res['seam_std_over_pairs']:.2f} ticks "
              f"({res['seam_over_noise_floor']:.2f}x same-input floor "
              f"{res['seam_noise_floor_ticks']:.2f}); gt one-step move over the same pair = "
              f"{res['seam_gt_one_step_ticks']:.2f} ticks (n={res['n_seam_gt_valid']})")
        print(f"  seam per channel {MOTOR_LABELS}")
        print("    mean " + " ".join(f"{x:7.1f}" for x in res["seam_mean_abs_ticks"]))


# ==========================================================================
# Self-test
# ==========================================================================

def self_test(args) -> int:
    ok = {"v": True}

    def check(name, cond, extra=""):
        ok["v"] = ok["v"] and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")

    print("== diag.input_ablation self-test ==")
    lp = load_policy(args.model_path, args.dataset_root, args.device)
    check("training task list loaded", len(lp.tasks) > 0, f"{len(lp.tasks)} instructions")
    # tasks.parquet keeps the sentences in the INDEX and an integer task_index in the column;
    # reading the wrong one yields '0','1','2'... which still "works" and silently makes the
    # language sweep meaningless. Assert they are sentences.
    check("the task list is sentences, not task indices",
          all(len(t.split()) >= 2 for t in lp.tasks[:5]), f"{lp.tasks[:2]}")
    sm = training_state_mean(lp)
    check("state MEAN_STD mean is finite and 5-D", sm.shape == (5,) and np.isfinite(sm).all(),
          f"{sm.round(1).tolist()}")

    chunk = int(lp.cfg.chunk_size)
    mi = MotionIndex.build(args.dataset_root, chunk)
    check("motion index covers the dataset", mi.n > 0 and np.isfinite(mi.motion).all(),
          f"n={mi.n} median={mi.pct(50):.1f} p95={mi.pct(95):.1f} max={mi.motion.max():.1f} ticks")

    d = dataset_frame(args.dataset_root, args.frame_index, chunk)
    frames, state, task = d["frames"], d["state"], d["task"]
    print(f"  frame #{d['index']}  task={task!r}  state={state.astype(int).tolist()}")
    gt_motion = float(np.abs(d["gt_chunk"][-1] - d["gt_chunk"][0]).mean())
    check("parquet motion == motion recomputed from the loaded gt chunk",
          abs(gt_motion - float(mi.motion[d["index"]])) < 1e-6,
          f"gt={gt_motion:.4f} parquet={float(mi.motion[d['index']]):.4f}")

    pidx = mi.swap_partner(d["index"])
    check("swap partner is a real frame from a different episode",
          mi.episode[pidx] != mi.episode[d["index"]],
          f"#{d['index']}(ep{mi.episode[d['index']]}) <- #{pidx}(ep{mi.episode[pidx]})")
    dp = dataset_frame(args.dataset_root, pidx, chunk)
    check("swap frames are not identical to the baseline frames",
          any(not np.array_equal(a, b) for a, b in zip(frames, dp["frames"])))

    # 1. paired determinism -- the whole method rests on this
    n0 = make_noise(lp, 0)
    a, _ = predict_ticks(lp, frames, state, task, noise=n0)
    b, _ = predict_ticks(lp, frames, state, task, noise=n0)
    check("same input + same noise -> IDENTICAL chunk (so a paired difference has 0 error)",
          float(np.abs(a - b).max()) == 0.0, f"max|delta|={float(np.abs(a - b).max()):.6f}")

    # 2. the sampler really is stochastic (otherwise the floor would be meaningless)
    c, _ = predict_ticks(lp, frames, state, task, noise=make_noise(lp, 12345))
    fl = float(np.abs(a - c).mean())
    check("different noise -> different chunk (a real sampling spread exists)", fl > 0.0,
          f"mean|delta|={fl:.2f} ticks")

    # 3. the identity component this script exists to remove is real and large: at the SAME
    #    timestep the label sits almost on top of the state input.
    gap = float(mi.identity_gap.mean())
    astd = float(lp.action_std.mean())
    check("dataset-wide mean|action_t - state_t| is a small fraction of the action std "
          "(this is WHY the raw state ablation over-reported)", gap < 0.10 * astd,
          f"{gap:.2f} ticks = {100 * gap / astd:.1f}% of mean action std {astd:.1f}")
    gap0 = float(np.abs(d["gt_chunk"][0] - state).mean())
    check("that gap is reproduced by the loaded frame's own gt_chunk[0]",
          abs(gap0 - float(mi.identity_gap[d["index"]])) < 1e-6,
          f"frame gap={gap0:.2f} ticks")

    # 4. wiring: each input must be able to move the output at all. A zero here is a SILENT
    #    FAILURE (an input that never reaches the model), not a scientific finding.
    for cond in ("img_swap", "state_swap", "lang_empty", "img_gray"):
        f2, s2, t2 = apply_condition(cond, frames, state, task, lp, sm, lp.tasks[0],
                                     swap_frames=dp["frames"], swap_state=dp["state"])
        p, _ = predict_ticks(lp, f2, s2, t2, noise=n0)
        dd = float(np.abs(a - p).mean())
        check(f"condition '{cond}' changes the prediction (input is wired in)", dd > 0.0,
              f"raw mean|delta|={dd:.2f} ticks")

    # 5. residual == raw for the non-state conditions (the transform must be a no-op there)
    f2, s2, t2 = apply_condition("img_swap", frames, state, task, lp, sm, lp.tasks[0],
                                 swap_frames=dp["frames"], swap_state=dp["state"])
    p, _ = predict_ticks(lp, f2, s2, t2, noise=n0)
    m = pair_metrics(a, state, p, condition_state("img_swap", state, sm, dp["state"]), lp)
    check("residual == raw for an image condition (state cancels)",
          abs(m["resid_mean_abs_ticks_all"] - m["raw_mean_abs_ticks_all"]) < 1e-9,
          f"{m['resid_mean_abs_ticks_all']:.4f} vs {m['raw_mean_abs_ticks_all']:.4f}")
    m2 = pair_metrics(a, state,
                      predict_ticks(lp, *apply_condition("state_swap", frames, state, task, lp, sm,
                                                         lp.tasks[0], swap_frames=dp["frames"],
                                                         swap_state=dp["state"]), noise=n0)[0],
                      condition_state("state_swap", state, sm, dp["state"]), lp)
    check("residual != raw for a state condition (the identity component is removed)",
          m2["resid_mean_abs_ticks_all"] != m2["raw_mean_abs_ticks_all"],
          f"resid={m2['resid_mean_abs_ticks_all']:.1f} raw={m2['raw_mean_abs_ticks_all']:.1f} ticks")

    # 6. small real table
    idx = [args.frame_index % mi.n, (args.frame_index + 733) % mi.n]
    res = run_ablation(lp, idx, list(DEFAULT_CONDITIONS), args.dataset_root, n_noise=2, seed=0,
                       mi=mi, verbose=True)
    res["stratum_desc"] = {"ALL": "(self-test: 2 frames)"}
    print_ablation(res)
    check("every condition produced a finite number",
          all(np.isfinite(v["resid_mean_abs_ticks_all"])
              for v in res["strata"]["ALL"]["conditions"].values()))

    tres = run_temporal(lp, idx, args.dataset_root, exec_steps=args.exec_steps, n_noise=1)
    print_temporal(tres)
    check("temporal consistency computed", np.isfinite(tres["raw_mean_abs_ticks_all"]))

    print(f"== self-test {'PASSED' if ok['v'] else 'FAILED'} ==")
    return 0 if ok["v"] else 1


# ==========================================================================
# Entry
# ==========================================================================

def default_out_path(mode: str) -> str:
    return str(_HERE / "runs" / "diag" / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frame-index", type=int, default=500)
    ap.add_argument("--n-frames", type=int, default=128,
                    help="ALL stratum: frames sampled evenly across the dataset")
    ap.add_argument("--n-move-frames", type=int, default=48,
                    help="MOVE stratum: extra frames drawn from the highest-motion frames")
    ap.add_argument("--move-pct", type=float, default=MOVE_PCT_DEFAULT,
                    help="MOVE = global top this%% of 16-step gt displacement (ARBITRARY)")
    ap.add_argument("--still-pct", type=float, default=STILL_PCT_DEFAULT,
                    help="STILL = frames of ALL below this displacement percentile (ARBITRARY)")
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    ap.add_argument("--n-noise", type=int, default=8,
                    help="extra draws per frame used to size the sampling spread")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap-task", default=None, help="instruction used by the lang_swap condition")
    ap.add_argument("--exec-steps", type=int, default=8, help="--temporal replan stride")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override the flow-matching denoising steps (inference-only)")
    ap.add_argument("--temporal-independent-noise", action="store_true",
                    help="--temporal: draw fresh noise per replan (what the robot experiences)")
    ap.add_argument("--n-avg", type=int, default=1,
                    help="--temporal: average this many independent draws per prediction")
    ap.add_argument("--out", default=None, help="JSON output path (default: runs/diag/<mode>_<ts>.json)")
    ap.add_argument("--no-out", action="store_true", help="do not write the JSON artifact")
    ap.add_argument("--lang-sweep", action="store_true")
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args)

    lp = load_policy(args.model_path, args.dataset_root, args.device)
    if args.num_steps is not None:
        lp.policy.config.num_steps = int(args.num_steps)
        print(f"[num_steps] overridden -> {lp.policy.config.num_steps} (inference-only)")
    chunk = int(lp.cfg.chunk_size)
    mode = "lang_sweep" if args.lang_sweep else ("temporal" if args.temporal else "ablation")

    if args.lang_sweep:
        res = run_lang_sweep(lp, args.frame_index, args.dataset_root, seed=args.seed,
                             n_noise=args.n_noise)
        print_lang_sweep(res)
    else:
        mi = MotionIndex.build(args.dataset_root, chunk)
        if args.temporal:
            n = mi.n
            step = max(1, n // max(1, args.n_frames))
            indices = [(args.frame_index + i * step) % n for i in range(max(1, args.n_frames))]
            res = run_temporal(lp, indices, args.dataset_root, exec_steps=args.exec_steps,
                               seed=args.seed, n_noise=args.n_noise,
                               paired_noise=not args.temporal_independent_noise,
                               n_avg=args.n_avg)
            print_temporal(res)
        else:
            indices, strata, desc, arb = build_frame_plan(
                mi, args.frame_index, args.n_frames, args.n_move_frames,
                args.move_pct, args.still_pct)
            conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
            res = run_ablation(lp, indices, conds, args.dataset_root, n_noise=args.n_noise,
                               seed=args.seed, swap_task=args.swap_task, mi=mi, strata=strata,
                               meta={"stratum_desc": desc, "arbitrary_choices": arb})
            print_ablation(res)

    if not args.no_out:
        out = args.out or default_out_path(mode)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2))
        print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
