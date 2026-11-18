#!/usr/bin/env python3
"""How much of the commanded motion is just the flow-matching noise draw? And can it be bought down?

THE PROBLEM
-----------
SmolVLA's action head is a flow-matching sampler: `sample_actions()` starts from a random
`noise` tensor and Euler-integrates it. Feed the SAME observation twice and you get two DIFFERENT
action chunks. Measured on `runs/d2_unfrozen/checkpoints/004000`, that spread is a large fraction
of the residual signal `mean|pred - state|` -- i.e. of the motion the policy is actually asking
for. The client replans every `exec_steps` (0.8 s at 10 Hz with the default 8), so a fresh draw
happens several times per grasp: the arm is repeatedly yanked toward a *different* plan. That is
a plausible, and so far unmeasured, contributor to the closed-loop grasp failures.

Nothing here trains anything. The two knobs are inference-only:

  num_steps   `SmolVLAPolicy.config.num_steps`, read at
              `modeling_smolvla.py:757` (`num_steps = self.config.num_steps`) and handed to
              `euler_integrate(..., num_steps)` in `policies/common/flow_matching.py:61`. The
              TRAINING loss (`VLAFlowMatching.forward`) never integrates -- it samples ONE random
              timestep per batch element and regresses the velocity. So num_steps appears nowhere
              in the training graph and NO RETRAINING IS NEEDED to change it. This script asserts
              that at runtime (`--self-test` checks the config field is honoured by checking that
              the number of denoise passes actually changes).
  n_avg       average N independently-drawn chunks. Pure post-processing of the sampler output.
              Averaging is NOT free: if the policy is genuinely multi-modal (go left / go right)
              the mean of the modes is a third thing that may be worse than either, so accuracy
              against ground truth is reported next to the spread, per channel.
              The N draws can also be run as ONE batched forward pass (`--batch-list`), which is
              far cheaper than N sequential calls; that path is measured, and checked against the
              sequential path draw-for-draw.

EXPERIMENTS
-----------
A  --steps    for each num_steps, M independent draws per frame:
              spread (mean pairwise |chunk_i - chunk_j|), spread/signal, per-channel error vs the
              (pad-masked) ground truth, and measured wall-clock latency vs the 100 ms budget of
              a 10 Hz loop.
B  --average  derived from the SAME draws: partition the M draws into disjoint groups of N,
              average each group, and measure the spread and accuracy of that averaged estimator
              for N in --n-avg-list. Cost is reported as N x the measured single-call latency
              (the calls are sequential; nothing here batches them).
C  --temporal receding-horizon jitter: chunk(t)[k:] vs chunk(t+k)[:-k] with INDEPENDENT noise on
              each side (what the robot actually gets), reusing `diag.input_ablation.run_temporal`.

    CUDA_VISIBLE_DEVICES=0 python -m diag.sampling_noise --self-test
    CUDA_VISIBLE_DEVICES=0 python -m diag.sampling_noise            # A + B + C
    CUDA_VISIBLE_DEVICES=0 python -m diag.sampling_noise --timing-only   # latency on an idle GPU

Results are ALWAYS written to JSON (--out, default runs/diag/sampling_noise_<stamp>.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from diag.input_ablation import (  # noqa: E402
    MOTOR_LABELS,
    MotionIndex,
    build_frame_plan,
    print_temporal,
    run_temporal,
    signal_metrics,
)
from diag.smolvla_introspect import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH,
    LoadedPolicy,
    build_observation,
    dataset_frame,
    load_policy,
    make_noise,
    predict_ticks,
)

# --- ARBITRARY CHOICES (all printed at runtime) ---------------------------------------------
NUM_STEPS_LIST = (10, 20, 50, 100)   # 10 = the shipped default in configuration_smolvla.py
N_AVG_LIST = (1, 2, 4, 8)
N_DRAWS_DEFAULT = 16                 # draws per frame; must be >= max(N_AVG_LIST) * 2 for a spread
LATENCY_BUDGET_MS = 100.0            # 10 Hz control loop
TIMING_WARMUP = 3
TIMING_REPS = 12


# ==========================================================================
# Latency
# ==========================================================================

def measure_latency(lp: LoadedPolicy, frames, state, task, reps: int = TIMING_REPS,
                    warmup: int = TIMING_WARMUP) -> dict:
    """Wall-clock of one /predict, the way the deployment server does it.

    `predict_ticks` ends in `.cpu()`, which synchronises, so timing the call is honest; an extra
    `cuda.synchronize()` is issued anyway. The breakdown pass is separate because its syncs
    perturb the total.
    """
    for i in range(warmup):
        predict_ticks(lp, frames, state, task, noise=make_noise(lp, 900_000 + i))
    torch.cuda.synchronize()
    ts = []
    for i in range(reps):
        t0 = time.perf_counter()
        predict_ticks(lp, frames, state, task, noise=make_noise(lp, 910_000 + i))
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)

    # breakdown (own syncs; do not compare its sum to the total above)
    parts = {"obs_build_ms": [], "preprocess_ms": [], "denoise_ms": [], "postprocess_ms": []}
    for i in range(max(3, reps // 3)):
        noise = make_noise(lp, 920_000 + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs = build_observation(lp, frames, state, task)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lp.policy.reset()
        lp.preprocessor.reset()
        lp.postprocessor.reset()
        with torch.no_grad():
            batch = lp.preprocessor(obs)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            chunk_norm = lp.policy.predict_action_chunk(batch, noise=noise)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            lp.postprocessor(chunk_norm)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        parts["obs_build_ms"].append((t1 - t0) * 1000)
        parts["preprocess_ms"].append((t2 - t1) * 1000)
        parts["denoise_ms"].append((t3 - t2) * 1000)
        parts["postprocess_ms"].append((t4 - t3) * 1000)
    return {
        "num_steps": int(lp.cfg.num_steps),
        "mean_ms": float(ts.mean()), "median_ms": float(np.median(ts)),
        "p95_ms": float(np.percentile(ts, 95)), "min_ms": float(ts.min()),
        "max_ms": float(ts.max()), "reps": int(reps),
        "fits_100ms_budget": bool(np.percentile(ts, 95) < LATENCY_BUDGET_MS),
        "breakdown_median_ms": {k: float(np.median(v)) for k, v in parts.items()},
    }


# ==========================================================================
# Experiment A/B: draws per frame -> spread + accuracy, for every N
# ==========================================================================

def draw_chunks(lp: LoadedPolicy, d: dict, n_draws: int, seed0: int) -> np.ndarray:
    """(n_draws, T, 5) tick chunks: same observation, independent flow-matching noise."""
    out = []
    for j in range(n_draws):
        p, _ = predict_ticks(lp, d["frames"], d["state"], d["task"],
                             noise=make_noise(lp, seed0 + j))
        out.append(p)
    return np.stack(out)


def batched_draw_chunks(lp: LoadedPolicy, frames, state, task, n: int,
                        noise: torch.Tensor | None = None, seed0: int = 0) -> np.ndarray:
    """(n, T, 5) tick chunks from ONE forward pass: the observation row is expanded to n, and
    the flow-matching noise carries n independent rows.

    This is a diagnostic of what a batched server COULD do; it does not change the policy. The
    per-row result is not bit-identical to the sequential path because cuBLAS picks different
    reduction orders at different batch sizes -- `measure_batching` quantifies that gap instead
    of assuming it away.
    """
    obs = build_observation(lp, frames, state, task)
    lp.policy.reset()
    lp.preprocessor.reset()
    lp.postprocessor.reset()
    with torch.no_grad():
        batch = lp.preprocessor(obs)
        big = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.shape[0] == 1:
                big[k] = v.expand(n, *v.shape[1:])
            elif isinstance(v, list) and len(v) == 1:
                big[k] = v * n
            else:
                big[k] = v
        if noise is None:
            noise = make_noise(lp, seed0, bsize=n)
        chunk = lp.postprocessor(lp.policy.predict_action_chunk(big, noise=noise))
    return chunk.detach().float().cpu().numpy().astype(np.float64)


def measure_batching(lp: LoadedPolicy, d: dict, batch_list, reps: int = 10,
                     warmup: int = 3) -> dict:
    """Latency per batch size + how far the batched draws land from the sequential ones."""
    frames, state, task = d["frames"], d["state"], d["task"]
    out = {"num_steps": int(lp.cfg.num_steps), "by_batch": {}}
    nmax = max(batch_list)
    noise = make_noise(lp, 777_000, bsize=nmax)
    seq = np.stack([predict_ticks(lp, frames, state, task, noise=noise[j:j + 1])[0]
                    for j in range(nmax)])
    bat = batched_draw_chunks(lp, frames, state, task, nmax, noise=noise)
    out["equivalence_vs_sequential"] = {
        "n": int(nmax),
        "max_abs_ticks": float(np.abs(bat - seq).max()),
        "mean_abs_ticks": float(np.abs(bat - seq).mean()),
        "note": ("non-zero because batched matmuls reduce in a different order; compare it "
                 "against the sampling spread before treating it as an error"),
    }
    for n in batch_list:
        for i in range(warmup):
            batched_draw_chunks(lp, frames, state, task, n, seed0=780_000 + i)
        torch.cuda.synchronize()
        ts = []
        for i in range(reps):
            t0 = time.perf_counter()
            batched_draw_chunks(lp, frames, state, task, n, seed0=790_000 + 31 * i)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.asarray(ts)
        out["by_batch"][str(n)] = {
            "median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "per_sample_ms": float(np.median(ts) / n),
            "fits_100ms_budget": bool(np.percentile(ts, 95) < LATENCY_BUDGET_MS),
        }
    return out


def print_batching(res: dict, seq_latency_ms: float) -> None:
    eq = res["equivalence_vs_sequential"]
    print(f"\n  === N draws in ONE batched forward pass (num_steps={res['num_steps']}) ===")
    print(f"  batched vs sequential, same noise rows: max|d|={eq['max_abs_ticks']:.2f} "
          f"mean|d|={eq['mean_abs_ticks']:.2f} ticks ({eq['note']})")
    print(f"  {'batch N':>8s} {'median ms':>10s} {'p95 ms':>8s} {'per-sample':>11s} "
          f"{'vs N sequential':>16s} {'<100ms':>7s}")
    for n, v in res["by_batch"].items():
        seq = seq_latency_ms * int(n)
        print(f"  {int(n):8d} {v['median_ms']:10.1f} {v['p95_ms']:8.1f} "
              f"{v['per_sample_ms']:11.1f} {seq:12.1f} ms {('YES' if v['fits_100ms_budget'] else 'no'):>7s}")


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def _pairwise_spread(chunks: np.ndarray) -> float:
    """mean over unordered pairs of mean|a - b| in ticks. Same definition as the ablation floor."""
    k = len(chunks)
    if k < 2:
        return float("nan")
    return float(np.mean([np.abs(chunks[i] - chunks[j]).mean() for i, j in combinations(range(k), 2)]))


def _pairwise_spread_per_ch(chunks: np.ndarray) -> np.ndarray:
    k = len(chunks)
    if k < 2:
        return np.full(chunks.shape[-1], np.nan)
    return np.mean([np.abs(chunks[i] - chunks[j]).mean(axis=0)
                    for i, j in combinations(range(k), 2)], axis=0)


def _pairwise_spread_per_step(chunks: np.ndarray) -> np.ndarray:
    """(T,) spread at each horizon step, averaged over motors. Shows the far half of a chunk
    blurring out, which is the whole reason a short chunk looks good on a step-averaged loss."""
    k = len(chunks)
    if k < 2:
        return np.full(chunks.shape[1], np.nan)
    return np.mean([np.abs(chunks[i] - chunks[j]).mean(axis=1)
                    for i, j in combinations(range(k), 2)], axis=0)


def _err_vs_gt(chunk: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(5,) mean abs error in ticks over the NON-PADDED horizon steps only."""
    if not valid.any():
        return np.full(chunk.shape[-1], np.nan)
    return np.abs(chunk[valid] - gt[valid]).mean(axis=0)


def _err_vs_gt_per_step(chunk: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T,) mean abs error per horizon step, averaged over motors; NaN on padded steps."""
    e = np.abs(chunk - gt).mean(axis=1)
    e[~valid] = np.nan
    return e


def frame_stats(lp: LoadedPolicy, d: dict, chunks: np.ndarray, n_avg_list,
                horizon: int | None = None) -> dict:
    """Spread + accuracy of the draws for one frame.

    `horizon` truncates BOTH the predictions and the ground truth to the first `horizon` steps.
    That is the only way to compare checkpoints trained with different chunk_size: the deployment
    loop executes n_action_steps rows and throws the rest away, so a step-averaged score over the
    full chunk grades a short-chunk model on an easier (nearer) horizon. Default None = the whole
    predicted chunk, the historical behaviour.

    The ground truth is truncated to the same length. `d` may legitimately carry a LONGER gt than
    the prediction (e.g. one frame cache shared by several chunk_size), but never a shorter one --
    that is an error, not something to broadcast around.
    """
    gt, valid = d["gt_chunk"], ~d["gt_is_pad"]
    h = int(horizon) if horizon is not None else int(chunks.shape[1])
    if chunks.shape[1] < h or gt.shape[0] < h:
        raise ValueError(f"horizon {h} exceeds the available chunk ({chunks.shape[1]}) or "
                         f"ground truth ({gt.shape[0]}) length")
    chunks, gt, valid = chunks[:, :h], gt[:h], valid[:h]
    state = d["state"]
    sig = float(np.mean([signal_metrics(c, state, lp)["signal_mean_abs_ticks_all"] for c in chunks]))
    rec = {"index": int(d["index"]), "task": d["task"], "n_valid_steps": int(valid.sum()),
           "horizon": int(chunks.shape[1]),
           "motion_ticks": float(np.abs(gt[-1] - gt[0]).mean()),
           "signal_mean_abs_ticks_all": sig, "n_avg": {}}
    m = len(chunks)
    for N in n_avg_list:
        g = m // N
        if g < 1:
            continue
        groups = np.stack([chunks[i * N:(i + 1) * N].mean(axis=0) for i in range(g)])
        with warnings.catch_warnings():   # all-padded step -> NaN column, which is correct
            warnings.simplefilter("ignore", RuntimeWarning)
            err_step = np.nanmean(
                np.stack([_err_vs_gt_per_step(x, gt, valid) for x in groups]), axis=0)
        rec["n_avg"][str(N)] = {
            "n_groups": int(g),
            "spread_ticks": _pairwise_spread(groups),
            "spread_ticks_per_ch": _pairwise_spread_per_ch(groups).tolist(),
            "spread_ticks_per_step": _pairwise_spread_per_step(groups).tolist(),
            "err_ticks_per_ch": np.mean([_err_vs_gt(x, gt, valid) for x in groups],
                                        axis=0).tolist(),
            "err_ticks_per_step": err_step.tolist(),
            "signal_mean_abs_ticks_all": float(np.mean(
                [signal_metrics(x, state, lp)["signal_mean_abs_ticks_all"] for x in groups])),
        }
    return rec


def run_steps_sweep(lp: LoadedPolicy, indices: list[int], dataset_root: str,
                    num_steps_list, n_draws: int, n_avg_list, seed: int = 0,
                    strata: dict | None = None, do_timing: bool = True,
                    verbose: bool = True) -> dict:
    chunk_size = int(lp.cfg.chunk_size)
    orig_steps = int(lp.cfg.num_steps)
    frames_cache = {}
    for idx in indices:
        frames_cache[idx] = dataset_frame(dataset_root, idx, chunk_size)
    res = {"num_steps_list": list(num_steps_list), "n_draws_per_frame": int(n_draws),
           "n_avg_list": list(n_avg_list), "n_frames": len(indices), "by_num_steps": {}}
    try:
        for ns in num_steps_list:
            lp.policy.config.num_steps = int(ns)
            if int(lp.cfg.num_steps) != int(ns):
                raise RuntimeError(f"setting config.num_steps={ns} did not stick "
                                   f"(reads back {lp.cfg.num_steps}); the sweep would be a lie.")
            t0 = time.time()
            per_frame = []
            for n, idx in enumerate(indices):
                d = frames_cache[idx]
                chunks = draw_chunks(lp, d, n_draws, seed0=seed + 100_000 + 1000 * n)
                per_frame.append(frame_stats(lp, d, chunks, n_avg_list))
                if verbose:
                    print(f"  [num_steps={ns}] [{n + 1}/{len(indices)}] frame {d['index']} "
                          f"spread={per_frame[-1]['n_avg']['1']['spread_ticks']:.1f} ticks",
                          flush=True)
            entry = {"per_frame": per_frame, "elapsed_s": round(time.time() - t0, 1),
                     "strata": {}}
            if do_timing:
                d0 = frames_cache[indices[0]]
                entry["latency"] = measure_latency(lp, d0["frames"], d0["state"], d0["task"])
            for sname, sidx in (strata or {"ALL": indices}).items():
                sel = [r for r in per_frame if r["index"] in set(i for i in sidx)]
                if sel:
                    entry["strata"][sname] = _summarize_draws(sel, n_avg_list)
            res["by_num_steps"][str(ns)] = entry
    finally:
        lp.policy.config.num_steps = orig_steps
    return res


def _summarize_draws(recs: list[dict], n_avg_list) -> dict:
    out = {"n_frames": len(recs),
           "motion_ticks_mean": float(np.mean([r["motion_ticks"] for r in recs])),
           "signal_mean_abs_ticks_all": float(np.mean(
               [r["signal_mean_abs_ticks_all"] for r in recs])),
           "n_avg": {}}
    for N in n_avg_list:
        sel = [r["n_avg"][str(N)] for r in recs if str(N) in r["n_avg"]]
        if not sel:
            continue
        sp = float(np.mean([s["spread_ticks"] for s in sel]))
        sig = float(np.mean([s["signal_mean_abs_ticks_all"] for s in sel]))
        err = np.mean([s["err_ticks_per_ch"] for s in sel], axis=0)
        entry = {
            "spread_ticks": sp,
            "spread_over_signal": sp / sig if sig > 0 else float("nan"),
            "spread_ticks_per_ch": np.mean([s["spread_ticks_per_ch"] for s in sel],
                                           axis=0).tolist(),
            "err_ticks_per_ch": err.tolist(),
            "err_ticks_mean": float(err.mean()),
            "signal_mean_abs_ticks_all": sig,
            "n_groups_mean": float(np.mean([s["n_groups"] for s in sel])),
        }
        # per-step curves only aggregate if every frame was scored on the same horizon
        for key, fn in (("spread_ticks_per_step", np.mean), ("err_ticks_per_step", np.nanmean)):
            vals = [s[key] for s in sel if key in s]
            if len(vals) == len(sel) and len({len(v) for v in vals}) == 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    entry[key] = fn(np.asarray(vals, dtype=np.float64), axis=0).tolist()
        out["n_avg"][str(N)] = entry
    return out


def print_steps_sweep(res: dict, stratum: str = "ALL") -> None:
    print(f"\n  === sampling spread vs denoising steps -- stratum {stratum} ===")
    # per-stratum count, not the union count in res['n_frames'] (union = 36, ALL/STILL/MOVE = 24/11/12)
    _n_str = "?"
    for _e in res["by_num_steps"].values():
        if stratum in _e.get("strata", {}):
            _n_str = str(_e["strata"][stratum]["n_frames"])
            break
    print(f"  {res['n_draws_per_frame']} independent noise draws per frame, "
          f"{_n_str} frames in this stratum (pool total {res['n_frames']})")
    print(f"  spread = mean over draw pairs of mean|chunk_i - chunk_j| (ticks); "
          f"signal = mean|chunk - state|")
    print(f"  {'num_steps':>9s} {'spread':>8s} {'signal':>8s} {'spr/sig':>8s} "
          f"{'lat med':>8s} {'lat p95':>8s} {'<100ms':>7s}   err vs gt (ticks) {MOTOR_LABELS}")
    for ns in res["num_steps_list"]:
        e = res["by_num_steps"].get(str(ns))
        if not e or stratum not in e["strata"]:
            continue
        s = e["strata"][stratum]["n_avg"].get("1")
        if not s:
            continue
        lat = e.get("latency", {})
        errs = " ".join(f"{x:6.1f}" for x in s["err_ticks_per_ch"])
        print(f"  {ns:9d} {s['spread_ticks']:8.2f} {s['signal_mean_abs_ticks_all']:8.2f} "
              f"{s['spread_over_signal']:8.3f} "
              f"{lat.get('median_ms', float('nan')):8.1f} {lat.get('p95_ms', float('nan')):8.1f} "
              f"{('YES' if lat.get('fits_100ms_budget') else 'no'):>7s}   {errs}")
    if res["by_num_steps"]:
        b = next(iter(res["by_num_steps"].values())).get("latency", {}).get("breakdown_median_ms")
        if b:
            print(f"  latency breakdown at num_steps="
                  f"{next(iter(res['by_num_steps'].keys()))}: " +
                  "  ".join(f"{k}={v:.1f}ms" for k, v in b.items()))


def run_batched_stats(lp: LoadedPolicy, indices: list[int], dataset_root: str, n_draws: int,
                      n_avg_list, strata: dict | None = None, seed: int = 0) -> dict:
    """Same spread/accuracy statistics as the sequential sweep, but from batched draws.

    If the two agree, the cheap batched path can be recommended without re-deriving anything.
    """
    chunk_size = int(lp.cfg.chunk_size)
    per_frame = []
    for n, idx in enumerate(indices):
        d = dataset_frame(dataset_root, idx, chunk_size)
        chunks = batched_draw_chunks(lp, d["frames"], d["state"], d["task"], n_draws,
                                     seed0=seed + 300_000 + 1000 * n)
        per_frame.append(frame_stats(lp, d, chunks, n_avg_list))
    out = {"n_draws_per_frame": int(n_draws), "num_steps": int(lp.cfg.num_steps), "strata": {}}
    for sname, sidx in (strata or {"ALL": indices}).items():
        sel = [r for r in per_frame if r["index"] in set(sidx)]
        if sel:
            out["strata"][sname] = _summarize_draws(sel, n_avg_list)
    return out


def print_average_sweep(res: dict, stratum: str = "ALL") -> None:
    print(f"\n  === sample averaging (N chunks averaged) -- stratum {stratum} ===")
    print(f"  cost is SEQUENTIAL: N x the measured single-call latency (no batching here)")
    print(f"  {'num_steps':>9s} {'N':>3s} {'spread':>8s} {'spr/sig':>8s} {'err mean':>9s} "
          f"{'est ms':>8s} {'<100ms':>7s}   err vs gt (ticks) {MOTOR_LABELS}")
    for ns in res["num_steps_list"]:
        e = res["by_num_steps"].get(str(ns))
        if not e or stratum not in e["strata"]:
            continue
        lat = e.get("latency", {}).get("median_ms", float("nan"))
        for N in res["n_avg_list"]:
            s = e["strata"][stratum]["n_avg"].get(str(N))
            if not s:
                continue
            est = lat * N
            errs = " ".join(f"{x:6.1f}" for x in s["err_ticks_per_ch"])
            print(f"  {ns:9d} {N:3d} {s['spread_ticks']:8.2f} {s['spread_over_signal']:8.3f} "
                  f"{s['err_ticks_mean']:9.2f} {est:8.1f} "
                  f"{('YES' if est < LATENCY_BUDGET_MS else 'no'):>7s}   {errs}")


# ==========================================================================
# Self-test
# ==========================================================================

def self_test(args) -> int:
    ok = {"v": True}

    def check(name, cond, extra=""):
        ok["v"] = ok["v"] and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")

    print("== diag.sampling_noise self-test ==")
    lp = load_policy(args.model_path, args.dataset_root, args.device)
    chunk = int(lp.cfg.chunk_size)
    d = dataset_frame(args.dataset_root, args.frame_index, chunk)

    # 1. num_steps is inference-only: it must reach the sampler, and it must NOT be baked into
    #    the checkpoint in any way that would need retraining. Count the denoise passes.
    counts = {}
    orig = lp.policy.model.denoise_step
    for ns in (4, 10):
        lp.policy.config.num_steps = ns
        n = {"k": 0}

        def counting(*a, _n=n, **kw):
            _n["k"] += 1
            return orig(*a, **kw)

        lp.policy.model.denoise_step = counting
        try:
            predict_ticks(lp, d["frames"], d["state"], d["task"], noise=make_noise(lp, 0))
        finally:
            lp.policy.model.denoise_step = orig
        counts[ns] = n["k"]
    lp.policy.config.num_steps = 10
    check("config.num_steps controls the number of Euler denoise steps at inference",
          counts == {4: 4, 10: 10}, f"denoise_step calls {counts}")

    # 2. the noise really is the only thing that differs between draws
    a, _ = predict_ticks(lp, d["frames"], d["state"], d["task"], noise=make_noise(lp, 7))
    b, _ = predict_ticks(lp, d["frames"], d["state"], d["task"], noise=make_noise(lp, 7))
    c, _ = predict_ticks(lp, d["frames"], d["state"], d["task"], noise=make_noise(lp, 8))
    check("same noise -> identical chunk", float(np.abs(a - b).max()) == 0.0)
    check("different noise -> different chunk", float(np.abs(a - c).mean()) > 0.0,
          f"mean|d|={float(np.abs(a - c).mean()):.2f} ticks")

    # 3. gt padding is exposed, otherwise accuracy is scored against fabricated repeats
    check("gt_is_pad is available and boolean", d["gt_is_pad"].dtype == np.bool_
          and d["gt_is_pad"].shape == (chunk,), f"{int(d['gt_is_pad'].sum())} padded steps")

    # 4. averaging arithmetic: N=1 groups must reproduce the raw draws exactly
    chunks = draw_chunks(lp, d, 4, seed0=1234)
    st = frame_stats(lp, d, chunks, (1, 2, 4))
    check("N=1 spread equals the raw pairwise spread",
          abs(st["n_avg"]["1"]["spread_ticks"] - _pairwise_spread(chunks)) < 1e-9)
    check("N=4 collapses to a single group (no spread defined)",
          st["n_avg"]["4"]["n_groups"] == 1 and np.isnan(st["n_avg"]["4"]["spread_ticks"]))
    man = np.abs(chunks[0:2].mean(0) - chunks[2:4].mean(0)).mean()
    check("N=2 spread matches a hand-computed group difference",
          abs(st["n_avg"]["2"]["spread_ticks"] - man) < 1e-9,
          f"{st['n_avg']['2']['spread_ticks']:.4f} vs {man:.4f}")

    # 4b. horizon truncation must score ONLY the first h steps -- this is what makes checkpoints
    #     with different chunk_size comparable, so a silent off-by-one here would fake the result
    h = max(2, min(8, chunk - 1))
    sth = frame_stats(lp, d, chunks, (1,), horizon=h)
    man_sp = _pairwise_spread(chunks[:, :h])
    v = ~d["gt_is_pad"][:h]
    man_err = np.mean([np.abs(c[:h][v] - d["gt_chunk"][:h][v]).mean(axis=0) for c in chunks],
                      axis=0)
    check(f"horizon={h} truncates the spread to the first {h} steps",
          abs(sth["n_avg"]["1"]["spread_ticks"] - man_sp) < 1e-9 and sth["horizon"] == h,
          f"{sth['n_avg']['1']['spread_ticks']:.4f} vs {man_sp:.4f}")
    check(f"horizon={h} truncates the pad-masked gt error too",
          np.allclose(man_err, sth["n_avg"]["1"]["err_ticks_per_ch"], atol=1e-9),
          f"per-ch {np.round(sth['n_avg']['1']['err_ticks_per_ch'], 2).tolist()}")
    check(f"horizon={h} per-step curves have length {h}",
          len(sth["n_avg"]["1"]["spread_ticks_per_step"]) == h
          and len(sth["n_avg"]["1"]["err_ticks_per_step"]) == h)
    check("horizon larger than the chunk is refused rather than silently padded",
          _raises(lambda: frame_stats(lp, d, chunks, (1,), horizon=chunk + 1)))

    # 5. latency measurement runs and is plausible
    lat = measure_latency(lp, d["frames"], d["state"], d["task"], reps=5, warmup=2)
    check("latency measured", np.isfinite(lat["median_ms"]) and lat["median_ms"] > 0,
          f"median {lat['median_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms at num_steps="
          f"{lat['num_steps']}")
    check("the denoise stage dominates the latency breakdown",
          lat["breakdown_median_ms"]["denoise_ms"] > lat["breakdown_median_ms"]["preprocess_ms"],
          " ".join(f"{k}={v:.1f}" for k, v in lat["breakdown_median_ms"].items()))

    print(f"== self-test {'PASSED' if ok['v'] else 'FAILED'} ==")
    return 0 if ok["v"] else 1


# ==========================================================================
# Entry
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frame-index", type=int, default=500)
    ap.add_argument("--n-frames", type=int, default=24, help="ALL stratum frames")
    ap.add_argument("--n-move-frames", type=int, default=12, help="MOVE stratum frames")
    ap.add_argument("--move-pct", type=float, default=5.0)
    ap.add_argument("--still-pct", type=float, default=50.0)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    ap.add_argument("--num-steps-list", default=",".join(str(x) for x in NUM_STEPS_LIST))
    ap.add_argument("--n-avg-list", default=",".join(str(x) for x in N_AVG_LIST))
    ap.add_argument("--exec-steps", type=int, default=8)
    ap.add_argument("--temporal-num-steps", default="10,100",
                    help="num_steps values for the receding-horizon experiment")
    ap.add_argument("--temporal-n-avg", default="1,4",
                    help="n_avg values for the receding-horizon experiment")
    ap.add_argument("--temporal-frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-list", default="1,2,4,8,16",
                    help="batch sizes for the one-forward-pass averaging measurement")
    ap.add_argument("--no-batching", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--timing-only", action="store_true",
                    help="measure latency only (run this on an otherwise idle GPU)")
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-out", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args)

    num_steps_list = [int(x) for x in args.num_steps_list.split(",") if x.strip()]
    n_avg_list = [int(x) for x in args.n_avg_list.split(",") if x.strip()]
    lp = load_policy(args.model_path, args.dataset_root, args.device)
    chunk = int(lp.cfg.chunk_size)
    mi = MotionIndex.build(args.dataset_root, chunk)
    indices, strata, desc, arb = build_frame_plan(mi, args.frame_index, args.n_frames,
                                                  args.n_move_frames, args.move_pct,
                                                  args.still_pct)
    arb = list(arb) + [
        f"num_steps swept over {num_steps_list} (10 = shipped default) -- ARBITRARY",
        f"n_avg swept over {n_avg_list} -- ARBITRARY",
        f"{args.n_draws} noise draws per frame -- ARBITRARY",
        f"latency budget {LATENCY_BUDGET_MS:.0f} ms = one 10 Hz control period -- from the rig, "
        f"not arbitrary",
    ]
    res = {"model_path": lp.model_path, "dataset_root": args.dataset_root,
           "config_num_steps_default": int(lp.cfg.num_steps),
           "chunk_size": chunk, "exec_steps": args.exec_steps,
           "action_std_ticks": lp.action_std.round(2).tolist(),
           "stratum_desc": desc, "arbitrary_choices": arb,
           "latency_budget_ms": LATENCY_BUDGET_MS}
    for a in arb:
        print(f"  [ARBITRARY] {a}")

    if args.timing_only:
        d = dataset_frame(args.dataset_root, args.frame_index, chunk)
        lat = {}
        orig = int(lp.cfg.num_steps)
        try:
            for ns in num_steps_list:
                lp.policy.config.num_steps = ns
                lat[str(ns)] = measure_latency(lp, d["frames"], d["state"], d["task"])
                m = lat[str(ns)]
                print(f"  num_steps={ns:4d}  median {m['median_ms']:7.1f} ms  p95 "
                      f"{m['p95_ms']:7.1f} ms  fits100ms={m['fits_100ms_budget']}  "
                      + " ".join(f"{k}={v:.1f}" for k, v in m["breakdown_median_ms"].items()),
                      flush=True)
        finally:
            lp.policy.config.num_steps = orig
        res["latency_only"] = lat
    else:
        sweep = run_steps_sweep(lp, indices, args.dataset_root, num_steps_list, args.n_draws,
                                n_avg_list, seed=args.seed, strata=strata,
                                do_timing=not args.no_timing)
        res["steps_sweep"] = sweep
        for st in ("ALL", "MOVE", "STILL"):
            if any(st in e["strata"] for e in sweep["by_num_steps"].values()):
                print_steps_sweep(sweep, st)
                print_average_sweep(sweep, st)

        if not args.no_batching:
            batch_list = [int(x) for x in args.batch_list.split(",") if x.strip()]
            d0 = dataset_frame(args.dataset_root, indices[0], chunk)
            bres = measure_batching(lp, d0, batch_list)
            seq_ms = (sweep["by_num_steps"][str(num_steps_list[0])]
                      .get("latency", {}).get("median_ms", float("nan")))
            print_batching(bres, seq_ms)
            bres["stats"] = run_batched_stats(lp, indices, args.dataset_root, args.n_draws,
                                              n_avg_list, strata=strata, seed=args.seed)
            print("\n  batched-draw statistics (compare against the sequential table above)")
            for st in ("ALL", "MOVE", "STILL"):
                s = bres["stats"]["strata"].get(st)
                if not s:
                    continue
                row = "  ".join(f"N={N}: spread={s['n_avg'][str(N)]['spread_ticks']:.2f} "
                                f"err={s['n_avg'][str(N)]['err_ticks_mean']:.2f}"
                                for N in n_avg_list if str(N) in s["n_avg"])
                print(f"    {st:<6s} {row}")
            res["batching"] = bres

        if not args.no_temporal:
            res["temporal"] = []
            n = mi.n
            step = max(1, n // max(1, args.temporal_frames))
            tidx = [(args.frame_index + i * step) % n for i in range(args.temporal_frames)]
            orig = int(lp.cfg.num_steps)
            try:
                for ns in [int(x) for x in args.temporal_num_steps.split(",") if x.strip()]:
                    for na in [int(x) for x in args.temporal_n_avg.split(",") if x.strip()]:
                        lp.policy.config.num_steps = ns
                        t = run_temporal(lp, tidx, args.dataset_root,
                                         exec_steps=args.exec_steps, seed=args.seed,
                                         n_noise=2, paired_noise=False, n_avg=na)
                        print_temporal(t)
                        t.pop("rows", None)
                        res["temporal"].append(t)
            finally:
                lp.policy.config.num_steps = orig

    if not args.no_out:
        out = args.out or str(_HERE / "runs" / "diag" /
                              f"sampling_noise_{time.strftime('%Y%m%d_%H%M%S')}.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2))
        print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
