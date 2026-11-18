#!/usr/bin/env python3
"""Score a checkpoint by its FINAL deployed output on the pinned held-out episodes (read-only).

Why: for Coda runs eval_loss is the main-AE flow loss only -- it never sees the correction AE.
Every one of the 87 smolvla_ext runs (and the baselines) is scored here the same way: the
checkpoint is loaded exactly like the server does (diag.smolvla_introspect.load_policy, which
attaches PVI / Coda / ROCKET from the smolvla_ext.json sidecar), so a Coda checkpoint is scored
on its 5+5 / 10+10 output and a PVI one with its injections.

Metrics (each in raw Dynamixel ticks, 1 rev = 4096):
  gt_err     mean |predicted joint - demonstrated joint| over the first EXEC steps of the chunk,
             padded ground-truth steps masked out; per motor and overall, averaged over draws.
  spread     same input asked DRAWS times with different flow-matching noise: mean pairwise
             |chunk_i - chunk_j| over the first EXEC steps. Larger = the arm shakes more.
  seam       execute EXEC steps, re-predict at t+EXEC with fresh noise (what the server does):
             |chunk(t)[EXEC-1] - chunk(t+EXEC)[0]|, next to the demonstrator's own one-step move.
Frames: FRAMES_PER_EP evenly placed frames in each held-out episode for gt_err / spread, and
SEAM_PER_EP re-plan pairs per episode. Same frames and same noise seeds for every checkpoint.
Helpers reused from diag/sampling_noise.py (draw_chunks, frame_stats, _summarize_draws) and
diag/input_ablation.py (run_temporal), the code behind diag/deploy_compare.py's M1 / M2 / M3b.

usage:
  python diag/ext_eval.py --ckpt runs/ext_C_F_c5/checkpoints/006000 [--label X]
         [--manifest runs/ext_C_F_c5/run_manifest.json | --episodes 4,15,...]
         [--out results/ext_eval/X.json] [--summary results/ext_eval/summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diag.input_ablation import run_temporal  # noqa: E402
from diag.sampling_noise import _summarize_draws, draw_chunks, frame_stats  # noqa: E402
from diag.smolvla_introspect import dataset_frame, load_policy  # noqa: E402

D9_EVAL_IDX = "4,15,23,29,42,45,61,64,78,84,100,110,128,137,172,198,230,237"  # train_ext.sh
# --- ARBITRARY CHOICES (recorded in the output) ---
EXEC = 8               # n_action_steps of every run (deployment executes 8 rows, then re-plans)
DRAWS = 8              # same as the project's draw-spread measurement
FRAMES_PER_EP = 4      # at 10%, 35%, 60%, 85% of each held-out episode
SEAM_PER_EP = 2        # re-plan pairs at 30% and 70% of each held-out episode
SEAM_NOISE = 4         # independent draws for the seam noise floor
SEED = 0


def episode_spans(root: str) -> dict[int, tuple[int, int]]:
    spans = {}
    for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")):
        d = pd.read_parquet(f, columns=["episode_index", "dataset_from_index", "dataset_to_index"])
        for e, a, b in d.itertuples(index=False):
            spans[int(e)] = (int(a), int(b))
    return spans


def frame_plan(spans, episodes):
    m12, m3 = [], []
    for e in episodes:
        a, b = spans[e]
        n = b - a
        m12 += [a + int(f * (n - 1)) for f in (0.10, 0.35, 0.60, 0.85)][:FRAMES_PER_EP]
        for f in (0.30, 0.70)[:SEAM_PER_EP]:
            t = a + int(f * (n - 1))
            if t + EXEC < b:
                m3.append(t)
    return m12, m3


def episodes_from(args) -> list[int]:
    if args.episodes:
        return [int(x) for x in args.episodes.split(",")]
    if args.manifest:
        flags = json.loads(Path(args.manifest).read_text()).get("flags", "")
        m = re.search(r"--eval-episode-indices\s+([\d,]+)", flags)
        if not m:
            raise SystemExit(f"no --eval-episode-indices in {args.manifest} flags")
        return [int(x) for x in m.group(1).split(",")]
    return [int(x) for x in D9_EVAL_IDX.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (…/checkpoints/006000) or its pretrained_model")
    ap.add_argument("--label", default="")
    ap.add_argument("--data", default="data/lerobot_v3_d9_0912")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--summary", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ck = Path(args.ckpt)
    pm = ck if (ck / "config.json").is_file() else ck / "pretrained_model"
    label = args.label or (f"{pm.parents[2].name}@{pm.parent.name}" if pm.name == "pretrained_model"
                           else str(pm))
    eps = episodes_from(args)
    spans = episode_spans(args.data)
    missing = [e for e in eps if e not in spans]
    if missing:
        raise SystemExit(f"episodes {missing} not in {args.data}")
    m12_idx, m3_idx = frame_plan(spans, eps)

    t0 = time.time()
    lp = load_policy(str(pm), args.data, args.device)
    cfg = lp.policy.config
    ext = getattr(lp.policy, "_smolvla_ext", None)
    chunk = int(cfg.chunk_size)
    per_frame = []
    for n, idx in enumerate(m12_idx):
        d = dataset_frame(args.data, int(idx), chunk)
        chunks = draw_chunks(lp, d, DRAWS, seed0=SEED + 100_000 + 1000 * n)
        per_frame.append(frame_stats(lp, d, chunks, (1,), horizon=EXEC))
    m12 = _summarize_draws(per_frame, (1,))["n_avg"]["1"]
    m3 = run_temporal(lp, m3_idx, args.data, exec_steps=EXEC, seed=SEED, n_noise=SEAM_NOISE,
                      paired_noise=False, n_avg=1, overlap_steps=EXEC)
    res = {
        "label": label, "checkpoint": str(pm), "dataset": args.data,
        "num_steps": int(cfg.num_steps),
        "ext": ext.enabled if ext is not None else [],
        "coda_steps": int(ext.coda_num_steps) if ext is not None and ext.coda else None,
        "heldout_episodes": eps, "n_frames_m12": len(m12_idx), "n_pairs_seam": len(m3_idx),
        "choices": {"exec": EXEC, "draws": DRAWS, "frames_per_ep": FRAMES_PER_EP,
                    "seam_per_ep": SEAM_PER_EP, "seam_noise": SEAM_NOISE, "seed": SEED,
                    "arbitrary": True},
        "gt_err_ticks": m12["err_ticks_mean"], "gt_err_ticks_per_motor": m12["err_ticks_per_ch"],
        "spread_ticks": m12["spread_ticks"], "spread_ticks_per_motor": m12["spread_ticks_per_ch"],
        "seam_ticks": m3.get("seam_mean_abs_ticks_all"),
        "seam_noise_floor_ticks": m3.get("seam_noise_floor_ticks"),
        "seam_gt_one_step_ticks": m3.get("seam_gt_one_step_ticks"),
        "elapsed_s": round(time.time() - t0, 1),
        "definitions": {
            "gt_err_ticks": f"mean |pred - demo| over first {EXEC} steps, padded steps masked, mean of {DRAWS} draws",
            "spread_ticks": f"same input x{DRAWS} noises, mean pairwise |chunk_i - chunk_j|, first {EXEC} steps",
            "seam_ticks": f"|chunk(t)[{EXEC - 1}] - chunk(t+{EXEC})[0]|, independent noise",
        },
        "per_frame_m12": per_frame,
    }
    print(f"[ext_eval] {label}: gt_err {res['gt_err_ticks']:.2f}  spread {res['spread_ticks']:.2f}  "
          f"seam {res['seam_ticks']:.2f} (demo one-step {res['seam_gt_one_step_ticks']:.2f})  ticks  "
          f"[{len(m12_idx)} frames, {len(m3_idx)} pairs, num_steps {res['num_steps']}"
          f"{'+' + str(res['coda_steps']) if res['coda_steps'] else ''}, {res['elapsed_s']:.0f}s]", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    if args.summary:
        p = Path(args.summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        row = {k: res[k] for k in ("label", "checkpoint", "num_steps", "coda_steps", "gt_err_ticks",
                                   "spread_ticks", "seam_ticks", "seam_gt_one_step_ticks",
                                   "n_frames_m12", "n_pairs_seam")}
        row["ext"] = "+".join(res["ext"]) or "none"
        row["gt_err_per_motor"] = " ".join(f"{x:.2f}" for x in res["gt_err_ticks_per_motor"])
        new = not p.exists()
        with p.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    del lp
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
