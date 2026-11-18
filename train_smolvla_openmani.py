#!/usr/bin/env python
"""Thin wrapper: run lerobot's own training loop with diag/ instrumentation attached.

This does NOT reimplement training. It calls
``lerobot.scripts.lerobot_train.train()`` exactly as ``lerobot-train`` does, and
the entire lerobot CLI is passed through untouched.

Why a wrapper and not a callback / a patch of the lerobot repo
--------------------------------------------------------------
* ``lerobot_train.train()`` has **no callback or hook mechanism** (checked:
  ``src/lerobot/scripts/lerobot_train.py`` has no callback plumbing; the only
  extension points are wandb and the metrics tracker).
* The policy and the optimizer are both created *inside* ``train()``, so a wrapper
  cannot simply build a policy, attach hooks and hand it over.
* Editing the lerobot repo is out (it is an editable install; upstream would
  conflict).

So the least invasive option left is to rebind two names **inside the
lerobot_train module namespace** before calling ``train()``:

  ``lerobot_train.make_policy``    -> capture the policy the moment it is built,
                                      attach the probe's forward hooks, and run the
                                      inline preflight against the live config +
                                      dataset metadata.
  ``lerobot_train.update_policy``  -> bracket each training step with
                                      ``probe.begin_step`` / ``probe.end_step`` and
                                      register the optimizer step pre-hook (grads
                                      are zeroed inside ``update_policy``, so they
                                      have to be read from inside it).

Both wrappers delegate to the originals; no lerobot behaviour is changed.

Resource monitoring reuses the scripts already in this repo
(``resource_monitor.sh``, ``watchdog.sh``) instead of reimplementing them; pass
``--diag-resource-log FILE`` to have the wrapper start ``resource_monitor.sh``
alongside training and stop it on exit.

Usage::

    python train_smolvla_openmani.py \
        --diag-out runs/probe.jsonl --diag-every 50 --diag-pred-every 500 \
        -- \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=me/openmani \
        --batch_size=8 --steps=20000 --output_dir=outputs/smolvla_openmani

Everything after ``--`` (or simply everything the wrapper does not recognise) is
forwarded verbatim to lerobot-train.
"""

from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys

import numpy as np
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--diag-out", default="diag_probe.jsonl")
    ap.add_argument("--diag-every", type=int, default=50)
    ap.add_argument("--diag-pred-every", type=int, default=0,
                    help="run the action-chunk prediction check every N steps (0=off). "
                         "It costs one full denoising rollout, so keep it rare.")
    ap.add_argument("--diag-mode", choices=["joint", "ee"], default="joint")
    ap.add_argument("--diag-delta-baseline", type=float, default=None,
                    help="external per-step motion baseline in physical units "
                         "(EE data measured on this project: 0.0185 m median |dxyz|)")
    ap.add_argument("--diag-gripper-index", type=int, default=-1)
    ap.add_argument("--diag-expert-layers", type=int, nargs="*", default=[0, 1, 8, 15])
    ap.add_argument("--diag-rss-warn-gb", type=float, default=25.0)
    ap.add_argument("--diag-tag", default="")
    ap.add_argument("--diag-resource-log", default=None,
                    help="start the existing resource_monitor.sh writing to this CSV")
    ap.add_argument("--diag-inline-preflight", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--diag-abort-on-preflight-fail", action="store_true")
    ap.add_argument("--no-diag", action="store_true", help="disable all instrumentation")
    ap.add_argument("--camera-dropout-p", type=float, default=0.0,
                    help="per-sample probability of blanking ONE randomly chosen camera "
                         "during training (0 = off). Which view is dropped is uniform over "
                         "the observation.images.* keys; never more than one at a time, so "
                         "the policy always keeps a usable view.")
    ap.add_argument("--camera-dropout-fill", choices=["gray", "zeros", "noise"], default="gray",
                    help="what a dropped view is replaced with. gray=0.5 constant (no "
                         "information, no fake lighting extreme), zeros=black, noise=uniform.")
    ap.add_argument("--camera-dropout-seed", type=int, default=0)
    ap.add_argument("--state-noise-p", type=float, default=0.0,
                    help="per-sample probability of corrupting observation.state during "
                         "training (0 = off). Counterpart of --camera-dropout-p: the "
                         "measured problem is that the policy leans on its own joints "
                         "far harder than on the cameras.")
    ap.add_argument("--state-noise-sigma", type=float, default=0.0,
                    help="gaussian sigma for a corrupted sample, as a FRACTION of the "
                         "normalised state's own scale (1.0 = one standard deviation). "
                         "0 means replace the state with the training mean instead of "
                         "adding noise, i.e. a hard state dropout.")
    ap.add_argument("--state-noise-seed", type=int, default=0)
    ap.add_argument("--old-episode-weight", type=float, default=1.0,
                    help="down-weight the episodes before --old-episode-boundary by this factor "
                         "(1.0 = off). Implemented as the fraction of old FRAMES each epoch "
                         "contains, redrawn every epoch, because lerobot's sampler takes a "
                         "boolean episode mask and SmolVLA exposes no per-sample loss.")
    ap.add_argument("--old-episode-boundary", type=int, default=0,
                    help="episodes [0, N) count as old. On a merged dataset this is the episode "
                         "count of the FIRST source (merge appends).")
    ap.add_argument("--gripper-goal-relabel", action="store_true",
                    help="rewrite the gripper ACTION label while it is closing or holding to "
                         "--gripper-close-target. The recorder saved only Present_Position, so "
                         "a held object leaves the label at the contact value (~1710 for the "
                         "small tape) although the keyboard goal kept rising by 192/tick to 2700 "
                         "(dex_ref/control_keyboard.py). Opening and resting-open are untouched.")
    ap.add_argument("--aux-keypose-weight", type=float, default=0.0,
                    help="0 = off. Weight of the state-blocked hindsight keypose loss: an extra head "
                         "reads ONLY the VLM's image + language tokens (the prefix mask already keeps "
                         "them from attending to the state token) and regresses the arm joints "
                         "(id11-24, normalised by the dataset state stats) at the NEXT gripper event "
                         "of the demo (grasp or release onset; episode end if none). Training only; "
                         "the head is dropped from saved checkpoints, so deployment is unchanged.")
    ap.add_argument("--aux-head-hidden", type=int, default=256)
    # Per-element flow-matching loss weights (2026-10-05). Training batches only, so eval_loss
    # stays the plain mean and remains comparable with unweighted runs. The product of the
    # enabled factors is renormalised to mean 1 over the batch's valid (non-pad) elements.
    ap.add_argument("--loss-step-decay", type=float, default=0.0,
                    help="0 = off. Chunk step k (0-based) weighted gamma**k (user: 0.9).")
    ap.add_argument("--loss-delta-weight", type=float, default=0.0,
                    help="0 = off. Element (frame f, joint j) weighted 1 + A*min(|a[f+1,j]-a[f,j]| "
                         "/ mean_j|da|, cap): joints that change fast in the demo get more loss. "
                         "Per-joint mean normalisation makes a 3-tick arm step and a 15-tick "
                         "gripper step comparable. Uses the relabelled gripper if relabel is on.")
    ap.add_argument("--loss-delta-cap", type=float, default=5.0,
                    help="upper clip of the normalised |da| score (ARBITRARY).")
    ap.add_argument("--loss-episode-min", type=float, default=1.0,
                    help="1 = off. Cosine decay over the episode: frame position p=f/(L-1), "
                         "weight = m + (1-m)*(1+cos(pi*p))/2 -> 1 at start, m at end (user: 0.5).")
    ap.add_argument("--loss-early-peak", type=float, default=1.0,
                    help="1 = off. Episode-start window: frames f < W weighted K, then a cosine "
                         "taper back to 1 over T frames, 1 afterwards. Motivation (2026-10-10): the "
                         "base yaw that points the arm at the object settles by frame 17 (median, d9 "
                         "0912), i.e. within ~3 executed chunks of 8 steps.")
    ap.add_argument("--loss-early-frames", type=int, default=24,
                    help="W: window length in frames (user: 3 chunks x 8 executed steps = 24).")
    ap.add_argument("--loss-early-taper", type=int, default=8,
                    help="T: cosine taper length after the window, frames (ARBITRARY).")
    ap.add_argument("--gripper-close-target", type=float, default=2700.0)
    ap.add_argument("--gripper-open-thresh", type=float, default=1150.0,
                    help="gripper ticks at or below this count as open (demos rest at 1000).")
    ap.add_argument("--gripper-tol", type=float, default=3.0,
                    help="ticks of jitter tolerated when deciding closing/holding vs opening.")
    ap.add_argument("--gripper-index", type=int, default=4)
    ap.add_argument("--old-episode-seed", type=int, default=0,
                    help="seed for the per-epoch old-frame draw.")
    ap.add_argument("--smooth-loss-weight", type=float, default=0.0,
                    help="weight of a temporal-smoothness penalty on the predicted action chunk "
                         "(0 = off). Measured motivation: the policy's output is flat across "
                         "frequency (white noise, 86-104 ticks) while the demonstrations fall "
                         "monotonically 79 -> 15, so the error that needs killing is the part "
                         "that changes fast from step to step.")
    ap.add_argument("--smooth-loss-order", type=int, choices=[1, 2], default=1,
                    help="1 = penalise the first difference along the chunk axis (velocity), "
                         "2 = the second difference (acceleration).")
    ap.add_argument("--smooth-loss-max-channel", type=int, default=5,
                    help="apply the smoothness penalty to the first N action channels only "
                         "(default 5 = the motors). On the 6-D done_flag datasets the flag "
                         "channel has 16.7x the motors' mean squared first difference "
                         "(MEASURED: 0.0529 vs 0.0032 normalized, all 125,979 frames of "
                         "lerobot_v3_d6_done), so including it would spend ~77%% of the "
                         "penalty flattening a channel that is supposed to be a step.")
    ap.add_argument("--eval-split-mode", choices=["last", "random"], default="last",
                    help="how --dataset.eval_split picks the held-out episodes. "
                         "last (DEFAULT, lerobot's own behaviour, byte-for-byte unchanged): "
                         "the LAST ceil(n*eval_split) episodes of each task by episode index. "
                         "random: a seeded random draw of the same size per task. Use random "
                         "on a MERGED dataset: episode index there encodes which source the "
                         "episode came from (the appended dataset always sits last), so 'last' "
                         "silently hands the whole eval set to one source. MEASURED on "
                         "d3_joint(972ep) + d4_0829(108ep) at eval_split=0.05: 'last' puts 72 "
                         "of 90 eval episodes (80%) in the 108-episode half, which is 17.5% of "
                         "the training frames.")
    ap.add_argument("--eval-split-seed", type=int, default=0,
                    help="RNG seed for the per-task draw and the task order used by "
                         "--eval-episodes (ARBITRARY default; the split is logged and "
                         "re-derivable from it).")
    ap.add_argument("--eval-episode-indices", default="",
                    help="comma-separated episode indices to hold out, INSTEAD of drawing them. "
                         "Use this to give two runs on different datasets the IDENTICAL "
                         "held-out episodes, which is the only way their eval_loss values are "
                         "comparable at all. On a merged dataset remember the offset: merge "
                         "appends, so episode i of the second source is 'first source size' + i. "
                         "Overrides --eval-episodes and --eval-split-mode.")
    ap.add_argument("--eval-episodes", type=int, default=0,
                    help="hold out EXACTLY this many episodes in total (0 = off, use "
                         "eval_split's per-task fraction). lerobot's own split cannot go "
                         "below one episode per task -- ceil() guarantees it -- so on a "
                         "54-task dataset eval_split alone never yields fewer than 54 "
                         "episodes. The episodes are spread one per task over a "
                         "seed-shuffled task order, so a request smaller than the task "
                         "count covers that many DISTINCT tasks with one episode each.")
    # smolvla_ext (2026-10-11): PVI / Coda / ROCKET attached to the stock policy at runtime
    # (lerobot/ untouched; settings saved as pretrained_model/smolvla_ext.json). No --ext-* module
    # flag = the extension code is never imported and training is exactly as before.
    ap.add_argument("--ext-pvi", action="store_true",
                    help="PVI: copy expert reading SigLIP tokens, zero-init per-layer injections")
    ap.add_argument("--ext-coda", action="store_true",
                    help="Coda: correction AE (copy of the main AE, zero-init output) after the main AE. "
                         "Pair with --policy.num_steps=5 (5 main + --ext-coda-steps correction).")
    ap.add_argument("--ext-rocket", action="store_true",
                    help="ROCKET: align VLM layer inputs with precomputed VGGT features (training only)")
    ap.add_argument("--ext-main-ae", choices=["train", "freeze", "lr"], default="train",
                    help="main action expert (lm_expert + 4 action projections): train at optimizer_lr, "
                         "freeze, or its own group at --ext-main-ae-lr. VLM/SigLIP/connector/state_proj "
                         "are not affected.")
    ap.add_argument("--ext-main-ae-lr", type=float, default=1e-5)
    ap.add_argument("--ext-new-lr", type=float, default=1e-4, help="lr of every new extension module")
    ap.add_argument("--ext-attn", choices=["eager", "sdpa"], default="eager",
                    help="attention kernel (smolvla_ext/fast.py); sdpa = fused masked attention")
    ap.add_argument("--ext-compile", choices=["off", "layers", "full"], default="off",
                    help="torch.compile per MLP / vision tower / connector, applied after attach")
    ap.add_argument("--ext-rocket-targets", default=None, help="VGGT teacher dir (targets.npy, filled.npy, meta.json)")
    ap.add_argument("--ext-rocket-pairs", default="3:4,7:11,11:17,15:23",
                    help="VLM layer whose INPUT is aligned : VGGT aggregator layer, comma separated")
    ap.add_argument("--ext-rocket-alpha", type=float, default=0.125)
    ap.add_argument("--ext-coda-steps", type=int, default=5)
    return ap


def _inline_preflight(cfg, ds_meta, abort: bool) -> None:
    """Run the dataset/config half of diag.preflight against the LIVE objects."""
    from diag import preflight as pf

    rep = pf.Report()
    try:
        meta = pf.DatasetMeta(Path(ds_meta.root), getattr(ds_meta, "repo_id", None))
    except Exception as e:  # noqa: BLE001
        print(f"[diag] inline preflight skipped: {e!r}")
        return
    rename_map = {}
    pf.check_features(rep, cfg, meta, rename_map)
    pf.check_stats(rep, cfg, meta, rename_map)
    pf.check_chunking(rep, cfg, meta)
    pf.check_channel_order(rep, meta, -1)
    print("[diag] inline preflight (live config + dataset metadata):")
    print(rep.render())
    if rep.failed:
        msg = "[diag] INLINE PREFLIGHT FAILED: " + ", ".join(r.name for r in rep.failed)
        if abort:
            raise SystemExit(msg)
        print(msg + "  (continuing; pass --diag-abort-on-preflight-fail to stop here)")


def _apply_camera_dropout(batch, keys, p, fill, gen):
    """Blank one randomly chosen camera per sample, in place, with probability p.

    Applied to the training batch only (eval never routes through update_policy), so the
    held-out loss stays measured on complete observations and remains comparable to runs
    without dropout. At most one view is dropped per sample: dropping two of three would
    leave a single viewpoint and change the task rather than test redundancy.
    """
    if not keys:
        return 0
    first = batch[keys[0]]
    n = first.shape[0]
    dev = first.device
    hit = torch.rand(n, generator=gen, device="cpu") < p
    if not bool(hit.any()):
        return 0
    which = torch.randint(len(keys), (n,), generator=gen)
    dropped = 0
    for j, k in enumerate(keys):
        sel = hit & (which == j)
        if not bool(sel.any()):
            continue
        idx = sel.nonzero(as_tuple=True)[0].to(dev)
        img = batch[k]
        if fill == "zeros":
            img[idx] = 0.0
        elif fill == "noise":
            img[idx] = torch.rand(
                (idx.numel(), *img.shape[1:]), device=dev, dtype=img.dtype
            )
        else:
            img[idx] = 0.5
        dropped += int(idx.numel())
    return dropped


def _apply_state_noise(batch, p, sigma_frac, state_std, gen):
    """Corrupt observation.state on a fraction of the training samples, in place.

    Why: input_ablation on d19_full measured state_swap at 1.34x the commanded motion
    against img_swap 0.67x and lang_swap 0.33x -- the policy extrapolates its own joints
    and barely reads the scene. Making the state unreliable on some samples removes the
    shortcut and is the state-side analogue of --camera-dropout-p.

    Two modes, mutually exclusive per sample and both in distribution in the sense that
    they keep the tensor on the same scale as real states:
      noise  x <- x + N(0, sigma_frac * per-channel std)
      drop   x <- the training mean (normalised zero), i.e. "I do not know where I am"
    `p` is the probability a sample is touched at all; `drop_share` of those get the mean.

    Training batches only (eval never routes through update_policy), so the held-out loss
    stays measured on clean observations and remains comparable to runs without this.
    """
    key = "observation.state"
    x = batch.get(key)
    if x is None or not torch.is_tensor(x):
        return 0
    n = x.shape[0]
    hit = torch.rand(n, generator=gen, device="cpu") < p
    if not bool(hit.any()):
        return 0
    idx = hit.nonzero(as_tuple=True)[0].to(x.device)
    if sigma_frac > 0:
        std = state_std.to(x.device, x.dtype) if state_std is not None else torch.ones_like(x[0])
        noise = torch.randn((idx.numel(), *x.shape[1:]), device=x.device, dtype=x.dtype)
        x[idx] = x[idx] + sigma_frac * std * noise
    else:
        x[idx] = 0.0
    return int(idx.numel())


def _gripper_intent(states, episodes, open_thr, tol):
    """Per frame t: True if the operator's gripper goal at t is 'closed'.

    Uses the WHOLE episode, so a pause is attributed to the last real movement:
    closing step (s[t+1] > s[t] + tol) -> closed; opening step -> open;
    stationary -> keep the previous mode (episode start: closed iff s[0] > open_thr).
    A pause half-way through opening stays 'open'; a stall on an object stays 'closed'.
    """
    intent = np.zeros(len(states), dtype=bool)
    for ep in np.unique(episodes):
        ix = np.nonzero(episodes == ep)[0]
        s = states[ix]
        mode = bool(s[0] > open_thr)
        for j in range(len(ix)):
            if j + 1 < len(ix):
                d = s[j + 1] - s[j]
                if d > tol:
                    mode = True
                elif d < -tol:
                    mode = False
            intent[ix[j]] = mode
    return intent


def _frame_table(root, open_thr, tol, idx):
    """Global frame index -> (gripper intent 'closed', episode, arm state) from the parquet."""
    import glob as _glob
    import pandas as _pd

    files = sorted(_glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    df = _pd.concat(_pd.read_parquet(f, columns=["index", "episode_index", "observation.state"])
                    for f in files).sort_values("index")
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    ep = df["episode_index"].to_numpy().astype(np.int64)
    gi = df["index"].to_numpy().astype(np.int64)
    intent = _gripper_intent(st[:, idx], ep, open_thr, tol)
    base, n = int(gi.min()), int(gi.max() - gi.min()) + 1
    t_int = np.zeros(n, bool); t_ep = np.full(n, -1, np.int64)
    t_int[gi - base] = intent; t_ep[gi - base] = ep
    return {"base": base, "intent": t_int, "ep": t_ep, "st": st, "gi": gi, "epi": ep,
            "intent_raw": intent}


class _RelabelPre:
    """Wraps lerobot's preprocessor: relabels the RAW gripper action, then normalises.

    Runs in the MAIN process on the collated batch. (A LeRobotDataset.__getitem__ patch does
    NOT reach lerobot's dataloader workers: they are started with 'spawn' and re-import the
    class -- found 2026-09-30, the first v2_base ran without relabel and was discarded.)
    action[b, k] is the frame index[b] + k; it is relabelled where that frame's intent is
    'closed', it lies in the same episode, and it is not padding.
    """

    def __init__(self, pre, tab, target, idx):
        self._pre, self._tab, self._target, self._idx = pre, tab, float(target), idx
        self._calls = self._steps = self._valid = 0

    def __call__(self, batch, *a, **kw):
        act = batch.get("action") if isinstance(batch, dict) else None
        if torch.is_tensor(act) and "index" in batch:
            t = self._tab
            g = batch["index"].detach().cpu().numpy().astype(np.int64).reshape(-1) - t["base"]
            K = act.shape[1] if act.dim() == 3 else 1
            pos = np.clip(g[:, None] + np.arange(K)[None, :], 0, len(t["intent"]) - 1)
            m = t["intent"][pos] & (t["ep"][pos] == t["ep"][np.clip(g, 0, len(t["ep"]) - 1)][:, None])
            pad = batch.get("action_is_pad")
            if torch.is_tensor(pad):
                m &= ~pad.detach().cpu().numpy().astype(bool).reshape(m.shape)
            self._valid += int(m.size)
            if m.any():
                act = act.clone()
                mt = torch.from_numpy(m).to(act.device)
                if act.dim() == 3:
                    ch = act[..., self._idx]
                    ch[mt] = self._target
                    act[..., self._idx] = ch
                else:
                    act[mt[:, 0], self._idx] = self._target
                batch["action"] = act
                self._steps += int(m.sum())
            self._calls += 1
            if self._calls in (1, 10, 100, 1000) or self._calls % 5000 == 0:
                print(f"[aug] gripper relabel: {self._steps}/{self._valid} chunk steps set to "
                      f"{self._target} after {self._calls} batches", flush=True)
        return self._pre(batch, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._pre, name)


def _attach_gripper_relabel(lerobot_train, root, target, open_thr, tol, idx):
    tab = _frame_table(root, open_thr, tol, idx)
    orig = lerobot_train.make_pre_post_processors

    @functools.wraps(orig)
    def _mk(*a, **kw):
        pre, post = orig(*a, **kw)
        return _RelabelPre(pre, tab, target, idx), post

    lerobot_train.make_pre_post_processors = _mk
    print(f"[aug] gripper relabel on (main process): intent 'closed' -> {target} "
          f"(open<= {open_thr}, tol {tol}, channel {idx}); {int(tab['intent_raw'].sum())}/"
          f"{len(tab['intent_raw'])} frames 'closed' in {root}", flush=True)
    return tab


def _loss_weight_table(root, relabel, target, open_thr, tol, gidx, cap):
    """Per global frame: normalised |da| score per action channel and the episode position p."""
    import glob as _glob
    import pandas as _pd

    files = sorted(_glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    df = _pd.concat(_pd.read_parquet(f, columns=["index", "episode_index", "frame_index",
                                                 "action", "observation.state"])
                    for f in files).sort_values("index")
    act = np.stack(df["action"].to_numpy()).astype(np.float64)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    ep = df["episode_index"].to_numpy().astype(np.int64)
    fi = df["frame_index"].to_numpy().astype(np.int64)
    gi = df["index"].to_numpy().astype(np.int64)
    if relabel:  # same rule as _RelabelPre, so the weights see the labels the loss sees
        act[_gripper_intent(st[:, gidx], ep, open_thr, tol), gidx] = target
    nxt = np.vstack([act[1:], act[-1:]])
    same = np.append(ep[1:] == ep[:-1], False)
    da = np.abs(nxt - act) * same[:, None]           # 0 on each episode's last frame
    mean_da = da[same].mean(0)
    score = np.minimum(da / np.maximum(mean_da, 1e-9), cap)
    L = np.zeros_like(fi)
    for e in np.unique(ep):
        m = ep == e
        L[m] = fi[m].max() + 1
    p = fi / np.maximum(L - 1, 1)
    base, n = int(gi.min()), int(gi.max() - gi.min()) + 1
    t_score = np.zeros((n, act.shape[1]), np.float32); t_p = np.zeros(n, np.float32)
    t_fi = np.zeros(n, np.float32)
    t_score[gi - base] = score; t_p[gi - base] = p; t_fi[gi - base] = fi
    return {"base": base, "score": t_score, "p": t_p, "fi": t_fi, "mean_da": mean_da,
            "capped": float((da / np.maximum(mean_da, 1e-9) > cap).mean())}


def _attach_loss_weight(policy, holder):
    """Multiply the per-element flow-matching loss (B, T, max_action_dim) by holder['w'].

    holder['w'] is set by the batch hook for training batches only and consumed once, so eval
    batches (which never pass through update_policy) see the unweighted loss.
    """
    inner = policy.model
    orig = inner.forward

    @functools.wraps(orig)
    def _fwd(*a, **kw):
        losses = orig(*a, **kw)
        w = holder.pop("w", None)
        if w is not None and policy.training:
            full = torch.ones_like(losses)
            full[:, :, : w.shape[-1]] = w.to(losses.dtype)
            losses = losses * full
            holder["applied"] = holder.get("applied", 0) + 1
        return losses

    inner.forward = _fwd
    print("[lossw] per-element loss weights attached to the flow-matching forward", flush=True)
    return policy


def _keypose_table(root, open_thr, tol, gidx_col=4):
    """Global frame index -> normalised arm joints at the next gripper event (hindsight label)."""
    import glob as _glob
    import json as _json
    import pandas as _pd

    files = sorted(_glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    df = _pd.concat(_pd.read_parquet(f, columns=["index", "episode_index", "observation.state"])
                    for f in files).sort_values("index")
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    ep = df["episode_index"].to_numpy(); gi = df["index"].to_numpy().astype(np.int64)
    intent = _gripper_intent(st[:, gidx_col], ep, open_thr, tol)
    stats = _json.loads(Path(root, "meta/stats.json").read_text())["observation.state"]
    mean = np.asarray(stats["mean"], np.float64)[:4]; std = np.asarray(stats["std"], np.float64)[:4]
    tgt = np.zeros((len(st), 4), np.float32)
    n_event = 0
    for e in np.unique(ep):
        ix = np.nonzero(ep == e)[0]
        it = intent[ix]
        sw = [j for j in range(1, len(ix)) if it[j] != it[j - 1]]
        n_event += len(sw)
        swset, nxt = set(sw), len(ix) - 1
        for j in range(len(ix) - 1, -1, -1):
            tgt[ix[j]] = ((st[ix[nxt], :4] - mean) / std).astype(np.float32)
            if j in swset:
                nxt = j
    base = int(gi.min())
    table = np.zeros((int(gi.max()) - base + 1, 4), np.float32)
    table[gi - base] = tgt
    print(f"[aux] keypose table from {root}: {len(gi)} frames, {len(np.unique(ep))} episodes, "
          f"{n_event} gripper events")
    return base, table


class AuxKeyposeHead(torch.nn.Module):
    """Attention-pool over image tokens + masked mean over language tokens -> 4 joints."""

    def __init__(self, d, hidden, n_img):
        super().__init__()
        self.n_img = n_img
        self.q = torch.nn.Parameter(torch.randn(d) / d ** 0.5)
        self.k = torch.nn.Linear(d, d, bias=False)
        self.mlp = torch.nn.Sequential(torch.nn.LayerNorm(2 * d), torch.nn.Linear(2 * d, hidden),
                                       torch.nn.GELU(), torch.nn.Linear(hidden, 4))

    def forward(self, prefix, lang_mask):
        x = prefix.float()
        img = x[:, : self.n_img]
        a = torch.softmax(self.k(img) @ self.q / img.shape[-1] ** 0.5, dim=1)
        img_pool = (a.unsqueeze(-1) * img).sum(1)
        L = lang_mask.shape[1]
        lang = x[:, self.n_img : self.n_img + L]
        m = lang_mask.to(x.dtype).unsqueeze(-1)
        lang_pool = (lang * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.mlp(torch.cat([img_pool, lang_pool], -1))


def _attach_aux_keypose(policy, weight, hidden, n_img):
    """Hook the SmolVLA training forward: capture the prefix output, add weight * aux MSE."""
    inner = policy.model
    vwe = inner.vlm_with_expert
    d = int(vwe.config.text_config.hidden_size)
    policy.aux_head = AuxKeyposeHead(d, hidden, n_img).to(next(policy.parameters()).device)
    cap = {"on": False, "prefix": None}

    orig_vwe = vwe.forward

    @functools.wraps(orig_vwe)
    def _vwe(*a, **kw):
        out = orig_vwe(*a, **kw)
        if cap["on"]:
            emb = out[0]
            if isinstance(emb, (list, tuple)) and emb[0] is not None:
                cap["prefix"] = emb[0]
        return out

    vwe.forward = _vwe

    orig_fwd = policy.forward

    @functools.wraps(orig_fwd)
    def _fwd(batch, *a, **kw):
        tgt = batch.get("aux.keypose") if isinstance(batch, dict) else None
        if not (policy.training and tgt is not None):
            return orig_fwd(batch, *a, **kw)
        cap["on"], cap["prefix"] = True, None
        try:
            loss, ld = orig_fwd(batch, *a, **kw)
        finally:
            cap["on"] = False
        pre = cap.pop("prefix", None); cap["prefix"] = None
        if pre is None:
            raise RuntimeError("[aux] prefix output was not captured; SmolVLA forward changed")
        lm = batch["observation.language.attention_mask"]
        if pre.shape[1] != n_img + lm.shape[1] + 1:
            raise RuntimeError(f"[aux] prefix length {pre.shape[1]} != {n_img} image + "
                               f"{lm.shape[1]} language + 1 state tokens; layout assumption broken")
        pred = policy.aux_head(pre, lm)
        aux = torch.nn.functional.mse_loss(pred, tgt.to(pred.dtype))
        ld["aux_keypose_loss"] = float(aux.detach())
        if loss.dim() == 0:
            loss = loss + weight * aux
        else:
            loss = loss + weight * torch.nn.functional.mse_loss(
                pred, tgt.to(pred.dtype), reduction="none").mean(-1)
        return loss, ld

    policy.forward = _fwd

    orig_sd = policy.state_dict

    def _sd(*a, **kw):
        sd = orig_sd(*a, **kw)
        for k in [k for k in sd if k.startswith("aux_head.")]:
            del sd[k]
        return sd

    policy.state_dict = _sd
    n = sum(p.numel() for p in policy.aux_head.parameters())
    print(f"[aux] keypose head attached: weight={weight} d={d} hidden={hidden} "
          f"image tokens={n_img} params={n}; dropped from saved checkpoints")


class SmoothnessPenalty:
    """Penalise fast step-to-step variation in the predicted action chunk.

    SmolVLA regresses the flow-matching velocity at one sampled timestep, so there is no
    predicted action to smooth directly. There is an exact one: with lerobot's convention
    (`x_t = t*noise + (1-t)*a`, `u_t = noise - a`) the clean action is recovered as
    `a = x_t - t*u_t`, so the model's implied action is `a_hat = x_t - t*v_t`. Penalising
    `diff(a_hat) - diff(a_gt)` is therefore well defined at any timestep.

    Note what this does NOT do: because `a_hat - a = -t*(v_t - u_t)`, the penalty only sees the
    part of the velocity error that VARIES along the chunk. A constant offset is invisible to
    it and stays the base MSE's job -- which is the point, since the measured defect is
    high-frequency, not bias.

    Applied in training only. Eval batches go through the same `policy.forward`, and adding a
    term there would make eval_loss incomparable with runs that have the penalty off.
    """

    def __init__(self, policy, weight: float, order: int, max_channel: int = 5):
        self.policy, self.weight, self.order = policy, float(weight), int(order)
        self.x_t = None
        self.time = None
        self.v_t = None
        self.max_channel = int(max_channel)
        model = policy.model
        orig_embed = model.embed_suffix

        def embed_suffix(noisy_actions, timestep, *a, **kw):
            self.x_t, self.time = noisy_actions, timestep
            return orig_embed(noisy_actions, timestep, *a, **kw)

        model.embed_suffix = embed_suffix
        model.action_out_proj.register_forward_hook(
            lambda _m, _i, out: setattr(self, "v_t", out))

    def term(self, batch):
        """-> scalar tensor in the graph, or None if this step cannot be scored."""
        if self.x_t is None or self.v_t is None or self.time is None:
            return None
        dim = int(self.policy.config.action_feature.shape[0])
        t = self.time.to(torch.float32)[:, None, None]
        x_t = self.x_t.to(torch.float32)[..., :dim]
        v_t = self.v_t.to(torch.float32)[..., :dim]
        a_hat = x_t - t * v_t
        a_gt = self.policy.prepare_action(batch).to(torch.float32)[..., :dim]
        # Motor channels only. On a 6-D dataset the done_flag channel carries 16.7x the mean
        # squared first difference of the motors (0.0529 vs 0.0032 in normalized units, measured
        # over all 125,979 frames of lerobot_v3_d6_done): left in, it would take ~77% of this
        # penalty's target energy and the smoothing the arm needs would barely reach it. A step
        # function is also exactly what the done channel is SUPPOSED to be.
        n = min(self.max_channel, a_hat.shape[-1], a_gt.shape[-1])
        a_hat, a_gt = a_hat[..., :n], a_gt[..., :n]
        if a_hat.shape[1] < self.order + 1:
            return None
        d_hat, d_gt = a_hat, a_gt
        for _ in range(self.order):
            d_hat = d_hat[:, 1:] - d_hat[:, :-1]
            d_gt = d_gt[:, 1:] - d_gt[:, :-1]
        err = (d_hat - d_gt) ** 2
        pad = batch.get("action_is_pad")
        if pad is not None:
            # a difference is only real if every step it touches is real
            valid = ~pad
            for _ in range(self.order):
                valid = valid[:, 1:] & valid[:, :-1]
            valid = valid.unsqueeze(-1).to(err.dtype)
            denom = (valid.sum() * err.shape[-1]).clamp(min=1.0)
            return (err * valid).sum() / denom
        return err.mean()

    def clear(self):
        self.x_t = self.time = self.v_t = None


def _attach_smoothness(policy, weight: float, order: int, max_channel: int = 5):
    """Wrap policy.forward so the penalty rides along with the flow-matching loss."""
    pen = SmoothnessPenalty(policy, weight, order, max_channel)
    orig_forward = policy.forward

    @functools.wraps(orig_forward)
    def forward(batch, *a, **kw):
        pen.clear()
        loss, out = orig_forward(batch, *a, **kw)
        if policy.training:
            extra = pen.term(batch)
            if extra is not None:
                out["smooth_loss"] = float(extra.detach())
                loss = loss + weight * extra
        return loss, out

    policy.forward = forward
    print(f"[smooth] temporal-smoothness penalty attached: weight={weight} order={order} "
          f"(training batches only)")
    return pen


def _attach_source_weight(lerobot_train, weight: float, boundary: int, seed: int) -> None:
    """Down-weight the episodes BEFORE `boundary` by `weight`, without touching lerobot/.

    WHY NOT THE OBVIOUS THINGS
    --------------------------
    * Repeating an episode index to oversample it does nothing: `EpisodeAwareSampler` turns
      `episode_indices_to_use` into a BOOLEAN MASK (`sampler.py:80-82`), so duplicates collapse.
    * There is no per-sample loss to scale: SmolVLA's forward returns `losses.mean()`, and the
      training loop only ever sees that scalar.
    So the weight is applied where it is actually representable -- in WHICH FRAMES THE EPOCH
    CONTAINS. Every epoch keeps all the new frames and a fresh random `weight` fraction of the
    old ones. In expectation each old frame contributes `weight` as many gradient steps as a
    new one, which is what a sample weight means; the difference from a true per-sample weight
    is variance, not bias.

    The subset is redrawn per epoch from (seed, epoch), so over a run the model still sees ALL
    the old data -- unlike physically deleting `1-weight` of the episodes, which would also
    shrink the old data's variety, not just its influence.

    `boundary` is an episode index: episodes [0, boundary) are "old". That matches how
    `merge_datasets_v3.py` appends, and the caller is expected to read it off the manifest.
    """
    from lerobot.datasets import sampler as _sampler_mod

    orig_cls = lerobot_train.EpisodeAwareSampler

    class WeightedEpisodeSampler(orig_cls):
        def __init__(self, from_idx, to_idx, *a, **kw):
            super().__init__(from_idx, to_idx, *a, **kw)
            self._from = np.asarray(from_idx, dtype=np.int64)
            self._w_starts = self._starts.copy()
            self._w_cum = self._cum_lengths.copy()
            # episode id of each retained entry, recovered from its start frame
            lens = np.diff(np.concatenate([[0], self._cum_lengths]))
            ep_of_entry = np.searchsorted(self._from, self._w_starts, side="right") - 1
            self._is_old = ep_of_entry < boundary
            self._lens = lens
            n_old = int(lens[self._is_old].sum())
            n_new = int(lens[~self._is_old].sum())
            self._n_old_ep = int(self._is_old.sum())
            self._n_new_ep = int((~self._is_old).sum())
            self._epoch_frames = int(round(n_new + weight * n_old))
            print(f"[weight] old episodes [0,{boundary}) x{weight}: "
                  f"{self._n_old_ep} old / {self._n_new_ep} new episodes, "
                  f"{n_old} old / {n_new} new frames -> {self._epoch_frames} frames per epoch "
                  f"({100.0 * weight * n_old / max(1, self._epoch_frames):.1f}% old)")
            if self._n_new_ep == 0 or self._n_old_ep == 0:
                raise SystemExit(f"[weight] boundary {boundary} leaves one side empty "
                                 f"({self._n_old_ep} old, {self._n_new_ep} new) -- wrong boundary?")

        def _epoch_positions(self, epoch: int) -> torch.Tensor:
            """All new frames + a `weight` fraction of the old ones, drawn for THIS epoch."""
            g = torch.Generator().manual_seed(
                int(np.random.SeedSequence([seed, epoch, 0x5748]).generate_state(1, dtype=np.uint64)[0])
            )
            keep = []
            base = 0
            for i, ln in enumerate(self._lens):
                ln = int(ln)
                pos = torch.arange(base, base + ln)
                if self._is_old[i]:
                    k = int(round(weight * ln))
                    if k > 0:
                        pos = pos[torch.randperm(ln, generator=g)[:k]]
                    else:
                        pos = pos[:0]
                keep.append(pos)
                base += ln
            out = torch.cat(keep)
            return out[torch.randperm(out.numel(), generator=g)]

        def __len__(self):
            return self._epoch_frames

        def __iter__(self):
            epoch = self._epoch
            self._epoch += 1
            pos = self._epoch_positions(epoch)
            for k in pos[self._start_index:].tolist():
                yield self._frame_index(int(k))
            self._start_index = 0

    lerobot_train.EpisodeAwareSampler = WeightedEpisodeSampler
    _sampler_mod.EpisodeAwareSampler = WeightedEpisodeSampler


def _attach_eval_split(lerobot_train, mode: str, seed: int, n_episodes: int,
                       explicit: list[int] | None = None) -> None:
    """Control WHICH episodes --dataset.eval_split holds out, and HOW MANY.

    lerobot's `make_train_eval_datasets` (datasets/factory.py:150) builds, per task, the list
    of episodes IN THE ORDER THEY APPEAR in `cfg.dataset.episodes` and holds out the tail:
        n_eval = math.ceil(len(eps) * eval_split); train = eps[:-n_eval]; eval = eps[-n_eval:]

    Two things are wrong with that on our merged dataset:
      * `merge_datasets` APPENDS, so every episode of the second source has a higher index
        than every episode of the first and the tail comes from one source only. MEASURED on
        d3_joint(972) + d4_0829(108) at eval_split=0.05: 72 of 90 eval episodes (80%) land in
        the 108-episode half, which is 17.5% of the training frames.
      * `ceil` gives EVERY task at least one eval episode, so with 54 tasks the eval set can
        never be smaller than 54 -- there is no eval_split that yields 40.

    Neither is fixed by reimplementing the split. Instead:
      * ORDER: `resolve_episode_indices` (datasets/utils.py:113) documents that "the input
        order is preserved" and `LeRobotDataset.episodes` stores the list as given, so a
        per-task-shuffled permutation passed through `cfg.dataset.episodes` moves a seeded
        random draw into the tail lerobot takes.
      * COUNT: `math.ceil` appears exactly ONCE in that module (factory.py:176, inside the
        per-task loop), so `factory.math` is swapped for a shim that returns our precomputed
        per-task count, one call per task, in the order our permutation fixes.

    Both are assumptions about someone else's code, so the resulting split is compared against
    the draw we intended and any disagreement aborts the run instead of training on an
    unknown split.
    """
    import math as _math
    import random as _random

    orig = lerobot_train.make_train_eval_datasets

    @functools.wraps(orig)
    def hooked(cfg, *a, **kw):
        from lerobot.datasets import factory as _factory
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        split = float(getattr(cfg.dataset, "eval_split", 0.0) or 0.0)
        if split == 0.0:
            print("[eval-split] eval_split=0 -> nothing is held out; this hook is a no-op")
            return orig(cfg, *a, **kw)

        meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=getattr(cfg.dataset, "revision", None),
            repo_type=getattr(cfg.dataset, "repo_type", None),
        )
        base = list(cfg.dataset.episodes) if cfg.dataset.episodes is not None \
            else list(range(meta.total_episodes))
        excl = set(getattr(cfg.dataset, "exclude_episodes", None) or [])
        base = [e for e in base if e not in excl]
        # SORT before grouping. This hook writes its permutation into cfg.dataset.episodes and
        # lerobot persists the whole config into every checkpoint's train_config.json, so on
        # --resume `base` arrives ALREADY SHUFFLED. Shuffling a shuffled list with the same seed
        # gives a different tail: measured 2/40 overlap with the original split, i.e. 38 held-out
        # episodes silently become training data. Sorting first makes the input to the shuffle
        # identical on a fresh run and on a resume, so the split is reproducible from the seed.
        base = sorted(base)

        ep_tasks = meta.episodes["tasks"]
        per_task: dict[str, list[int]] = {}
        for e in base:
            key = ep_tasks[e][0] if ep_tasks[e] else ""
            per_task.setdefault(key, []).append(e)

        rng = _random.Random(seed)
        task_order = sorted(per_task)
        if mode == "random":
            for key in task_order:
                rng.shuffle(per_task[key])

        # how many episodes each task contributes to eval, in task_order
        if n_episodes > 0:
            if n_episodes >= len(base):
                raise ValueError(f"--eval-episodes {n_episodes} >= {len(base)} episodes")
            # one per task, round-robin over a shuffled task order, so a request smaller
            # than the number of tasks covers as many DISTINCT tasks as it can afford
            # rather than spending several episodes on a few tasks.
            pick_order = list(task_order)
            _random.Random(seed + 1).shuffle(pick_order)
            n_eval = dict.fromkeys(task_order, 0)
            i = 0
            while sum(n_eval.values()) < n_episodes:
                key = pick_order[i % len(pick_order)]
                if n_eval[key] < len(per_task[key]) - 1:
                    n_eval[key] += 1
                i += 1
                if i > n_episodes * len(pick_order) + len(pick_order):
                    raise ValueError("cannot allocate the requested eval episodes")
            counts = [n_eval[k] for k in task_order]
        else:
            counts = [_math.ceil(len(per_task[k]) * split) for k in task_order]

        if explicit:
            # Pin the split. Every named episode must exist and must not be the only episode of
            # its task (lerobot needs at least one training episode per task).
            missing = [e for e in explicit if e not in set(base)]
            if missing:
                raise ValueError(f"--eval-episode-indices: {missing} not in the dataset "
                                 f"(or excluded); dataset has {len(base)} episodes")
            want = set(explicit)
            for key in task_order:
                eps = per_task[key]
                if want.issuperset(eps):
                    raise ValueError(f"--eval-episode-indices would hold out EVERY episode of "
                                     f"task {key!r}")
            counts = [sum(1 for e in per_task[k] if e in want) for k in task_order]
            perm, intended = [], set(explicit)
            for key in task_order:
                eps = [e for e in per_task[key] if e not in want] + \
                      [e for e in per_task[key] if e in want]
                perm.extend(eps)
            print(f"[eval-split] PINNED to {len(explicit)} given episode indices "
                  f"(--eval-episode-indices); the seeded draw is not used")
        else:
            perm, intended = [], set()
            for key, c in zip(task_order, counts):
                eps = per_task[key]
                perm.extend(eps)
                if c:
                    intended.update(eps[len(eps) - c:])

        class _CeilShim:
            """Stands in for `math` inside factory for the duration of one call."""

            def __init__(self, values):
                self._values, self._i = list(values), 0

            def ceil(self, _x):
                if self._i >= len(self._values):
                    raise RuntimeError(
                        "[eval-split] factory.math.ceil was called more times than there are "
                        "tasks; lerobot's split no longer has the shape this hook assumes."
                    )
                v = self._values[self._i]
                self._i += 1
                return v

            def __getattr__(self, name):
                return getattr(_math, name)

        cfg.dataset.episodes = perm
        cfg.dataset.exclude_episodes = None
        real_math, _factory.math = _factory.math, _CeilShim(counts)
        try:
            train_ds, eval_ds = orig(cfg, *a, **kw)
        finally:
            _factory.math = real_math

        got = set(eval_ds.episodes or [])
        if got != intended:
            raise RuntimeError(
                f"[eval-split] the split did not take: lerobot held out {len(got)} episodes, "
                f"{len(got - intended)} of which we did not choose. The assumptions behind "
                "this hook no longer hold -- refusing to train on an unknown split."
            )
        n_task = sum(1 for k, c in zip(task_order, counts) if c)
        print(f"[eval-split] mode={mode} seed={seed} eval_split={split}"
              f"{f' eval_episodes={n_episodes}' if n_episodes > 0 else ''}: "
              f"{len(train_ds.episodes or [])} train / {len(got)} eval episodes, "
              f"{n_task} of {len(task_order)} tasks represented in eval")
        print(f"[eval-split] held-out episode indices: {sorted(got)}")
        return train_ds, eval_ds

    lerobot_train.make_train_eval_datasets = hooked


def _ext_from_args(args, passthrough):
    """ExtConfig from the --ext-* flags, or None when no extension module is requested."""
    on = args.ext_pvi or args.ext_coda or args.ext_rocket
    if not on:
        dflt = build_arg_parser().parse_args([])
        changed = [n for n in ("ext_main_ae", "ext_main_ae_lr", "ext_new_lr", "ext_rocket_targets",
                               "ext_rocket_pairs", "ext_rocket_alpha", "ext_coda_steps",
                               "ext_attn", "ext_compile")
                   if getattr(args, n) != getattr(dflt, n)]
        if changed:
            raise SystemExit(f"[ext] {changed} given without --ext-pvi/--ext-coda/--ext-rocket")
        return None
    clash = [n for n, v in (("--smooth-loss-weight", args.smooth_loss_weight > 0),
                            ("--aux-keypose-weight", args.aux_keypose_weight > 0),
                            ("loss weights (--loss-*)", args.loss_step_decay > 0 or args.loss_delta_weight > 0
                             or args.loss_episode_min < 1.0 or args.loss_early_peak != 1.0)) if v]
    if clash:
        # these wrap the same forward/embed_suffix the extension replaces; never validated together
        raise SystemExit(f"[ext] not combinable (untested interaction): {clash}")
    from smolvla_ext import ExtConfig, parse_pairs

    if args.ext_rocket and not args.ext_rocket_targets:
        raise SystemExit("[ext] --ext-rocket needs --ext-rocket-targets DIR")
    return ExtConfig(pvi=args.ext_pvi, coda=args.ext_coda, rocket=args.ext_rocket,
                     main_ae=args.ext_main_ae, main_ae_lr=args.ext_main_ae_lr, new_lr=args.ext_new_lr,
                     coda_num_steps=args.ext_coda_steps,
                     rocket_pairs=[list(p) for p in parse_pairs(args.ext_rocket_pairs)],
                     rocket_alpha=args.ext_rocket_alpha, rocket_target_dir=args.ext_rocket_targets,
                     attn_impl=args.ext_attn, compile_mode=args.ext_compile)


def _ext_attach_policy(ext, policy, cfg, ds_meta, log_every):
    """Attach the extension right after make_policy (weights loaded, so copies take them)."""
    from smolvla_ext import attach
    from smolvla_ext.attach import camera_hw_from_features

    hw = camera_hw_from_features(ds_meta.features, list(policy.config.image_features)) \
        if ds_meta is not None else None
    pp = getattr(cfg, "pretrained_path", None)
    pp = pp if pp is not None and Path(str(pp)).is_dir() else None
    attach(policy, ext, pretrained_path=pp, camera_hw=hw, log_every=log_every)
    return policy


def _ext_install_rocket_io(lerobot_train, ext):
    """ROCKET teacher targets: read in the dataloader workers (TeacherDataset around the TRAIN set
    only) and carried across the preprocessor, which drops unknown top-level keys (CarryKeysPre)."""
    from smolvla_ext.rocket import CarryKeysPre, TeacherDataset

    orig_dl = lerobot_train.make_dataloaders

    @functools.wraps(orig_dl)
    def _dl(cfg, dataset, *a, **kw):
        cams = list(dataset.meta.camera_keys)
        import json as _json
        _m = _json.loads((Path(ext.rocket_target_dir) / "meta.json").read_text())
        if int(_m["num_frames"]) != int(dataset.meta.total_frames) or not _m.get("complete", False):
            raise SystemExit(f"[ext] teacher file {ext.rocket_target_dir}: num_frames {_m['num_frames']} vs "
                             f"dataset {dataset.meta.total_frames}, complete={_m.get('complete')}")
        wrapped = TeacherDataset(dataset, ext.rocket_target_dir, cams, ext.vggt_layers, ext.rocket_dim)
        print(f"[ext] rocket teacher targets read in dataloader workers from {ext.rocket_target_dir} "
              f"(cams {cams}, VGGT layers {ext.vggt_layers}, format {_m.get('format', 'fp16')})", flush=True)
        return orig_dl(cfg, wrapped, *a, **kw)

    lerobot_train.make_dataloaders = _dl
    orig_pp = lerobot_train.make_pre_post_processors

    @functools.wraps(orig_pp)
    def _pp(*a, **kw):
        pre, post = orig_pp(*a, **kw)
        return CarryKeysPre(pre), post

    lerobot_train.make_pre_post_processors = _pp


def main() -> int:
    ap = build_arg_parser()
    args, passthrough = ap.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]
    ext = _ext_from_args(args, passthrough)
    ext_log_every = 200  # lerobot's default log_freq
    for _i, _a in enumerate(passthrough):
        if _a.startswith("--log_freq="):
            ext_log_every = int(_a.split("=", 1)[1])
        elif _a == "--log_freq" and _i + 1 < len(passthrough):
            ext_log_every = int(passthrough[_i + 1])

    drop = {"p": float(args.camera_dropout_p), "keys": None, "n": 0, "reported": False,
            "fill": args.camera_dropout_fill,
            "gen": torch.Generator().manual_seed(args.camera_dropout_seed)}
    sdrop = {"p": float(args.state_noise_p), "sigma": float(args.state_noise_sigma),
             "n": 0, "reported": False,
             "gen": torch.Generator().manual_seed(args.state_noise_seed)}


    aux = {"on": args.aux_keypose_weight > 0, "base": 0, "table": None, "dev": None}
    if aux["on"]:
        _root = next((x.split("=", 1)[1] for x in passthrough if x.startswith("--dataset.root=")), None)
        if _root is None:
            raise SystemExit("[aux] --aux-keypose-weight needs --dataset.root=<path> in the args")
        aux["base"], aux["table"] = _keypose_table(_root, args.gripper_open_thresh, args.gripper_tol,
                                                   args.gripper_index)

    _root_arg = next((x.split("=", 1)[1] for x in passthrough if x.startswith("--dataset.root=")), None)

    def _relabel_attach(lerobot_train):
        if args.gripper_goal_relabel:
            if _root_arg is None:
                raise SystemExit("[aug] --gripper-goal-relabel needs --dataset.root=<path>")
            _attach_gripper_relabel(lerobot_train, _root_arg, args.gripper_close_target,
                                    args.gripper_open_thresh, args.gripper_tol, args.gripper_index)

    def _aux_batch(batch):
        """Attach the hindsight keypose target, looked up by global frame index."""
        if not aux["on"] or not isinstance(batch, dict):
            return
        gi = batch["index"]
        dev = batch["observation.state"].device
        if aux["dev"] is None:
            aux["dev"] = torch.from_numpy(aux["table"]).to(dev)
        pos = (gi.to(dev).long().reshape(-1) - aux["base"]).clamp(0, aux["dev"].shape[0] - 1)
        batch["aux.keypose"] = aux["dev"][pos]

    lw = {"on": args.loss_step_decay > 0 or args.loss_delta_weight > 0 or args.loss_episode_min < 1.0
                or args.loss_early_peak != 1.0,
          "tab": None, "dev": None, "n": 0, "holder": {}}
    if lw["on"]:
        if _root_arg is None:
            raise SystemExit("[lossw] loss weights need --dataset.root=<path> in the args")
        lw["tab"] = _loss_weight_table(_root_arg, args.gripper_goal_relabel, args.gripper_close_target,
                                       args.gripper_open_thresh, args.gripper_tol, args.gripper_index,
                                       args.loss_delta_cap)
        print(f"[lossw] on: step_decay={args.loss_step_decay} delta_weight={args.loss_delta_weight} "
              f"(cap {args.loss_delta_cap}, {100 * lw['tab']['capped']:.2f}% of |da| capped; "
              f"mean|da| per channel {np.round(lw['tab']['mean_da'], 2).tolist()} ticks) "
              f"episode_min={args.loss_episode_min} early_peak={args.loss_early_peak} "
              f"(window {args.loss_early_frames} + taper {args.loss_early_taper} frames); "
              f"renormalised to mean 1 per batch", flush=True)

    def _lw_batch(batch):
        """Build the (B, T, D) weight for this training batch and hand it to the forward hook."""
        if not lw["on"] or not isinstance(batch, dict):
            return
        act = batch["action"]
        B, T, D = act.shape
        dev = act.device
        if lw["dev"] is None:
            lw["dev"] = (torch.from_numpy(lw["tab"]["score"]).to(dev),
                         torch.from_numpy(lw["tab"]["p"]).to(dev),
                         torch.from_numpy(lw["tab"]["fi"]).to(dev))
        sc, pp, ff = lw["dev"]
        g = batch["index"].to(dev).long().reshape(-1) - lw["tab"]["base"]
        pos = (g[:, None] + torch.arange(T, device=dev)[None, :]).clamp(0, pp.shape[0] - 1)
        w = torch.ones(B, T, D, device=dev, dtype=torch.float32)
        if args.loss_step_decay > 0:
            w = w * (args.loss_step_decay ** torch.arange(T, device=dev, dtype=torch.float32))[None, :, None]
        if args.loss_delta_weight > 0:
            w = w * (1.0 + args.loss_delta_weight * sc[pos][..., :D])
        if args.loss_episode_min < 1.0:
            m = args.loss_episode_min
            w = w * (m + (1 - m) * 0.5 * (1 + torch.cos(torch.pi * pp[pos])))[..., None]
        if args.loss_early_peak != 1.0:
            W, Tp = float(args.loss_early_frames), float(max(args.loss_early_taper, 1))
            u = ((ff[pos] - W) / Tp).clamp(0, 1)          # 0 inside the window, 1 after the taper
            g = 0.5 * (1 + torch.cos(torch.pi * u))       # 1 -> 0
            w = w * (1 + (args.loss_early_peak - 1) * g)[..., None]
        pad = batch.get("action_is_pad")
        valid = (~pad.to(dev)).float()[..., None].expand_as(w) if torch.is_tensor(pad) else torch.ones_like(w)
        w = w / ((w * valid).sum() / valid.sum().clamp_min(1)).clamp_min(1e-8)
        lw["holder"]["w"] = w
        lw["n"] += 1
        if lw["n"] in (1, 10, 100, 1000) or lw["n"] % 5000 == 0:
            vs = valid.sum((0, 2)).clamp_min(1)
            per_k = ((w * valid).sum((0, 2)) / vs).cpu().numpy()
            per_j = ((w * valid).sum((0, 1)) / valid.sum((0, 1)).clamp_min(1)).cpu().numpy()
            print(f"[lossw] batch {lw['n']} (applied in forward so far: "
                  f"{lw['holder'].get('applied', 0)}): weight by chunk step "
                  f"{[round(float(x), 2) for x in per_k]} by channel "
                  f"{[round(float(x), 2) for x in per_j]} max {float(w.max()):.2f}", flush=True)
            if args.loss_early_peak != 1.0:
                ew = (ff[pos] < args.loss_early_frames).float()[..., None].expand_as(w) * valid
                print(f"[lossw] early window: {100 * float(ew.sum() / valid.sum()):.1f}% of elements "
                      f"carry {100 * float((w * ew).sum() / (w * valid).sum()):.1f}% of the weight",
                      flush=True)

    def _lw_policy(pol):
        if lw["on"]:
            _attach_loss_weight(pol, lw["holder"])
        return pol

    def _aux_policy(pol):
        if aux["on"]:
            _attach_aux_keypose(pol, args.aux_keypose_weight, args.aux_head_hidden, 3 * 64)
        return pol

    def _drop_batch(batch):
        """Blank one camera per sample. Returns nothing; edits the batch in place."""
        if drop["p"] <= 0 or not isinstance(batch, dict):
            return
        if drop["keys"] is None:
            drop["keys"] = sorted(
                k for k, v in batch.items()
                if k.startswith("observation.images") and torch.is_tensor(v) and v.dim() >= 4
            )
            print(f"[aug] camera dropout p={drop['p']} fill={drop['fill']} "
                  f"over {len(drop['keys'])} views: {drop['keys']}")
            if not drop["keys"]:
                print("[aug] WARNING: no observation.images.* tensors in batch; dropout is a no-op")
        drop["n"] += _apply_camera_dropout(batch, drop["keys"], drop["p"], drop["fill"], drop["gen"])

    def _state_batch(batch):
        """Corrupt observation.state on some samples. Edits the batch in place."""
        if sdrop["p"] <= 0 or not isinstance(batch, dict):
            return
        if not sdrop["reported"]:
            sdrop["reported"] = True
            how = (f"gaussian sigma={sdrop['sigma']} x normalised state scale"
                   if sdrop["sigma"] > 0 else "hard dropout to the training mean")
            print(f"[aug] state noise p={sdrop['p']}: {how}")
            if "observation.state" not in batch:
                print("[aug] WARNING: no observation.state in batch; state noise is a no-op")
        # the batch is already normalised here, so unit scale IS the per-channel std
        sdrop["n"] += _apply_state_noise(batch, sdrop["p"], sdrop["sigma"], None, sdrop["gen"])

    if args.no_diag:
        from lerobot.scripts import lerobot_train
        _relabel_attach(lerobot_train)

        _explicit = [int(x) for x in args.eval_episode_indices.replace(",", " ").split()]
        if _explicit or args.eval_split_mode == "random" or args.eval_episodes > 0:
            _attach_eval_split(lerobot_train, args.eval_split_mode,
                               args.eval_split_seed, args.eval_episodes, _explicit)

        if args.old_episode_weight != 1.0:
            _attach_source_weight(lerobot_train, args.old_episode_weight,
                                  args.old_episode_boundary, args.old_episode_seed)

        if ext is not None:
            _ope = lerobot_train.make_policy

            @functools.wraps(_ope)
            def _mpe(*a, **kw):
                pol = _ope(*a, **kw)
                return _ext_attach_policy(ext, pol, kw.get("cfg", a[0] if a else None),
                                          kw.get("ds_meta"), ext_log_every)

            lerobot_train.make_policy = _mpe
            if ext.rocket:
                _ext_install_rocket_io(lerobot_train, ext)

        if args.smooth_loss_weight > 0:
            _op = lerobot_train.make_policy

            @functools.wraps(_op)
            def _mp(*a, **kw):
                pol = _op(*a, **kw)
                _attach_smoothness(pol, args.smooth_loss_weight, args.smooth_loss_order,
                                   args.smooth_loss_max_channel)
                return pol

            lerobot_train.make_policy = _mp

        if lw["on"]:
            _op3 = lerobot_train.make_policy

            @functools.wraps(_op3)
            def _mp3(*a, **kw):
                return _lw_policy(_op3(*a, **kw))

            lerobot_train.make_policy = _mp3

        if aux["on"]:
            _op2 = lerobot_train.make_policy

            @functools.wraps(_op2)
            def _mp2(*a, **kw):
                return _aux_policy(_op2(*a, **kw))

            lerobot_train.make_policy = _mp2

        if drop["p"] > 0 or sdrop["p"] > 0 or aux["on"] or lw["on"]:
            _orig_up = lerobot_train.update_policy

            @functools.wraps(_orig_up)
            def _up(*a, **kw):
                _b = kw.get("batch", a[2] if len(a) > 2 else None)
                _drop_batch(_b); _state_batch(_b); _aux_batch(_b); _lw_batch(_b)
                return _orig_up(*a, **kw)

            lerobot_train.update_policy = _up
        sys.argv = [sys.argv[0], *passthrough]
        lerobot_train.main()
        return 0

    from diag.probe import Probe, ProbeConfig
    from lerobot.scripts import lerobot_train
    _relabel_attach(lerobot_train)

    _explicit = [int(x) for x in args.eval_episode_indices.replace(",", " ").split()]
    if _explicit or args.eval_split_mode == "random" or args.eval_episodes > 0:
        _attach_eval_split(lerobot_train, args.eval_split_mode,
                           args.eval_split_seed, args.eval_episodes, _explicit)

    if args.old_episode_weight != 1.0:
        _attach_source_weight(lerobot_train, args.old_episode_weight,
                              args.old_episode_boundary, args.old_episode_seed)

    state: dict = {"probe": None, "step": 0, "opt_attached": False}

    orig_make_policy = lerobot_train.make_policy
    orig_update_policy = lerobot_train.update_policy

    if ext is not None and ext.rocket:
        _ext_install_rocket_io(lerobot_train, ext)

    @functools.wraps(orig_make_policy)
    def make_policy_hooked(*a, **kw):
        policy = orig_make_policy(*a, **kw)
        cfg = kw.get("cfg", a[0] if a else None)
        ds_meta = kw.get("ds_meta")
        if ext is not None:  # before the probe, so it sees the final module tree
            _ext_attach_policy(ext, policy, cfg, ds_meta, ext_log_every)
        if args.diag_inline_preflight and cfg is not None and ds_meta is not None:
            try:
                _inline_preflight(cfg, ds_meta, args.diag_abort_on_preflight_fail)
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"[diag] inline preflight error (ignored): {e!r}")

        action_mean = action_std = action_names = None
        if ds_meta is not None:
            try:
                st = (ds_meta.stats or {}).get("action", {})
                action_mean = [float(x) for x in st["mean"]]
                action_std = [float(x) for x in st["std"]]
            except Exception:  # noqa: BLE001 - stats may be absent; probe stays in normalized units
                action_mean = action_std = None
            action_names = (ds_meta.features.get("action", {}) or {}).get("names")

        pcfg = ProbeConfig(
            out_path=args.diag_out, every=args.diag_every, pred_every=args.diag_pred_every,
            tag=args.diag_tag, mode=args.diag_mode, delta_baseline=args.diag_delta_baseline,
            gripper_index=args.diag_gripper_index,
            expert_layers=tuple(args.diag_expert_layers),
            rss_warn_gb=args.diag_rss_warn_gb,
            action_mean=action_mean, action_std=action_std,
            extra={"action_names": action_names} if action_names else {},
        )
        probe = Probe(policy, pcfg)
        hooked = probe.attach()
        state["probe"] = probe
        print(f"[diag] probe attached to {len(hooked)} modules, writing {args.diag_out} "
              f"(every={args.diag_every}, pred_every={args.diag_pred_every})")
        print(f"[diag] action unnormalization stats: "
              f"{'loaded' if action_mean else 'NOT available -> magnitudes stay normalized'}")
        if args.smooth_loss_weight > 0:
            _attach_smoothness(policy, args.smooth_loss_weight, args.smooth_loss_order,
                               args.smooth_loss_max_channel)
        return _aux_policy(_lw_policy(policy))

    @functools.wraps(orig_update_policy)
    def update_policy_hooked(*a, **kw):
        probe = state["probe"]
        if probe is None:
            _b = kw.get("batch", a[2] if len(a) > 2 else None)
            _drop_batch(_b); _state_batch(_b); _aux_batch(_b); _lw_batch(_b)
            return orig_update_policy(*a, **kw)
        batch = kw.get("batch", a[2] if len(a) > 2 else None)
        _drop_batch(batch)
        _state_batch(batch)
        _aux_batch(batch)
        _lw_batch(batch)
        optimizer = kw.get("optimizer", a[3] if len(a) > 3 else None)
        if optimizer is not None and not state["opt_attached"]:
            ok = probe.attach_optimizer(optimizer)
            state["opt_attached"] = True
            print(f"[diag] optimizer step pre-hook registered: {ok}")
        step = state["step"]
        probe.begin_step(step, batch)
        try:
            out = orig_update_policy(*a, **kw)
        finally:
            probe.end_step()
            if args.diag_pred_every > 0 and step % args.diag_pred_every == 0 and batch is not None:
                try:
                    probe.prediction_check(batch, step=step)
                except Exception as e:  # noqa: BLE001
                    print(f"[diag] prediction check failed (ignored): {e!r}")
            state["step"] = step + 1
        return out

    lerobot_train.make_policy = make_policy_hooked
    lerobot_train.update_policy = update_policy_hooked

    monitor = None
    if args.diag_resource_log:
        script = REPO / "resource_monitor.sh"
        if script.exists():
            monitor = subprocess.Popen([str(script), args.diag_resource_log, "15"])
            print(f"[diag] started {script} -> {args.diag_resource_log} (pid {monitor.pid})")
        else:
            print(f"[diag] {script} not found; skipping resource monitor")
    else:
        print("[diag] no --diag-resource-log: for machine-wide RAM/GPU logging run\n"
              f"       {REPO / 'resource_monitor.sh'} <csv> 15\n"
              f"       {REPO / 'watchdog.sh'} --log <trainlog> --check-stall --check-resource")

    sys.argv = [sys.argv[0], *passthrough]
    try:
        lerobot_train.main()
    finally:
        if drop["p"] > 0:
            print(f"[aug] camera dropout: {drop['n']} views blanked over the run")
        if sdrop["p"] > 0:
            print(f"[aug] state noise: {sdrop['n']} states corrupted over the run")
        if state["probe"] is not None:
            state["probe"].close()
        if monitor is not None:
            monitor.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
