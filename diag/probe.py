#!/usr/bin/env python
"""Runtime anomaly probe for SmolVLA (lerobot) training / inference.

What it records (one JSONL line per record, console gets a one-line summary):

1. ``kind="step"``  -- every ``--every`` optimizer steps:
   * input sanity: state min/max/mean/std/NaN, image dtype+range, ``action_is_pad``
     ratio, which camera keys are actually present in the batch vs. which the
     policy config will actually consume;
   * forward-hook stats for ``state_proj`` / ``action_in_proj`` / ``action_out_proj``
     / ``action_time_mlp_*`` and the action-expert q/k/v/o + MLP projections:
     output RMS, min, max, NaN/Inf count, dead-unit fraction;
   * zero-pad integrity: SmolVLA pads state (5 or 6 dims) and action to
     ``max_state_dim`` / ``max_action_dim`` (=32) with **zeros**
     (``lerobot/policies/common/vla_utils.py:pad_vector``). The probe asserts the
     tail slots of the tensors actually entering ``state_proj`` / ``action_in_proj``
     are still exactly zero. Contamination there is silent and fatal;
   * per-group gradient L2 norms (see ``GRAD_GROUPS`` below) captured with an
     optimizer step pre-hook, i.e. *after* ``clip_grad_norm_``. Ratios between
     groups are the meaningful quantity;
   * in-process resources (VmRSS, VmSwap, system MemAvailable, torch CUDA mem).

2. ``kind="pred"`` -- every ``--pred-every`` steps (0 = off):
   * predicted action chunk diagnostics. **Channel-wise only.** Aggregate MSE is
     deliberately not reported as a headline number: a "gripper stays put"
     predictor scores well on aggregate MSE because the gripper channel has tiny
     variance. Reported instead: per-channel MSE, per-channel Pearson r,
     gripper-transition-only MSE, and per-step motion magnitude of the prediction
     vs. the same statistic measured on the ground-truth actions in the same
     batch (plus an optional external baseline).

Resource monitoring: the repo already ships ``resource_monitor.sh`` (system-wide
CSV + WARN lines) and ``watchdog.sh`` (trips on RAM/disk/GPU thresholds). Those
are *not* reimplemented here -- run them alongside training. This probe only adds
the part they cannot see: the **training process's own** RSS/swap and torch CUDA
allocator numbers, aligned to the same JSONL timeline as the tensor stats.

Usage (library):

    from diag.probe import Probe, ProbeConfig
    probe = Probe(policy, ProbeConfig(out_path="run.jsonl", every=50))
    probe.attach()
    probe.attach_optimizer(optimizer)
    ...
    probe.begin_step(step, batch)
    <one training step>
    probe.end_step()

Usage (self test, no training):

    python -m diag.probe --self-test --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Arbitrary thresholds. Every number here was CHOSEN BY HAND, not measured.
# They only control *warnings*; nothing is silently modified.
# ---------------------------------------------------------------------------
DEAD_UNIT_EPS = 1e-6  # arbitrary: |out| <= eps over the whole batch => channel counted dead
DEAD_UNIT_WARN_FRAC = 0.5  # arbitrary: warn when >50% of a layer's channels are dead
PAD_TAIL_TOL = 0.0  # not arbitrary: zero padding must be exactly zero
IMG_RANGE_LO, IMG_RANGE_HI = 0.0, 1.0  # lerobot_train divides uint8 by 255 before the policy
IMG_RANGE_TOL = 1e-3  # arbitrary tolerance on the [0,1] range check
RSS_WARN_GB = 25.0  # arbitrary: machine has 31 GB shared with other projects
MAG_RATIO_WARN = 0.5  # arbitrary: warn when median |predicted step| < 0.5x |ground-truth step|
GRIPPER_TRANSITION_THRESH = 0.1  # arbitrary, in normalized (std) units of the gripper channel
EE_DELTA_BASELINE_M = 0.0185  # NOT arbitrary: median per-step |dxyz| measured on the user's EE data
GRAD_GROUP_RATIO_WARN = 1e-3  # arbitrary: warn when state_proj grad norm < 1e-3 x total

# ---------------------------------------------------------------------------
# Gradient groups.
#
# Classification rule (checked against the loaded module tree, not guessed):
#   state_proj      : model.state_proj.*                      -- the ONLY path by which
#                     proprioception reaches the action expert.
#   vlm_readout_kv  : model.vlm_with_expert.lm_expert.layers.<i>.self_attn.{k,v}_proj
#                     for the CROSS-ATTENTION layers only. smolvlm_with_expert.py:122-136
#                     replaces k_proj/v_proj on every layer with
#                     ``layer_idx % self_attn_every_n_layers != 0`` by an nn.Linear whose
#                     in_features == VLM kv width (960/3=320), not the expert width (720).
#                     Those are literally the weights that read the VLM (image+language)
#                     key/value stream into the action expert. Verified at attach time by
#                     comparing in_features against expert_hidden_size.
#   action_io       : model.action_in_proj / action_out_proj / action_time_mlp_{in,out}
#   expert_other    : the rest of lm_expert (self-attn layers, MLPs, norms)
#   vlm             : model.vlm_with_expert.vlm.*  (frozen unless train_expert_only=False)
#   other           : anything unclassified (should stay empty; reported if not)
#
# state_proj vs vlm_readout_kv is the state-anchoring measurement: if the expert
# learns to read the state channel and ignore the VLM stream (or vice-versa), the
# ratio of these two grad norms shows it while the scalar loss does not.
# ---------------------------------------------------------------------------
GRAD_GROUPS = ("state_proj", "vlm_readout_kv", "action_io", "expert_other", "vlm", "other")


def _finite_stats(t: torch.Tensor) -> dict[str, Any]:
    """min/max/mean/std/NaN/Inf of a tensor, computed in fp32 on-device."""
    with torch.no_grad():
        tf = t.detach().float()
        n_nan = int(torch.isnan(tf).sum().item())
        n_inf = int(torch.isinf(tf).sum().item())
        finite = tf[torch.isfinite(tf)]
        if finite.numel() == 0:
            return {"min": None, "max": None, "mean": None, "std": None,
                    "n_nan": n_nan, "n_inf": n_inf, "numel": int(tf.numel())}
        return {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()) if finite.numel() > 1 else 0.0,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "numel": int(tf.numel()),
        }


def _module_out_stats(out: torch.Tensor) -> dict[str, Any]:
    """Output stats for a Linear: RMS, range, NaN/Inf, dead-channel fraction."""
    with torch.no_grad():
        o = out.detach().float()
        flat = o.reshape(-1, o.shape[-1])
        st = _finite_stats(o)
        st["rms"] = float(torch.sqrt(torch.mean(torch.nan_to_num(flat) ** 2)))
        chan_absmax = torch.nan_to_num(flat).abs().amax(dim=0)
        st["dead_frac"] = float((chan_absmax <= DEAD_UNIT_EPS).float().mean())
        st["n_chan"] = int(flat.shape[-1])
        return st


def _proc_resources() -> dict[str, Any]:
    """In-process RSS/swap + system MemAvailable + torch CUDA allocator numbers.

    Deliberately NOT a replacement for resource_monitor.sh / watchdog.sh (those
    cover the whole machine, continuously). This is the per-process view that
    those scripts cannot provide, sampled on the same timeline as the tensor stats.
    """
    out: dict[str, Any] = {}
    try:
        vals = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmSwap:", "VmHWM:")):
                k, v = line.split(":", 1)
                vals[k] = int(v.strip().split()[0])  # kB
        out["rss_gb"] = round(vals.get("VmRSS", 0) / 1024 / 1024, 3)
        out["rss_peak_gb"] = round(vals.get("VmHWM", 0) / 1024 / 1024, 3)
        out["proc_swap_gb"] = round(vals.get("VmSwap", 0) / 1024 / 1024, 3)
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                out["sys_mem_avail_gb"] = round(int(line.split()[1]) / 1024 / 1024, 3)
            elif line.startswith("SwapFree:"):
                out["sys_swap_free_gb"] = round(int(line.split()[1]) / 1024 / 1024, 3)
    except OSError:
        pass
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            out["cuda_dev"] = int(dev)
            out["cuda_alloc_gb"] = round(torch.cuda.memory_allocated(dev) / 1024**3, 3)
            out["cuda_reserved_gb"] = round(torch.cuda.memory_reserved(dev) / 1024**3, 3)
            out["cuda_max_alloc_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1024**3, 3)
        except RuntimeError:
            pass
    return out


@dataclass
class ProbeConfig:
    out_path: str = "diag_probe.jsonl"
    every: int = 50
    pred_every: int = 0            # 0 = never run the (expensive) action-chunk prediction check
    tag: str = ""
    expert_layers: tuple[int, ...] = (0, 1, 8, 15)  # arbitrary sample: first SA, first XA, mid, last
    hook_expert_mlp: bool = True
    mode: str = "joint"            # "joint" (absolute targets) or "ee" (per-step deltas)
    delta_baseline: float | None = None   # external per-step magnitude baseline, physical units
    gripper_index: int = -1
    rss_warn_gb: float = RSS_WARN_GB
    dead_warn_frac: float = DEAD_UNIT_WARN_FRAC
    gripper_transition_thresh: float = GRIPPER_TRANSITION_THRESH
    mag_ratio_warn: float = MAG_RATIO_WARN
    console: bool = True
    action_mean: list[float] | None = None  # for un-normalizing predictions (MEAN_STD only)
    action_std: list[float] | None = None
    # Inference mode: there is no optimizer, no loss, no ground-truth action chunk and no
    # `action_is_pad`, so the two warnings that only make sense while training are muted.
    # Everything else (input sanity, module stats, dead units, zero-pad integrity,
    # prediction magnitudes) is identical to the training path.
    inference: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Probe:
    """Forward-hook + grad monitor. Never mutates the policy or the batch."""

    def __init__(self, policy: torch.nn.Module, cfg: ProbeConfig):
        self.policy = policy
        self.cfg = cfg
        self._handles: list[Any] = []
        self._opt_handles: list[Any] = []
        self._buf: dict[str, Any] = {}
        self._grad_buf: dict[str, Any] = {}
        self._record_now = False
        self._step = -1
        self._t0 = 0.0
        self._warn_counts: dict[str, int] = {}
        # lerobot creates output_dir only later in train(), so a probe path inside it would not
        # exist yet -- make the parents ourselves rather than requiring the caller to pre-create
        # the run directory (a pre-existing output_dir changes lerobot's resume behaviour).
        Path(cfg.out_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._out = open(cfg.out_path, "a", buffering=1)  # line buffered, no big in-RAM buffer

        core = getattr(policy, "module", policy)  # unwrap DDP/accelerate if present
        self.core = getattr(core, "_orig_mod", core)  # unwrap torch.compile
        self.pcfg = self.core.config

        # Real (unpadded) dims -- SmolVLA pads these to max_state_dim / max_action_dim.
        # NOTE: config.input_features["observation.state"] is NOT trustworthy here. factory.py:311
        # only fills input_features when it is empty, so a pretrained checkpoint keeps whatever dim
        # it was trained with (smolvla_base says 6) even when the dataset supplies more (our EE
        # sincos state is 7). Using the stale dim made the pad-tail check flag the real 7th channel
        # as "contaminated padding" -- verified false positive: pad_vector leaves slots 7..31 at
        # exactly 0 while slot 6 carries grip_norm. So the real dim is read off the live tensor in
        # check_padding(); this config value is only a fallback for the first record.
        try:
            self.real_state_dim = int(self.pcfg.input_features["observation.state"].shape[0])
        except (KeyError, AttributeError):
            self.real_state_dim = None
        try:
            self.real_action_dim = int(self.pcfg.action_feature.shape[0])
        except (KeyError, AttributeError, TypeError):
            self.real_action_dim = None
        self.max_state_dim = getattr(self.pcfg, "max_state_dim", None)
        self.max_action_dim = getattr(self.pcfg, "max_action_dim", None)
        self.cfg_image_keys = list(getattr(self.pcfg, "image_features", {}) or {})

        self._param_group = self._classify_params()

    # ------------------------------------------------------------------ setup

    def _classify_params(self) -> dict[int, str]:
        """Assign every parameter to exactly one gradient group (see GRAD_GROUPS)."""
        group_of: dict[int, str] = {}
        expert_hidden = None
        vwe = None
        try:
            vwe = self.core.model.vlm_with_expert
            expert_hidden = int(vwe.expert_hidden_size)
        except AttributeError:
            pass

        xattn_kv: set[int] = set()
        if vwe is not None:
            for layer in vwe.lm_expert.layers:
                for attr in ("k_proj", "v_proj"):
                    m = getattr(layer.self_attn, attr, None)
                    if m is None or not hasattr(m, "in_features"):
                        continue
                    # cross-attention layers are the ones whose k/v were rebuilt to take the
                    # VLM kv width instead of the expert width (smolvlm_with_expert.py:122-136)
                    if expert_hidden is not None and m.in_features != expert_hidden:
                        xattn_kv.update(id(p) for p in m.parameters())

        for name, p in self.core.named_parameters():
            if id(p) in xattn_kv:
                g = "vlm_readout_kv"
            elif name.startswith("model.state_proj"):
                g = "state_proj"
            elif name.startswith(("model.action_in_proj", "model.action_out_proj",
                                  "model.action_time_mlp_in", "model.action_time_mlp_out")):
                g = "action_io"
            elif name.startswith("model.vlm_with_expert.lm_expert"):
                g = "expert_other"
            elif name.startswith("model.vlm_with_expert.vlm"):
                g = "vlm"
            else:
                g = "other"
            group_of[id(p)] = g
        return group_of

    def target_modules(self) -> dict[str, torch.nn.Module]:
        """Modules to hook. Resolved against the real module tree; missing ones are skipped."""
        want = ["model.state_proj", "model.action_in_proj", "model.action_out_proj",
                "model.action_time_mlp_in", "model.action_time_mlp_out"]
        for li in self.cfg.expert_layers:
            base = f"model.vlm_with_expert.lm_expert.layers.{li}"
            want += [f"{base}.self_attn.{a}_proj" for a in ("q", "k", "v", "o")]
            if self.cfg.hook_expert_mlp:
                want += [f"{base}.mlp.{a}_proj" for a in ("gate", "up", "down")]
        found: dict[str, torch.nn.Module] = {}
        for n in want:
            try:
                found[n] = self.core.get_submodule(n)
            except AttributeError:
                continue
        return found

    def attach(self) -> list[str]:
        """Register forward hooks. Returns the list of hooked module paths."""
        mods = self.target_modules()
        for name, m in mods.items():
            self._handles.append(m.register_forward_hook(self._make_hook(name)))
        return list(mods)

    def attach_optimizer(self, optimizer: Any) -> bool:
        """Register a step pre-hook so grads can be read while they still exist.

        ``update_policy`` in lerobot_train.py calls ``optimizer.zero_grad()``
        immediately after ``optimizer.step()``, so grads are gone by the time a
        wrapper around ``update_policy`` regains control. A step pre-hook fires
        with grads populated (and after ``accelerator.clip_grad_norm_``).

        Accelerate's ``AcceleratedOptimizer`` inherits ``register_step_pre_hook`` from
        ``torch.optim.Optimizer`` but never runs ``Optimizer.__init__``, so calling it
        raises ``AttributeError: _optimizer_step_pre_hooks``. The real torch optimizer
        underneath is therefore tried first, and every candidate is guarded.
        """
        inner = getattr(optimizer, "optimizer", None)
        candidates = [c for c in (inner, optimizer) if c is not None]
        for opt in candidates:
            reg = getattr(opt, "register_step_pre_hook", None)
            if reg is None:
                continue
            try:
                self._opt_handles.append(reg(self._opt_pre_hook))
            except Exception as e:  # noqa: BLE001
                print(f"[probe] step pre-hook not supported on {type(opt).__name__}: {e!r}",
                      file=sys.stderr)
                continue
            return True
        return False

    def detach(self) -> None:
        for h in self._handles + self._opt_handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001 - handle may already be dead
                pass
        self._handles.clear()
        self._opt_handles.clear()

    def close(self) -> None:
        self.detach()
        try:
            self._out.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ hooks

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._record_now:
                return
            try:
                out = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(out):
                    return
                rec = _module_out_stats(out)
                # Zero-pad integrity on the input side.
                #
                # state_proj: its input is exactly pad_vector(state, 32), so the tail MUST be
                #   exactly 0. Anything else is contamination -> hard warning.
                # action_in_proj: during *training* its input is the flow-matching interpolant
                #   x_t = t*noise + (1-t)*padded_action with noise drawn over all 32 dims, so a
                #   non-zero tail there is BY DESIGN and must not be warned about. The tail of
                #   the padded ground-truth action is checked separately and directly in
                #   Probe.check_padding().
                inp = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
                if inp is not None:
                    rec["in_shape"] = list(inp.shape)
                    if name.endswith("state_proj") and self.real_state_dim is not None \
                            and inp.shape[-1] > self.real_state_dim:
                        tail = inp.detach()[..., self.real_state_dim:]
                        rec["pad_tail_absmax"] = float(torch.nan_to_num(tail).abs().max())
                        rec["pad_tail_dims"] = [self.real_state_dim, int(inp.shape[-1])]
                    elif name.endswith("action_in_proj") and self.real_action_dim is not None \
                            and inp.shape[-1] > self.real_action_dim:
                        tail = inp.detach()[..., self.real_action_dim:]
                        rec["input_tail_absmax_informational"] = float(
                            torch.nan_to_num(tail).abs().max())
                if name.endswith("action_out_proj") and self.real_action_dim is not None \
                        and out.shape[-1] > self.real_action_dim:
                    # what the model emits in the slots that get thrown away
                    tail = out.detach()[..., self.real_action_dim:]
                    rec["out_tail_absmax"] = float(torch.nan_to_num(tail).abs().max())
                self._buf.setdefault("modules", {})[name] = rec
            except Exception as e:  # noqa: BLE001 - a probe must never kill training
                self._buf.setdefault("modules", {})[name] = {"probe_error": repr(e)}
        # never traced by torch.compile (smolvla_ext compile_mode=layers compiles hooked modules):
        # tracing it causes graph breaks and dynamo warnings that echo this source ("n_nan") into
        # the training log, which verify_run's NaN check then reports.
        return torch.compiler.disable(hook)

    def _opt_pre_hook(self, optimizer, *args, **kwargs):  # noqa: ARG002
        if not self._record_now or self._grad_buf:
            return
        try:
            sums = dict.fromkeys(GRAD_GROUPS, 0.0)
            counts = dict.fromkeys(GRAD_GROUPS, 0)
            withgrad = dict.fromkeys(GRAD_GROUPS, 0)
            n_nan = 0
            for p in self.core.parameters():
                g = self._param_group.get(id(p), "other")
                counts[g] += p.numel()
                if p.grad is None:
                    continue
                withgrad[g] += p.numel()
                gn = float(torch.nan_to_num(p.grad.detach().float()).pow(2).sum())
                if not math.isfinite(gn):
                    n_nan += 1
                    continue
                sums[g] += gn
            total = math.sqrt(sum(sums.values()))
            self._grad_buf = {
                "total_norm": total,
                "n_nonfinite_params": n_nan,
                "by_group": {
                    g: {
                        "norm": math.sqrt(sums[g]),
                        # ratio of L2 norms, NOT a partition: these do not sum to 1
                        "ratio_to_total_norm": (math.sqrt(sums[g]) / total) if total > 0 else None,
                        "n_params": counts[g],
                        "n_params_with_grad": withgrad[g],
                    }
                    for g in GRAD_GROUPS
                    if counts[g] > 0
                },
                "note": "measured at optimizer.step() time, i.e. AFTER clip_grad_norm_; "
                        "ratios between groups are the reliable quantity",
            }
        except Exception as e:  # noqa: BLE001
            self._grad_buf = {"probe_error": repr(e)}

    # ------------------------------------------------------------------ steps

    def should_record(self, step: int) -> bool:
        return self.cfg.every > 0 and step % self.cfg.every == 0

    def begin_step(self, step: int, batch: dict[str, Any] | None = None) -> None:
        self._step = step
        self._record_now = self.should_record(step)
        self._buf = {}
        self._grad_buf = {}
        if not self._record_now:
            return
        self._t0 = time.perf_counter()
        if batch is not None:
            self._buf["inputs"] = self.inspect_batch(batch)
            self._buf["pad_check"] = self.check_padding(batch)

    @torch.no_grad()
    def check_padding(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Directly exercise SmolVLA's own padding helpers and assert the tails are zero.

        This calls ``policy.prepare_state`` / ``policy.prepare_action`` (pure
        ``pad_vector`` calls, negligible cost) rather than inferring padding from a
        hook, because the ``action_in_proj`` input during training is the noisy
        flow-matching interpolant and its tail is legitimately non-zero.
        """
        out: dict[str, Any] = {}
        for kind, fn, real, mx in (
            ("state", getattr(self.core, "prepare_state", None), self.real_state_dim,
             self.max_state_dim),
            ("action", getattr(self.core, "prepare_action", None), self.real_action_dim,
             self.max_action_dim),
        ):
            key = "observation.state" if kind == "state" else "action"
            if fn is None or not torch.is_tensor(batch.get(key)) or mx is None:
                out[kind] = {"status": "skipped"}
                continue
            try:
                src = batch[key].detach()
                # Trust the live tensor over config.input_features (see __init__ note): the
                # checkpoint's declared dim can be stale, which shifts the tail boundary.
                real_live = int(src.shape[-1])
                if real is not None and real_live != real:
                    out.setdefault("_notes", []).append(
                        f"{kind}: using live dim {real_live}, config declared {real}")
                real = real_live
                # Correct the cached dim so the state_proj forward hook (which only ever sees the
                # already-padded 32-wide input and cannot infer the real width itself) uses the
                # live boundary from here on.
                if kind == "state":
                    self.real_state_dim = real
                else:
                    self.real_action_dim = real
                padded = fn({key: src})
                tail = padded[..., real:]
                out[kind] = {
                    "real_dim": real,
                    "padded_dim": int(padded.shape[-1]),
                    "tail_absmax": float(torch.nan_to_num(tail).abs().max()) if tail.numel() else 0.0,
                    "tail_all_zero": bool(tail.numel() == 0 or bool((tail == 0).all())),
                }
            except Exception as e:  # noqa: BLE001
                out[kind] = {"probe_error": repr(e)}
        return out

    def end_step(self, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self._record_now:
            return None
        rec: dict[str, Any] = {
            "kind": "step",
            "ts": time.time(),
            "step": self._step,
            "tag": self.cfg.tag,
            "wall_s": round(time.perf_counter() - self._t0, 4),
        }
        rec.update(self._buf)
        if self._grad_buf:
            rec["grads"] = self._grad_buf
        elif self.cfg.inference:
            rec["grads"] = {"status": "skipped (inference: no_grad, no optimizer)"}
        else:
            rec["grads"] = {"status": "not captured (optimizer step pre-hook did not fire)"}
        rec["resources"] = _proc_resources()
        if extra:
            rec["extra"] = extra
        rec["warnings"] = self._collect_warnings(rec)
        self._emit(rec)
        self._record_now = False
        return rec

    # ------------------------------------------------------------------ input sanity

    def inspect_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {}
        st = batch.get("observation.state")
        if torch.is_tensor(st):
            s = _finite_stats(st)
            s["shape"] = list(st.shape)
            s["dtype"] = str(st.dtype)
            per_ch = st.detach().float().reshape(-1, st.shape[-1])
            s["per_channel_mean"] = [round(float(v), 6) for v in per_ch.mean(0)]
            s["per_channel_std"] = [round(float(v), 6) for v in per_ch.std(0)] \
                if per_ch.shape[0] > 1 else None
            info["state"] = s

        ac = batch.get("action")
        if torch.is_tensor(ac):
            a = _finite_stats(ac)
            a["shape"] = list(ac.shape)
            per_ch = ac.detach().float().reshape(-1, ac.shape[-1])
            a["per_channel_std"] = [round(float(v), 6) for v in per_ch.std(0)] \
                if per_ch.shape[0] > 1 else None
            info["action"] = a

        pad = batch.get("action_is_pad")
        if torch.is_tensor(pad):
            info["action_is_pad_frac"] = round(float(pad.detach().float().mean()), 6)
            info["action_is_pad_shape"] = list(pad.shape)
        else:
            info["action_is_pad_frac"] = None  # absent => loss is NOT masked

        # lerobot puts companion mask keys next to the image keys
        # (observation.images.<cam>_is_pad / _padding_mask); they are not cameras.
        batch_img_keys = sorted(
            k for k in batch
            if k.startswith("observation.images.")
            and not k.endswith(("_is_pad", "_padding_mask"))
            and torch.is_tensor(batch[k]) and batch[k].ndim >= 4
        )
        used = [k for k in self.cfg_image_keys if k in batch]
        info["cameras"] = {
            "in_batch": batch_img_keys,
            "in_policy_config": list(self.cfg_image_keys),
            # modeling_smolvla.py:prepare_images builds its list from config.image_features
            # and silently ignores anything else that happens to be in the batch.
            "used_by_policy": used,
            "dropped_from_batch": sorted(set(batch_img_keys) - set(self.cfg_image_keys)),
            "missing_from_batch": sorted(set(self.cfg_image_keys) - set(batch_img_keys)),
            "empty_cameras_cfg": int(getattr(self.pcfg, "empty_cameras", 0)),
        }
        imgs = {}
        for k in batch_img_keys:
            v = batch[k]
            if not torch.is_tensor(v):
                continue
            s = _finite_stats(v)
            s["shape"] = list(v.shape)
            s["dtype"] = str(v.dtype)
            imgs[k] = s
        info["images"] = imgs
        for k in ("observation.language.tokens", "observation.language.attention_mask"):
            if torch.is_tensor(batch.get(k)):
                info.setdefault("language", {})[k] = {
                    "shape": list(batch[k].shape),
                    "n_valid_tokens": int(batch[k].sum()) if k.endswith("attention_mask") else None,
                }
        return info

    # ------------------------------------------------------------------ prediction check

    @torch.no_grad()
    def prediction_check(self, batch: dict[str, Any], step: int | None = None) -> dict[str, Any]:
        """Channel-wise diagnostics on a predicted action chunk. Never aggregate MSE.

        Expects an already-preprocessed (normalized) batch, i.e. the same object
        that would be handed to ``policy.forward``. Returns (and emits) the record.
        """
        rec: dict[str, Any] = {"kind": "pred", "ts": time.time(),
                               "step": self._step if step is None else step, "tag": self.cfg.tag}
        was_training = self.core.training
        try:
            pred = self.core.predict_action_chunk(dict(batch))
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
            self._emit(rec)
            return rec
        finally:
            # predict_action_chunk() calls self.eval() and pushes into the policy's action
            # queue; undo both so a probe call never changes training/deployment behaviour.
            if hasattr(self.core, "reset"):
                self.core.reset()
            if was_training:
                self.core.train()
        return self._score_prediction(rec, pred, batch)

    @torch.no_grad()
    def record_prediction(
        self,
        pred: torch.Tensor,
        batch: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Score an action chunk the caller ALREADY computed. Same record as
        ``prediction_check``, minus the forward pass.

        Exists for the deployment server: it has just run ``predict_action_chunk`` to
        answer the request, and re-running it inside the probe would double the latency
        (~0.4 s per request on this machine) and, worse, score a *different* sample than
        the one actually sent to the motors (flow-matching sampling is stochastic).

        ``pred`` must be in the SAME space as ``batch["action"]`` -- i.e. normalized, the
        raw ``predict_action_chunk`` output, before the policy postprocessor. Supply
        ``cfg.action_mean``/``action_std`` to have the record reported in physical units.
        """
        rec: dict[str, Any] = {"kind": "pred", "ts": time.time(),
                               "step": self._step if step is None else step, "tag": self.cfg.tag}
        return self._score_prediction(rec, pred, batch if batch is not None else {})

    @torch.no_grad()
    def _score_prediction(self, rec: dict[str, Any], pred: torch.Tensor,
                          batch: dict[str, Any]) -> dict[str, Any]:
        gt = batch.get("action")
        pred = pred.detach().float()          # (B, T, D) normalized action space
        rec["pred_shape"] = list(pred.shape)
        D = pred.shape[-1]
        names = self._action_names(D)

        mean, std = self._action_stats(D, pred.device)
        units = "physical" if mean is not None else "normalized"
        rec["units"] = units
        rec["unnormalization"] = ("MEAN_STD from supplied action stats" if mean is not None
                                  else "NOT APPLIED (no action stats given to the probe)")

        def denorm(x):
            return x * std + mean if mean is not None else x

        # --- validity mask -----------------------------------------------------------
        # Chunk entries past the end of an episode are padding (they repeat the last
        # action), so every statistic below is computed over VALID entries only.
        # Including them makes the ground-truth motion reference collapse toward 0,
        # which is precisely the failure mode this probe exists to expose.
        T = pred.shape[1]
        pad = batch.get("action_is_pad")
        if torch.is_tensor(pad):
            valid = ~pad.detach().bool()
            valid = valid[:, :T] if valid.ndim == 2 else valid.reshape(pred.shape[0], -1)[:, :T]
        else:
            valid = torch.ones(pred.shape[0], T, dtype=torch.bool, device=pred.device)
        rec["valid_entries"] = {"n_valid": int(valid.sum()), "n_total": int(valid.numel()),
                                "valid_frac": round(float(valid.float().mean()), 6),
                                "note": "all statistics below use valid (non-padded) entries only"}

        pred_phys = denorm(pred)
        vm = valid.unsqueeze(-1)

        def vsel(x, i):
            return x[..., i][valid]

        rec["pred_per_channel"] = {
            names[i]: {
                "absmean": round(float(vsel(pred_phys, i).abs().mean()), 8),
                "std": round(float(vsel(pred_phys, i).std()), 8) if int(valid.sum()) > 1 else None,
                "min": round(float(vsel(pred_phys, i).min()), 8),
                "max": round(float(vsel(pred_phys, i).max()), 8),
            }
            for i in range(D)
        } if int(valid.sum()) > 0 else {}
        rec["pred_chunk_l2_per_channel"] = {
            names[i]: round(float(vsel(pred_phys, i).pow(2).mean().sqrt()), 8) for i in range(D)
        } if int(valid.sum()) > 0 else {}
        del vm

        # --- per-step motion magnitude vs. an in-batch ground-truth reference -------
        mag = self._step_magnitude(pred_phys, valid)
        rec["pred_step_mag_median"] = mag
        if torch.is_tensor(gt):
            gt_phys = denorm(gt.detach().float()[:, :T, :D])
            gt_mag = self._step_magnitude(gt_phys, valid)
            rec["gt_step_mag_median"] = gt_mag
            if gt_mag and gt_mag > 0 and mag is not None:
                rec["step_mag_ratio_pred_over_gt"] = round(mag / gt_mag, 6)
        if self.cfg.delta_baseline is not None:
            rec["external_step_mag_baseline"] = self.cfg.delta_baseline
            if units != "physical":
                rec["external_baseline_note"] = (
                    "SKIPPED comparison: predictions are in normalized units, baseline is physical"
                )
            elif mag is not None:
                rec["step_mag_ratio_pred_over_baseline"] = round(mag / self.cfg.delta_baseline, 6)
        rec["step_magnitude_definition"] = (
            "mode=ee: ||a[:, t, 0:3]|| (actions are per-step deltas); "
            "mode=joint: |a[:, t+1, :] - a[:, t, :]| per channel, then median over all"
        )

        # --- channel-wise error, NEVER aggregate ------------------------------------
        if torch.is_tensor(gt) and int(valid.sum()) > 0:
            g = gt.detach().float()[:, :T, :D]
            err = (pred - g) ** 2
            rec["mse_per_channel_normalized"] = {
                names[i]: round(float(err[..., i][valid].mean()), 8) for i in range(D)
            }
            rec["pearson_r_per_channel"] = {
                names[i]: self._pearson(pred[..., i][valid], g[..., i][valid]) for i in range(D)
            }
            gi = self.cfg.gripper_index % D
            rec["gripper_channel"] = {"index": gi, "name": names[gi]}
            if T >= 2:
                # a transition needs both endpoints inside the episode
                pair_valid = valid[:, 1:] & valid[:, :-1]
                dg = (g[:, 1:, gi] - g[:, :-1, gi]).abs()
                mask = (dg > self.cfg.gripper_transition_thresh) & pair_valid
                off = (dg <= self.cfg.gripper_transition_thresh) & pair_valid
                n_tr = int(mask.sum())
                rec["gripper_transition"] = {
                    "threshold_normalized": self.cfg.gripper_transition_thresh,
                    "threshold_is_arbitrary": True,
                    "n_transition_frames": n_tr,
                    "n_valid_pairs": int(pair_valid.sum()),
                }
                e = (pred[:, 1:, gi] - g[:, 1:, gi]) ** 2
                rec["gripper_transition"]["mse_on_transition"] = (
                    round(float(e[mask].mean()), 8) if n_tr > 0 else None)
                rec["gripper_transition"]["mse_off_transition"] = (
                    round(float(e[off].mean()), 8) if int(off.sum()) > 0 else None)
            rec["note"] = ("aggregate MSE intentionally omitted: the low-variance gripper channel "
                           "is invisible in it")
        rec["resources"] = _proc_resources()
        rec["warnings"] = self._collect_warnings(rec)
        self._emit(rec)
        return rec

    def _step_magnitude(self, a: torch.Tensor, valid: torch.Tensor) -> float | None:
        """Median per-step motion over valid (non-padded) chunk entries."""
        if a.numel() == 0 or int(valid.sum()) == 0:
            return None
        if self.cfg.mode == "ee":
            # actions are per-step deltas: the commanded step size IS the action
            if a.shape[-1] < 3:
                return None
            v = a[..., 0:3].norm(dim=-1)[valid]
            return round(float(v.median()), 8) if v.numel() else None
        # absolute targets: motion is the difference between consecutive commands,
        # and both endpoints must be inside the episode
        if a.shape[1] < 2:
            return None
        pair_valid = valid[:, 1:] & valid[:, :-1]
        if int(pair_valid.sum()) == 0:
            return None
        d = (a[:, 1:, :] - a[:, :-1, :]).abs()
        v = d[pair_valid]
        return round(float(v.median()), 8) if v.numel() else None

    def _action_stats(self, D: int, device):
        if self.cfg.action_mean is None or self.cfg.action_std is None:
            return None, None
        m = torch.tensor(self.cfg.action_mean[:D], dtype=torch.float32, device=device)
        s = torch.tensor(self.cfg.action_std[:D], dtype=torch.float32, device=device)
        if m.numel() != D or s.numel() != D:
            return None, None
        return m, s

    def _action_names(self, D: int) -> list[str]:
        names = self.cfg.extra.get("action_names")
        if isinstance(names, list) and len(names) >= D:
            return [str(x) for x in names[:D]]
        return [f"a{i}" for i in range(D)]

    @staticmethod
    def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
        if x.numel() < 2:
            return None
        xc, yc = x - x.mean(), y - y.mean()
        den = float(xc.norm() * yc.norm())
        if den == 0:
            return None
        return round(float((xc * yc).sum()) / den, 6)

    # ------------------------------------------------------------------ warnings + io

    def _collect_warnings(self, rec: dict[str, Any]) -> list[str]:
        w: list[str] = []
        inp = rec.get("inputs", {})
        st = inp.get("state")
        if st and (st.get("n_nan") or st.get("n_inf")):
            w.append(f"state has {st['n_nan']} NaN / {st['n_inf']} Inf")
        cams = inp.get("cameras")
        if cams:
            if cams["dropped_from_batch"]:
                w.append("cameras in batch but NOT in policy config (silently dropped by "
                         f"prepare_images): {cams['dropped_from_batch']}")
            if cams["missing_from_batch"]:
                w.append("cameras in policy config but MISSING from batch: "
                         f"{cams['missing_from_batch']} (empty_cameras="
                         f"{cams['empty_cameras_cfg']})")
            if not cams["used_by_policy"]:
                w.append("NO camera from the policy config is present in the batch")
        for k, s in (inp.get("images") or {}).items():
            if s.get("max") is None:
                continue
            if "uint8" in s.get("dtype", ""):
                w.append(f"{k} is uint8 at the policy input (expected float in [0,1])")
            elif s["max"] > IMG_RANGE_HI + IMG_RANGE_TOL or s["min"] < IMG_RANGE_LO - IMG_RANGE_TOL:
                w.append(f"{k} range [{s['min']:.4g},{s['max']:.4g}] outside [0,1]")
        if inp.get("action_is_pad_frac") is None and "inputs" in rec and not self.cfg.inference:
            w.append("batch has no 'action_is_pad': chunk padding is NOT masked out of the loss")
        elif inp.get("action_is_pad_frac", 0) and inp["action_is_pad_frac"] > 0.5:
            w.append(f"action_is_pad fraction = {inp['action_is_pad_frac']:.3f} "
                     "(>50% of the chunk entries are padding)")

        for name, m in (rec.get("modules") or {}).items():
            if "probe_error" in m:
                w.append(f"{name}: probe error {m['probe_error']}")
                continue
            if m.get("n_nan") or m.get("n_inf"):
                w.append(f"{name}: {m['n_nan']} NaN / {m['n_inf']} Inf in output")
            if m.get("dead_frac", 0) > self.cfg.dead_warn_frac:
                w.append(f"{name}: {m['dead_frac']:.1%} of output channels dead")
            if m.get("pad_tail_absmax") is not None and m["pad_tail_absmax"] > PAD_TAIL_TOL:
                w.append(f"{name}: zero-pad tail is NOT zero (max|tail|="
                         f"{m['pad_tail_absmax']:.3g}, dims {m.get('pad_tail_dims')})")

        for kind, pc in (rec.get("pad_check") or {}).items():
            if not isinstance(pc, dict):  # e.g. the "_notes" list added by check_padding
                continue
            if pc.get("tail_all_zero") is False:
                w.append(f"pad_check[{kind}]: slots {pc['real_dim']}..{pc['padded_dim']} are not "
                         f"zero (max|tail|={pc['tail_absmax']:.3g}) -- padded dims are contaminated")
            if "probe_error" in pc:
                w.append(f"pad_check[{kind}]: {pc['probe_error']}")

        gr = rec.get("grads", {}).get("by_group")
        if gr:
            sp = gr.get("state_proj", {}).get("ratio_to_total_norm")
            if sp is not None and sp < GRAD_GROUP_RATIO_WARN:
                w.append(f"state_proj grad is {sp:.2e} of total grad norm "
                         f"(< {GRAD_GROUP_RATIO_WARN:g}): state is barely being used")
            if gr.get("state_proj", {}).get("n_params_with_grad", 0) == 0 \
                    and gr.get("state_proj", {}).get("n_params", 0) > 0:
                w.append("state_proj has NO gradient (train_state_proj=False?)")
            if "vlm_readout_kv" in gr and gr["vlm_readout_kv"].get("n_params_with_grad", 0) == 0:
                w.append("vlm_readout_kv (expert cross-attn k/v) has NO gradient")
        if rec.get("grads", {}).get("n_nonfinite_params"):
            w.append(f"{rec['grads']['n_nonfinite_params']} parameters have non-finite grads")

        res = rec.get("resources", {})
        if res.get("rss_gb", 0) > self.cfg.rss_warn_gb:
            w.append(f"RSS {res['rss_gb']:.1f} GB > threshold {self.cfg.rss_warn_gb} GB")

        r = rec.get("step_mag_ratio_pred_over_gt")
        if r is not None and r < self.cfg.mag_ratio_warn:
            w.append(f"predicted per-step motion is {r:.2f}x the ground-truth motion in the same "
                     f"batch (< {self.cfg.mag_ratio_warn}): under-actuated / mean-collapsed output")
        rb = rec.get("step_mag_ratio_pred_over_baseline")
        if rb is not None and rb < self.cfg.mag_ratio_warn:
            w.append(f"predicted per-step motion is {rb:.2f}x the external baseline "
                     f"{self.cfg.delta_baseline}")
        for k, v in (rec.get("pearson_r_per_channel") or {}).items():
            if v is not None and v < 0.0:
                w.append(f"channel {k}: negative correlation with ground truth (r={v})")
        return w

    def _emit(self, rec: dict[str, Any]) -> None:
        try:
            self._out.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[probe] failed to write record: {e!r}", file=sys.stderr)
        if not self.cfg.console:
            return
        w = rec.get("warnings") or []
        head = f"[probe {rec['kind']} step={rec.get('step')}]"
        if rec["kind"] == "step":
            mods = rec.get("modules") or {}
            sp = (mods.get("model.state_proj") or {}).get("rms")
            ao = (mods.get("model.action_out_proj") or {}).get("rms")
            g = rec.get("grads", {}).get("by_group", {})
            gsp = g.get("state_proj", {}).get("norm")
            gkv = g.get("vlm_readout_kv", {}).get("norm")
            print(f"{head} state_proj.rms={_f(sp)} action_out.rms={_f(ao)} "
                  f"grad[state_proj]={_f(gsp)} grad[vlm_readout_kv]={_f(gkv)} "
                  f"rss={rec.get('resources', {}).get('rss_gb')}GB warn={len(w)}")
        else:
            print(f"{head} step_mag pred={rec.get('pred_step_mag_median')} "
                  f"gt={rec.get('gt_step_mag_median')} "
                  f"ratio={rec.get('step_mag_ratio_pred_over_gt')} warn={len(w)}")
        for msg in w:
            print(f"  [probe WARN] {msg}")


def _f(x):
    return "None" if x is None else f"{x:.4g}"


# ---------------------------------------------------------------------------
# Self test: random tensors through a real SmolVLA policy, no dataset, no training.
# ---------------------------------------------------------------------------


def _self_test(args) -> int:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)
    cfg = SmolVLAConfig.from_pretrained(args.policy_path)
    # NOTE: this is the ONLY way that actually changes chunk_size -- passing
    # chunk_size=... straight to from_pretrained is silently ignored (verified).
    cfg.chunk_size = args.chunk_size
    cfg.n_action_steps = args.chunk_size
    if args.small_images:
        cfg.resize_imgs_with_padding = (args.image_size, args.image_size)
    cfg.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=cfg)
    policy.to(device)
    policy.train()
    print(f"[self-test] chunk_size={policy.config.chunk_size} "
          f"n_action_steps={policy.config.n_action_steps} "
          f"resize={policy.config.resize_imgs_with_padding}")
    tot = sum(p.numel() for p in policy.parameters())
    tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[self-test] params total={tot} trainable={tr} ({100 * tr / tot:.2f}%)")

    pcfg = ProbeConfig(out_path=args.out, every=1, tag="self-test", mode=args.mode,
                       expert_layers=tuple(args.expert_layers), console=True)
    probe = Probe(policy, pcfg)
    hooked = probe.attach()
    print(f"[self-test] hooked {len(hooked)} modules: {hooked}")

    B = args.batch_size
    T = policy.config.chunk_size
    sdim = policy.config.input_features["observation.state"].shape[0]
    adim = policy.config.action_feature.shape[0]
    g = torch.Generator().manual_seed(0)
    batch: dict[str, Any] = {
        "observation.state": torch.randn(B, sdim, generator=g).to(device),
        "action": torch.randn(B, T, adim, generator=g).to(device),
        "action_is_pad": torch.zeros(B, T, dtype=torch.bool).to(device),
    }
    batch["action_is_pad"][:, T // 2:] = True
    H = W = policy.config.resize_imgs_with_padding[0]
    for k in policy.config.image_features:
        batch[k] = torch.rand(B, 3, H, W, generator=g).to(device)
    if args.extra_camera:
        # a camera the policy config does NOT know about -> must be reported as dropped
        batch["observation.images.camera_wrist"] = torch.rand(B, 3, H, W, generator=g).to(device)
    tok = policy.model.vlm_with_expert.processor.tokenizer
    enc = tok(["pick up the block"] * B, return_tensors="pt",
              padding="max_length", max_length=policy.config.tokenizer_max_length, truncation=True)
    batch["observation.language.tokens"] = enc["input_ids"].to(device)
    # tokenizer_processor.py:207 casts the mask to bool; smolvlm_with_expert's eager attention
    # does torch.where(mask, ...) and hard-crashes on an int mask.
    batch["observation.language.attention_mask"] = enc["attention_mask"].to(device).bool()

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5)
    print("[self-test] optimizer step pre-hook registered:", probe.attach_optimizer(opt))

    for step in range(args.steps):
        probe.begin_step(step, batch)
        loss, out = policy.forward(batch)
        loss.backward()
        opt.step()
        opt.zero_grad()
        probe.end_step(extra={"loss": float(loss.detach())})
        print(f"[self-test] step {step} loss={float(loss.detach()):.5f}")

    if args.pred:
        probe.prediction_check(batch, step=args.steps)
    probe.close()
    print(f"[self-test] wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run random tensors through a real SmolVLA policy and dump probe records")
    ap.add_argument("--policy-path", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="diag_probe_selftest.jsonl")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--mode", choices=["joint", "ee"], default="joint")
    ap.add_argument("--expert-layers", type=int, nargs="*", default=[0, 1, 8, 15])
    ap.add_argument("--small-images", action="store_true",
                    help="shrink resize_imgs_with_padding (self-test speed only, NOT for training)")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--extra-camera", action="store_true",
                    help="add a camera key the policy config does not know about")
    ap.add_argument("--pred", action="store_true", help="also run the action-chunk prediction check")
    args = ap.parse_args()
    if not args.self_test:
        ap.print_help()
        return 2
    return _self_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
