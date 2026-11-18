#!/usr/bin/env python3
"""Shared SmolVLA introspection plumbing for `diag/attention_map.py` and `diag/input_ablation.py`.

Everything in here is READ-ONLY with respect to the policy: the two hooks it installs
(`vlm_with_expert.eager_attention_forward` and `vlm_with_expert.forward`) wrap the original
callables, record, and return the original result unchanged. `detach()` restores them.

Facts below were MEASURED on `runs/d2_unfrozen/checkpoints/004000` (2026-08-16), not assumed;
every one of them is re-derived at runtime by `PrefixLayout.measure()` and asserted, so a
different checkpoint fails loudly instead of producing a wrong overlay.

    prefix tokens (241) = camera1[0:64] camera2[64:128] camera3[128:192] lang[192:240] state[240]
    tokens per image    = 64 = 8x8 grid   (SigLIP 512/16 = 32x32 patches, connector
                                            scale_factor=4 -> pixel shuffle 4x4 -> 8x8)
    language            = 48 tokens (tokenizer_max_length=48, pad_language_to='max_length')
    state               = 1 token  (embed_prefix appends state_proj(state) last)
    expert query tokens = 16 = chunk_size (embed_suffix: one token per action step)

Image geometry (modeling_smolvla.prepare_images -> vla_utils.resize_with_pad):
    the padding is on the **LEFT and TOP**, not centered and not bottom-right.
    240x320 -> ratio=max(320/512, 240/512)=0.625 -> resized 384x512 -> pad_top=128, pad_left=0.
    Measured: rows 0..127 of the prepared tensor are exactly -1.0 (the pad value after the
    [0,1]->[-1,1] shift) and row 128 is the first image row. So token rows 0 and 1 of the 8x8
    grid are PURE PADDING and carry no image information -- 16 of the 64 tokens per camera.

Where the attention comes from
    `smolvlm_with_expert.get_attention_interface()` returns `eager_attention_forward`
    unconditionally (smolvlm_with_expert.py:511-513), which materialises `probs = softmax(...)`.
    So the probabilities really exist and can be captured; there is no sdpa fast path to defeat.
    Three call shapes appear per /predict (measured counts for num_steps=10, 16 layers):
      (q=241, k=241) x16   prefix prefill: the VLM attending to itself. No action query.
      (q=16,  k=257) x80   expert SELF-attn layers (even layer_idx, self_attn_every_n_layers=2):
                           keys = 241 prefix + 16 action tokens.
      (q=16,  k=241) x80   expert CROSS-attn layers (odd layer_idx): keys = prefix only.
    The last two are "the policy reading the image"; `KIND_CROSS` is the primary signal.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

DEFAULT_MODEL_PATH = "/home/vcl/Desktop/minjun/vla/runs/d2_unfrozen/checkpoints/004000/pretrained_model"
DEFAULT_DATASET_ROOT = "/home/vcl/Desktop/minjun/vla/data/lerobot_v3_d2_joint"

KIND_PREFILL = "prefill_self"   # VLM prefix self-attention (q == k == prefix)
KIND_SELF = "expert_self"       # expert self-attn layer (k = prefix + suffix)
KIND_CROSS = "expert_cross"     # expert cross-attn layer (k = prefix)


# ==========================================================================
# Model loading
# ==========================================================================

@dataclass
class LoadedPolicy:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: str
    model_path: str
    action_mean: np.ndarray
    action_std: np.ndarray
    tasks: list[str]

    @property
    def cfg(self):
        return self.policy.config

    @property
    def vwe(self):
        return self.policy.model.vlm_with_expert

    @property
    def image_keys(self) -> list[str]:
        return list(self.policy.config.image_features)


def load_policy(model_path: str = DEFAULT_MODEL_PATH,
                dataset_root: str | None = DEFAULT_DATASET_ROOT,
                device: str = "cuda") -> LoadedPolicy:
    """Load the checkpoint exactly the way `smolvla_server_openmani.py` does.

    Same caveat as the server: `SmolVLAPolicy.from_pretrained(path, chunk_size=...)` silently
    ignores the kwarg, so the config object is the only thing that can change anything.
    """
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = SmolVLAConfig.from_pretrained(model_path)
    cfg.device = device
    if (Path(str(model_path)) / "smolvla_ext.json").is_file():
        # PVI / Coda / ROCKET checkpoint: same loader as the server (smolvla_ext.load_policy)
        import sys as _sys
        _repo = str(Path(__file__).resolve().parents[1])
        if _repo not in _sys.path:
            _sys.path.insert(0, _repo)
        from smolvla_ext import load_policy as _ext_load_policy
        policy = _ext_load_policy(model_path, device, config=cfg)
    else:
        policy = SmolVLAPolicy.from_pretrained(model_path, config=cfg)
    policy.to(device)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    mean, std = _action_stats(post, int(policy.config.action_feature.shape[0]))
    return LoadedPolicy(policy=policy, preprocessor=pre, postprocessor=post, device=device,
                        model_path=str(model_path), action_mean=mean, action_std=std,
                        tasks=load_tasks(dataset_root))


def _action_stats(postprocessor, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """MEAN_STD action stats out of the unnormalizer, so predictions can be read in ticks."""
    for s in getattr(postprocessor, "steps", []):
        st = (getattr(s, "_tensor_stats", {}) or {}).get("action")
        if st and "mean" in st and "std" in st:
            m = st["mean"].detach().float().cpu().numpy().reshape(-1)[:action_dim]
            v = st["std"].detach().float().cpu().numpy().reshape(-1)[:action_dim]
            return m.astype(np.float64), v.astype(np.float64)
    raise RuntimeError(
        "no MEAN_STD action stats found in the postprocessor. Without them a normalized "
        "prediction cannot be reported in ticks; refusing to guess.")


def load_tasks(dataset_root: str | None) -> list[str]:
    """The instruction distribution the policy was trained on (meta/tasks.*).

    Mirrors `smolvla_server_openmani.load_dataset_tasks` deliberately, including the detail that
    trips people up: in our v3 tasks.parquet the SENTENCES ARE THE INDEX (`df.index.name ==
    'task'`) and the only column is the integer `task_index`. Reading `df.columns[0]` returns
    '0','1','2',... which look like a valid task list and silently turn a language sweep into a
    sweep over digit strings.
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
            print(f"[tasks][warn] {pq} has index={df.index.name!r} columns={list(df.columns)}; "
                  f"no 'task' index or column -> no task list")
        except Exception as e:
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
            except Exception:
                continue
        return out
    return []


# ==========================================================================
# Observation building (identical path to the deployment server)
# ==========================================================================

def build_observation(lp: LoadedPolicy, frames: list[np.ndarray], state_ticks: np.ndarray,
                      task: str) -> dict:
    """frames: list of (H,W,3) uint8 RGB in image_keys order. state_ticks: (5,) RAW TICKS."""
    from lerobot.policies.utils import prepare_observation_for_inference

    keys = lp.image_keys
    if len(frames) != len(keys):
        raise ValueError(f"need {len(keys)} frames for {keys}, got {len(frames)}")
    obs: dict = {"observation.state": np.asarray(state_ticks, dtype=np.float32).reshape(-1)}
    for key, f in zip(keys, frames):
        obs[key] = f if f.dtype == np.uint8 else f.astype(np.uint8)
    return prepare_observation_for_inference(
        obs, torch.device(lp.device), task, "openmani4_follower")


def predict_ticks(lp: LoadedPolicy, frames: list[np.ndarray], state_ticks: np.ndarray,
                  task: str, noise: torch.Tensor | None = None) -> tuple[np.ndarray, dict]:
    """-> (chunk in TICKS float64 [chunk_size, 5], batch).

    `noise` pins the flow-matching draw. SmolVLA's sampler calls `sample_noise()` when noise is
    None, so two calls on the SAME input differ. Every comparison in `input_ablation.py` passes
    the same noise tensor to both sides; the seed-to-seed spread is measured separately as the
    reference floor.
    """
    obs = build_observation(lp, frames, state_ticks, task)
    lp.policy.reset()
    lp.preprocessor.reset()
    lp.postprocessor.reset()
    with torch.no_grad():
        batch = lp.preprocessor(obs)
        chunk_norm = lp.policy.predict_action_chunk(batch, noise=noise)
        chunk = lp.postprocessor(chunk_norm)
    arr = chunk.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float64), batch


def make_noise(lp: LoadedPolicy, seed: int, bsize: int = 1) -> torch.Tensor:
    """Deterministic flow-matching noise, shape (b, chunk_size, max_action_dim)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    shape = (bsize, int(lp.cfg.chunk_size), int(lp.cfg.max_action_dim))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(lp.device)


# ==========================================================================
# Prefix token layout -- measured, never assumed
# ==========================================================================

@dataclass
class PrefixLayout:
    n_images: int
    tokens_per_image: int
    grid_h: int
    grid_w: int
    lang_start: int
    lang_len: int
    state_index: int
    total: int
    n_action_tokens: int
    target_h: int
    target_w: int
    lang_token_ids: list[int] = field(default_factory=list)
    lang_mask: list[int] = field(default_factory=list)

    def image_span(self, i: int) -> tuple[int, int]:
        if not 0 <= i < self.n_images:
            raise IndexError(f"image {i} out of range (n_images={self.n_images})")
        return i * self.tokens_per_image, (i + 1) * self.tokens_per_image

    @property
    def image_end(self) -> int:
        return self.n_images * self.tokens_per_image

    def describe(self) -> str:
        parts = [f"img{i}[{self.image_span(i)[0]}:{self.image_span(i)[1]}]"
                 for i in range(self.n_images)]
        parts.append(f"lang[{self.lang_start}:{self.lang_start + self.lang_len}]")
        parts.append(f"state[{self.state_index}]")
        return (f"prefix={self.total} tokens: " + " ".join(parts)
                + f" | grid={self.grid_h}x{self.grid_w} per image"
                + f" | expert queries={self.n_action_tokens}")

    @staticmethod
    def measure(lp: LoadedPolicy, frames: list[np.ndarray], state_ticks: np.ndarray,
                task: str) -> "PrefixLayout":
        """Run one forward pass with embed_image / embed_prefix wrapped and read the truth off it."""
        model = lp.policy.model
        vwe = lp.vwe
        rec: dict[str, Any] = {"img": [], "prefix": None}

        orig_embed_image = vwe.embed_image
        orig_embed_prefix = model.embed_prefix

        def embed_image(img):
            out = orig_embed_image(img)
            rec["img"].append(int(out.shape[1]))
            return out

        def embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
            e, p, a = orig_embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
            rec["prefix"] = {
                "n_images": len(images),
                "lang_len": int(lang_tokens.shape[1]),
                "total": int(e.shape[1]),
                "lang_token_ids": lang_tokens[0].detach().cpu().tolist(),
                "lang_mask": lang_masks[0].detach().cpu().to(torch.int64).tolist(),
                "state_tokens": 1 if state is not None and state.ndim == 2 else int(
                    state.shape[1] if state is not None and state.ndim == 3 else 1),
            }
            return e, p, a

        vwe.embed_image = embed_image
        model.embed_prefix = embed_prefix
        try:
            predict_ticks(lp, frames, state_ticks, task, noise=make_noise(lp, 0))
        finally:
            vwe.embed_image = orig_embed_image
            model.embed_prefix = orig_embed_prefix

        pr = rec["prefix"]
        if pr is None or not rec["img"]:
            raise RuntimeError("embed_prefix/embed_image were never called; the model structure "
                               "changed and the token layout cannot be derived.")
        toks = set(rec["img"][:pr["n_images"]])
        if len(toks) != 1:
            raise RuntimeError(f"cameras produced different token counts {rec['img']}; the "
                               f"per-image span assumption does not hold.")
        per_img = toks.pop()
        g = int(round(math.sqrt(per_img)))
        if g * g != per_img:
            raise RuntimeError(f"{per_img} image tokens is not a square grid; cannot map tokens "
                               f"to pixels. Refusing to guess.")
        n_img = pr["n_images"]
        st = pr["state_tokens"]
        computed = n_img * per_img + pr["lang_len"] + st
        if computed != pr["total"]:
            raise RuntimeError(
                f"token accounting failed: {n_img}x{per_img} images + {pr['lang_len']} lang + "
                f"{st} state = {computed} but embed_prefix returned {pr['total']}. Something "
                f"else (image start/end special tokens?) is in the prefix; the per-camera spans "
                f"would be wrong, so refusing to produce an overlay.")
        if lp.cfg.add_image_special_tokens:
            raise RuntimeError("config.add_image_special_tokens=True: each image is wrapped in "
                               "extra tokens. The accounting above happens to pass only when it "
                               "is False. Update PrefixLayout.measure before trusting overlays.")
        tw, th = lp.cfg.resize_imgs_with_padding  # config stores (width, height)
        return PrefixLayout(
            n_images=n_img, tokens_per_image=per_img, grid_h=g, grid_w=g,
            lang_start=n_img * per_img, lang_len=pr["lang_len"],
            state_index=n_img * per_img + pr["lang_len"], total=pr["total"],
            n_action_tokens=int(lp.cfg.chunk_size), target_h=int(th), target_w=int(tw),
            lang_token_ids=pr["lang_token_ids"], lang_mask=pr["lang_mask"],
        )


# ==========================================================================
# Token grid <-> original pixels
# ==========================================================================

@dataclass
class PadGeometry:
    """Exact inverse of `vla_utils.resize_with_pad` (LEFT/TOP padding)."""
    src_h: int
    src_w: int
    target_h: int
    target_w: int
    resized_h: int
    resized_w: int
    pad_top: int
    pad_left: int

    @staticmethod
    def compute(src_h: int, src_w: int, target_h: int, target_w: int) -> "PadGeometry":
        if src_h == target_h and src_w == target_w:
            return PadGeometry(src_h, src_w, target_h, target_w, target_h, target_w, 0, 0)
        ratio = max(src_w / target_w, src_h / target_h)
        rh, rw = int(src_h / ratio), int(src_w / ratio)
        return PadGeometry(src_h, src_w, target_h, target_w, rh, rw,
                           max(0, target_h - rh), max(0, target_w - rw))

    def valid_token_mask(self, gh: int, gw: int) -> np.ndarray:
        """(gh,gw) bool: does this grid cell overlap the real image (not the pad band)?"""
        cell_h, cell_w = self.target_h / gh, self.target_w / gw
        m = np.zeros((gh, gw), dtype=bool)
        for r in range(gh):
            r0, r1 = r * cell_h, (r + 1) * cell_h
            for c in range(gw):
                c0, c1 = c * cell_w, (c + 1) * cell_w
                m[r, c] = (r1 > self.pad_top) and (c1 > self.pad_left)
        return m

    def grid_to_source(self, grid: np.ndarray) -> np.ndarray:
        """(gh,gw) token map -> (src_h, src_w) map over the ORIGINAL frame.

        Upsample to the padded canvas with nearest (a token IS a constant-value block), crop the
        pad band away, then resize to the source resolution.
        """
        gh, gw = grid.shape
        rows = np.minimum((np.arange(self.target_h) * gh) // self.target_h, gh - 1)
        cols = np.minimum((np.arange(self.target_w) * gw) // self.target_w, gw - 1)
        canvas = grid[rows][:, cols]                             # (target_h, target_w)
        crop = canvas[self.pad_top:, self.pad_left:]             # (resized_h, resized_w)
        if crop.shape == (self.src_h, self.src_w):
            return crop
        ri = np.minimum((np.arange(self.src_h) * crop.shape[0]) // self.src_h, crop.shape[0] - 1)
        ci = np.minimum((np.arange(self.src_w) * crop.shape[1]) // self.src_w, crop.shape[1] - 1)
        return crop[ri][:, ci]

    def describe(self) -> str:
        return (f"{self.src_h}x{self.src_w} -> resize {self.resized_h}x{self.resized_w} -> "
                f"pad LEFT={self.pad_left} TOP={self.pad_top} -> {self.target_h}x{self.target_w}")


# ==========================================================================
# Attention capture
# ==========================================================================

@dataclass
class AttnRecord:
    kind: str
    pass_idx: int      # which vlm_with_expert.forward() call (0 = prefix prefill)
    layer_idx: int
    probs: np.ndarray  # (heads, q, k) float32


class AttentionCapture:
    """Wrap `eager_attention_forward` (+ `forward` for pass/layer indexing) and keep the probs.

    Recomputes softmax on the captured q/k rather than reaching into the original function, so it
    stays correct if the interface is refactored -- and the recomputation is checked against the
    original's output in `--self-test` via the attention rows summing to 1.
    """

    def __init__(self, lp: LoadedPolicy, keep: tuple[str, ...] = (KIND_CROSS, KIND_SELF),
                 max_records: int = 4096):
        self.lp = lp
        self.keep = tuple(keep)
        self.max_records = int(max_records)
        self.records: list[AttnRecord] = []
        self.unclassified: list[tuple[int, int]] = []
        self._orig_eager: Callable | None = None
        self._orig_forward: Callable | None = None
        self._pass = -1
        self._layer = 0
        self.prefix_len: int | None = None
        self.n_action_tokens: int | None = None

    # -- lifecycle ---------------------------------------------------------
    def attach(self) -> "AttentionCapture":
        vwe = self.lp.vwe
        if self._orig_eager is not None:
            return self
        self._orig_eager = vwe.eager_attention_forward
        self._orig_forward = vwe.forward
        orig_eager, orig_forward = self._orig_eager, self._orig_forward

        def forward(*a, **kw):
            self._pass += 1
            self._layer = 0
            return orig_forward(*a, **kw)

        def eager(attention_mask, batch_size, head_dim, q, k, v):
            out = orig_eager(attention_mask, batch_size, head_dim, q, k, v)
            try:
                self._capture(attention_mask, head_dim, q, k)
            finally:
                self._layer += 1
            return out

        vwe.forward = forward
        vwe.eager_attention_forward = eager
        return self

    def detach(self) -> None:
        vwe = self.lp.vwe
        if self._orig_eager is not None:
            vwe.eager_attention_forward = self._orig_eager
            self._orig_eager = None
        if self._orig_forward is not None:
            vwe.forward = self._orig_forward
            self._orig_forward = None

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    def reset(self) -> None:
        self.records.clear()
        self.unclassified.clear()
        self.prefix_len = None
        self.n_action_tokens = None
        self._pass = -1
        self._layer = 0

    # -- capture -----------------------------------------------------------
    def _capture(self, attention_mask, head_dim, q, k) -> None:
        q_len, k_len = int(q.shape[1]), int(k.shape[1])
        if q_len == k_len:
            # prefix prefill: the VLM attending to itself. This ALWAYS runs first inside
            # sample_actions(), which is what pins prefix_len for the calls after it.
            kind = KIND_PREFILL
            if self.prefix_len is None:
                self.prefix_len = k_len
        elif self.prefix_len is None:
            raise RuntimeError(
                f"attention call with q={q_len}, k={k_len} before any prefill call: the prefix "
                f"length is unknown, so expert self-attn and cross-attn cannot be told apart. "
                f"Attach the capture before calling predict, not in the middle of one.")
        elif k_len == self.prefix_len:
            kind = KIND_CROSS
        elif k_len == self.prefix_len + q_len:
            kind = KIND_SELF
        else:
            self.unclassified.append((q_len, k_len))
            return
        if kind != KIND_PREFILL and self.n_action_tokens is None:
            self.n_action_tokens = q_len
        if kind not in self.keep or len(self.records) >= self.max_records:
            return

        # Re-derive probs exactly as eager_attention_forward does (GQA expand, fp32, mask, softmax).
        vwe = self.lp.vwe
        n_heads = vwe.num_attention_heads
        n_kv = vwe.num_key_value_heads
        groups = n_heads // n_kv
        b, s = int(k.shape[0]), k.shape[1]
        kk = k[:, :, :, None, :].expand(b, s, n_kv, groups, head_dim).reshape(b, s, n_kv * groups, head_dim)
        qs = q.to(torch.float32).transpose(1, 2)
        ks = kk.to(torch.float32).transpose(1, 2)
        w = torch.matmul(qs, ks.transpose(2, 3)) * (head_dim ** -0.5)
        big_neg = torch.finfo(w.dtype).min
        w = torch.where(attention_mask[:, None, :, :], w, big_neg)
        probs = torch.softmax(w, dim=-1)[0].detach().float().cpu().numpy()
        self.records.append(AttnRecord(kind=kind, pass_idx=self._pass, layer_idx=self._layer,
                                       probs=probs))

    # -- reduction ---------------------------------------------------------
    def reduce(self, kind: str = KIND_CROSS, layers: tuple[int, ...] | None = None,
               passes: tuple[int, ...] | None = None, prefix_len: int | None = None) -> np.ndarray:
        """-> (prefix_len,) attention mass each PREFIX token receives, averaged over heads,
        action-query tokens, selected layers and selected denoise passes.

        Rows of `probs` sum to 1 over ALL keys, so for KIND_SELF the returned vector does NOT sum
        to 1 (the missing mass went to the action tokens themselves; see `self_mass`).
        """
        sel = [r for r in self.records if r.kind == kind
               and (layers is None or r.layer_idx in layers)
               and (passes is None or r.pass_idx in passes)]
        if not sel:
            raise ValueError(f"no captured records for kind={kind} layers={layers} passes={passes}"
                             f" (have {sorted({(r.kind, r.layer_idx) for r in self.records})})")
        n = prefix_len if prefix_len is not None else self.prefix_len
        acc = np.zeros(n, dtype=np.float64)
        for r in sel:
            acc += r.probs[:, :, :n].mean(axis=(0, 1)).astype(np.float64)
        return acc / len(sel)

    def self_mass(self, layers: tuple[int, ...] | None = None) -> float:
        """Fraction of attention the SELF layers spend on the action tokens themselves."""
        sel = [r for r in self.records if r.kind == KIND_SELF
               and (layers is None or r.layer_idx in layers)]
        if not sel or self.prefix_len is None:
            return float("nan")
        return float(np.mean([r.probs[:, :, self.prefix_len:].sum(axis=-1).mean() for r in sel]))

    def summary(self) -> dict:
        kinds: dict[str, dict] = {}
        for r in self.records:
            d = kinds.setdefault(r.kind, {"n": 0, "layers": set(), "passes": set(), "shape": None})
            d["n"] += 1
            d["layers"].add(r.layer_idx)
            d["passes"].add(r.pass_idx)
            d["shape"] = tuple(r.probs.shape)
        return {k: {"n": v["n"], "layers": sorted(v["layers"]),
                    "n_passes": len(v["passes"]), "probs_shape": v["shape"]}
                for k, v in kinds.items()}


# ==========================================================================
# Attention -> named segments
# ==========================================================================

def split_attention(vec: np.ndarray, layout: PrefixLayout) -> dict:
    """Split a (prefix_len,) attention vector into per-camera grids + language + state."""
    if vec.shape[0] < layout.total:
        raise ValueError(f"attention vector has {vec.shape[0]} entries, prefix is {layout.total}")
    out: dict[str, Any] = {"grids": [], "mass": {}}
    total = float(vec[:layout.total].sum())
    for i in range(layout.n_images):
        a, b = layout.image_span(i)
        g = vec[a:b].reshape(layout.grid_h, layout.grid_w)
        out["grids"].append(g)
        out["mass"][f"camera{i + 1}"] = float(vec[a:b].sum())
    lang = vec[layout.lang_start:layout.lang_start + layout.lang_len]
    out["lang"] = lang
    out["mass"]["language"] = float(lang.sum())
    out["mass"]["state"] = float(vec[layout.state_index])
    out["mass"]["_prefix_total"] = total
    out["mass_frac"] = {k: (v / total if total > 0 else float("nan"))
                        for k, v in out["mass"].items() if not k.startswith("_")}
    return out


def decode_lang_tokens(lp: LoadedPolicy, layout: PrefixLayout) -> list[str]:
    tok = lp.vwe.processor.tokenizer
    return [tok.decode([t]) for t in layout.lang_token_ids]


# ==========================================================================
# Dataset frames
# ==========================================================================

_DS_CACHE: dict = {}


def dataset_frame(dataset_root: str, index: int, chunk_size: int = 16) -> dict:
    """One real frame + its ground-truth action chunk (same loader as the server's self-test).

    The LeRobotDataset object is cached per (root, chunk_size): building it costs seconds, and
    the ablation/temporal sweeps ask for dozens of frames.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_root).expanduser()
    key = (str(root), int(chunk_size))
    if key not in _DS_CACHE:
        meta = json.loads((root / "meta" / "info.json").read_text())
        fps = int(meta.get("fps", 10))
        _DS_CACHE[key] = (LeRobotDataset(
            f"local/{root.name}", root=str(root),
            delta_timestamps={"action": [i / fps for i in range(chunk_size)]}), fps)
    ds, fps = _DS_CACHE[key]
    item = ds[index % len(ds)]
    frames = []
    for c in (1, 2, 3):
        img = item[f"observation.images.camera{c}"]
        frames.append((img.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8))
    gt = item["action"].numpy().astype(np.float64)
    # `action_is_pad[k]` marks steps past the episode end, where LeRobotDataset repeats the last
    # real action. Scoring a prediction against those fabricated steps is not an error measurement,
    # so callers that compare against ground truth must mask them out.
    pad = item.get("action_is_pad")
    pad = (pad.numpy().astype(bool).reshape(-1) if pad is not None
           else np.zeros(gt.shape[0], dtype=bool))
    return {
        "n": len(ds), "index": index % len(ds), "task": item["task"],
        "state": item["observation.state"].numpy().astype(np.float64),
        "gt_chunk": gt, "gt_is_pad": pad,
        "episode_index": int(item["episode_index"]) if "episode_index" in item else -1,
        "frames": frames, "fps": fps,
    }
