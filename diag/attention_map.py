#!/usr/bin/env python3
"""Where is the SmolVLA policy looking? -- attention heatmaps over the 3 camera images.

WHAT IS EXTRACTED (and why that one)
------------------------------------
The action expert's attention over the VLM prefix, i.e. "how much weight does the policy put
on each image token / language token / the state token while producing the action chunk".
That is one softmax away from the question we actually care about; the SigLIP encoder's own
self-attention only says which patches the *image tower* found mutually interesting, which is
policy-independent.

  primary   `expert_cross` -- odd layers (self_attn_every_n_layers=2, attention_mode=cross_attn).
            Query = 16 action tokens, keys = the 241 prefix tokens ONLY, so each row is a clean
            probability distribution over (images, language, state). Default.
  secondary `expert_self`  -- even layers. Keys = 241 prefix + the 16 action tokens, so the row
            mass is shared with the action tokens; `--kind expert_self` reports the leftover
            "attention spent on itself" separately.
  optional  `--vit-attn`   -- SigLIP self-attention, re-running the vision tower with
            output_attentions=True. 32x32 patch resolution (pre pixel-shuffle). Reported as the
            column mean (how much each patch is attended TO). Weaker signal; off by default.

These are real probabilities, not a proxy: `smolvlm_with_expert.get_attention_interface()`
returns `eager_attention_forward` unconditionally (smolvlm_with_expert.py:511-513), so the
softmax is materialised on every layer and can be captured. Verified by --self-test (rows of the
captured cross-attention sum to 1.0).

TOKEN -> PIXEL MAPPING (measured, see diag/smolvla_introspect.py docstring)
--------------------------------------------------------------------------
64 tokens per camera = an 8x8 grid over the 512x512 padded canvas (SigLIP 512/16 = 32x32
patches, connector scale_factor=4 -> 4x4 pixel shuffle -> 8x8). `resize_with_pad` pads on the
LEFT and TOP, so a 240x320 frame lands at rows 128..511 of the canvas and the TOP TWO token rows
(16 of the 64 tokens) are pure padding. The overlay crops them away; `--report` prints how much
attention mass went to those dead tokens, which is a free sanity check on the whole mapping.
Everything above is re-derived at runtime and asserted -- a different checkpoint raises instead
of drawing a wrong picture.

USAGE
-----
    # offline: one dataset frame -> PNG with the 3 cameras side by side
    CUDA_VISIBLE_DEVICES=1 python -m diag.attention_map --frame-index 500 --out-dir runs/attn

    # a few frames + the numeric report
    ... --frame-index 500,1200,4000 --report

    # verification (no PNG needed): layout, probability sums, causal token<->pixel check
    CUDA_VISIBLE_DEVICES=1 python -m diag.attention_map --self-test

Online mode lives in the server: `smolvla_server_openmani.py --attn-every N --attn-out DIR`
uses `AttentionRecorder` from this module (default off; it costs a full extra forward pass).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from diag.smolvla_introspect import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_PATH,
    KIND_CROSS,
    KIND_SELF,
    AttentionCapture,
    LoadedPolicy,
    PadGeometry,
    PrefixLayout,
    dataset_frame,
    load_policy,
    make_noise,
    predict_ticks,
    split_attention,
)


# ==========================================================================
# Rendering
# ==========================================================================

def _colormap(x: np.ndarray) -> np.ndarray:
    """(H,W) in [0,1] -> (H,W,3) uint8. Hand-rolled 'jet'-ish ramp: no matplotlib dependency."""
    x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255.0).round().astype(np.uint8)


def overlay(frame_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """frame (H,W,3) uint8 + heat (H,W) already normalized to [0,1] -> blended uint8."""
    hm = _colormap(heat).astype(np.float32)
    base = frame_rgb.astype(np.float32)
    w = (alpha * np.clip(heat, 0, 1))[..., None]
    return np.clip(base * (1 - w) + hm * w, 0, 255).astype(np.uint8)


def _draw_grid(img: np.ndarray, gh: int, gw: int, geom: PadGeometry) -> np.ndarray:
    """Faint lines at the token-cell boundaries as they land on the ORIGINAL frame."""
    out = img.copy()
    h, w = out.shape[:2]
    cell_h, cell_w = geom.target_h / gh, geom.target_w / gw
    scale_r = h / max(geom.resized_h, 1)
    scale_c = w / max(geom.resized_w, 1)
    for r in range(gh + 1):
        y = int(round((r * cell_h - geom.pad_top) * scale_r))
        if 0 <= y < h:
            out[y] = (0.5 * out[y] + 0.5 * np.array([255, 255, 255])).astype(np.uint8)
    for c in range(gw + 1):
        x = int(round((c * cell_w - geom.pad_left) * scale_c))
        if 0 <= x < w:
            out[:, x] = (0.5 * out[:, x] + 0.5 * np.array([255, 255, 255])).astype(np.uint8)
    return out


def _hstack(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    parts = []
    for i, im in enumerate(images):
        if im.shape[0] != h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
            im = np.concatenate([im, pad], axis=0)
        if i:
            parts.append(np.zeros((h, gap, 3), dtype=np.uint8))
        parts.append(im)
    return np.concatenate(parts, axis=1)


def save_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


# ==========================================================================
# One frame -> attention
# ==========================================================================

def attention_for_frame(lp: LoadedPolicy, layout: PrefixLayout, frames: list[np.ndarray],
                        state: np.ndarray, task: str, kind: str = KIND_CROSS,
                        layers: tuple[int, ...] | None = None,
                        passes: tuple[int, ...] | None = None,
                        seed: int = 0) -> dict:
    """Run one prediction with attention captured; return the split attention + the chunk."""
    cap = AttentionCapture(lp, keep=(KIND_CROSS, KIND_SELF))
    with cap:
        t0 = time.perf_counter()
        chunk, batch = predict_ticks(lp, frames, state, task, noise=make_noise(lp, seed))
        ms = (time.perf_counter() - t0) * 1000.0
    if cap.prefix_len != layout.total:
        raise RuntimeError(f"captured prefix_len={cap.prefix_len} but the measured layout says "
                           f"{layout.total}; refusing to map tokens to pixels.")
    vec = cap.reduce(kind=kind, layers=layers, passes=passes)
    out = split_attention(vec, layout)
    # The layout is measured ONCE, so its cached lang_token_ids belong to the frame it was
    # measured on. Take THIS frame's tokens off the batch instead -- otherwise the per-token
    # report silently prints the wrong instruction's tokens.
    out["lang_token_ids"], out["lang_mask"] = _batch_lang(batch, layout)
    out["chunk_ticks"] = chunk
    out["capture_summary"] = cap.summary()
    out["unclassified"] = cap.unclassified
    out["inference_ms"] = ms
    out["kind"] = kind
    if kind == KIND_SELF:
        out["mass"]["action_tokens_self"] = cap.self_mass(layers=layers)
    return out


def _batch_lang(batch: dict, layout: PrefixLayout) -> tuple[list[int], list[int]]:
    """This frame's language tokens + pad mask, straight off the preprocessed batch."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    try:
        ids = batch[OBS_LANGUAGE_TOKENS][0].detach().cpu().tolist()
        msk = batch[OBS_LANGUAGE_ATTENTION_MASK][0].detach().cpu().to(int).tolist()
    except (KeyError, IndexError, AttributeError):
        return list(layout.lang_token_ids), list(layout.lang_mask)
    if len(ids) != layout.lang_len:
        raise RuntimeError(f"this frame has {len(ids)} language tokens but the layout was "
                           f"measured with {layout.lang_len}; the prefix spans would shift.")
    return ids, msk


def render_frame(frames: list[np.ndarray], split: dict, layout: PrefixLayout,
                 normalize: str = "per-camera", alpha: float = 0.55,
                 grid: bool = True) -> np.ndarray:
    """3 cameras side by side, each with its attention heatmap overlaid.

    normalize='per-camera' scales each camera's map to its own max (shows WHERE inside that view);
    'global' uses one scale across cameras (also shows WHICH camera gets more attention -- but a
    camera with little mass then looks empty). Both are useful; the mass fractions in --report
    are the quantitative answer to "which camera".
    """
    geoms = [PadGeometry.compute(f.shape[0], f.shape[1], layout.target_h, layout.target_w)
             for f in frames]
    valid = [g.valid_token_mask(layout.grid_h, layout.grid_w) for g in geoms]
    masked = [np.where(v, g, 0.0) for g, v in zip(split["grids"], valid)]
    gmax = max(float(m.max()) for m in masked) or 1.0
    panels = []
    for i, (f, gm, ge) in enumerate(zip(frames, masked, geoms)):
        scale = (float(gm.max()) or 1.0) if normalize == "per-camera" else gmax
        heat = ge.grid_to_source(gm / scale)
        img = overlay(f, heat, alpha=alpha)
        if grid:
            img = _draw_grid(img, layout.grid_h, layout.grid_w, ge)
        panels.append(img)
    return _hstack(panels)


def report_lines(split: dict, layout: PrefixLayout, lp: LoadedPolicy,
                 frames: list[np.ndarray], top_k: int = 8) -> list[str]:
    lines = []
    mf = split["mass_frac"]
    lines.append("  attention mass (fraction of the softmax rows, kind=%s):"
                 % split.get("kind", KIND_CROSS))
    for k in [f"camera{i + 1}" for i in range(layout.n_images)] + ["language", "state"]:
        lines.append(f"    {k:<10s} {split['mass'][k]:.4f}  ({100 * mf[k]:5.1f}%)")
    if "action_tokens_self" in split["mass"]:
        lines.append(f"    {'self(act)':<10s} {split['mass']['action_tokens_self']:.4f}  "
                     f"(share of the row that never reaches the prefix)")
    img_frac = sum(mf[f"camera{i + 1}"] for i in range(layout.n_images))
    lines.append(f"    -> vision {100 * img_frac:.1f}%  language {100 * mf['language']:.1f}%  "
                 f"state {100 * mf['state']:.1f}%")

    # padding sanity: mass that landed on token cells which contain no image at all.
    for i, f in enumerate(frames):
        ge = PadGeometry.compute(f.shape[0], f.shape[1], layout.target_h, layout.target_w)
        v = ge.valid_token_mask(layout.grid_h, layout.grid_w)
        g = split["grids"][i]
        tot = float(g.sum()) or 1.0
        lines.append(f"    camera{i + 1} pad-token mass {100 * float(g[~v].sum()) / tot:5.1f}% "
                     f"of its own ({int((~v).sum())}/{v.size} cells are padding; {ge.describe()})")

    ids = split.get("lang_token_ids") or layout.lang_token_ids
    msk = split.get("lang_mask") or layout.lang_mask
    tok = lp.vwe.processor.tokenizer
    toks = [tok.decode([t]) for t in ids]
    lang = split["lang"]
    mask = np.asarray(msk, dtype=bool)
    pad_mass = float(lang[~mask].sum())
    lines.append(f"    language mass on PAD positions {pad_mass:.6f} "
                 f"({100 * pad_mass / (float(lang.sum()) or 1.0):.2f}% of the language mass) "
                 f"-- should be ~0; a large value means the pad mask is not reaching attention")
    order = np.argsort(-lang)[:top_k]
    lines.append(f"  top-{top_k} language tokens (of {layout.lang_len}, "
                 f"{int(np.sum(msk))} unmasked):")
    for j in order:
        flag = "" if msk[j] else "  [PAD]"
        lines.append(f"    #{j:<3d} {toks[j]!r:<16s} {lang[j]:.5f}{flag}")
    return lines


# ==========================================================================
# Optional: SigLIP self-attention
# ==========================================================================

def vit_attention(lp: LoadedPolicy, frames: list[np.ndarray], state: np.ndarray,
                  task: str) -> list[np.ndarray] | None:
    """Last-layer SigLIP self-attention, column mean = how much each patch is attended TO.

    Returns one (p,p) map per camera at PATCH resolution (32x32 for 512/16), or None when the
    installed transformers build refuses output_attentions (sdpa without an eager fallback).
    """
    import torch

    from diag.smolvla_introspect import build_observation

    obs = build_observation(lp, frames, state, task)
    lp.policy.reset()
    lp.preprocessor.reset()
    vm = lp.vwe.get_vlm_model().vision_model
    # SiglipAttention picks its kernel from `config._attn_implementation`; under the default
    # sdpa the probabilities are never materialised and `attentions` comes back empty. Force
    # eager for this call only, then restore -- the policy must keep running on sdpa (no
    # flash-attn on sm_120, and eager everywhere would slow every /predict down).
    cfgs = [c for c in (getattr(vm, "config", None), getattr(lp.vwe.vlm, "config", None)) if c]
    cfgs += [getattr(c, "vision_config", None) for c in cfgs]
    cfgs = [c for c in cfgs if c is not None]
    saved = [(c, getattr(c, "_attn_implementation", None)) for c in cfgs]
    maps = []
    try:
        for c in cfgs:
            c._attn_implementation = "eager"
        with torch.no_grad():
            batch = lp.preprocessor(obs)
            imgs, _ = lp.policy.prepare_images(batch)
            for im in imgs:
                try:
                    out = vm(pixel_values=im.to(dtype=vm.dtype), patch_attention_mask=None,
                             output_attentions=True)
                except Exception as e:
                    print(f"[vit] output_attentions unavailable ({type(e).__name__}: {e}); skipped")
                    return None
                att = getattr(out, "attentions", None)
                if not att or att[-1] is None:
                    print("[vit] the vision tower returned no attentions even with eager "
                          "attention forced; skipped (this build does not expose them).")
                    return None
                a = att[-1][0].float().cpu().numpy()          # (heads, n, n)
                col = a.mean(axis=0).mean(axis=0)             # attention RECEIVED per patch
                p = int(round(np.sqrt(col.shape[0])))
                if p * p != col.shape[0]:
                    print(f"[vit] {col.shape[0]} patches is not square; skipped")
                    return None
                maps.append(col.reshape(p, p))
    finally:
        for c, v in saved:
            if v is None:
                try:
                    delattr(c, "_attn_implementation")
                except Exception:
                    pass
            else:
                c._attn_implementation = v
    return maps


# ==========================================================================
# Online recorder (used by the server when --attn-every is set)
# ==========================================================================

class AttentionRecorder:
    """Save an attention PNG + a JSON line every Nth request. Costs one EXTRA forward pass.

    The server's own /predict already ran the policy; re-running it here with the capture
    attached keeps the served chunk untouched (and flow matching is stochastic, so the recorded
    attention belongs to a different noise draw than the served chunk -- that is stated in the
    JSON as `separate_forward_pass: true` rather than hidden).
    """

    def __init__(self, lp: LoadedPolicy, out_dir: str, every: int = 50, kind: str = KIND_CROSS,
                 save_png_files: bool = True):
        self.lp = lp
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(every))
        self.kind = kind
        self.save_png = bool(save_png_files)
        self.layout: PrefixLayout | None = None
        self._n = 0
        self._jsonl = (self.dir / "attention.jsonl").open("a", buffering=1)

    def maybe_record(self, frames: list[np.ndarray], state: np.ndarray, task: str,
                     request_id: int) -> dict | None:
        if request_id % self.every != 0:
            return None
        try:
            if self.layout is None:
                self.layout = PrefixLayout.measure(self.lp, frames, state, task)
            split = attention_for_frame(self.lp, self.layout, frames, state, task, kind=self.kind)
        except Exception as e:
            print(f"[attn][warn] request {request_id}: {type(e).__name__}: {e}")
            return None
        rec = {"request": int(request_id), "task": task, "kind": self.kind,
               "state_ticks": np.asarray(state).astype(int).tolist(),
               "mass": split["mass"], "mass_frac": split["mass_frac"],
               "separate_forward_pass": True}
        if self.save_png:
            p = self.dir / f"attn_{request_id:06d}.png"
            save_png(p, render_frame(frames, split, self.layout))
            rec["png"] = str(p)
        self._jsonl.write(json.dumps(rec) + "\n")
        self._n += 1
        return rec

    def close(self):
        try:
            self._jsonl.close()
        except Exception:
            pass


# ==========================================================================
# Self-test
# ==========================================================================

def _checker():
    state = {"ok": True}

    def check(name, cond, extra=""):
        state["ok"] = state["ok"] and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")

    return state, check


def self_test(args) -> int:
    import torch

    print("== diag.attention_map self-test ==")
    st, check = _checker()

    # --- 1. pure geometry, no model -------------------------------------------------
    ge = PadGeometry.compute(240, 320, 512, 512)
    check("resize_with_pad geometry for 240x320 -> 512x512",
          (ge.resized_h, ge.resized_w, ge.pad_top, ge.pad_left) == (384, 512, 128, 0),
          ge.describe())
    v = ge.valid_token_mask(8, 8)
    check("top 2 of 8 token rows are padding", (~v).sum() == 16 and not v[0].any()
          and not v[1].any() and v[2:].all(), f"{int((~v).sum())}/64 pad cells")
    probe = np.zeros((8, 8), dtype=np.float64)
    probe[2, 0] = 1.0                       # first real cell: top-left of the visible image
    m = ge.grid_to_source(probe)
    check("token (2,0) maps to the TOP-LEFT of the original frame",
          m[0, 0] == 1.0 and m[-1, -1] == 0.0 and m.sum() > 0,
          f"hot rows 0..{int(np.max(np.nonzero(m.any(axis=1))))} "
          f"cols 0..{int(np.max(np.nonzero(m.any(axis=0))))}")
    probe2 = np.zeros((8, 8))
    probe2[7, 7] = 1.0
    m2 = ge.grid_to_source(probe2)
    check("token (7,7) maps to the BOTTOM-RIGHT of the original frame",
          m2[-1, -1] == 1.0 and m2[0, 0] == 0.0)

    if args.no_model:
        print(f"== self-test (geometry only) {'PASSED' if st['ok'] else 'FAILED'} ==")
        return 0 if st["ok"] else 1

    # --- 2. model + real frame ------------------------------------------------------
    lp = load_policy(args.model_path, args.dataset_root, args.device)
    d = dataset_frame(args.dataset_root, args.self_test_index, int(lp.cfg.chunk_size))
    frames, state, task = d["frames"], d["state"], d["task"]
    print(f"  frame #{d['index']} of {d['n']}  task={task!r}  state={state.astype(int).tolist()}")

    layout = PrefixLayout.measure(lp, frames, state, task)
    print("  " + layout.describe())
    check("prefix accounting closes",
          layout.n_images * layout.tokens_per_image + layout.lang_len + 1 == layout.total,
          f"{layout.n_images}x{layout.tokens_per_image}+{layout.lang_len}+1={layout.total}")
    check("image tokens form a square grid", layout.grid_h * layout.grid_w == layout.tokens_per_image,
          f"{layout.grid_h}x{layout.grid_w}")
    check("one image key per camera", layout.n_images == len(lp.image_keys),
          f"{lp.image_keys}")

    cap = AttentionCapture(lp, keep=(KIND_CROSS, KIND_SELF))
    with cap:
        chunk, _ = predict_ticks(lp, frames, state, task, noise=make_noise(lp, 0))
    summ = cap.summary()
    print(f"  capture: {json.dumps(summ)}")
    check("cross-attention layers were captured", KIND_CROSS in summ, str(list(summ)))
    check("no unclassified attention calls", not cap.unclassified, str(cap.unclassified[:4]))
    check("captured prefix_len == measured layout", cap.prefix_len == layout.total,
          f"{cap.prefix_len} vs {layout.total}")
    check("expert query tokens == chunk_size", cap.n_action_tokens == layout.n_action_tokens,
          f"{cap.n_action_tokens}")
    cross = [r for r in cap.records if r.kind == KIND_CROSS]
    check("cross layers are the ODD layer indices (self_attn_every_n_layers=2)",
          all(r.layer_idx % 2 == 1 for r in cross),
          f"layers={sorted({r.layer_idx for r in cross})}")
    rows = cross[0].probs.sum(axis=-1)
    check("captured cross-attention rows are probabilities (sum to 1)",
          bool(np.allclose(rows, 1.0, atol=1e-4)),
          f"min={rows.min():.6f} max={rows.max():.6f}")
    sself = [r for r in cap.records if r.kind == KIND_SELF]
    check("self layers cover the prefix + the 16 action tokens",
          all(r.probs.shape[-1] == layout.total + layout.n_action_tokens for r in sself),
          f"k={sself[0].probs.shape[-1] if sself else '-'}")

    vec = cap.reduce(kind=KIND_CROSS)
    split = split_attention(vec, layout)
    check("split covers the whole prefix",
          abs(sum(split["mass"][k] for k in list(split["mass_frac"])) - vec[:layout.total].sum()) < 1e-6)

    # --- 3. CAUSAL token<->pixel check ----------------------------------------------
    # Perturb one 64x64 cell of the ORIGINAL frame and see which image token moves most.
    # This is the only check that can catch a row/column-major or a flipped-padding mistake.
    print("  causal token<->pixel check (perturb one cell, find the most-changed token):")
    base_tokens = _image_tokens(lp, frames, state, task)
    ge0 = PadGeometry.compute(frames[0].shape[0], frames[0].shape[1],
                              layout.target_h, layout.target_w)
    hits = 0
    trials = [(2, 0), (2, 7), (7, 0), (7, 7), (4, 3)]
    for (tr, tc) in trials:
        f2 = [f.copy() for f in frames]
        y0, y1, x0, x1 = _cell_source_box(ge0, layout, tr, tc, frames[0].shape[:2])
        if y1 <= y0 or x1 <= x0:
            continue
        f2[0][y0:y1, x0:x1] = 255 - f2[0][y0:y1, x0:x1]     # invert the cell
        tk = _image_tokens(lp, f2, state, task)
        dist = np.linalg.norm(tk[0] - base_tokens[0], axis=-1)  # (64,)
        got = int(np.argmax(dist))
        want = tr * layout.grid_w + tc
        ok = got == want
        hits += ok
        print(f"    cell (r={tr},c={tc}) src box y[{y0}:{y1}] x[{x0}:{x1}] -> "
              f"argmax token {got} (row-major expects {want}) {'OK' if ok else 'MISMATCH'}")
    check("token index == row-major (r*grid_w + c) for every probed cell",
          hits == len([t for t in trials]), f"{hits}/{len(trials)}")

    # --- 4. render ------------------------------------------------------------------
    img = render_frame(frames, split, layout)
    check("rendered panel has the right shape",
          img.ndim == 3 and img.shape[2] == 3 and img.shape[0] == frames[0].shape[0],
          f"{img.shape}")
    if args.out_dir:
        p = Path(args.out_dir) / "selftest_attention.png"
        save_png(p, img)
        print(f"  wrote {p}")
    for line in report_lines(split, layout, lp, frames):
        print(line)
    print(f"  predicted chunk[0] (ticks) = {np.rint(chunk[0]).astype(int).tolist()}")
    print(f"  gt chunk[0]        (ticks) = {np.rint(d['gt_chunk'][0]).astype(int).tolist()}")

    print(f"== self-test {'PASSED' if st['ok'] else 'FAILED'} ==")
    return 0 if st["ok"] else 1


def _cell_source_box(ge: PadGeometry, layout: PrefixLayout, r: int, c: int,
                     src_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """Grid cell (r,c) -> pixel box in the ORIGINAL frame (clipped to it)."""
    cell_h, cell_w = ge.target_h / layout.grid_h, ge.target_w / layout.grid_w
    sr = src_hw[0] / max(ge.resized_h, 1)
    sc = src_hw[1] / max(ge.resized_w, 1)
    y0 = int(np.floor((r * cell_h - ge.pad_top) * sr))
    y1 = int(np.ceil(((r + 1) * cell_h - ge.pad_top) * sr))
    x0 = int(np.floor((c * cell_w - ge.pad_left) * sc))
    x1 = int(np.ceil(((c + 1) * cell_w - ge.pad_left) * sc))
    return (max(0, y0), min(src_hw[0], y1), max(0, x0), min(src_hw[1], x1))


def _image_tokens(lp: LoadedPolicy, frames: list[np.ndarray], state: np.ndarray,
                  task: str) -> list[np.ndarray]:
    """The 64 post-connector tokens per camera, as numpy. Used by the causal mapping check."""
    import torch

    from diag.smolvla_introspect import build_observation

    obs = build_observation(lp, frames, state, task)
    lp.policy.reset()
    lp.preprocessor.reset()
    out = []
    with torch.no_grad():
        batch = lp.preprocessor(obs)
        imgs, _ = lp.policy.prepare_images(batch)
        for im in imgs:
            out.append(lp.vwe.embed_image(im)[0].float().cpu().numpy())
    return out


# ==========================================================================
# Entry
# ==========================================================================

def _parse_ints(s: str | None) -> tuple[int, ...] | None:
    if not s:
        return None
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frame-index", default="500",
                    help="dataset frame index/indices (comma separated) for offline mode")
    ap.add_argument("--task", default=None,
                    help="override the instruction (default: the frame's own label)")
    ap.add_argument("--out-dir", default="runs/attention")
    ap.add_argument("--kind", choices=(KIND_CROSS, KIND_SELF), default=KIND_CROSS,
                    help="expert_cross (default, keys = prefix only) or expert_self")
    ap.add_argument("--layers", default=None,
                    help="restrict to these layer indices (default: all layers of --kind)")
    ap.add_argument("--denoise-pass", default=None,
                    help="restrict to these vlm_with_expert.forward() pass indices; pass 0 is the "
                         "prefix prefill, so denoise steps are 1..num_steps")
    ap.add_argument("--normalize", choices=("per-camera", "global"), default="per-camera")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--no-grid", action="store_true", help="do not draw token-cell boundaries")
    ap.add_argument("--report", action="store_true", help="print the numeric attention report")
    ap.add_argument("--vit-attn", action="store_true",
                    help="also dump the SigLIP self-attention (patch resolution, secondary)")
    ap.add_argument("--seed", type=int, default=0, help="flow-matching noise seed")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-index", type=int, default=500)
    ap.add_argument("--no-model", action="store_true",
                    help="--self-test only: geometry checks that need no GPU/checkpoint")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args)

    lp = load_policy(args.model_path, args.dataset_root, args.device)
    indices = [int(x) for x in str(args.frame_index).split(",") if x.strip() != ""]
    layers = _parse_ints(args.layers)
    passes = _parse_ints(args.denoise_pass)
    out_dir = Path(args.out_dir)
    layout = None
    summary = []
    for idx in indices:
        d = dataset_frame(args.dataset_root, idx, int(lp.cfg.chunk_size))
        task = args.task or d["task"]
        if layout is None:
            layout = PrefixLayout.measure(lp, d["frames"], d["state"], task)
            print("[layout] " + layout.describe())
        split = attention_for_frame(lp, layout, d["frames"], d["state"], task, kind=args.kind,
                                    layers=layers, passes=passes, seed=args.seed)
        png = out_dir / f"attn_frame{d['index']:06d}_{args.kind}.png"
        save_png(png, render_frame(d["frames"], split, layout, normalize=args.normalize,
                                   alpha=args.alpha, grid=not args.no_grid))
        print(f"[frame {d['index']}] task={task!r} -> {png}  ({split['inference_ms']:.0f} ms)")
        if args.report:
            for line in report_lines(split, layout, lp, d["frames"]):
                print(line)
        if args.vit_attn:
            maps = vit_attention(lp, d["frames"], d["state"], task)
            if maps is not None:
                panels = []
                for f, m in zip(d["frames"], maps):
                    ge = PadGeometry.compute(f.shape[0], f.shape[1], layout.target_h, layout.target_w)
                    v = ge.valid_token_mask(m.shape[0], m.shape[1])
                    mm = np.where(v, m, 0.0)
                    panels.append(overlay(f, ge.grid_to_source(mm / (mm.max() or 1.0)), args.alpha))
                p = out_dir / f"vit_frame{d['index']:06d}.png"
                save_png(p, _hstack(panels))
                print(f"           SigLIP self-attention ({maps[0].shape[0]}x{maps[0].shape[1]} "
                      f"patches) -> {p}")
        summary.append({"index": int(d["index"]), "task": task, "png": str(png),
                        "mass": split["mass"], "mass_frac": split["mass_frac"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {len(summary)} frame(s) -> {out_dir}/attention_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
