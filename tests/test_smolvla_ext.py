"""CPU tests for smolvla_ext (PVI / Coda / ROCKET attached to a stock SmolVLAPolicy).

Specs §7 of SmolVLA_{PVI,Coda,ROCKET}_구현.md, adapted to the 2026-10-11 plan (VLM trains in every
variant, freeze = main AE only, ROCKET pairs). Random weights (load_vlm_weights=False), 3 cameras
(3, 240, 320), 6-D state/action.

To fit the RAM budget the SmolVLM2 config is shrunk at construction: text layers 32 -> N_LAYERS
(SmolVLA keeps the first num_vlm_layers=N_LAYERS), vision layers 12 -> 1, vocab 49280 -> 1024.
Widths (960 / 720 / 768 / 5x64 kv) and the 64-tokens-per-camera layout are unchanged.
ROCKET pairs are scaled to the shallow model (student layers 0..N-1 <-> VGGT 4, 11, 17, 23).

pytest is not installed in the lerobot env, so this file also runs standalone:
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=.:third_party \
        /home/vcl/anaconda3/envs/lerobot/bin/python tests/test_smolvla_ext.py [-k substring]
"""

from __future__ import annotations

import copy
import gc
import inspect
import json
import os
import pickle
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "third_party"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import pytest  # noqa: F401
except ImportError:  # standalone runner below
    pytest = None

import lerobot.policies.smolvla.smolvlm_with_expert as swe  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

from smolvla_ext import ExtConfig, attach, load_policy  # noqa: E402
from smolvla_ext.attach import MAIN_PREFIXES  # noqa: E402
from smolvla_ext.config import SIDECAR_NAME  # noqa: E402
from smolvla_ext.pvi import block_of_token, build_copy_cache, patch_valid_mask  # noqa: E402
from smolvla_ext.rocket import (  # noqa: E402
    VGGT_KEY, CarryKeysPre, TeacherDataset, TeacherTargets, align_loss, valid_token_mask)
from smolvla_ext.vlm_forward import ext_vlm_forward  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 1))

N_LAYERS = 4
VOCAB = int(os.environ.get("SMOLVLA_EXT_TEST_VOCAB", "1024"))  # 0 = keep 49280 (real tokenizer ids)
CAMS = [f"observation.images.camera{i}" for i in (1, 2, 3)]
HW = {k: [240, 320] for k in CAMS}
PAIRS = [[0, 4], [1, 11], [2, 17], [3, 23]]
CHUNK = 4
B = 2


class _Skip(Exception):
    pass


def skip(msg):
    if pytest is not None:
        pytest.skip(msg)
    raise _Skip(msg)


# ------------------------------------------------------------------ small model construction --
_RealAutoConfig = swe.AutoConfig


class _SmallAutoConfig:
    @staticmethod
    def from_pretrained(model_id, *a, **kw):
        c = _RealAutoConfig.from_pretrained(model_id, *a, **kw)
        c.text_config.num_hidden_layers = N_LAYERS
        c.vision_config.num_hidden_layers = 1
        if VOCAB:
            c.text_config.vocab_size = VOCAB
        return c


swe.AutoConfig = _SmallAutoConfig


def make_cfg(**over):
    kw = dict(load_vlm_weights=False, chunk_size=CHUNK, n_action_steps=CHUNK, num_steps=5,
              freeze_vision_encoder=False, train_expert_only=False, num_vlm_layers=N_LAYERS, device="cpu")
    kw.update(over)
    cfg = SmolVLAConfig(**kw)
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                          **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)) for k in CAMS}}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    return cfg


def make_policy(ext: ExtConfig | None = None, **over):
    set_seed(0)  # same seed -> identical weights for stock and extended policies
    p = SmolVLAPolicy(make_cfg(**over))
    if ext is not None:
        attach(p, ext, camera_hw=HW, verbose=False)
    return p


def ext_cfg(**kw):
    kw.setdefault("rocket_pairs", PAIRS)
    return ExtConfig(**kw)


def make_batch(seed=1, targets=False, n_layers=None):
    g = torch.Generator().manual_seed(seed)
    b = {k: torch.rand(B, 3, 240, 320, generator=g) for k in CAMS}
    b["observation.state"] = torch.randn(B, 6, generator=g)
    b["action"] = torch.randn(B, CHUNK, 6, generator=g)
    pad = torch.zeros(B, CHUNK, dtype=torch.bool)
    pad[1, -1] = True
    b["action_is_pad"] = pad
    b["observation.language.tokens"] = torch.randint(0, 1000, (B, 6), generator=g)
    m = torch.ones(B, 6, dtype=torch.bool)
    m[1, -2:] = False
    b["observation.language.attention_mask"] = m
    b["index"] = torch.tensor([3, 7])
    if targets:
        b[VGGT_KEY] = torch.randn(B, n_layers or len(PAIRS), 3, 64, 2048, generator=g).half()
    return b


def noise_time(seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, CHUNK, 32, generator=g), torch.rand(B, generator=g) * 0.9 + 0.05


def model_inputs(policy, batch):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return (images, img_masks, batch["observation.language.tokens"],
            batch["observation.language.attention_mask"], state)


def sample(policy, batch, noise, **kw):
    with torch.no_grad():
        return policy.model.sample_actions(*model_inputs(policy, batch), noise=noise, **kw)


def rtc_on(policy, horizon=2, delay=1, schedule="LINEAR", max_w=10.0):
    """Turn RTC on for this policy instance (what the server's configure_rtc does, in miniature:
    CHUNK=4 here, so horizon 2 / delay 1 / a 2-step tail mirrors the deployed 8 / 1 / 8)."""
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=horizon,
                                         prefix_attention_schedule=RTCAttentionSchedule(schedule),
                                         max_guidance_weight=max_w)
    policy.init_rtc_processor()      # also pushes the processor onto policy.model
    return delay, horizon


def prev_tail(seed=9, steps=2, dim=6):
    """The unexecuted tail of the previous chunk, 6-wide like the server's stored plan."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, steps, dim, generator=g)


def grad_is_zero(p):
    return p.grad is None or float(p.grad.abs().max()) == 0.0


def grad_nonzero(p):
    return p.grad is not None and float(p.grad.abs().max()) > 0.0


def free(*objs):
    for o in objs:
        del o
    gc.collect()


# ================================================================================== common ==
def test_vlm_forward_copy_bitwise():
    """smolvla_ext copy of SmolVLMWithExpertModel.forward with default args == the class method."""
    p = make_policy()
    flow = p.model
    vwe = flow.vlm_with_expert
    orig = type(vwe).forward
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    batch = make_batch()
    images, img_masks, lt, lm, state = model_inputs(p, batch)
    noise, time = noise_time()
    with torch.no_grad():
        pe, ppad, patt = flow.embed_prefix(images, img_masks, lt, lm, state=state)
        se, spad, satt = flow.embed_suffix(noise, time)
        pad = torch.cat([ppad, spad], 1)
        att = make_att_2d_masks(pad, torch.cat([patt, satt], 1))
        pos = torch.cumsum(pad, 1) - 1
        # joint training forward
        a, _ = orig(vwe, attention_mask=att, position_ids=pos, past_key_values=None,
                    inputs_embeds=[pe, se], use_cache=False)
        b, _ = ext_vlm_forward(vwe, attention_mask=att, position_ids=pos, past_key_values=None,
                               inputs_embeds=[pe, se], use_cache=False)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        # prefix prefill + one cached denoise step
        patt2 = make_att_2d_masks(ppad, patt)
        ppos = torch.cumsum(ppad, 1) - 1
        _, c1 = orig(vwe, attention_mask=patt2, position_ids=ppos, past_key_values=None,
                     inputs_embeds=[pe, None], use_cache=True)
        _, c2 = ext_vlm_forward(vwe, attention_mask=patt2, position_ids=ppos, past_key_values=None,
                                inputs_embeds=[pe, None], use_cache=True)
        for l1, l2 in zip(c1.layers, c2.layers):
            assert torch.equal(l1.keys, l2.keys) and torch.equal(l1.values, l2.values)
        v1 = type(flow).denoise_step(flow, ppad, c1, noise, time)
        v2 = type(flow).denoise_step(flow, ppad, c2, noise, time)
        assert torch.equal(v1, v2)
        # bound method on the instance, the denoise step through it
        vwe.forward = ext_vlm_forward.__get__(vwe)
        v3 = type(flow).denoise_step(flow, ppad, c2, noise, time)
        assert torch.equal(v1, v3)
    free(p)


def test_ext_off_equals_stock():
    """attach() with every module off: same loss, same samples, same keys, same trainable set."""
    a = make_policy()
    b = make_policy(ext_cfg())
    batch = make_batch()
    noise, time = noise_time()
    a.train(); b.train()
    la, da = a.forward(copy.copy(batch), noise=noise, time=time)
    lb, db = b.forward(copy.copy(batch), noise=noise, time=time)
    assert torch.equal(la, lb)
    for k in ("losses_after_forward", "losses_after_in_ep_bound", "losses_after_rm_padding", "loss"):
        assert da[k] == db[k], k
    pa, _ = a.forward(copy.copy(batch), noise=noise, time=time, reduction="none")
    pb, _ = b.forward(copy.copy(batch), noise=noise, time=time, reduction="none")
    assert torch.equal(pa, pb)
    a.eval(); b.eval()
    assert torch.equal(sample(a, batch, noise), sample(b, batch, noise))
    assert list(a.state_dict()) == list(b.state_dict())
    ta = [n for n, p in a.named_parameters() if p.requires_grad]
    tb = [n for n, p in b.named_parameters() if p.requires_grad]
    assert ta == tb
    groups = b.get_optim_params()
    got = sorted(id(p) for g in groups for p in g["params"])
    assert got == sorted(id(p) for p in b.parameters() if p.requires_grad)
    free(a, b)


# ===================================================================================== PVI ==
def test_pvi_masks_and_block_map():
    m = patch_valid_mask(240, 320, 512, 512, 16)
    assert m.shape == (1024,) and not m[:256].any() and m[256:].all()   # top 8 of 32 rows padding
    assert patch_valid_mask(512, 512, 512, 512).all()
    # connector.pixel_shuffle groups patch (r, c) into token (r // 4) * 8 + c // 4
    p = make_policy()
    conn = p.model.vlm_with_expert.get_vlm_model().connector
    x = torch.arange(1024, dtype=torch.float32).view(1, 1024, 1)
    out = conn.pixel_shuffle(x, conn.scale_factor)[0]              # [64, 16]
    bot = block_of_token(1024)
    for t in range(64):
        r0, c0 = 4 * (t // 8), 4 * (t % 8)
        want = {r * 32 + c for r in range(r0, r0 + 4) for c in range(c0, c0 + 4)}
        assert set(out[t].long().tolist()) == want
        assert set(torch.nonzero(bot == t).flatten().tolist()) == want
    assert torch.equal(block_of_token(64), torch.arange(64))
    free(p)


def test_pvi_zero_init_equivalence():
    a = make_policy()
    for main_ae in ("train", "freeze", "lr"):
        b = make_policy(ext_cfg(pvi=True, main_ae=main_ae))
        batch = make_batch()
        noise, time = noise_time()
        with torch.no_grad():
            la = a.model.forward(*model_inputs(a, batch), b.prepare_action(batch), noise, time)
            lb = b.model.forward(*model_inputs(b, batch), b.prepare_action(batch), noise, time)
        assert float((la - lb).abs().max()) < 1e-6
        a.eval(); b.eval()
        assert float((sample(a, batch, noise) - sample(b, batch, noise)).abs().max()) < 1e-6
        free(b)
    free(a)


def test_pvi_grad_order_and_groups():
    b = make_policy(ext_cfg(pvi=True, main_ae="lr"))
    br = b.model.bypass
    assert not br.copy_expert.norm.weight.requires_grad
    groups = b.get_optim_params()
    assert len(groups) == 3 and groups[1]["lr"] == 1e-5 and groups[2]["lr"] == 1e-4
    batch = make_batch()
    b.train()
    loss, _ = b.forward(copy.copy(batch))
    loss.backward()
    assert all(grad_nonzero(z.weight) for z in br.z)
    for n, prm in br.named_parameters():
        if not n.startswith("z."):
            assert grad_is_zero(prm), n
    opt = torch.optim.AdamW(b.get_optim_params(), lr=1e-4)
    opt.step(); opt.zero_grad()
    loss, _ = b.forward(copy.copy(batch))
    loss.backward()
    assert any(grad_nonzero(prm) for prm in br.copy_expert.parameters() if prm.requires_grad)
    assert grad_nonzero(br.w_v.weight)
    # freeze: main AE out, VLM still trains, copy norm out, exactly one group per param
    f = make_policy(ext_cfg(pvi=True, main_ae="freeze"))
    for n, prm in f.named_parameters():
        if n.startswith(MAIN_PREFIXES):
            assert not prm.requires_grad, n
    assert any(prm.requires_grad for n, prm in f.named_parameters() if ".vlm." in n)
    g = f.get_optim_params()
    assert len(g) == 2 and g[1]["lr"] == 1e-4
    free(b, f)


def test_pvi_state_kv_and_positions():
    b = make_policy(ext_cfg(pvi=True))
    flow = b.model
    batch = make_batch()
    images, img_masks, lt, lm, state = model_inputs(b, batch)
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
    with torch.no_grad():
        flow._ext_feats = []
        pe, ppad, patt = flow.embed_prefix(images, img_masks, lt, lm, state=state)
        feats = flow._ext_feats
        flow._ext_feats = None
        assert len(feats) == 3 and feats[0].shape[1:] == (1024, 768)
        _, pkv = flow.vlm_with_expert.forward(attention_mask=make_att_2d_masks(ppad, patt),
                                              position_ids=torch.cumsum(ppad, 1) - 1,
                                              past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
        cache, copy_pad = build_copy_cache(flow, feats, img_masks, state, ppad)
        # the state token is the LAST prefix token (no trailing padding: prefix_length=-1); its RoPE
        # position is (#valid tokens - 1), which differs from the sequence index when language pads
        assert ppad[:, -1].all() and int(ppad[1].sum()) < ppad.shape[1]
        k0, v0 = pkv.layers[0].keys, pkv.layers[0].values          # [B, H, S, D]
        assert torch.allclose(cache.layers[0].keys[:, :, -1], k0[:, :, -1], atol=1e-5)
        assert torch.allclose(cache.layers[0].values[:, :, -1], v0[:, :, -1], atol=1e-5)
        # aux key RoPE positions = prefix position of the connector token that pools the patch
        torch.nn.init.normal_(flow.bypass.w_k.weight, std=0.02)
        cache, _ = build_copy_cache(flow, feats, img_masks, state, ppad)
        ppos = torch.cumsum(ppad, 1) - 1
        bot = block_of_token(1024)
        for k in range(3):
            u = feats[k].to(flow.bypass.w_k.weight.dtype) + flow.bypass.cam_emb[k]
            key = flow.bypass.w_k(u).view(B, 1024, 5, 64)
            got = cache.layers[0].keys[:, :, k * 1024:(k + 1) * 1024].transpose(1, 2)
            want = apply_rope(key, ppos[:, k * 64 + bot]).to(got.dtype)
            assert torch.allclose(got, want, atol=1e-6)
            assert not torch.allclose(got, apply_rope(key, ppos[:, k * 64 + 1 + bot]).to(got.dtype),
                                      atol=1e-6)   # the check can fail: a shifted position differs
        # mask: padding patches and missing cameras off, state on
        assert copy_pad.shape == (B, 3 * 1024 + 1)
        assert not copy_pad[:, :256].any() and copy_pad[:, 256:1024].all() and copy_pad[:, -1].all()
    free(b)


# ==================================================================================== Coda ==
def test_coda_zero_init_and_initial_loss():
    a = make_policy()
    a.eval()
    for main_ae in ("freeze", "lr"):
        b = make_policy(ext_cfg(coda=True, main_ae=main_ae))
        batch = make_batch()
        noise, time = noise_time()
        b.eval()
        assert float((sample(a, batch, noise) - sample(b, batch, noise)).abs().max()) < 1e-6
        # initial correction loss = (A - a*)^2, in train mode too (train mode does not change A)
        b.train()
        actions = b.prepare_action(batch)
        with torch.no_grad():
            lc = b.model.correction_losses(*model_inputs(b, batch), actions, noise=noise, time=time)
        A = sample(a, batch, noise)
        assert torch.allclose(lc, (A - actions) ** 2, atol=1e-5)
        free(b)
    free(a)


def test_coda_freeze_scope_and_groups():
    f = make_policy(ext_cfg(coda=True, main_ae="freeze"))
    corr = {n for n, _ in f.named_parameters() if n.startswith("model.correction.")}
    assert corr and all(prm.requires_grad for n, prm in f.named_parameters() if n in corr)
    for n, prm in f.named_parameters():
        if n.startswith(MAIN_PREFIXES):
            assert not prm.requires_grad, n
    assert any(prm.requires_grad for n, prm in f.named_parameters() if ".vlm." in n)   # plan: VLM trains
    assert f.model.state_proj.weight.requires_grad
    g = f.get_optim_params()
    assert len(g) == 2 and "lr" not in g[0] and g[1]["lr"] == 1e-4
    assert {id(p) for p in g[1]["params"]} == {id(p) for n, p in f.named_parameters() if n in corr}
    c = make_policy(ext_cfg(coda=True, main_ae="lr"))
    g = c.get_optim_params()
    assert len(g) == 3 and g[1]["lr"] == 1e-5
    main_ids = {id(p) for n, p in c.named_parameters() if n.startswith(MAIN_PREFIXES)}
    assert {id(p) for p in g[1]["params"]} == main_ids
    free(f, c)


def test_coda_grad_isolation():
    f = make_policy(ext_cfg(coda=True, main_ae="freeze"))
    corr = f.model.correction
    batch = make_batch()
    f.train()
    loss, ld = f.forward(copy.copy(batch))
    assert {"main_loss", "correction_loss"} <= set(ld)
    loss.backward()
    assert grad_nonzero(corr.action_out_proj.weight)
    for n, prm in corr.named_parameters():
        if not n.startswith("action_out_proj."):
            assert grad_is_zero(prm), n
    for n, prm in f.named_parameters():
        if n.startswith(MAIN_PREFIXES):
            assert prm.grad is None, n
    assert grad_nonzero(f.model.state_proj.weight)       # VLM side trained through L_m
    opt = torch.optim.AdamW(f.get_optim_params(), lr=1e-4)
    opt.step(); opt.zero_grad()
    loss, _ = f.forward(copy.copy(batch))
    loss.backward()
    assert grad_nonzero(corr.lm_expert.layers[0].self_attn.k_proj.weight)   # self-attn layer
    assert grad_nonzero(corr.lm_expert.layers[1].self_attn.k_proj.weight)   # cross-attn layer
    assert grad_nonzero(corr.action_in_proj.weight)
    free(f)
    # L_c alone reaches only the correction AE; L_m alone never reaches it
    c = make_policy(ext_cfg(coda=True, main_ae="lr"))
    with torch.no_grad():
        c.model.correction.action_out_proj.weight.normal_(std=0.01)
    c.train()
    actions = c.prepare_action(batch)
    c.model.correction_losses(*model_inputs(c, batch), actions).mean().backward()
    for n, prm in c.named_parameters():
        if not n.startswith("model.correction."):
            assert prm.grad is None, n
    c.zero_grad(set_to_none=True)
    c.model.forward(*model_inputs(c, batch), actions).mean().backward()
    for n, prm in c.named_parameters():
        if n.startswith("model.correction."):
            assert prm.grad is None, n
    free(c)


def test_coda_cache_rollback_and_eval_loss():
    b = make_policy(ext_cfg(coda=True, main_ae="freeze"))
    flow = b.model
    with torch.no_grad():
        flow.correction.action_out_proj.weight.normal_(std=0.01)
    batch = make_batch()
    noise, time = noise_time()
    from smolvla_ext.flow import ext_prefill
    with torch.no_grad():
        ppad, pkv, _ = ext_prefill(flow, *model_inputs(b, batch))
        v1 = flow.denoise_step(ppad, pkv, noise, time, head=flow.correction)
        v2 = flow.denoise_step(ppad, pkv, noise, time, head=flow.correction)
    assert torch.equal(v1, v2) and float(v1.abs().max()) > 0
    assert pkv.layers[0].keys.shape[2] == ppad.shape[1]
    # eval: returned loss is L_m only, identical to the stock policy
    a = make_policy()
    a.eval(); b.eval()
    with torch.no_grad():
        la, _ = a.forward(copy.copy(batch), noise=noise, time=time)
        lb, db = b.forward(copy.copy(batch), noise=noise, time=time)
    assert torch.equal(la, lb) and "correction_loss" not in db
    free(a, b)


# ================================================================================== ROCKET ==
def test_rocket_masks():
    for hw in ((240, 320), (480, 640)):
        m = valid_token_mask(*hw)
        assert not m[:16].any() and m[16:].all()
    assert valid_token_mask(512, 512).all()


def test_rocket_pooling():
    try:
        import vggt.heads.utils  # noqa: F401
    except ImportError:
        skip("vggt not importable (needs PYTHONPATH=third_party)")
    from smolvla_ext.rocket import apply_vggt_pos_embed, pool_to_grid
    f = torch.randn(2, 2048, 37, 37)
    out = pool_to_grid(f)
    assert out.shape == (2, 64, 2048)
    pe = apply_vggt_pos_embed(f)
    assert torch.allclose(out[:, 0], pe[:, :, 0, 0], atol=1e-5)
    assert torch.allclose(out[:, 63], pe[:, :, 36, 36], atol=1e-5)
    assert torch.allclose(out[:, 7], pe[:, :, 0, 36], atol=1e-5)


def test_rocket_alpha0_equivalence_and_capture():
    a = make_policy()
    b = make_policy(ext_cfg(rocket=True, rocket_alpha=0.0))
    batch = make_batch(targets=True)
    noise, time = noise_time()
    a.train(); b.train()
    # capture position: input of text layer k (pre-hook on its input_layernorm), first 192 tokens
    seen = {}
    layers = b.model.vlm_with_expert.get_vlm_model().text_model.layers
    hooks = [layers[k].input_layernorm.register_forward_pre_hook(
        lambda m, args, k=k: seen.__setitem__(k, args[0].detach().clone())) for k, _ in PAIRS]
    caught = []
    orig_align = sys.modules["smolvla_ext.attach"].align_loss

    def spy(proj, hiddens, *rest):
        caught.extend(h.detach().clone() for h in hiddens)
        return orig_align(proj, hiddens, *rest)

    sys.modules["smolvla_ext.attach"].align_loss = spy
    try:
        la, _ = a.forward(copy.copy(batch), noise=noise, time=time)
        lb, db = b.forward(copy.copy(batch), noise=noise, time=time)
    finally:
        sys.modules["smolvla_ext.attach"].align_loss = orig_align
        for h in hooks:
            h.remove()
    assert float((la - lb).abs().max()) < 1e-6 and abs(db["main_loss"] - float(la)) < 1e-6
    assert 0.0 < db["align_loss"] < 2.0
    assert len(caught) == len(PAIRS)
    for (k, _), h in zip(PAIRS, caught):
        assert h.shape == (B, 192, 960)
        assert torch.allclose(h, seen[k][:, :192])
    a.eval(); b.eval()
    assert float((sample(a, batch, noise) - sample(b, batch, noise)).abs().max()) < 1e-6
    free(a, b)


def test_rocket_grad_scope_and_groups():
    b = make_policy(ext_cfg(rocket=True))
    flow = b.model
    batch = make_batch(targets=True)
    b.train()
    images, img_masks, lt, lm, state = model_inputs(b, batch)
    flow._ext_capture = []
    flow.forward(images, img_masks, lt, lm, state, b.prepare_action(batch))
    hiddens, flow._ext_capture = flow._ext_capture, None
    per_sample, _ = align_loss(flow.align_proj, hiddens, batch[VGGT_KEY], flow.align_proj.valid, img_masks)
    per_sample.mean().backward()
    assert grad_nonzero(flow.align_proj.fc1.weight)
    tl = flow.vlm_with_expert.get_vlm_model().text_model.layers
    assert grad_nonzero(tl[N_LAYERS - 2].mlp.down_proj.weight)
    assert not any(prm.requires_grad for prm in tl[N_LAYERS - 1].parameters())
    for prm in list(flow.vlm_with_expert.lm_expert.parameters()) + list(flow.state_proj.parameters()):
        assert grad_is_zero(prm)
    g = b.get_optim_params()
    assert len(g) == 3 and "lr" not in g[1] and g[2]["lr"] == 1e-4
    assert {id(p) for p in g[2]["params"]} == {id(p) for p in flow.align_proj.parameters()}
    free(b)


def _write_targets(d, n=16, cams=CAMS, layers=(4, 11, 17, 23), dim=8):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    t = np.lib.format.open_memmap(d / "targets.npy", mode="w+", dtype=np.float16,
                                  shape=(n, len(layers), len(cams), 64, dim))
    for i in range(n):
        t[i] = i
    t.flush()
    np.save(d / "filled.npy", np.ones(n, dtype=bool))
    (d / "meta.json").write_text(json.dumps({"camera_keys": list(cams), "dim": dim,
                                             "vggt_layers": list(layers)}))
    return d


class _FakeDS:
    def __init__(self, n):
        self.n = n
        self.meta = "meta-obj"

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"index": torch.tensor(i), "x": torch.zeros(1)}


def test_teacher_reading(tmp_path=None):
    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    d = _write_targets(tmp_path / "t")
    tt = TeacherTargets(d, CAMS, [4, 11, 17, 23], 8)
    out = tt(torch.tensor([5, 2, 9]))
    assert out.shape == (3, 4, 3, 64, 8) and out[:, 0, 0, 0, 0].tolist() == [5.0, 2.0, 9.0]
    for bad in (dict(camera_keys=CAMS[::-1], vggt_layers=[4, 11, 17, 23], dim=8),
                dict(camera_keys=CAMS, vggt_layers=[23, 23, 23, 23], dim=8),
                dict(camera_keys=CAMS, vggt_layers=[4, 11, 17, 23], dim=2048)):
        try:
            TeacherTargets(d, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"no ValueError for {bad}")
    ds = TeacherDataset(_FakeDS(16), d, CAMS, [4, 11, 17, 23], 8)
    ds2 = pickle.loads(pickle.dumps(ds))                 # what a spawn worker receives
    item = ds2[6]
    assert item[VGGT_KEY].dtype == torch.float16 and float(item[VGGT_KEY].max()) == 6.0
    assert len(ds2) == 16 and ds2.meta == "meta-obj"     # attribute delegation
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=1,
                                         multiprocessing_context="spawn", sampler=[3, 1, 4, 15])
    b = next(iter(loader))
    assert b[VGGT_KEY].shape == (4, 4, 3, 64, 8)
    assert b[VGGT_KEY][:, 0, 0, 0, 0].tolist() == [3.0, 1.0, 4.0, 15.0]


def test_teacher_int8_copy(tmp_path=None):
    """quantize_targets: padding cells dropped (zeros on read), valid cells within int8 error,
    TeacherDataset / TeacherTargets read the int8 copy in the same float16 [L, C, 64, D] layout."""
    if not (REPO / "data/lerobot_v3_d9_0912/meta/info.json").is_file():
        skip("needs data/lerobot_v3_d9_0912/meta/info.json for the camera size")
    from smolvla_ext.quantize_targets import main as q8
    from smolvla_ext.rocket import TargetFile
    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    d = _write_targets(tmp_path / "f16", n=10)
    t = np.load(d / "targets.npy", mmap_mode="r+")
    t[:] = np.random.default_rng(0).standard_normal(t.shape).astype(np.float16)
    t.flush()
    m = json.loads((d / "meta.json").read_text())
    m.update(root=str(REPO / "data/lerobot_v3_d9_0912"), canvas=512, grid=8, complete=True, num_frames=10)
    (d / "meta.json").write_text(json.dumps(m))
    q8(["--src", str(d), "--out", str(tmp_path / "q8"), "--chunk", "4", "--device", "cpu"])
    a, b = TargetFile(d), TargetFile(tmp_path / "q8")
    v = valid_token_mask(240, 320).numpy()
    x, y = a.get(6).astype(np.float32), b.get(6).astype(np.float32)
    assert y.shape == x.shape and np.abs(y[:, :, ~v]).max() == 0
    assert np.abs(x[:, :, v] - y[:, :, v]).max() <= np.abs(x).max() / 127 * 0.51 + 1e-2
    ds = TeacherDataset(_FakeDS(10), tmp_path / "q8", CAMS, [4, 11, 17, 23], 8)
    item = pickle.loads(pickle.dumps(ds))[6]
    assert item[VGGT_KEY].dtype == torch.float16 and np.allclose(item[VGGT_KEY].numpy(), b.get(6))
    tt = TeacherTargets(tmp_path / "q8", CAMS, [4, 11, 17, 23], 8)
    assert tuple(tt(torch.tensor([1, 6])).shape) == (2, 4, 3, 64, 8)
    qm = json.loads((tmp_path / "q8" / "meta.json").read_text())["quantization"]
    assert qm["cos_min"] > 0.999


def test_teacher_key_survives_real_preprocessor():
    """lerobot's SmolVLA preprocessor drops unknown top-level keys; CarryKeysPre keeps ours."""
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    cfg = make_cfg()
    stats = {"observation.state": {"mean": torch.zeros(6), "std": torch.ones(6)},
             "action": {"mean": torch.zeros(6), "std": torch.ones(6)}}
    pre, _ = make_smolvla_pre_post_processors(cfg, dataset_stats=stats)

    def raw():
        b = make_batch(targets=True)
        del b["observation.language.tokens"], b["observation.language.attention_mask"]
        b["task"] = ["pick the tape\n", "place it"]
        return b

    out = pre(raw())
    assert VGGT_KEY not in out                            # dropped by batch_to_transition
    wrapped = CarryKeysPre(pre)
    rb = raw()
    t = rb[VGGT_KEY].clone()
    out = wrapped(rb)
    assert torch.equal(out[VGGT_KEY], t) and out[VGGT_KEY].dtype == torch.float16
    assert "observation.language.tokens" in out and wrapped.carried == 1


# ============================================================================ save / load ==
def _one_step(p, batch):
    p.train()
    opt = torch.optim.AdamW(p.get_optim_params(), lr=1e-3)
    loss, _ = p.forward(copy.copy(batch))
    loss.backward()
    opt.step()
    opt.zero_grad()


def test_save_load_roundtrip(tmp_path=None):
    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    from safetensors.torch import load_file, save_file
    for name, kw in (("PCL", dict(pvi=True, coda=True, main_ae="lr")),
                     ("RC", dict(rocket=True, coda=True))):
        p = make_policy(ext_cfg(**kw))
        batch = make_batch(targets=True)
        _one_step(p, batch)
        d = tmp_path / name
        p.save_pretrained(d)
        assert (d / SIDECAR_NAME).is_file()
        side = json.loads((d / SIDECAR_NAME).read_text())
        assert side["camera_hw"] == HW and side["main_num_steps"] == 5
        sd_file = load_file(str(d / "model.safetensors"))
        ext_keys = [k for k in p.state_dict() if k.startswith(("model.bypass.", "model.correction.",
                                                                 "model.align_proj."))]
        assert ext_keys and all(k in sd_file for k in ext_keys)
        q = load_policy(d, device="cpu", verbose=False)
        assert q._smolvla_ext.enabled == p._smolvla_ext.enabled
        for k in ext_keys:
            assert torch.equal(q.state_dict()[k], p.state_dict()[k]), k
        noise, _ = noise_time()
        p.eval()
        assert torch.equal(sample(p, batch, noise), sample(q, batch, noise))
        # one extension tensor missing -> strict load refuses
        del sd_file[ext_keys[0]]
        save_file(sd_file, str(d / "model.safetensors"))
        try:
            load_policy(d, device="cpu", verbose=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("load_policy accepted a checkpoint with a missing extension tensor")
        free(p, q, sd_file)
    # stock checkpoint + attach(pretrained_path=...) -> copies of the main AE, zero output, no load
    s = make_policy()
    with torch.no_grad():
        s.model.action_in_proj.weight.add_(0.1)          # make "loaded" weights differ from init
    s.save_pretrained(tmp_path / "stock")
    assert not (tmp_path / "stock" / SIDECAR_NAME).exists()
    t = SmolVLAPolicy.from_pretrained(str(tmp_path / "stock"), config=make_cfg())
    attach(t, ext_cfg(coda=True, pvi=True), pretrained_path=tmp_path / "stock", camera_hw=HW, verbose=False)
    flow = t.model
    assert torch.equal(flow.correction.action_in_proj.weight, flow.action_in_proj.weight)
    assert torch.equal(flow.correction.action_in_proj.weight, s.model.action_in_proj.weight)
    assert float(flow.correction.action_out_proj.weight.abs().max()) == 0.0
    for (n1, p1), (n2, p2) in zip(flow.bypass.copy_expert.state_dict().items(),
                                  flow.vlm_with_expert.lm_expert.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)
    assert float(flow.bypass.z[0].weight.abs().max()) == 0.0 and float(flow.bypass.cam_emb.abs().max()) == 0.0
    # resume: attaching to an extension checkpoint loads its extension tensors instead of copying
    p = make_policy(ext_cfg(coda=True, main_ae="freeze"))
    _one_step(p, make_batch())
    p.save_pretrained(tmp_path / "cf")
    r = SmolVLAPolicy.from_pretrained(str(tmp_path / "cf"), config=make_cfg())
    attach(r, ext_cfg(coda=True, main_ae="freeze"), pretrained_path=tmp_path / "cf", camera_hw=HW,
           verbose=False)
    assert torch.equal(r.model.correction.action_out_proj.weight, p.model.correction.action_out_proj.weight)
    assert float(r.model.correction.action_out_proj.weight.abs().max()) > 0
    free(s, t, p, r)


# ================================================================================== combos ==
def test_pvi_coda_combo():
    a = make_policy()
    a.eval()
    b = make_policy(ext_cfg(pvi=True, coda=True, main_ae="freeze"))
    batch = make_batch()
    noise, time = noise_time()
    b.eval()
    assert float((sample(a, batch, noise) - sample(b, batch, noise)).abs().max()) < 1e-6
    g = b.get_optim_params()
    assert len(g) == 2 and g[1]["lr"] == 1e-4
    new_ids = {id(p) for n, p in b.named_parameters()
               if n.startswith(("model.bypass.", "model.correction.")) and p.requires_grad}
    assert {id(p) for p in g[1]["params"]} == new_ids
    # make PVI active; the candidate must be the main AE WITH PVI injections
    flow = b.model
    with torch.no_grad():
        for z in flow.bypass.z:
            z.weight.normal_(std=0.05)
        flow.bypass.w_v.weight.normal_(std=0.05)
    flow._ext.coda = False
    A_pvi = sample(b, batch, noise)                       # main AE + PVI only
    flow._ext.coda = True
    assert float((A_pvi - sample(a, batch, noise)).abs().max()) > 1e-4   # PVI changed the output
    actions = b.prepare_action(batch)
    with torch.no_grad():
        lc = flow.correction_losses(*model_inputs(b, batch), actions, noise=noise, time=time)
    assert torch.allclose(lc, (A_pvi - actions) ** 2, atol=1e-5)
    # the correction AE never runs the copy expert (no PVI injection into it)
    calls = []
    h = flow.bypass.copy_expert.layers[0].self_attn.q_proj.register_forward_hook(lambda *a_: calls.append(1))
    from smolvla_ext.flow import ext_prefill
    with torch.no_grad():
        ppad, pkv, cp = ext_prefill(flow, *model_inputs(b, batch))
        n0 = len(calls)
        flow.denoise_step(ppad, pkv, noise, time, head=flow.correction)
        assert len(calls) == n0
        flow.denoise_step(ppad, pkv, noise, time, copy_cache=cp)
        assert len(calls) == n0 + 1
    h.remove()
    # training loss = L_m + L_c, gradients reach bypass and correction, not the frozen main AE
    b.train()
    loss, ld = b.forward(copy.copy(batch), noise=noise, time=time)
    assert abs(float(loss) - (ld["main_loss"] + ld["correction_loss"])) < 1e-5
    loss.backward()
    assert grad_nonzero(flow.bypass.z[0].weight) and grad_nonzero(flow.correction.action_out_proj.weight)
    assert all(p.grad is None for n, p in b.named_parameters() if n.startswith(MAIN_PREFIXES))
    free(a, b)


def test_rocket_coda_combo():
    a = make_policy()
    a.eval()
    b = make_policy(ext_cfg(rocket=True, coda=True))
    batch = make_batch(targets=True)
    noise, time = noise_time()
    b.eval()
    assert float((sample(a, batch, noise) - sample(b, batch, noise)).abs().max()) < 1e-6
    g = b.get_optim_params()
    assert len(g) == 3 and "lr" not in g[1] and g[2]["lr"] == 1e-4
    new_ids = {id(p) for n, p in b.named_parameters() if n.startswith(("model.align_proj.", "model.correction."))}
    assert {id(p) for p in g[2]["params"]} == new_ids
    b.train()
    loss, ld = b.forward(copy.copy(batch), noise=noise, time=time)
    want = ld["main_loss"] + ld["correction_loss"] + 0.125 * ld["align_loss"]
    assert abs(float(loss) - want) < 1e-5
    loss.backward()
    assert grad_nonzero(b.model.align_proj.fc1.weight) and grad_nonzero(b.model.correction.action_out_proj.weight)
    b.eval()
    with torch.no_grad():
        le, de = b.forward(copy.copy(batch), noise=noise, time=time)   # eval: L_m only, no targets needed
        la, _ = a.forward(copy.copy(batch), noise=noise, time=time)
    assert torch.equal(le, la) and "align_loss" not in de
    free(a, b)


def test_rocket_pvi_combo():
    """R+P (and R+P+C): zero-init equivalence, ROCKET capture unchanged by PVI, losses add up,
    gradients reach bypass + projector, frozen main AE gets none, groups partition the params."""
    for kw in (dict(rocket=True, pvi=True, main_ae="freeze"),
               dict(rocket=True, pvi=True, coda=True, main_ae="lr")):
        a = make_policy()
        a.eval()
        b = make_policy(ext_cfg(**kw))
        batch = make_batch(targets=True)
        noise, time = noise_time()
        b.eval()
        assert float((sample(a, batch, noise) - sample(b, batch, noise)).abs().max()) < 1e-6, kw
        g = b.get_optim_params()
        n_groups = 2 if kw["main_ae"] == "freeze" else 3
        assert len(g) == n_groups and g[-1]["lr"] == 1e-4
        if kw["main_ae"] == "lr":
            assert g[1]["lr"] == 1e-5
        pre = tuple(p for m, p in (("pvi", "model.bypass."), ("coda", "model.correction."),
                                   ("rocket", "model.align_proj.")) if kw.get(m))
        new_ids = {id(p) for n, p in b.named_parameters() if n.startswith(pre) and p.requires_grad}
        assert {id(p) for p in g[-1]["params"]} == new_ids
        # PVI active: ROCKET capture (VLM prefix) must not depend on the expert-side injection
        flow = b.model
        with torch.no_grad():
            for z in flow.bypass.z:
                z.weight.normal_(std=0.05)
            flow.bypass.w_v.weight.normal_(std=0.05)
        b.train()
        loss, ld = b.forward(copy.copy(batch), noise=noise, time=time)
        want = ld["main_loss"] + ld.get("correction_loss", 0.0) + 0.125 * ld["align_loss"]
        assert abs(float(loss) - want) < 1e-5, kw
        loss.backward()
        assert grad_nonzero(flow.align_proj.fc1.weight) and grad_nonzero(flow.bypass.z[0].weight)
        if kw["main_ae"] == "freeze":
            assert all(p.grad is None for n, p in b.named_parameters() if n.startswith(MAIN_PREFIXES))
        else:
            assert grad_nonzero(flow.action_out_proj.weight)
        c = make_policy(ext_cfg(rocket=True))           # same base weights, ROCKET only
        # the projector's random init draws from the RNG after the PVI modules in b -> copy it
        c.model.align_proj.load_state_dict(flow.align_proj.state_dict())
        c.train()
        _, lc = c.forward(copy.copy(batch), noise=noise, time=time)
        assert abs(lc["align_loss"] - ld["align_loss"]) < 1e-5   # same VLM, same capture
        free(a, b, c)


def test_rocket_early_layer_capture():
    """'working layers' variant: capture the INPUT of layers 1..k (= output of 0..k-1)."""
    pairs = [[1, 4], [2, 11], [3, 17]]                     # 4-layer test model: inputs of 1, 2, 3
    b = make_policy(ext_cfg(rocket=True, rocket_pairs=pairs))
    batch = make_batch(targets=True, n_layers=len(pairs))
    noise, time = noise_time()
    seen = {}
    layers = b.model.vlm_with_expert.get_vlm_model().text_model.layers
    hooks = [layers[k].input_layernorm.register_forward_pre_hook(
        lambda m, args, k=k: seen.__setitem__(k, args[0].detach().clone())) for k, _ in pairs]
    caught = []
    orig_align = sys.modules["smolvla_ext.attach"].align_loss

    def spy(proj, hiddens, *rest):
        caught.extend(h.detach().clone() for h in hiddens)
        return orig_align(proj, hiddens, *rest)

    sys.modules["smolvla_ext.attach"].align_loss = spy
    b.train()
    try:
        loss, ld = b.forward(copy.copy(batch), noise=noise, time=time)
    finally:
        sys.modules["smolvla_ext.attach"].align_loss = orig_align
        for h in hooks:
            h.remove()
    assert len(caught) == len(pairs)
    for (k, _), h in zip(pairs, caught):
        assert torch.allclose(h, seen[k][:, :192])
    loss.backward()
    assert grad_nonzero(layers[0].mlp.down_proj.weight)  # output of layer 0 is aligned -> layer 0 learns
    free(b)


def test_coda_10_steps_equivalence():
    """Coda 10+10 at zero init == stock 10-step sampling; candidate uses 10 main steps."""
    a = make_policy(num_steps=10)
    b = make_policy(ext_cfg(coda=True, coda_num_steps=10, main_ae="freeze"), num_steps=10)
    a.eval(); b.eval()
    batch = make_batch()
    noise, time = noise_time()
    A = sample(a, batch, noise)
    assert float((A - sample(b, batch, noise)).abs().max()) < 1e-6
    a5 = make_policy(num_steps=5)
    a5.eval()
    assert float((A - sample(a5, batch, noise)).abs().max()) > 1e-4   # 10 vs 5 steps really differ
    actions = b.prepare_action(batch)
    with torch.no_grad():
        lc = b.model.correction_losses(*model_inputs(b, batch), actions, noise=noise, time=time)
    assert torch.allclose(lc, (A - actions) ** 2, atol=1e-5)
    free(a, b, a5)


# ============================================================================ Coda + RTC ==
# ExtConfig.coda_rtc (2026-10-17): where the RTC guidance acts on a Coda checkpoint. Inference
# only. Pull per Euler step = max_guidance_weight / num_steps, so max_w=2 keeps every step a
# contraction here; the deployed w=10 on a 5-step checkpoint pulls 2.0 (overshoot) and is swept
# on the GPU instead (plan: results/rtc_coda).
def test_coda_rtc_refusals():
    """Default off; bad value refused; both refusal sites fire; a placement lets it through."""
    assert ExtConfig(coda=True).coda_rtc == "off"
    assert ExtConfig.from_dict({"pvi": True, "coda": True, "coda_num_steps": 5}).coda_rtc == "off"
    try:
        ExtConfig(coda=True, coda_rtc="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("coda_rtc=bogus accepted")
    # attach-time refusal (attach.py _check_config): rtc_config already enabled
    set_seed(0)
    cfg = make_cfg()
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    cfg.rtc_config = RTCConfig(enabled=True, execution_horizon=2,
                               prefix_attention_schedule=RTCAttentionSchedule("LINEAR"))
    pol = SmolVLAPolicy(cfg)
    try:
        attach(pol, ext_cfg(coda=True), camera_hw=HW, verbose=False)
    except ValueError as e:
        assert "coda_rtc" in str(e)
    else:
        raise AssertionError("attach accepted Coda with RTC enabled and coda_rtc=off")
    attach(pol, ext_cfg(coda=True, coda_rtc="main"), camera_hw=HW, verbose=False)   # allowed
    free(pol)
    # predict-time refusal (flow.sample_actions)
    b = make_policy(ext_cfg(coda=True, main_ae="freeze"))
    b.eval()
    rtc_on(b)
    batch, (noise, _) = make_batch(), noise_time()
    try:
        sample(b, batch, noise)
    except RuntimeError as e:
        assert "coda_rtc" in str(e)
    else:
        raise AssertionError("sample_actions ran Coda + RTC with coda_rtc=off")
    b.model._ext.coda_rtc = "main"
    sample(b, batch, noise)                                                        # allowed
    free(b)


def test_coda_rtc_none_prefix_bitwise():
    """With no previous chunk the option cannot change anything: RTCProcessor returns v_t
    untouched when prev_chunk_left_over is None (modeling_rtc.py:167-170). Bitwise."""
    b = make_policy(ext_cfg(coda=True, main_ae="freeze"))
    with torch.no_grad():
        b.model.correction.action_out_proj.weight.normal_(std=0.01)
    b.eval()
    batch, (noise, _) = make_batch(), noise_time()
    ref = sample(b, batch, noise)                      # RTC off
    delay, horizon = rtc_on(b)
    for mode in ("main", "correction", "both"):
        b.model._ext.coda_rtc = mode
        assert torch.equal(ref, sample(b, batch, noise)), mode
        assert torch.equal(ref, sample(b, batch, noise, prev_chunk_left_over=None,
                                       inference_delay=delay, execution_horizon=horizon)), mode
    free(b)


def test_coda_rtc_placement_reaches_the_right_integrate():
    """The guidance must wrap exactly the stage the placement names. coda_num_steps=3 vs
    num_steps=5 so the two integrates cannot be confused."""
    b = make_policy(ext_cfg(coda=True, coda_num_steps=3, main_ae="freeze"))
    b.eval()
    flow = b.model
    batch, (noise, _) = make_batch(), noise_time()
    delay, horizon = rtc_on(b, max_w=2.0)
    prev = prev_tail()
    log = []
    proc = flow.rtc_processor
    orig_proc, orig_step = proc.denoise_step, flow.denoise_step

    def spy_proc(*a, **kw):
        log.append(("guided", None))
        return orig_proc(*a, **kw)

    def spy_step(_self, *a, **kw):
        log.append(("step", kw.get("head") is flow.correction))
        return orig_step(*a, **kw)

    try:
        proc.denoise_step = spy_proc
        flow.denoise_step = types.MethodType(spy_step, flow)
        for mode, n_guided, heads in (("main", 5, [False] * 5),
                                      ("correction", 3, [True] * 3),
                                      ("both", 8, [False] * 5 + [True] * 3)):
            flow._ext.coda_rtc = mode
            log.clear()
            sample(b, batch, noise, prev_chunk_left_over=prev,
                   inference_delay=delay, execution_horizon=horizon)
            steps = [e for e in log if e[0] == "step"]
            guided = [log[i + 1][1] for i, e in enumerate(log)
                      if e[0] == "guided" and i + 1 < len(log) and log[i + 1][0] == "step"]
            assert len(steps) == 8, (mode, len(steps))          # 5 main + 3 correction, always
            assert len(guided) == n_guided and guided == heads, (mode, guided)
        # RTC off: no guided call at all
        flow._ext.coda_rtc = "off"
        b.config.rtc_config = None
        b.init_rtc_processor()
        flow.rtc_processor = None
        log.clear()
        sample(b, batch, noise)
        assert not [e for e in log if e[0] == "guided"]
    finally:
        proc.denoise_step = orig_proc
        flow.denoise_step = orig_step
    free(b)


def test_coda_rtc_pulls_step0_toward_prev():
    """correction / both pull the EXECUTED output's first step toward the previous plan."""
    b = make_policy(ext_cfg(coda=True, coda_num_steps=3, main_ae="freeze"))
    with torch.no_grad():
        b.model.correction.action_out_proj.weight.normal_(std=0.01)
    b.eval()
    flow = b.model
    batch, (noise, _) = make_batch(), noise_time()
    prev = prev_tail()
    base = sample(b, batch, noise)                               # unguided Coda output
    d0 = float((base[:, 0, :6] - prev[:, 0]).abs().mean())
    delay, horizon = rtc_on(b, max_w=2.0)
    kw = dict(prev_chunk_left_over=prev, inference_delay=delay, execution_horizon=horizon)
    for mode in ("correction", "both"):
        flow._ext.coda_rtc = mode
        out = sample(b, batch, noise, **kw)
        d = float((out[:, 0, :6] - prev[:, 0]).abs().mean())
        assert d < d0, (mode, d, d0)
    # main: stage 1 is pulled; how far the correction then moves step 0 is the quantity the GPU
    # measurement calls D -- printed, not asserted (random correction weights have no sign).
    flow._ext.coda_rtc = "main"
    out_main = sample(b, batch, noise, **kw)
    flow._ext.coda = False
    try:
        stage1 = sample(b, batch, noise, **kw)
    finally:
        flow._ext.coda = True
    assert float((stage1[:, 0, :6] - prev[:, 0]).abs().mean()) < d0
    print(f"    [coda_rtc] D (correction moves step 0) = "
          f"{float((out_main[:, 0, :6] - stage1[:, 0, :6]).abs().mean()):.4f} normalized units")
    free(b)


def test_coda_rtc_cache_and_grad_hygiene():
    """The extra autograd pass must not leak a graph, touch .grad, or disturb the prefix K/V."""
    b = make_policy(ext_cfg(coda=True, coda_num_steps=3, main_ae="lr"))
    with torch.no_grad():
        b.model.correction.action_out_proj.weight.normal_(std=0.01)
    b.eval()
    flow = b.model
    flow._ext.coda_rtc = "both"
    batch, (noise, time) = make_batch(), noise_time()
    delay, horizon = rtc_on(b, max_w=2.0)
    prev = prev_tail()
    kw = dict(prev_chunk_left_over=prev, inference_delay=delay, execution_horizon=horizon)
    out = sample(b, batch, noise, **kw)
    assert out.grad_fn is None
    assert all(p.grad is None for p in b.parameters())
    assert torch.equal(out, sample(b, batch, noise, **kw))        # no state carried between calls
    # cache still cropped back to the prefix after a guided correction step (cf. :434)
    from smolvla_ext.flow import ext_prefill
    with torch.no_grad():
        ppad, pkv, _ = ext_prefill(flow, *model_inputs(b, batch))
    flow.rtc_processor.denoise_step(
        x_t=noise, prev_chunk_left_over=prev, inference_delay=delay, time=0.6,
        original_denoise_step_partial=lambda x: flow.denoise_step(ppad, pkv, x, time,
                                                                  head=flow.correction),
        execution_horizon=horizon)
    assert pkv.layers[0].keys.shape[2] == ppad.shape[1]
    assert all(p.grad is None for p in b.parameters())
    # a stray grad-enabled caller must not explode either
    out2 = flow.sample_actions(*model_inputs(b, batch), noise=noise, **kw)
    assert torch.isfinite(out2).all()
    b.zero_grad(set_to_none=True)
    free(b)


def test_coda_rtc_pads_unused_channels_toward_zero():
    """KNOWN QUIRK, locked in on purpose: the previous plan is 6-wide while x_t is 32-wide, so
    modeling_rtc.py:193-197 zero-pads it and the guidance also drags the 26 unused channels
    toward 0. Pre-existing for RTC (not a Coda regression). If a lerobot bump changes the
    padding, this test fails here instead of silently in deployment."""
    b = make_policy(ext_cfg(coda=True, coda_num_steps=3, main_ae="freeze"))
    with torch.no_grad():
        b.model.correction.action_out_proj.weight.normal_(std=0.01)
    b.eval()
    batch, (noise, _) = make_batch(), noise_time()
    prev = prev_tail()
    base = sample(b, batch, noise)
    delay, horizon = rtc_on(b, max_w=2.0)
    b.model._ext.coda_rtc = "correction"
    out = sample(b, batch, noise, prev_chunk_left_over=prev,
                 inference_delay=delay, execution_horizon=horizon)
    assert float(out[:, 0, 6:].abs().mean()) < float(base[:, 0, 6:].abs().mean())
    free(b)


def test_coda_rtc_sidecar_roundtrip(tmp_path=None):
    """The key round-trips, but a saved placement is NEVER honoured unless the caller asks."""
    tmp_path = Path(tmp_path or tempfile.mkdtemp())
    p = make_policy(ext_cfg(coda=True, main_ae="lr", coda_rtc="both"))
    d = tmp_path / "coda_rtc"
    p.save_pretrained(d)
    side = json.loads((d / SIDECAR_NAME).read_text())
    assert side["coda_rtc"] == "both"
    assert load_policy(d, device="cpu", verbose=False)._smolvla_ext.coda_rtc == "off"
    assert load_policy(d, device="cpu", verbose=False,
                       coda_rtc="correction")._smolvla_ext.coda_rtc == "correction"
    side.pop("coda_rtc")
    assert ExtConfig.from_dict(side).coda_rtc == "off"           # the 87 existing sidecars
    try:
        ExtConfig.from_dict({**side, "nope": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("from_dict accepted an unknown key")
    assert "coda_rtc=both" in ExtConfig(coda=True, coda_rtc="both").summary()
    assert "coda_rtc" not in ExtConfig(coda=True).summary()      # pre-2026-10-17 log line intact
    free(p)


def test_sdpa_matches_eager():
    """attn_impl=sdpa (fused masked attention) ~= eager: sampling, training losses and gradients,
    incl. padded language tokens (query rows with no allowed key). Not bitwise: the VLM / expert
    weights are bf16 and eager rounds probs @ v in bf16 where SDPA does not (an fp32 SDPA differs by
    the same ~7e-3 after 4 layers x 5 steps, measured 2026-10-11). The decisive check is the
    real-checkpoint comparison in ticks against the draw spread (diag)."""
    for kw in (dict(), dict(pvi=True, coda=True, rocket=True, main_ae="lr")):
        a = make_policy(ext_cfg(**kw))
        b = make_policy(ext_cfg(attn_impl="sdpa", **kw))
        b.load_state_dict(a.state_dict())                # identical weights (RNG draws differ per module order)
        batch = make_batch(targets=True)
        noise, time = noise_time()
        a.eval(); b.eval()
        assert float((sample(a, batch, noise) - sample(b, batch, noise)).abs().max()) < 3e-2, kw
        a.train(); b.train()
        la, da = a.forward(copy.copy(batch), noise=noise, time=time)
        lb, db = b.forward(copy.copy(batch), noise=noise, time=time)
        assert torch.isfinite(lb) and abs(float(la) - float(lb)) < 1e-2 * max(1.0, float(la)), (kw, float(la), float(lb))
        la.backward(); lb.backward()
        ga = a.model.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.grad
        gb = b.model.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.grad
        cos = float(torch.nn.functional.cosine_similarity(ga.flatten().float(), gb.flatten().float(), dim=0))
        assert torch.isfinite(gb).all() and cos > 0.99, cos
        free(a, b)


# ======================================================================= standalone runner ==
if __name__ == "__main__":
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: 0)
             if n.startswith("test_") and callable(f) and sel in n]
    tests.sort(key=lambda nf: nf[1].__code__.co_firstlineno)
    passed = failed = skipped = 0
    for n, f in tests:
        kwargs = {"tmp_path": Path(tempfile.mkdtemp(prefix=n + "_"))} \
            if "tmp_path" in inspect.signature(f).parameters else {}
        try:
            f(**kwargs)
            passed += 1
            print(f"PASS {n}", flush=True)
        except _Skip as e:
            skipped += 1
            print(f"SKIP {n}: {e}", flush=True)
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}\n{traceback.format_exc()}", flush=True)
        finally:
            if kwargs:
                shutil.rmtree(kwargs["tmp_path"], ignore_errors=True)  # saved checkpoints ~1 GB
        gc.collect()
    print(f"== {passed} passed, {failed} failed, {skipped} skipped ==")
    sys.exit(1 if failed else 0)
