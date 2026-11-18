"""PVI: a copy of the action expert that reads SigLIP features directly (SmolVLA_PVI_구현.md §2-4).

Differences from the spec (all forced by "lerobot/ is not edited"):
* SigLIP `last_hidden_state` is taken with a forward PRE-hook on the connector instead of a new
  `embed_image(return_pre_connector=True)` argument. Same tensor, no extra compute.
* The copy-expert cache is `PrefixKVCache` (one K/V tensor shared by all 16 layers) instead of a
  `DynamicCache` with 16 copies of the same [B, 5, 3073, 64] tensors. The spec allows a cache
  substitute with this interface (§4-5 end); it only saves memory.
* Padding masks come from the RAW camera (H, W) (ExtConfig.camera_hw), not config.image_features.
* Only `bypass_source="siglip"` is implemented (the connector control variant §4-8 is not needed).
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


def block_of_token(n: int) -> torch.Tensor:
    """Feature-token index p -> index (0..63) of the connector token that pools it.

    pixel_shuffle (scale 4) pools a 4x4 block of the 32x32 SigLIP grid into token
    (r // 4) * 8 + (c // 4). For n == 64 this is the identity.
    """
    side = int(round(math.sqrt(n)))
    assert side * side == n and side % 8 == 0, n
    s = side // 8
    p = torch.arange(n)
    r, c = p // side, p % side
    return (r // s) * 8 + (c // s)


def patch_valid_mask(h, w, target_w, target_h, patch=16):
    """[target_h/patch * target_w/patch] bool: False for SigLIP patches that are pure padding
    when an h x w image is resize_with_pad-ed (left/top padding) into target_w x target_h."""
    ratio = max(w / target_w, h / target_h)
    rh, rw = int(h / ratio), int(w / ratio)
    top, left = (target_h - rh) // patch, (target_w - rw) // patch  # rows / cols that are ALL padding
    m = torch.zeros(target_h // patch, target_w // patch, dtype=torch.bool)
    m[top:, left:] = True
    return m.flatten()


class _Layer:
    __slots__ = ("keys", "values")

    def __init__(self, k, v):
        self.keys, self.values = k, v


class PrefixKVCache:
    """Minimal stand-in for `DynamicCache` holding the SAME prefix K/V for every layer.

    Interface used by SmolVLMWithExpertModel: `update(k, v, layer_idx)` (append along seq and
    return the concatenation), `layers[i].keys/.values`, `crop(n)`, truthiness (len)."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, num_layers: int):
        self.layers = [_Layer(keys, values) for _ in range(num_layers)]

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        lay = self.layers[layer_idx]
        lay.keys = torch.cat([lay.keys, key_states], dim=-2)
        lay.values = torch.cat([lay.values, value_states], dim=-2)
        return lay.keys, lay.values

    def crop(self, max_length: int):
        for lay in self.layers:
            lay.keys = lay.keys[..., :max_length, :]
            lay.values = lay.values[..., :max_length, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].keys.shape[-2]

    def __len__(self):
        return len(self.layers)


class BypassBranch(nn.Module):
    """copy_expert (copy of lm_expert) + w_k, w_v, cam_emb, z_i. Everything but the copy starts at 0."""

    def __init__(self, lm_expert, n_cams, feat_dim, kv_dim, expert_dim, num_layers, valid_masks):
        super().__init__()
        self.copy_expert = copy.deepcopy(lm_expert)
        for p in self.copy_expert.parameters():
            p.requires_grad_(True)
        # The last norm's output is never used (z_i reads the pre-norm stream): freeze it so it is
        # not an unused trainable parameter (spec §3 correction).
        self.copy_expert.norm.weight.requires_grad_(False)
        self.cam_emb = nn.Parameter(torch.zeros(n_cams, feat_dim))
        self.w_k = nn.Linear(feat_dim, kv_dim, bias=False)
        self.w_v = nn.Linear(feat_dim, kv_dim, bias=False)
        self.z = nn.ModuleList(nn.Linear(expert_dim, expert_dim) for _ in range(num_layers))
        for m in (self.w_k, self.w_v, *self.z):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # [n_cams, n_feat] patch validity (not saved: recomputed from camera_hw)
        self.register_buffer("valid", valid_masks, persistent=False)
        n_feat = valid_masks.shape[1]
        self.register_buffer("block_of_token", block_of_token(n_feat), persistent=False)

    @torch.no_grad()
    def init_from_main(self, lm_expert):
        self.copy_expert.load_state_dict(lm_expert.state_dict())


def build_copy_cache(flow, feats, img_masks, state, prefix_pad_masks):
    """Copy-expert condition: [aux K/V of every camera ; VLM-layer-0 K/V of the state token]."""
    from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

    br = flow.bypass
    vwe = flow.vlm_with_expert
    tcfg = vwe.config.text_config
    H_kv, D = tcfg.num_key_value_heads, tcfg.head_dim
    B = state.shape[0]
    prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    n_tok_cam = 64  # connector tokens per camera (asserted at attach)

    ks, vs, masks = [], [], []
    for k, f in enumerate(feats):
        u = f.detach().to(br.w_k.weight.dtype) + br.cam_emb[k]
        n = u.shape[1]
        key = br.w_k(u).view(B, n, H_kv, D)
        val = br.w_v(u).view(B, n, H_kv, D)
        tok = k * n_tok_cam + br.block_of_token.to(prefix_pos.device)
        key = apply_rope(key, prefix_pos[:, tok])
        ks.append(key)
        vs.append(val)
        masks.append(br.valid[k][None, :].to(img_masks[k].device) & img_masks[k][:, None])

    # state: exactly what VLM layer 0 computes for the state token (before any mixing)
    layer0 = vwe.get_vlm_model().text_model.layers[0]
    s = flow.state_proj(state)[:, None, :]
    g = layer0.input_layernorm(s).to(layer0.self_attn.k_proj.weight.dtype)
    k_st = layer0.self_attn.k_proj(g).view(B, 1, H_kv, D)
    v_st = layer0.self_attn.v_proj(g).view(B, 1, H_kv, D)
    k_st = apply_rope(k_st, prefix_pad_masks.sum(-1, keepdim=True) - 1)  # state = last valid token
    ks.append(k_st)
    vs.append(v_st)
    masks.append(torch.ones(B, 1, dtype=torch.bool, device=state.device))

    dtype = br.copy_expert.layers[0].self_attn.q_proj.weight.dtype
    K = torch.cat(ks, dim=1).to(dtype).transpose(1, 2)  # [B, H, S_c, D] (DynamicCache layout)
    V = torch.cat(vs, dim=1).to(dtype).transpose(1, 2)
    cache = PrefixKVCache(K, V, vwe.num_expert_layers)
    return cache, torch.cat(masks, dim=1)


def run_copy(flow, suffix_embs, suffix_pad_masks, suffix_att_masks, prefix_pad_masks, cache, copy_pad):
    """Run the copy expert on the suffix against the copy cache -> per-layer injections z_i(h_c)."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    B, L = suffix_pad_masks.shape
    S_c = copy_pad.shape[1]
    att = torch.cat(
        [copy_pad[:, None, :].expand(B, L, S_c), make_att_2d_masks(suffix_pad_masks, suffix_att_masks)],
        dim=2,
    )
    pos = prefix_pad_masks.sum(-1, keepdim=True) + torch.cumsum(suffix_pad_masks, dim=1) - 1
    _, _, outs = flow.vlm_with_expert.forward(
        attention_mask=att,
        position_ids=pos,
        past_key_values=cache,
        inputs_embeds=[None, suffix_embs],
        use_cache=True,
        expert_model=flow.bypass.copy_expert,
        return_expert_layer_outputs=True,
    )
    cache.crop(S_c)  # even layers appended the suffix K/V
    return [z(h.to(z.weight.dtype)) for z, h in zip(flow.bypass.z, outs)]
