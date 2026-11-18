"""Speed options for an attached SmolVLA (user, 2026-10-11): SDPA attention + torch.compile.

attn_impl = "sdpa": SmolVLMWithExpertModel.eager_attention_forward (fp32 matmul + softmax over a
  boolean block mask) is replaced, on the instance, by torch.nn.functional.scaled_dot_product_attention
  with the same mask. The flash-attn package cannot take SmolVLA's prefix/suffix block mask; SDPA
  picks a fused kernel that accepts a mask (memory-efficient / cuDNN) where available.
  Padding query rows have no allowed key at all: eager returns a uniform average there, SDPA would
  return NaN. Those rows are pointed at key 0 instead -- they are padding, masked out for every
  real query and for the loss, so real outputs are unchanged.
  Numerics: q/k are no longer upcast to fp32 (bf16 kernels accumulate in fp32). Measured against
  eager before use (diag + tests).
compile_mode = "layers": nn.Module.compile() (in place, state_dict keys unchanged) on every MLP of
  the VLM text layers, the main / correction / copy experts, plus the vision tower and connector.
  The whole flow forward is NOT compiled: it is Python control flow over KV caches, hooks and
  extension branches (graph breaks / recompiles). lerobot's own compile_model cannot be used --
  it compiles the stock methods in __init__, which attach() then replaces.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F  # noqa: N812


def sdpa_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
    """Drop-in for SmolVLMWithExpertModel.eager_attention_forward (same arguments / output)."""
    groups = self.num_attention_heads // self.num_key_value_heads
    dt = value_states.dtype
    q = query_states.transpose(1, 2).to(dt)                                   # B H Lq D
    k = key_states.transpose(1, 2).to(dt).repeat_interleave(groups, dim=1)   # same head order as eager's expand
    v = value_states.transpose(1, 2).repeat_interleave(groups, dim=1)
    m = attention_mask[:, None, :, :]
    empty = ~m.any(dim=-1, keepdim=True)                                      # padding query rows
    first = torch.zeros(m.shape[-1], dtype=torch.bool, device=m.device)
    first[0] = True
    m = m | (empty & first)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=head_dim ** -0.5)
    return out.transpose(1, 2).reshape(batch_size, -1, self.num_attention_heads * head_dim)


def _compile_targets(flow):
    vwe = flow.vlm_with_expert
    vlm = vwe.get_vlm_model()
    mods = [layer.mlp for layer in vlm.text_model.layers]
    experts = [vwe.lm_expert]
    if getattr(flow, "correction", None) is not None:
        experts.append(flow.correction.lm_expert)
    if getattr(flow, "bypass", None) is not None:
        experts.append(flow.bypass.copy_expert)
    for e in experts:
        mods += [layer.mlp for layer in e.layers]
    mods.append(vlm.vision_model)
    mods.append(vlm.connector)
    return mods


def apply_fast(policy, ext, verbose=True):
    flow = policy.model
    vwe = flow.vlm_with_expert
    done = []
    if ext.attn_impl == "sdpa":
        vwe.get_attention_interface = types.MethodType(
            lambda self: types.MethodType(sdpa_attention_forward, self), vwe)
        done.append("attn=sdpa")
    if ext.compile_mode == "layers":
        mods = _compile_targets(flow)
        for m in mods:
            m.compile()
        done.append(f"compile=layers({len(mods)} modules)")
    elif ext.compile_mode == "full":
        # NOT USABLE (2026-10-11, torch 2.9.1): inductor BackendCompilerFailed "'SymInt' object has
        # no attribute 'size'" on the first training step. Kept only so the failure is documented;
        # the chain uses compile_mode="layers".
        vwe.forward = torch.compile(vwe.forward, dynamic=True)
        vlm = vwe.get_vlm_model()
        vlm.vision_model.compile()
        done.append("compile=full(vlm_with_expert.forward + vision tower)")
    if done and verbose:
        print(f"[ext] fast: {' '.join(done)}", flush=True)
    return done
