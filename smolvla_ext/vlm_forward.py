"""Copy of `SmolVLMWithExpertModel.forward` (lerobot smolvlm_with_expert.py, git 9e14584) with four
extra keyword arguments. Bound to an instance with `types.MethodType`; lerobot is not edited.

With every extra argument at its default the computation is the original one, op for op (the
test suite checks bitwise equality against the class method).

    expert_model                 run this module in the expert slot instead of `self.lm_expert`
                                 (PVI copy expert, Coda correction expert). Its layers and final
                                 norm are used.
    layer_injections             list (one per layer) of tensors added to the expert stream at the
                                 END of each layer (after the MLP residual, before the final norm).
    return_expert_layer_outputs  also return the expert stream after every layer (after the
                                 injection, before the final norm).
    capture_prefix_layers        also return the INPUT of these VLM layers (= previous layer's
                                 output), full prefix, still in the autograd graph.

Return value: `(outputs_embeds, past_key_values)` as the original; with
`return_expert_layer_outputs` and/or `capture_prefix_layers` the extra lists follow in that order.
"""

from __future__ import annotations

import torch


def ext_vlm_forward(
    self,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: list[torch.FloatTensor] = None,
    use_cache: bool | None = None,
    expert_model=None,
    layer_injections=None,
    return_expert_layer_outputs: bool = False,
    capture_prefix_layers=(),
):
    from transformers import DynamicCache

    models = [self.get_vlm_model().text_model, self.lm_expert if expert_model is None else expert_model]
    model_layers = self.get_model_layers(models)
    for hidden_states in inputs_embeds:
        # TODO this is very inefficient
        # dtype is always the same, batch size too (if > 1 len)
        # device could be trickier in multi gpu edge cases but that's it
        if hidden_states is None:
            continue
        batch_size = hidden_states.shape[0]

    # Prefix prefill: no cache was passed, so create one and fill it (every layer runs
    # self-attention over the prefix). When a filled cache is passed (denoising), layers
    # read from it instead.
    fill_kv_cache = use_cache and past_key_values is None
    if fill_kv_cache:
        past_key_values = DynamicCache()

    expert_layer_outputs = []
    captured = []

    # RMSNorm
    num_layers = self.num_vlm_layers
    head_dim = self.vlm.config.text_config.head_dim
    for layer_idx in range(num_layers):
        if layer_idx in capture_prefix_layers:
            if inputs_embeds[0] is None:
                raise AssertionError("capture_prefix_layers needs the prefix stream (training forward)")
            captured.append(inputs_embeds[0])
        if (
            fill_kv_cache
            or "cross" not in self.attention_mode
            or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
        ):
            att_outputs, past_key_values = self.forward_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        else:
            att_outputs, past_key_values = self.forward_cross_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            att_output = (
                att_outputs[i] if i < len(att_outputs) else att_outputs[0]
            )  # in case of self_attn
            if hidden_states is not None:
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue
                end = start + hidden_states.shape[1]

                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                att_out = att_output[:, start:end]
                out_emb = layer.self_attn.o_proj(att_out)

                out_emb += hidden_states
                after_first_residual = out_emb.clone()

                out_emb = layer.post_attention_layernorm(out_emb)
                out_emb = layer.mlp(out_emb)

                out_emb += after_first_residual

                outputs_embeds.append(out_emb)

                start = end if len(att_outputs) == 1 else 0
            else:
                outputs_embeds.append(None)

        inputs_embeds = outputs_embeds

        if layer_injections is not None:
            inj = layer_injections[layer_idx]
            if inj is not None:
                inputs_embeds[1] = inputs_embeds[1] + inj.to(inputs_embeds[1].dtype)
        if return_expert_layer_outputs:
            expert_layer_outputs.append(inputs_embeds[1])

    # final norm
    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is not None:
            out_emb = models[i].norm(hidden_states)
            outputs_embeds.append(out_emb)
        else:
            outputs_embeds.append(None)
    extras = []
    if return_expert_layer_outputs:
        extras.append(expert_layer_outputs)
    if capture_prefix_layers:
        extras.append(captured)
    if extras:
        return (outputs_embeds, past_key_values, *extras)
    return outputs_embeds, past_key_values
