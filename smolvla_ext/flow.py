"""Instance-method replacements for `VLAFlowMatching` (policy.model). Bound with types.MethodType
by attach(); lerobot is not edited. With every module off they compute exactly what the stock
methods compute (bitwise; see tests).

State the methods read off the instance:
    self._ext          ExtConfig
    self._ext_feats    None, or a list the connector pre-hook appends SigLIP features to (PVI)
    self._ext_capture  None, or a list that receives the ROCKET student hiddens of one forward
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from .pvi import build_copy_cache, run_copy
from .rocket import GRID


def _take_feats(self, n_images):
    feats = self._ext_feats
    self._ext_feats = None
    if feats is None or len(feats) != n_images:
        raise RuntimeError(f"[ext] PVI: captured {None if feats is None else len(feats)} SigLIP feature "
                           f"maps for {n_images} cameras; connector hook / embed_prefix changed?")
    return feats


def embed_suffix(self, noisy_actions, timestep, head=None):
    """Stock embed_suffix; `head` (Coda correction expert) supplies the three input projections."""
    from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding

    head = self if head is None else head
    embs = []
    pad_masks = []
    att_masks = []

    # Fuse timestep + action information using an MLP
    action_emb = head.action_in_proj(noisy_actions)
    device = action_emb.device
    bsize = action_emb.shape[0]
    dtype = action_emb.dtype
    # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
    time_emb = create_sinusoidal_pos_embedding(
        timestep,
        self.vlm_with_expert.expert_hidden_size,
        self.config.min_period,
        self.config.max_period,
        device=device,
    )
    time_emb = time_emb.type(dtype=dtype)

    time_emb = time_emb[:, None, :].expand_as(action_emb)
    action_time_emb = torch.cat([action_emb, time_emb], dim=2)

    action_time_emb = head.action_time_mlp_in(action_time_emb)
    action_time_emb = F.silu(action_time_emb)  # swish == silu
    action_time_emb = head.action_time_mlp_out(action_time_emb)

    # Add to input tokens
    embs.append(action_time_emb)

    bsize, action_time_dim = action_time_emb.shape[:2]
    action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
    pad_masks.append(action_time_mask)

    # Set attention masks so that image, language and state inputs do not attend to action tokens
    att_masks += [1] * self.config.chunk_size
    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
    att_masks = att_masks[None, :].expand(bsize, len(att_masks))
    return embs, pad_masks, att_masks


def ext_prefill(self, images, img_masks, lang_tokens, lang_masks, state):
    """Prefix K/V cache (as stock sample_actions) [+ PVI copy cache]. -> (pad_masks, cache, copy)."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    pvi = self._ext.pvi
    if pvi:
        self._ext_feats = []
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    feats = _take_feats(self, len(images)) if pvi else None
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    # Compute image and language key value cache
    _, past_key_values = self.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=self.config.use_cache,
    )
    copy = build_copy_cache(self, feats, img_masks, state, prefix_pad_masks) if pvi else None
    return prefix_pad_masks, past_key_values, copy


def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, head=None, copy_cache=None):
    """Stock denoise_step + `head` (Coda correction expert in the expert slot) + `copy_cache`
    (PVI: run the copy expert first and inject z_i into the main expert)."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep, head=head)

    layer_injections = None
    if copy_cache is not None:
        cache, copy_pad = copy_cache
        layer_injections = run_copy(
            self, suffix_embs, suffix_pad_masks, suffix_att_masks, prefix_pad_masks, cache, copy_pad
        )

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

    outputs_embeds, _ = self.vlm_with_expert.forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=self.config.use_cache,
        expert_model=None if head is None else head.lm_expert,
        layer_injections=layer_injections,
    )
    if past_key_values is not None:
        # Self-attention layers append suffix K/V in place; restore the prefix for the next step.
        past_key_values.crop(prefix_len)
    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -self.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = (self if head is None else head).action_out_proj(suffix_out)
    return v_t


def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    """Main-AE flow-matching loss (elementwise), stock math. PVI injections when on; ROCKET student
    hiddens are appended to self._ext_capture when the caller set it to a list."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    ext = self._ext
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)

    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    if ext.pvi:
        self._ext_feats = []
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

    layer_injections = None
    if ext.pvi:
        feats = _take_feats(self, len(images))
        cache, copy_pad = build_copy_cache(self, feats, img_masks, state, prefix_pad_masks)
        layer_injections = run_copy(
            self, suffix_embs, suffix_pad_masks, suffix_att_masks, prefix_pad_masks, cache, copy_pad
        )

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    capture = ext.rocket and self._ext_capture is not None
    out = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        layer_injections=layer_injections,
        capture_prefix_layers=tuple(ext.student_layers) if capture else (),
    )
    (_, suffix_out) = out[0]
    if capture:
        n_img = len(images) * GRID * GRID  # image tokens lead the prefix, camera order, 64 each
        self._ext_capture.extend(h[:, :n_img] for h in out[2])
    suffix_out = suffix_out[:, -self.config.chunk_size :]
    # Original openpi code, upcast attention output
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = self.action_out_proj(suffix_out)
    losses = F.mse_loss(u_t, v_t, reduction="none")
    return losses


def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
    """Stock sampling (main AE, num_steps, RTC args) [+PVI injections] [+Coda correction steps].

    With Coda, ExtConfig.coda_rtc says WHERE the RTC guidance acts (inference only, default
    "off" = refuse, as before): main = stage 1 only, correction = stage 2 only (the output
    the robot executes), both. The guidance needs nothing but a velocity field, x_t and the
    time (modeling_rtc.py:117-232), and stage 2 integrates the same t=1->0 schedule, so the
    same kwargs apply; its x_1 is the main AE chunk A instead of noise, and x1_t = x_t - t*v_t
    is still the predicted final chunk because the correction interpolates x_t = t*A+(1-t)*gt
    (correction_losses below). Pull per Euler step = max_guidance_weight / num_steps.
    """
    from lerobot.policies.common.flow_matching import euler_integrate

    ext = self._ext
    bsize = state.shape[0]
    device = state.device

    if noise is None:
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)
    rtc_on = self._rtc_enabled()
    placement = ext.coda_rtc if ext.coda else "main"
    if ext.coda and rtc_on and placement == "off":
        raise RuntimeError(
            "[ext] Coda + RTC is opt-in and ExtConfig.coda_rtc is 'off'. Pick where the guidance "
            "acts: main (the correction AE then moves the overlap freely) | correction (the "
            "executed output is guided) | both. Server: --coda-rtc; or run without --rtc. "
            "RuntimeError, not assert, so python -O cannot drop the check."
        )
    rtc_main = rtc_on and placement in ("main", "both")
    rtc_corr = rtc_on and ext.coda and placement in ("correction", "both")

    prefix_pad_masks, past_key_values, copy = ext_prefill(
        self, images, img_masks, lang_tokens, lang_masks, state
    )
    num_steps = self.config.num_steps

    actions = euler_integrate(
        lambda input_x_t, current_timestep: self.denoise_step(
            x_t=input_x_t,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=current_timestep,
            copy_cache=copy,
        ),
        noise,
        num_steps,
        rtc_processor=self.rtc_processor,
        rtc_enabled=rtc_main,
        inference_delay=kwargs.get("inference_delay"),
        prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
        execution_horizon=kwargs.get("execution_horizon"),
    )
    if not ext.coda:
        return actions
    # Empty dict when the correction stage is unguided: euler_integrate fires the debug tracker
    # for a non-None rtc_processor even with rtc_enabled=False (flow_matching.py:111-113), and
    # stage 2 passed nothing before this option existed.
    corr_rtc = {}
    if rtc_corr:
        corr_rtc = dict(
            rtc_processor=self.rtc_processor,
            rtc_enabled=True,
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )
    # correction AE: starts from the main AE output A, same t schedule, no PVI injection
    return euler_integrate(
        lambda input_x_t, current_timestep: self.denoise_step(
            x_t=input_x_t,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=current_timestep,
            head=self.correction,
        ),
        actions,
        ext.coda_num_steps,
        **corr_rtc,
    )


def correction_losses(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    """Coda loss (elementwise). Candidate A = current main AE (+PVI) integrated num_steps from noise,
    and the prefix K/V, both without grad: L_c reaches only the correction AE."""
    from lerobot.policies.common.flow_matching import euler_integrate

    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)
    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

    with torch.no_grad():
        prefix_pad_masks, past_key_values, copy = ext_prefill(
            self, images, img_masks, lang_tokens, lang_masks, state
        )
        cand = euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=current_timestep,
                copy_cache=copy,
            ),
            noise,
            self.config.num_steps,
        )

    time_expanded = time[:, None, None]
    x_t = time_expanded * cand + (1 - time_expanded) * actions  # t = 1 -> A, t = 0 -> ground truth
    u_t = cand - actions
    v_t = self.denoise_step(
        x_t=x_t,
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        timestep=time,
        head=self.correction,
    )
    return F.mse_loss(u_t, v_t, reduction="none")
