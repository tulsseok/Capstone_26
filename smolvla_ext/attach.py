"""attach(policy, ext): bolt PVI / Coda / ROCKET onto a stock, already-loaded SmolVLAPolicy.

Nothing in lerobot is edited. On the instance only:
  * VLAFlowMatching methods replaced (types.MethodType): embed_suffix, denoise_step, forward,
    sample_actions (+ new correction_losses); vlm_with_expert.forward -> smolvla_ext copy.
  * new submodules: model.bypass (PVI), model.correction (Coda), model.align_proj (ROCKET)
    -> saved in model.safetensors under those prefixes.
  * policy.forward (loss = L_m [+ L_c] [+ alpha L_align] in training, L_m only in eval),
    policy.get_optim_params (param groups), policy.prepare_images (camera size check),
    policy._save_pretrained (also writes smolvla_ext.json).

attach() must run AFTER the weights are loaded: the copies (PVI copy expert, Coda correction AE)
take the main AE's current values. If `pretrained_path` already holds extension weights (resume,
or starting from an extension checkpoint) those are loaded instead, strictly.
"""

from __future__ import annotations

import types
from pathlib import Path

import torch

from . import flow as _flow
from .coda import CorrectionExpert
from .config import EXT_PREFIX, MAIN_AE_MODULES, SIDECAR_NAME, ExtConfig
from .pvi import BypassBranch, patch_valid_mask
from .rocket import GRID, VGGT_KEY, AlignProjector, align_loss, valid_token_mask
from .vlm_forward import ext_vlm_forward

MAIN_PREFIXES = tuple(f"model.{m}." for m in MAIN_AE_MODULES)


def _log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


def camera_hw_from_features(features, keys):
    """{key: [H, W]} from LeRobot dataset features (info.json shape + names, e.g. [240, 320, 3])."""
    out = {}
    for k in keys:
        ft = features.get(k)
        if ft is None:
            raise ValueError(f"[ext] camera {k} not in the dataset features")
        shape, names = list(ft["shape"]), list(ft.get("names") or [])
        if "height" in names and "width" in names:
            out[k] = [int(shape[names.index("height")]), int(shape[names.index("width")])]
        elif len(shape) == 3 and shape[-1] in (1, 3, 4):
            out[k] = [int(shape[0]), int(shape[1])]
        else:
            raise ValueError(f"[ext] cannot read (H, W) of {k} from shape {shape} names {names}")
    return out


def _resolve_camera_hw(policy, ext, camera_hw):
    keys = list(policy.config.image_features)
    hw = camera_hw or ext.camera_hw
    if hw is None:
        hw = {k: list(policy.config.image_features[k].shape[1:3]) for k in keys}
        print(f"[ext] WARNING: no camera_hw given; using config.image_features shapes {hw}. "
              "These can differ from the real frames (d10_new says 256x256 for 240x320 data).", flush=True)
    hw = {k: [int(v[0]), int(v[1])] for k, v in hw.items()}
    missing = [k for k in keys if k not in hw]
    if missing:
        raise ValueError(f"[ext] camera_hw has no entry for {missing}")
    return {k: hw[k] for k in keys}


def _check_config(policy, ext):
    cfg = policy.config
    if cfg.add_image_special_tokens:
        raise ValueError("[ext] needs add_image_special_tokens=False (image tokens lead the prefix)")
    if (ext.pvi or ext.rocket) and cfg.empty_cameras:
        raise ValueError("[ext] PVI/ROCKET need empty_cameras=0")
    if ext.coda:
        if not cfg.use_cache:
            raise ValueError("[ext] Coda reads the prefix K/V cache: use_cache must be True")
        if cfg.rtc_config is not None and cfg.rtc_config.enabled and ext.coda_rtc == "off":
            raise ValueError("[ext] Coda and RTC are not used together unless ExtConfig.coda_rtc "
                             "picks a placement (main|correction|both); server: --coda-rtc")
    if ext.rocket and cfg.train_expert_only:
        raise ValueError("[ext] ROCKET needs train_expert_only=False (else L_align stops at the projector)")
    vwe = policy.model.vlm_with_expert
    if ext.pvi and vwe.num_expert_layers != vwe.num_vlm_layers:
        raise ValueError("[ext] PVI assumes one expert layer per VLM layer")
    if ext.rocket and max(ext.student_layers) >= vwe.num_vlm_layers:
        raise ValueError(f"[ext] ROCKET student layer >= num_vlm_layers {vwe.num_vlm_layers}")
    vc = vwe.config
    g = vc.vision_config.image_size // vc.vision_config.patch_size // vc.scale_factor
    if g != GRID:
        raise ValueError(f"[ext] expected {GRID}x{GRID} connector tokens per camera, got {g}x{g}")


def _ext_param_names(policy, ext):
    pre = tuple(EXT_PREFIX[m] for m in ext.enabled)
    return [k for k in policy.state_dict() if pre and k.startswith(pre)]


def load_ext_weights(policy, model_file, ext, strict=True):
    """Load every extension tensor of the enabled modules from a safetensors file.
    Returns the number of tensors loaded; raises if strict and any is missing."""
    from safetensors import safe_open

    want = _ext_param_names(policy, ext)
    sd = policy.state_dict()
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        have = set(f.keys())
        missing = [k for k in want if k not in have]
        if missing and strict:
            raise RuntimeError(f"[ext] {len(missing)} extension tensors missing in {model_file}: {missing[:5]}")
        n = 0
        with torch.no_grad():
            for k in want:
                if k in have:
                    t = f.get_tensor(k)
                    if tuple(t.shape) != tuple(sd[k].shape):
                        raise RuntimeError(f"[ext] shape mismatch for {k}: file {tuple(t.shape)} vs {tuple(sd[k].shape)}")
                    sd[k].copy_(t.to(sd[k].dtype))
                    n += 1
    return n


def _file_ext_keys(model_file, ext):
    from safetensors import safe_open

    pre = tuple(EXT_PREFIX[m] for m in ext.enabled)
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        return [k for k in f.keys() if pre and k.startswith(pre)]


def _set_main_ae_grad(policy, ext):
    if ext.main_ae == "freeze":
        for m in MAIN_AE_MODULES:
            for p in policy.model.get_submodule(m).parameters():
                p.requires_grad_(False)


# ------------------------------------------------------------------------- policy methods --
def ext_get_optim_params(self):
    """Param groups: [rest @ optimizer_lr] [main AE @ optimizer_lr | main_ae_lr] [new @ new_lr].
    Only trainable params are handed out; each exactly once (asserted)."""
    ext = self._smolvla_ext
    new_pre = tuple(EXT_PREFIX[m] for m in ext.enabled)
    rest, main, new = [], [], []
    names = {"rest": [], "main": [], "new": []}
    for n, p in self.named_parameters():
        if not p.requires_grad:
            continue
        if new_pre and n.startswith(new_pre):
            new.append(p); names["new"].append(n)
        elif n.startswith(MAIN_PREFIXES):
            main.append(p); names["main"].append(n)
        else:
            rest.append(p); names["rest"].append(n)
    if ext.main_ae == "freeze" and main:
        raise AssertionError(f"[ext] main AE is frozen but trainable: {names['main'][:5]}")
    trainable = [p for p in self.parameters() if p.requires_grad]
    ids = [id(p) for p in rest + main + new]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in trainable}:
        raise AssertionError("[ext] trainable parameters are not in exactly one group")
    groups = []
    desc = []
    if rest:
        groups.append({"params": rest})
        desc.append(f"rest={sum(p.numel() for p in rest)}@{self.config.optimizer_lr:g}")
    if main:
        g = {"params": main}
        lr = self.config.optimizer_lr
        if ext.main_ae == "lr":
            g["lr"] = lr = ext.main_ae_lr
        groups.append(g)
        desc.append(f"main_ae={sum(p.numel() for p in main)}@{lr:g}")
    else:
        desc.append("main_ae=frozen")
    if new:
        groups.append({"params": new, "lr": ext.new_lr})
        desc.append(f"new={sum(p.numel() for p in new)}@{ext.new_lr:g}")
    print(f"[ext] trainable groups: {' '.join(desc)} (each trainable param in exactly one group)",
          flush=True)
    return groups


def _reduce(self, losses, actions_is_pad, reduction):
    """Stock SmolVLAPolicy.forward masking/reduction of an elementwise [B, T, D] loss."""
    losses = losses[:, :, : self.config.action_feature.shape[0]]
    if actions_is_pad is not None:
        losses = losses * (~actions_is_pad).unsqueeze(-1)
    losses = losses[:, :, : self.config.max_action_dim]
    if reduction == "none":
        if actions_is_pad is None:
            return losses.mean(dim=(1, 2))
        num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
        return losses.sum(dim=(1, 2)) / num_valid
    if actions_is_pad is None:
        return losses.mean()
    num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
    return losses.sum() / num_valid


def ext_policy_forward(self, batch, noise=None, time=None, reduction: str = "mean"):
    """Stock SmolVLAPolicy.forward for L_m; in training adds Coda L_c and ROCKET alpha*L_align.
    In eval mode the returned loss is L_m only (eval_loss comparable with d10_new)."""
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    ext = self._smolvla_ext
    flow = self.model
    if self.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

    images, img_masks = self.prepare_images(batch)
    state = self.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    actions = self.prepare_action(batch)
    actions_is_pad = batch.get("action_is_pad")
    loss_dict = {}

    training = self.training
    want_align = ext.rocket and training
    targets = None
    if want_align:
        targets = batch.get(VGGT_KEY)
        if targets is None:
            raise KeyError(f"[ext] ROCKET training batch has no '{VGGT_KEY}' (TeacherDataset not attached?)")
    flow._ext_capture = [] if want_align else None
    try:
        losses = flow.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        hiddens = flow._ext_capture
    finally:
        flow._ext_capture = None
    original_action_dim = self.config.action_feature.shape[0]
    losses = losses[:, :, :original_action_dim]
    loss_dict["losses_after_forward"] = losses.clone().mean().item()

    if actions_is_pad is not None:
        in_episode_bound = ~actions_is_pad
        losses = losses * in_episode_bound.unsqueeze(-1)
        loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

    # Remove padding
    losses = losses[:, :, : self.config.max_action_dim]
    loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

    if reduction == "none":
        if actions_is_pad is None:
            loss = losses.mean(dim=(1, 2))
        else:
            num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
            loss = losses.sum(dim=(1, 2)) / num_valid
    else:
        if actions_is_pad is None:
            loss = losses.mean()
        else:
            num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
            loss = losses.sum() / num_valid

    if training and ext.any_module:
        st = self._ext_stats
        loss_dict["main_loss"] = loss.mean().item()
        st["main"] += loss_dict["main_loss"]
        if ext.coda:
            lc = flow.correction_losses(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
            corr = _reduce(self, lc, actions_is_pad, reduction)
            loss = loss + corr
            loss_dict["correction_loss"] = corr.mean().item()
            st["corr"] += loss_dict["correction_loss"]
        if ext.rocket:
            if len(hiddens) != len(ext.rocket_pairs):
                raise RuntimeError(f"[ext] captured {len(hiddens)} student layers, expected {len(ext.rocket_pairs)}")
            per_sample, per_pair = align_loss(flow.align_proj, hiddens, targets, flow.align_proj.valid, img_masks)
            loss = loss + ext.rocket_alpha * (per_sample if reduction == "none" else per_sample.mean())
            loss_dict["align_loss"] = float(per_sample.detach().mean())
            for (a, b), v in zip(ext.rocket_pairs, per_pair):
                loss_dict[f"align_loss_L{a}_V{b}"] = v
            st["align"] += loss_dict["align_loss"]
        st["n"] += 1
        st["calls"] += 1
        if st["n"] >= self._ext_log_every:
            n = st["n"]
            msg = f"[ext] losses main={st['main'] / n:.4f}"
            if ext.coda:
                msg += f" corr={st['corr'] / n:.4f}"
            if ext.rocket:
                msg += f" align={st['align'] / n:.4f}"
            print(msg + f" (mean of {n} training batches, through batch {st['calls']})", flush=True)
            for k in ("main", "corr", "align", "n"):
                st[k] = 0.0 if k != "n" else 0

    loss_dict["loss"] = loss.mean().item()
    return loss, loss_dict


def _camera_masks(policy, ext, h, w):
    """Padding masks the enabled modules would use for an h x w camera (tuple of tensors)."""
    out = []
    if ext.pvi:
        tw, th = policy.config.resize_imgs_with_padding
        patch = policy.model.vlm_with_expert.config.vision_config.patch_size
        out.append(patch_valid_mask(h, w, tw, th, patch))
    if ext.rocket:
        out.append(valid_token_mask(h, w))
    return tuple(out)


def _wrap_prepare_images(policy, ext):
    """Refuse camera frames whose padding layout differs from the one the masks were built for.
    Only the mask matters, so e.g. 480x640 passes for masks built at 240x320 (same aspect)."""
    if not (ext.pvi or ext.rocket):
        return
    orig = policy.prepare_images
    ref = {k: _camera_masks(policy, ext, h, w) for k, (h, w) in ext.camera_hw.items()}
    ok_shapes = {k: {tuple(hw)} for k, hw in ext.camera_hw.items()}

    def prepare_images(self, batch):
        for k in ref:
            x = batch.get(k)
            if not torch.is_tensor(x):
                continue
            hw = tuple(int(v) for v in x.shape[-2:])
            if hw in ok_shapes[k]:
                continue
            got = _camera_masks(self, ext, *hw)
            if any(not torch.equal(a, b) for a, b in zip(got, ref[k])):
                raise ValueError(f"[ext] {k} is {hw}: its padding mask differs from the one built for "
                                 f"{tuple(ext.camera_hw[k])} (ExtConfig.camera_hw)")
            ok_shapes[k].add(hw)
        return orig(batch)

    policy.prepare_images = types.MethodType(prepare_images, policy)


def _wrap_save(policy):
    orig = policy._save_pretrained

    def _save_pretrained(self, save_directory):
        orig(save_directory)
        if (Path(save_directory) / "config.json").is_file():  # main process wrote the checkpoint
            self._smolvla_ext.save(save_directory)

    policy._save_pretrained = types.MethodType(_save_pretrained, policy)


# ------------------------------------------------------------------------------- attach --
def attach(policy, ext: ExtConfig, pretrained_path=None, camera_hw=None, log_every: int = 200,
           verbose: bool = True):
    if getattr(policy, "_smolvla_ext", None) is not None:
        raise RuntimeError("[ext] already attached")
    _check_config(policy, ext)
    flow = policy.model
    vwe = flow.vlm_with_expert
    cfg = policy.config
    ext.camera_hw = _resolve_camera_hw(policy, ext, camera_hw)
    ext.main_num_steps = int(cfg.num_steps)
    keys = list(cfg.image_features)
    dev = flow.action_in_proj.weight.device

    # method replacements (also when no module is on: then they reproduce the stock computation)
    vwe.forward = types.MethodType(ext_vlm_forward, vwe)
    for name in ("embed_suffix", "denoise_step", "forward", "sample_actions", "correction_losses"):
        setattr(flow, name, types.MethodType(getattr(_flow, name), flow))
    flow._ext = ext
    flow._ext_feats = None
    flow._ext_capture = None

    if ext.pvi:
        vc = vwe.config
        tw, th = cfg.resize_imgs_with_padding
        patch = vc.vision_config.patch_size
        valid = torch.stack([patch_valid_mask(h, w, tw, th, patch) for h, w in ext.camera_hw.values()])
        n_feat = (vc.vision_config.image_size // patch) ** 2
        assert valid.shape[1] == n_feat, (valid.shape, n_feat)
        flow.bypass = BypassBranch(
            vwe.lm_expert, len(keys), vc.vision_config.hidden_size,
            vc.text_config.num_key_value_heads * vc.text_config.head_dim,
            vwe.expert_hidden_size, vwe.num_expert_layers, valid,
        ).to(dev)
        connector = vwe.get_vlm_model().connector

        def _grab_siglip(_mod, args):
            if flow._ext_feats is not None:
                flow._ext_feats.append(args[0])

        flow._ext_hook = connector.register_forward_pre_hook(_grab_siglip)
    if ext.coda:
        flow.correction = CorrectionExpert(flow).to(dev)
    if ext.rocket:
        flow.align_proj = AlignProjector(vwe.config.text_config.hidden_size, ext.rocket_dim).to(dev)
        flow.align_proj.register_buffer(
            "valid", torch.stack([valid_token_mask(h, w) for h, w in ext.camera_hw.values()]).to(dev),
            persistent=False)

    # extension weights from the start checkpoint, if it has them (resume / ext checkpoint)
    src = "copied from the loaded main AE / fresh init"
    if pretrained_path is not None and ext.any_module:
        mf = Path(pretrained_path) / "model.safetensors"
        if mf.is_file():
            found = _file_ext_keys(mf, ext)
            if found:
                n = load_ext_weights(policy, mf, ext, strict=True)
                src = f"loaded {n} tensors from {mf}"
    _set_main_ae_grad(policy, ext)

    policy._smolvla_ext = ext
    policy._ext_log_every = max(1, int(log_every))
    policy._ext_stats = {"main": 0.0, "corr": 0.0, "align": 0.0, "n": 0, "calls": 0}
    policy.forward = types.MethodType(ext_policy_forward, policy)
    policy.get_optim_params = types.MethodType(ext_get_optim_params, policy)
    _wrap_prepare_images(policy, ext)
    _wrap_save(policy)
    from smolvla_ext.fast import apply_fast
    apply_fast(policy, ext, verbose=verbose)

    n_new = {m: sum(p.numel() for n, p in policy.named_parameters() if n.startswith(EXT_PREFIX[m]))
             for m in ext.enabled}
    _log(f"[ext] on: {ext.summary()} | new params {n_new} | ext weights {src}", verbose)
    return policy


def load_policy(path, device="cpu", config=None, verbose=True, compile_ok=False, coda_rtc=None):
    """Inference loader. With a sidecar: stock from_pretrained -> attach per sidecar -> extension
    tensors loaded strictly from model.safetensors. Without a sidecar: plain SmolVLAPolicy.
    The sidecar's attn_impl is honoured (numerics); compile is off unless compile_ok.
    coda_rtc (inference only) overrides the sidecar; without it the sidecar value is IGNORED and
    forced to "off", so a checkpoint saved with a placement cannot silently change any caller
    that did not ask for it (diag/*, offline tools). The server passes --coda-rtc explicitly."""
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = Path(path)
    if config is None:
        config = SmolVLAConfig.from_pretrained(str(path))
        config.device = device
    policy = SmolVLAPolicy.from_pretrained(str(path), config=config)
    ext = ExtConfig.load(path)
    if ext is None:
        policy.to(device)
        policy.eval()
        return policy
    if ext.main_num_steps is not None and int(config.num_steps) != int(ext.main_num_steps):
        raise ValueError(f"[ext] config.num_steps={config.num_steps} but the sidecar was written with "
                         f"main_num_steps={ext.main_num_steps}")
    if not compile_ok:
        ext.compile_mode = "off"
    ext.coda_rtc = "off" if coda_rtc is None else str(coda_rtc)
    ext.__post_init__()      # validate the override (and re-validate the sidecar fields)
    attach(policy, ext, pretrained_path=None, verbose=False)
    n = load_ext_weights(policy, path / "model.safetensors", ext, strict=True)
    policy.to(device)
    policy.eval()
    _log(f"[ext] on: {ext.summary()} | load_policy {path} ({SIDECAR_NAME}): {n} extension tensors "
         f"loaded strictly from model.safetensors", verbose)
    return policy


__all__ = ["attach", "load_policy", "load_ext_weights", "ext_get_optim_params", "MAIN_PREFIXES"]
