# SmolVLA Coda Implementation Spec: Correction AE After the Main AE

> ⚠ Changes made in the 2026-10-11 meeting and implementation are in Section 10 at the end, which takes priority (if it differs from the body of this document, follow Section 10).

> Written 2026.10.11. Canonical design = end of Section 3-3 of `ref/summary/SmolVLA_추가모듈.md` (user's Notion design summary, 2nd edition, decided 10.11). This file is the instruction sheet for turning that design into code.
> Code checked against:
> - LeRobot `src/lerobot/policies/smolvla/` (`modeling_smolvla.py`, `smolvlm_with_expert.py`, `configuration_smolvla.py`)
> - `policies/common/flow_matching.py`, `policies/pretrained.py`, `scripts/lerobot_train.py`
> Source paper = Coda (arXiv 2609.21216, code not released). Close-reading notes in `ref/summary/SmolVLA_추가모듈_선행연구.md` ## 23.
> Companion documents = `SmolVLA_PVI_구현.md` (PVI), `SmolVLA_ROCKET_구현.md` (ROCKET). The three proposals are tested separately.

## 0. To the person (agent) receiving this document

- Your job is the implementation in Sections 3-6 and passing the tests in Section 7. Training in Section 8 is run after the user gives the step count.
- The LeRobot version on the server may differ from the version this document was checked against.
  - Locations are written as `file::function` instead of line numbers.
  - If a name differs, find the function with the same role, change it, and report what you changed.
- Do not remove user modifications that already exist in the server code (e.g., the path that trains only the VLM at a low LR, the group split in `get_optim_params`). When `correction_enabled=False`, those modifications must behave as they do now.
- The design values (copy scope, zero init, 5 + 5 steps, LR 1e-4 and 1e-5, loss form) were set by the user. If something forces a change, do not change it; report it.
- Do not put this in the same code branch as the PVI or ROCKET implementations. The `expert_model` argument in Section 4-3 has the same name and meaning as in PVI spec 4-4.

## 1. Intent

User's Notion design summary (2026.10.10, 2nd edition, original text):

> * Problem: cases where the gripper slightly misses the object
> * Attach a correction module next to the action module so it moves more precisely.
> * Division of roles: action module (big strides) + correction module (small nudges)
> * Assume the action module's output is still not fully denoised, and denoise from there
> * Copy the action module's weights (so they are trainable), original module FREEZE or Low LR
> * Input: VLM KV Cross-ATT (same as the original), action module output A
> * Output: corrected value of the action module, A+a
> * Loss: what the action module got wrong (a loss per module, detach) EX) (ground truth - action module) - correction = Loss
> * Zero-init only the correction module output (at the very start only the main AE's output is used; a structure that fixes only what is wrong)
> * Same time input (same denoising steps. … split 5 each)

"FREEZE or Low LR" was settled by the user's answer on 2026.10.11: *"We'll do both."* Both variants are built (Section 8).

The structure has the same form as Coda (a corrector that reads the frozen policy's candidate action + VLM prefix KV, starts from a copy of the AE weights, output layer zero-initialized). Coda reported 64.4 → 72.6 at 10 steps on frozen official SmolVLA (simulation).

Three conditions to keep:
1. At the moment training starts, the policy output equals the main AE's 5-step output. Guaranteed by starting the correction AE's output projection at 0.
2. Gradients of the main AE and the correction AE do not mix. The candidate A that the correction AE receives is produced without gradients.
3. Inference cost is the same as the current 10 steps. Action expert calls are main 5 + correction 5 = 10 [computed, one correction AE call = one main AE call].

## 2. One-page summary

```
3 cameras, command, state → VLM 16 layers ── prefix K/V (once per observation) ──┬→ main AE 16 layers        ε ── 5 steps ──→ A
                                                                                  └→ correction AE 16 layers  A ── 5 steps ──→ A + a
```

Terms: main AE = the existing action expert and its input/output projections (`vlm_with_expert.lm_expert`, `action_in_proj`, `action_time_mlp_in`, `action_time_mlp_out`, `action_out_proj`). Correction AE = a copy of those five modules. The correction AE takes the main AE's place and reads the same prefix K/V in the same way (self-attention on even layers, cross-attention on odd layers).

```
Inference
  main AE:        x ← ε,   for each t = 1.0, 0.8, 0.6, 0.4, 0.2   x ← x − 0.2 × v_m(x, t)     end = A
  correction AE:  y ← A,   for the same t                         y ← y − 0.2 × v_c(y, t)     end = A + a

Training (correction AE)
  A    = main AE 5-step output (fresh noise ε, no gradient)
  t    ~ sample_time  (same as main AE: Beta(1.5, 1) × 0.999 + 0.001)
  y_t  = t × A + (1 − t) × a⋆                       A at t = 1, ground truth at t = 0
  u    = A − a⋆
  L_c  = (v_c(y_t, t) − u)²   elementwise, same padding mask as the current loss

FREEZE variant:  L = L_c
Low LR variant:  L = L_c + L_m,   L_m = current flow matching loss as is (x_t = t × ε + (1 − t) × a⋆, target ε − a⋆)
```

- If v_c matches u at every t, then after 5 steps y = a⋆, i.e., the correction a = a⋆ − A. This fits the same target as the user's formula "(ground truth − action module) − correction", in velocity units.
- The loss is the same MSE as SmolVLA flow matching (user: "same as the main AE"). Coda used L1 since it is a one-shot correction.
- The time input follows the same rule as the main AE (user: "same time input"). The Coda SmolVLA variant dropped the time condition.

## 3. Components and sizes

| Component | Shape | Initial value | FREEZE variant LR | Low LR variant LR |
|---|---|---|---|---|
| `correction.lm_expert` | Same structure as `lm_expert` (16 layers, width 720) | Copy of the loaded main AE | 1e-4 | 1e-4 |
| `correction.action_in_proj` | `nn.Linear(32, 720)` | Copy | 1e-4 | 1e-4 |
| `correction.action_time_mlp_in` | `nn.Linear(1440, 720)` | Copy | 1e-4 | 1e-4 |
| `correction.action_time_mlp_out` | `nn.Linear(720, 720)` | Copy | 1e-4 | 1e-4 |
| `correction.action_out_proj` | `nn.Linear(720, 32)` | Weight and bias 0 | 1e-4 | 1e-4 |
| Main AE (originals of the five above) | | Current fine-tuned version | Frozen | 1e-5 |
| VLM, SigLIP, connector, `state_proj` | | Current fine-tuned version | Frozen | Frozen |

- Numbers are read from the config: 720 = `vlm_with_expert.expert_hidden_size`, 32 = `config.max_action_dim`, 16 = `vlm_with_expert.num_expert_layers`.
- Correction AE = 99,849,312 parameters (`lm_expert` 98,245,840 + four projections 1,603,472) [counted on CPU with the default config]. Only the 23,072 parameters of the output projection start at 0; the rest start as copies.
- The correction AE's final norm (`lm_expert.norm`) is used before the output projection, so every correction parameter receives gradients. The multi-GPU training "unused parameters" error does not occur.
- The VLM and `state_proj` are frozen in both variants, because the prefix K/V is computed once per observation and read by both the main AE and the correction AE. If they were unfrozen, the K/V read by the correction AE would need a separate detach.
- No gate, magnitude penalty, or LayerNorm is added (not in the design).

## 4. Places to change

### 4-1. `configuration_smolvla.py::SmolVLAConfig`

```python
correction_enabled: bool = False            # when off: same computation and same parameter list as the original
correction_num_steps: int = 5               # number of correction AE denoise steps (main AE uses num_steps)
correction_main_lr: float | None = None     # None = main AE frozen (FREEZE variant), value = train main AE together at that LR (Low LR variant)
```

In `__post_init__`, if `correction_enabled`:
- Assert that `use_cache` is True. The correction AE reads the prefix K/V cache.
- Assert `correction_num_steps >= 1`.

Why it is not named `main_expert_lr`: the same name in the PVI spec means "`optimizer_lr` if None". Keep them separate so the meanings do not collide when the two are merged later.

### 4-2. New file `smolvla/correction_expert.py`

```python
import copy

import torch
from torch import nn

PROJ_NAMES = ("action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj")
MAIN_PREFIXES = ("vlm_with_expert.lm_expert.",) + tuple(f"{n}." for n in PROJ_NAMES)  # names relative to VLAFlowMatching


class CorrectionExpert(nn.Module):
    """Copy of the main AE (action expert + four input/output projections). Only the output projection starts at 0."""

    def __init__(self, flow):
        super().__init__()
        self.lm_expert = copy.deepcopy(flow.vlm_with_expert.lm_expert)  # values at construction are temporary; init_from_main after loading
        for name in PROJ_NAMES:
            setattr(self, name, copy.deepcopy(getattr(flow, name)))
        self.zero_output()

    @torch.no_grad()
    def init_from_main(self, flow):
        self.lm_expert.load_state_dict(flow.vlm_with_expert.lm_expert.state_dict())
        for name in PROJ_NAMES:
            getattr(self, name).load_state_dict(getattr(flow, name).state_dict())
        self.zero_output()

    @torch.no_grad()
    def zero_output(self):
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)
```

- `flow` is `VLAFlowMatching`. Do not store it as an attribute (avoids a circular module reference).
- Checkpoint keys become `model.correction.*`.

### 4-3. `smolvlm_with_expert.py::SmolVLMWithExpertModel.forward`

Add an argument `expert_model=None`. If given, that model is placed in the expert slot.

```python
models = [self.get_vlm_model().text_model, self.lm_expert if expert_model is None else expert_model]
```

- Both the layer list (`get_model_layers`) and the final norm (`models[i].norm`) come from `models`, so this one line makes the correction AE's layers and norm get used.
- With the default value, behavior is the same as now.

### 4-4. `modeling_smolvla.py::VLAFlowMatching`

**`__init__`**: place after the existing `self.set_requires_grad()` (before compile).

```python
if config.correction_enabled:
    self.correction = CorrectionExpert(self)
    trainable = ("correction.",)
    if config.correction_main_lr is not None:          # Low LR variant: main AE is trained too
        trainable += MAIN_PREFIXES
    for name, p in self.named_parameters():
        p.requires_grad = name.startswith(trainable)
```

- VLM, SigLIP, connector, and `state_proj` get `requires_grad=False` in both variants, regardless of the values of `train_expert_only`, `freeze_vision_encoder`, `train_state_proj`.
- The frozen part must give the same output even in train mode (candidate A has the same distribution as at inference). Test 3 in Section 7 checks this. If it differs on the server checkpoint, report the dropout settings.

**`_prefill`** (new method): moves the prefix cache computation from `sample_actions` as is.

```python
def _prefill(self, images, img_masks, lang_tokens, lang_masks, state):
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past_key_values = self.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=self.config.use_cache,
    )
    return prefix_pad_masks, past_key_values
```

**`embed_suffix(self, noisy_actions, timestep, head=None)`**: put `head = self if head is None else head` as the first line, and change the three places `self.action_in_proj`, `self.action_time_mlp_in`, `self.action_time_mlp_out` to `head.`. The time embedding and masks stay as they are.

**`denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, head=None)`**: change only three places.
- `self.embed_suffix(x_t, timestep, head=head)`
- `self.vlm_with_expert.forward(..., expert_model=None if head is None else head.lm_expert)`
- `v_t = (self if head is None else head).action_out_proj(suffix_out)`

The cache rollback (`past_key_values.crop(prefix_len)`) stays as is. Even layers append the correction AE's action-token K/V after the cache and it is stripped here.

**`sample_actions`**: replace the prefix computation with `_prefill`, and add correction AE integration after main AE integration.

```python
if self.config.correction_enabled:
    assert not self._rtc_enabled(), "RTC and the correction AE are not used together"
prefix_pad_masks, past_key_values = self._prefill(images, img_masks, lang_tokens, lang_masks, state)
actions = euler_integrate(...)                         # current code as is (main AE, num_steps, RTC arguments)
if not self.config.correction_enabled:
    return actions
return euler_integrate(
    lambda input_x_t, current_timestep: self.denoise_step(
        x_t=input_x_t,
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        timestep=current_timestep,
        head=self.correction,
    ),
    actions,                                           # main AE output A in place of the noise
    self.config.correction_num_steps,
)
```

- `euler_integrate` starts at t = 1 and goes with dt = −1 / number of steps. With 5 steps, t = 1.0, 0.8, 0.6, 0.4, 0.2, the same as the main AE.
- RTC aligns the previous chunk on the main AE side, so the correction would change that result. It is out of scope for this document, so it is blocked.

**`correction_losses`** (new method, training):

```python
def correction_losses(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
    """Correction AE loss (elementwise). Candidate A is produced by the current main AE with the same step count as inference."""
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)
    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

    with torch.no_grad():  # no gradient for the candidate and prefix (per-module detach)
        prefix_pad_masks, past_key_values = self._prefill(images, img_masks, lang_tokens, lang_masks, state)
        cand = euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=current_timestep,
            ),
            noise,
            self.config.num_steps,
        )

    time_expanded = time[:, None, None]
    x_t = time_expanded * cand + (1 - time_expanded) * actions  # A at t = 1, ground truth at t = 0
    u_t = cand - actions
    v_t = self.denoise_step(
        x_t=x_t,
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        timestep=time,
        head=self.correction,
    )
    return F.mse_loss(u_t, v_t, reduction="none")
```

- The correction `denoise_step` is outside `no_grad`. Even layers concatenate the correction AE's K/V (with gradient) after the cache (no gradient) and attend over it; odd layers re-project the cached K/V with the correction AE's `k_proj`, `v_proj`. Gradients flow only to correction AE parameters.
- If `DynamicCache.update` in the server's transformers version stores detached tensors, test 5 in Section 7 fails (even-layer `k_proj` gradient is 0). In that case follow the cache substitute method at the end of PVI spec 4-5 and report it.
- One training step = prefix once + main AE 5 times (no gradient) + correction AE once (with gradient).

### 4-5. `modeling_smolvla.py::SmolVLAPolicy`

**`forward`**: replace the single line `losses = self.model.forward(...)` with the code below. Move the `original_action_dim` computation before it. Leave the subsequent padding mask and `reduction` code as is.

```python
original_action_dim = self.config.action_feature.shape[0]
if self.config.correction_enabled:
    losses = self.model.correction_losses(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
    loss_dict["correction_loss"] = losses[:, :, :original_action_dim].mean().item()
    if self.config.correction_main_lr is not None:     # Low LR variant: main AE keeps the current loss as is
        main_losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["main_loss"] = main_losses[:, :, :original_action_dim].mean().item()
        losses = losses + main_losses
else:
    losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
```

- Both logged values are means without the padding mask, like the existing `losses_after_forward`.
- The two losses in the Low LR variant share no parameters. Even when summed and backpropagated once, gradients flow separately to each.
- If `noise` and `time` are passed, both losses receive the same values (for testing). If None, each samples its own.

**`get_optim_params`**:

```python
def get_optim_params(self):
    cfg = self.config
    if not cfg.correction_enabled:
        return self.parameters()                       # original (if the server has user modifications, they stay on this path)
    main_prefixes = tuple(f"model.{p}" for p in MAIN_PREFIXES)
    allowed = ("model.correction.",) + (main_prefixes if cfg.correction_main_lr is not None else ())
    bad = [n for n, p in self.named_parameters() if p.requires_grad and not n.startswith(allowed)]
    assert not bad, f"Parameters that should be frozen are being trained: {bad[:5]}"
    groups = [{"params": list(self.model.correction.parameters())}]          # optimizer_lr
    if cfg.correction_main_lr is not None:
        main = [p for n, p in self.named_parameters() if n.startswith(main_prefixes)]
        groups.append({"params": main, "lr": cfg.correction_main_lr})
    return groups
```

- The assertion runs once when the optimizer is built (after policy construction and loading). If the server code turns `requires_grad` back on after construction, it stops here. If it stops, report that location.
- `optim/optimizers.py::AdamWConfig.build` accepts the group list and per-group `lr` as is. `CosineDecayWithWarmupSchedulerConfig` is a `LambdaLR`, so it multiplies each group's own LR by the same factor.

### 4-6. Checkpoint loading: override `SmolVLAPolicy._load_as_safetensor`

The default of `policies/pretrained.py::PreTrainedPolicy.from_pretrained` is `strict=False`. So loading a checkpoint without `correction` keys (the current fine-tuned version) passes without error, and the correction AE keeps its pre-load weights. Define a classmethod with the same name on `SmolVLAPolicy`, and add the copy after the original body.

```python
@classmethod
def _load_as_safetensor(cls, model, model_file, map_location, strict):
    missing_keys, unexpected_keys = load_model_as_safetensor(
        model, model_file, strict=strict, device=resolve_safetensors_device(map_location)
    )
    log_model_loading_keys(missing_keys, unexpected_keys)
    corr_keys = [k for k in model.state_dict() if k.startswith("model.correction.")]
    missing_corr = [k for k in missing_keys if k.startswith("model.correction.")]
    if corr_keys and len(missing_corr) == len(corr_keys):   # only when the checkpoint has no correction AE at all
        model.model.correction.init_from_main(model.model)
    elif missing_corr:
        raise RuntimeError(f"Only some correction keys are missing: {missing_corr[:5]}")
    return model
```

- Names to import: `load_model_as_safetensor` = `safetensors.torch.load_model`, `resolve_safetensors_device` = `lerobot.utils.device_utils`, `log_model_loading_keys` = `lerobot.policies.utils` (same as the imports at the top of `pretrained.py`).
- When re-loading a trained version or resuming training (`--resume`), the `correction` keys exist, so no copy is made.
- If keys other than `model.correction.*` are missing, stop and report.

## 5. Properties to preserve

1. With `correction_enabled=False`: same output, same `state_dict` keys, same `requires_grad`, same optimizer input as the original.
2. Right after enabling (training step 0), the `sample_actions` output equals the output of integrating only the main AE for `num_steps`. The correction velocity is exactly 0.
3. Right after enabling, correction loss = (A − a⋆)² (elementwise).
4. FREEZE variant: only `model.correction.*` parameters receive gradients. After training, all other tensors are the same as in the starting checkpoint.
5. Low LR variant: the correction loss sends no gradient to the main AE, and the main AE loss sends no gradient to the correction AE.

## 6. Things not changed

- Main AE loss (`VLAFlowMatching.forward`), time sampling (`sample_time`), `embed_prefix`, causal mask between action tokens (`att_masks` in `embed_suffix`).
- VLM, SigLIP, connector, `state_proj` are not trained (both variants).
- No auxiliary features (e.g., SigLIP before the connector) are fed to the correction AE. That path is PVI.
- The original noise ε is not fed to the correction AE input (user's input definition). In Coda the variant without ε also scored higher (SmolVLA 2, 5, 10 steps 71.4 / 75.2 / 74.8, variant with it 69.4 / 72.0 / 72.6).
- The correction AE does not read the main AE's hidden states. It receives only the output A.
- No measurement features other than the loss logs (`correction_loss`, `main_loss`).
- Not merged with PVI or ROCKET into one policy.

## 7. Tests `tests/policies/smolvla/test_correction_expert.py`

Run on CPU.
- VLM weights are not needed (`load_vlm_weights=False`; properties are checked with random weights).
- For the config template, refer to `test_smolvla_rtc.py` in the same folder. Three cameras (3, 240, 320), state and action of about 6 dims is enough. `num_steps=5`, `correction_num_steps=5`.
- Variant A = `correction_enabled=False`. Variant B = enabled (FREEZE, `correction_main_lr=None`). Variant C = enabled (Low LR, `correction_main_lr=1e-5`). Build with the same seed, load A's `state_dict` into B and C with `strict=False`, then `model.correction.init_from_main(model)`.

1. **Disabled variant equals original**: A's `state_dict` has no `correction` keys, and `get_optim_params()` gives the same parameters as `self.parameters()`.
2. **Zero-init equivalence**: with the same batch and same noise, the max difference between B's `sample_actions` and A's `sample_actions` (both `num_steps=5`) is below 1e-6.
3. **Initial loss**: put B in `train()` mode and fix noise ε and time t. `B.model.correction_losses(..., noise=ε, time=t)` is `allclose` to `(A.model.sample_actions(..., noise=ε) − actions)²` (A in `eval()`). This also checks that train mode on the frozen part does not change the candidate.
4. **Freeze scope and optimizer groups**:
   - B: all parameter names with `requires_grad` start with `model.correction.`, and they are all of the correction AE parameters. `get_optim_params()` has one group.
   - C: the five main AE modules are added to the above. Two groups, and the second group's `lr` is 1e-5. Every trainable parameter is in exactly one group.
5. **Gradient order and separation**:
   - In B, backward the `policy.forward(batch)` loss once. The `correction.action_out_proj` gradient is nonzero, all other correction parameter gradients are 0, and gradients of parameters outside the correction are None.
   - Take one step with `torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)` and backward again. The gradients of `self_attn.k_proj` in `correction.lm_expert.layers[0]` (self-attention layer), `self_attn.k_proj` in `layers[1]` (cross-attention layer), and `correction.action_in_proj` are nonzero.
   - In C, backward only the mean of `correction_losses`: the five main AE module gradients are None. Clear gradients and backward only the mean of `model.forward`: the correction AE gradients are None.
6. **Cache rollback**: set B's `correction.action_out_proj` to small nonzero random values. Feeding the same (x, t) into `denoise_step(..., head=model.correction)` twice with the `_prefill` result gives the same output. Afterwards, `past_key_values.layers[0].keys.shape[2]` equals the prefix length.
7. **Save and load**:
   - After one training step, `save_pretrained` then `from_pretrained` of B gives the same `correction` weights.
   - Loading the A checkpoint with `from_pretrained` under the enabled config gives a correction AE whose `lm_expert` and three projections equal the main AE, with `action_out_proj` at 0.
   - Loading a B checkpoint file with one `model.correction.` key removed raises `RuntimeError`.

To check on the GPU server:
- Run test 2 once with the real checkpoint (current fine-tuned version). The enabled 5 + 5 variant and the disabled 5-step variant must give the same `sample_actions`.
- After FREEZE variant training, check that all tensors outside `model.correction.` in the resulting checkpoint are the same as the starting checkpoint (`torch.equal`).

## 8. Training variants (step count undecided, run after user instruction)

Keep the training command of the current fine-tuned version as is and change only the flags below.

| Variant | Starting checkpoint | Flags |
|---|---|---|
| FREEZE | Current fine-tuned version | `--policy.path=<current version> --policy.correction_enabled=true --policy.num_steps=5 --policy.correction_num_steps=5 --steps=<STEPS>` |
| Low LR | Current fine-tuned version | FREEZE flags + `--policy.correction_main_lr=1e-5` |
| AE-cont (comparison) | Current fine-tuned version | `--policy.path=<current version> --steps=<STEPS>` (current command as is) |

- Everything else (dataset, batch, schedule, seed) is the same across the three variants.
- The correction group LR is `optimizer_lr` (SmolVLA default 1e-4). The Coda SmolVLA variant was also 1e-4.
- The server's low-LR VLM path is not used in the FREEZE and Low LR variants (VLM frozen). AE-cont uses the current command as is, so it uses that path.
- AE-cont has the same form as Coda's control variant (training the AE more with the same training budget, SmolVLA 10 steps 69.2 vs Coda 72.6). It separates whether the correction AE's gain is just "more training".
- The current version's `config.json` has no new fields, so defaults (off) are used and the flags above override them. The resulting checkpoint stores `num_steps=5`, `correction_num_steps=5`, so it runs as 5 + 5 at deployment.

Settings for inference comparison (real-world evaluation is the user's job):

| Setting | Checkpoint | Denoise |
|---|---|---|
| Current policy | Current fine-tuned version | 10 |
| Main AE only, 5 steps | Current fine-tuned version, `num_steps=5` | 5 |
| FREEZE | FREEZE variant result | 5 + 5 |
| Low LR | Low LR variant result | 5 + 5 |
| AE-cont | AE-cont result | 10 |

"Main AE only, 5 steps" is the same computation with the correction AE removed. Coda also compared the base at 5 steps paired with the correction variant (SmolVLA 60.8 → 72.0).

## 9. Undecided items and cautions

- Undecided: `<STEPS>` and the schedule. Same as the undecided values in the PVI and ROCKET specs (current fine-tuned version values, to be obtained from the user). For reference, the Coda SmolVLA corrector used 18,716 steps, batch 256, LR 1e-4 (4 × 4090).
- Training data is the same demonstrations used to fine-tune the main AE. The Coda π0.5 variant also trained the corrector on the clean50 set used to fine-tune the base (5 steps 71.64 → 74.68).
- Remaining coupling in the Low LR variant: `lerobot_train.py` clips the gradient norm of all parameters together at 10 (`optimizer_grad_clip_norm`). If one side's gradient is large, the other side's step shrinks too [inference]. Left as in the original training.
- In the Low LR variant the main AE changes, so candidate A and the correction target (a⋆ − A) also change every step. All four source papers read (Coda, RoboDual, ActMem-VLA, Hume) either froze the base or trained in separate stages.
- Every difference from Coda follows the user's design: 5-step integration instead of one-shot correction, MSE instead of L1, time condition present, no ε input, training candidates produced with the same 5 steps as inference (the Coda SmolVLA variant trained with 10-step candidates).
- Training cost: each step adds 5 main AE passes without gradients. The FREEZE variant, which has no main AE gradient computation, uses less memory than current training [inference].

## 10. 2026-10-11 decisions and implementation (take priority over the body)

### 10-0. Common (same in all three specs)
- Implementation location: following the no-modification principle for `lerobot/`, implemented as an external extension module `smolvla_ext/` (`attach`, `load_policy`, sidecar `smolvla_ext.json`). The instructions in Section 4 of this document to modify `lerobot` files directly were not followed. Training entry point `train_smolvla_openmani.py --ext-*`, launcher `train_ext.sh`, chain `run_ext_chain.sh`.
- Recipe = d10_new: data `data/lerobot_v3_d9_0912` (place, 18 sentences = 6 objects × 3 locations), 6000 steps, batch 32, 1 GPU per variant (2 variants concurrently), fixed eval of 18 episodes, loss weighting and state noise off. Schedule = 200-step warmup + cosine, floor = peak × 0.025.
- Three start modes (crossed with every module and combination): F = `runs/d10_new/checkpoints/004750` + main AE frozen / L = d10_new + main AE 1e-5 / W = `lerobot/smolvla_base`, main AE 1e-4. Freezing applies only to the five main AE modules (`lm_expert`, `action_in_proj`, `action_time_mlp_in/out`, `action_out_proj`).
- VLM, SigLIP, connector, `state_proj` (rest group) are trained in every variant. LR: 1e-4 for variants including ROCKET, otherwise 1e-5 when starting from d10_new and 1e-4 when starting from web. New modules 1e-4.
- Variant layout (87 variants total): 3 starts × [R 4, P 1, C 2, RP 4, RC 8, PC 2, RPC 8]. ROCKET 4 = 2 VLM layers × 2 α, Coda 2 = 5+5 / 10+10. Settings for other variants are not chosen after seeing one variant's results (everything is crossed in advance). Queue `make_ext_queue.py`.
- Modules are combined (this document's "not merged into one policy" is discarded). Freezing is not a combination constraint. No "no module" control variant.
- Evaluation: eval_loss = main AE flow loss only. Variants are compared with `diag/ext_eval.py` (gt error, draw spread, and seam based on the final deployed output, fixed eval of 18 episodes).
- torch.compile is off for both training and inference. Deferred to a later stage.

### 10-1. Coda
- Steps: two options, 5+5 and 10+10 (user). 10+10 is `--policy.num_steps=10 --ext-coda-steps 10`.
- VLM: "VLM, SigLIP, connector, state_proj frozen" in the Section 3 table is discarded. Coda variants also train the VLM (VLM-frozen variants are excluded entirely, user). The correction loss L_c detaches the prefix K/V so it does not train the VLM (user decision). The VLM is trained only by L_m (and L_align in ROCKET variants).
- Starting points: the original FREEZE/Low LR (d10_new) + web start (W) — F/L/W in Section 10-0.
- The Section 8 AE-cont control variant is not run ("no module" control excluded).
- When combined with PVI, PVI injection goes only to the main AE (not to the correction AE).
- Evaluation: eval_loss does not reflect the correction AE. Checkpoints are compared with the final-output metrics of `diag/ext_eval.py` and the last checkpoint.
- Deployment: the `serve.sh` default of RTC on cannot be used together with Coda (`smolvla_ext/flow.py` assert). Turn RTC off when deploying a Coda variant.
