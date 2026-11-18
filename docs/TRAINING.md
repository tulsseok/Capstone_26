# Training Method and Settings

There are two stages. **(1) Base = d10_new**: fine-tune `lerobot/smolvla_base` on the 0912 data.
**(2) Coda = ext_C_F_c5**: attach a correction action expert (correction AE) on top of d10_new and train further.
The original settings are in `checkpoint_configs/<run>/train_config.json`, `run_manifest.json`, `smolvla_ext.json`.

## 1. Base (d10_new)
```bash
GPU=0 ./train_d10.sh new        # calls train_smolvla_openmani.py internally
```
| item | value |
|---|---|
| initial weights | `lerobot/smolvla_base` (450M, loaded with `--policy.path` — `--policy.type` is forbidden because the VLM would be random) |
| data | `data/lerobot_v3_d9_0912` 242 ep, no augmentation |
| held-out | 1 per task = 18 ep (random split, seed 0). The remaining 224 ep are used for training |
| training scope | full fine-tuning including the VLM (`freeze_vision_encoder=False`, `train_expert_only=False`) |
| steps / batch | 6000 / 32 (≈3.5 epoch), seed 1000 |
| optimizer | AdamW β(0.9, 0.95), eps 1e-8, wd 1e-10, grad clip 10 |
| lr | 1e-4, warmup 1000 → cosine decay (based on 30000 steps, floor 2.5e-6) |
| action | chunk 16, execute 8 (n_action_steps), flow matching Euler 10 step |
| loss | L_m = ‖v − (noise − GT)‖², padded region excluded, 5 real channels |
| selected checkpoint | step 4750 (held-out eval loss 0.0934) |
| time | 1x RTX PRO 6000 (96GB), about 3.2 h per the log (including eval and saving) |

## 2. Coda (ext_C_F_c5) — correction action expert
```bash
GPU=1 MODS=C START=F CS=5 ./train_ext.sh      # naming rule: ext_<module>_<start point>_c<coda steps>
```
Structure (`smolvla_ext/coda.py`, `flow.py`):
- Correction AE = a **copy** of the main AE, with only the output projection **zero-initialized** → at the start of training the output is identical to the original Base.
- The VLM prefix K·V is shared by both AEs.
- Inference: the main AE integrates 5 steps from noise to get candidate A → the correction AE **starts from A** and integrates 5 more steps (5+5 total).
- Loss: L = L_m + L_c (weights 1:1)
  - L_c = ‖v_c(x_t, t) − (A − GT)‖², x_t = t·A + (1−t)·GT. A and the prefix K·V are no_grad → **L_c trains only the correction AE**.

| item | value |
|---|---|
| initial weights | d10_new step 4750 |
| main AE | **frozen** (F = freeze) |
| lr | rest including VLM 1e-5 (floor 2.5e-7) / new module (correction AE) 1e-4 |
| Euler step | main 5 + correction 5 (`--policy.num_steps=5 --ext-coda-steps 5`) |
| rest | same as Base (6000 step, batch 32, seed 1000, same held-out 18 ep) |
| selected checkpoint | step 4000 (eval loss 0.0944) |
| parameters | about 550M (including the added correction AE) |
| time | about 2.3 h per the log |
| actual flags | see `flags` in `run_manifest.json` (includes `--ext-attn sdpa --ext-compile layers`) |

Offline evaluation (held-out, Base → Coda): joint gt error 28.0 → 25.7 tick, spread across draws 21.8 → 17.3 tick
(`diag/ext_eval.py`, result files not included in the repo).

## 3. Inference / deployment
- GPU server: `./serve.sh` (`smolvla_server_openmani.py`, port 8000). Pick a MODEL/DATA line and uncomment it.
  When the Coda checkpoint is used with RTC, the `--coda-rtc` placement must be specified (see the comments in serve.sh).
- Robot PC: `./client.sh "<instruction>" [--dry-run]` (`smolvla_client_local.py`, `openmani4_follower/`). The `SERVER` IP must be edited.
- 10 Hz control; out of the 16 predicted steps, the first 8 are executed and then it re-predicts. One prediction takes about 90 ms.
- Tests (CPU): `PYTHONPATH=. python tests/test_smolvla_ext.py` — judged by PASS/FAIL lines. Some ROCKET tests require the vggt package.

## 4. Code version notes
- The training scripts and `train_ext.sh` are identical to those at the time of Coda training (sha match). d10_new was trained with an earlier version of
  `train_smolvla_openmani.py` (sha 882b62b178d0). `smolvla_ext/attach.py` and others are later revisions.
- `smolvla_ext/` also contains PVI and ROCKET code (kept as-is because they are intertwined in the same files). **The target of this repo is Coda only**, and
  they are turned off with `pvi=false, rocket=false` in `smolvla_ext.json`.
- lerobot was not modified. All extensions are runtime attaches (`smolvla_ext/attach.py`).
