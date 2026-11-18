# Capstone_26 — SmolVLA + Coda on OpenManipulator (5-DoF)

Fine-tuning SmolVLA (450M, `lerobot/smolvla_base`) on real-robot demonstrations collected on 2026-09-12 ("0912"):

- **Base (`d10_new`)**: SmolVLA fine-tuned on the 0912 dataset.
- **Coda (`ext_C_F_c5`)**: Base plus a corrective action expert.

This repo contains the code, data format, and training configs for the whole pipeline:
**record (robot PC) → convert (LeRobot v3) → train (GPU server) → deploy (server + robot client)**.

## Documentation
| File | Contents |
|---|---|
| [docs/DATA.md](docs/DATA.md) | Raw recording format, conversion, LeRobot v3 layout, normalization |
| [docs/TRAINING.md](docs/TRAINING.md) | Base / Coda training setup, hyperparameters, inference and deployment |
| [docs/CODA_IMPLEMENTATION.md](docs/CODA_IMPLEMENTATION.md) | Coda design notes |
| [data/raw_0912/EPISODES.md](data/raw_0912/EPISODES.md) | 6 sample episodes, one per object |
| [robot_pc/README.md](robot_pc/README.md) | Teleoperation and recording on the robot PC |

## Repository layout

**1. Recording (robot PC)**
```
robot_pc/                      Gamepad / keyboard teleop + recorder → image_0~2/, joints.csv, metadata.csv
```

**2. Data conversion**
```
build_view0912.py              0912 raw episodes → conversion input view + English instruction labels
convert_h5_to_lerobot_v3.py    raw / h5 → LeRobot v3.0 (joint ticks, absolute_next actions)
```

**3. Training**
```
train_smolvla_openmani.py      Training entry point (wraps `lerobot train`: held-out split, probe, ext attach)
train_d10.sh                   Base:  GPU=0 ./train_d10.sh new
train_ext.sh                   Coda:  GPU=1 MODS=C START=F CS=5 ./train_ext.sh
smolvla_ext/                   Runtime extensions (coda.py, flow.py, attach.py, config.py, ...). lerobot is not modified
verify_run.py                  Training log checks
prune_watchdog.py              Checkpoint cleanup
```

**4. Inference and deployment**
```
smolvla_server_openmani.py, serve.sh     GPU inference server (HTTP :8000)
smolvla_client_local.py, client.sh       Robot PC client (3 cameras + Dynamixel)
vlm_done_server.py                       Optional external VLM that decides when the task is done
openmani4_follower/                      Robot class (XM430 ×5, IDs 11,22,23,24,25, raw ticks)
```

**5. Diagnostics and tests**
```
diag/                          preflight, probe, evaluation (ext_eval), attention maps, ...
tests/test_smolvla_ext.py      CPU unit tests
```

**6. Configs and data (metadata only)**
```
checkpoint_configs/            Configs, pre/post-processors, normalization stats for d10_new@4750 and ext_C_F_c5@4000
data/lerobot_v3_d9_0912/       Training dataset: meta/ + sidecar files
data/raw_0912/                 Instruction CSV, English label JSON, 6 full raw sample episodes (PNGs + CSVs)
```

## What is not included
Large files are left out. Where a folder is incomplete, a `STRUCTURE.txt` lists every original file with its size in bytes, so you can see what is missing.

| Excluded | Size | Listed in |
|---|---|---|
| Base weights `model.safetensors` (d10_new @ step 4750) | 907 MB | `checkpoint_configs/d10_new_004750/STRUCTURE.txt` |
| Coda weights `model.safetensors` (ext_C_F_c5 @ step 4000) | 1.1 GB | `checkpoint_configs/ext_C_F_c5_004000/STRUCTURE.txt` |
| Tokenizer files (`tokenizer/`) | 3.5 MB | same as above |
| Dataset frames `data/` (parquet) | 2.4 MB | `data/lerobot_v3_d9_0912/STRUCTURE.txt` |
| Dataset videos `videos/` (3 cameras, mp4) | ~650 MB | `data/lerobot_v3_d9_0912/STRUCTURE.txt` |
| Training logs (`runs/`), rollout videos (`results/`) | — | not listed |

Small processor files (`*processor.safetensors`, ~8 KB each) **are** included, because they hold the normalization stats needed for inference.

## Environment
Tested on conda env `lerobot` (GPU server):

| Package | Version |
|---|---|
| Python | 3.12.13 |
| torch / torchvision | 2.9.1+cu128 / 0.24.1 |
| transformers | 5.5.4 |
| **lerobot** | **0.6.2** (huggingface/lerobot commit `9e14584`, editable install) |
| numpy | 2.2.6 (must be < 2.3) |
| av | 15.1.0 (18.x breaks) |
| datasets / pandas / pyarrow | 4.8.5 / 2.3.3 / 25.0.1 |
| h5py / safetensors / accelerate | 3.16.0 / 0.8.0 / 1.14.0 |

- GPU: RTX PRO 6000 Blackwell (sm_120) → needs a CUDA 12.8 build of torch.
- Robot PC: Linux, Python 3.10, `dynamixel_sdk` (see [robot_pc/requirements.txt](robot_pc/requirements.txt)).

```bash
git clone https://github.com/huggingface/lerobot.git && (cd lerobot && git checkout 9e14584 && pip install -e .)
PYTHONPATH=. python tests/test_smolvla_ext.py        # expected: 30 passed / 1 skipped (vggt)
```

> Scripts contain absolute paths from the original machines (e.g. `/home/vcl/anaconda3/envs/lerobot/bin/python`, `PY=` in `robot_pc/run_gamepad.sh`). Change them for your environment.
