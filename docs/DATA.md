# Data Structure and Preprocessing (0912 collection, for Base training)

## 1. Raw data (recorder output)
```
data/dataset_new0912/
  robot_arm_tasks_positioned_0908.csv     # instruction CSV, 243 rows, 6 object types
  pick_place/<YYYYMMDD_HHMMSS>/           # one episode
    image_0/ image_1/ image_2/  0001.png ...   # 320x240 RGB, 3 cameras
    joints.csv                               # timestep,t_seconds,id11,id22,id23,id24,id25
    metadata.csv                             # key,value (person_name, task_name ...)
```
- 10 Hz (verified via `t_seconds`). Joint values are **raw Dynamixel ticks** (0~4095, center 2048, 4096/rev).
- Fixed motor order: `id11, id22, id23, id24, id25` (id25 = gripper, open ≈2700 / closed ≈1000).
- 6 object types: shot glass 27, big tape 27, small tape 27, pen 54, eraser 54, utility knife 54
  (the 3 long objects were recorded twice, vertical/horizontal; the instruction text has no orientation). The small tape was recorded first.
- Examples: `data/raw_0912/` holds one raw episode per object, unmodified (list in `data/raw_0912/EPISODES.md`).

## 2. Conversion pipeline
1. `build_view0912.py` : raw → symlink view at `data/_view0912/SEOK3/pick_place/<stamp>` +
   `reasoning_0912.json` (Korean CSV → English instruction `place the <object> <direction>`). This script fixes the mapping between recording order and CSV order.
2. `convert_h5_to_lerobot_v3.py --input-format raw` : view → LeRobot v3.0 dataset
   ```bash
   python convert_h5_to_lerobot_v3.py --input-format raw --input data/_view0912 \
     --out data/lerobot_v3_d9_0912 --repo-id local/d9_0912 \
     --fps 10 --space joint --joint-units tick --joint-action-mode absolute_next \
     --reasoning-json data/_view0912/reasoning_0912.json \
     --lang-level 1 --lang-min-match 0.99
   ```
   (The general form is the same as RUNBOOK section 4. The actual arguments are recorded in `data/lerobot_v3_d9_0912/conversion_report.json`)
3. Validation: `diag/check_direction_labels.py` (instructed direction ↔ final base yaw), `diag/preflight.py`.

## 3. Conversion result `data/lerobot_v3_d9_0912` (v3.0)
- 242 ep / 54,686 frames / 18 tasks (6 objects × 3 directions: left / right / in front of me). 1 bad take excluded.
- Included: `meta/` (info.json, stats.json, tasks.parquet, episodes/), `openmani_meta.json` (units / motor order sidecar),
  `conversion_report.json`. **Not included**: `data/` (frame parquet), `videos/` (AV1 mp4) — listed in `STRUCTURE.txt`.

| feature | shape | definition |
|---|---|---|
| `observation.state` | 5 | row t of `joints.csv`, tick |
| `action` | 5 | **absolute value of state[t+1]** (absolute_next). The last frame holds the last target |
| `observation.images.camera1/2/3` | 240x320x3 | same order as `image_0/1/2`. No resizing |

Camera placement (from the viewpoint of a person facing the robot): camera1 upper-right diagonal of the robot / camera2 front, low / camera3 upper-left diagonal of the robot.
SmolVLA has no camera-identity embedding, so **it runs without error even if the slots are swapped** — checking `/dev/video*` ↔ slot mapping at deployment is mandatory.

## 4. Normalization (training preprocessing)
lerobot policy preprocessor (`checkpoint_configs/*/policy_preprocessor.json`) order:
rename → batch → newline → tokenizer(max 48, pad) → device → **normalizer**.
- state, action: **MEAN_STD**, statistics = `meta/stats.json` (verified to match the checkpoint's normalizer safetensors, error <1.2e-4 tick)
  - action mean [2020.98, 2237.78, 1773.10, 2899.68, 2003.84] / std [352.50, 256.04, 361.64, 155.80, 616.15] tick
- Images: IDENTITY (0~1 float) → resize-with-pad to 512x512 inside SmolVLA.
- state/action are zero-padded to `max_state_dim = max_action_dim = 32`. Loss is computed only on the 5 real channels.
- Output: the postprocessor de-normalizes MEAN_STD → **absolute tick targets** → `openmani4_follower` writes them directly to Goal_Position.
- Known quirk: the checkpoint config still has state shape 6 (lerobot behavior where the smolvla_base value takes precedence).
  Since the 5 channels are zero-padded to 32, this was harmless at runtime (confirmed in both training and deployment).
