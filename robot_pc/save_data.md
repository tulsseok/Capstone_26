# Dataset Storage Structure

> Source reference file: `src_control/save_data.py`

---

## Configuration constants

| constant | current value | description |
|------|---------|------|
| `CAPSTONE_PERSON_NAME` | `"unknown_person"` | person name (reflected in directory name) |
| `CAPSTONE_TASK_NAME` | `"Pick and Place"` | task name (reflected in directory name) |
| `CAPSTONE_OBJECT_NAME` | `"unknown_object"` | object name (reflected in directory name) |
| `CAPSTONE_DATASET_META` | unset | set `person,task,object` at once |
| `CAPSTONE_DATASET_ROOT` | `"dataset"` | root storage path |
| `RECORD_FPS` | `10` | saving sample rate (Hz) |
| `JOINT_IDS` | `(11, 22, 23, 24, 25)` | order of joint IDs recorded in the CSV |
| `JOINTS_CSV_NAME` | `"joints.csv"` | CSV file name |
| `METADATA_CSV_NAME` | `"metadata.csv"` | session metadata file name |

---

## Directory structure

```
dataset/
└── <person_name>/
    └── <task_name>/
        └── <object_name>/
            └── <YYYYMMDD_HHMMSS>/
                ├── image_0/
                │   ├── 0001.png
                │   ├── 0002.png
                │   └── ...
                ├── image_1/
                │   └── <timestep:04d>.png
                ├── image_2/
                │   └── <timestep:04d>.png
                ├── joints.csv
                └── metadata.csv
```

- `image_0` ~ `image_2`: camera index order (`cap1`, `cap2`, `cap3`)
- File name format: `{timestep:04d}.png` (0001.png ~ 9999.png)
- `None` frames are skipped (no file created)

Example run:

```bash
CAPSTONE_DATASET_META=minjun,pick_and_place,cup DYNAMIXEL_PORT=/dev/ttyUSB0 python control_gamepad.py
```

---

## joints.csv columns

| column | type | description |
|------|------|------|
| `timestep` | int | 1-based frame index |
| `t_seconds` | float | `(timestep - 1) / RECORD_FPS` — relative time from 0 s |
| `id11` | int | base joint Dynamixel position value |
| `id22` | int | shoulder joint Dynamixel position value |
| `id23` | int | elbow joint Dynamixel position value |
| `id24` | int | wrist joint Dynamixel position value |
| `id25` | int | gripper joint Dynamixel position value |

---

## Sampling rules

| condition | behavior |
|------|------|
| `recording == False` | skip |
| any joint is `-1` (communication failure) | skip the entire timestep (no images or CSV row) |
| `perf_counter` elapsed < `1/RECORD_FPS` | skip (wait for next loop) |
| all of the above conditions pass | timestep +1, 3 PNGs + 1 CSV row flushed immediately |

---

## Session lifecycle

```
toggle_recording() called (X button / X key)
  → create session directory + image_0~2 folders
  → open joints.csv + write header row
  → recording = True

every loop → maybe_record_frame()

toggle_recording() called again / Ctrl+C / 'q'
  → finalize() or _close_session()
  → close CSV file
  → recording = False
```
