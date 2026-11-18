# robot_pc — Teleoperation and recording (data collection)

First step of the pipeline: **record (this folder)** → convert (`build_view0912.py`, `convert_h5_to_lerobot_v3.py`) → train.

These are the files actually used on the robot PC. `control_gamepad.py` has not changed since 2026-09-06, so this is the version used for the 0912 recordings.

## Run
Linux, Python 3.10, Dynamixel on `/dev/ttyUSB0`.
```bash
pip install -r requirements.txt
python record_home_pose.py     # once: torque OFF, pose the arm by hand + open gripper → SPACE → home_pose.json
./run_gamepad.sh               # Xbox controller teleop + recording (--swap: swap image_0 ↔ image_1)
python control_keyboard.py     # alternative: mouse + keyboard teleop + recording
```
Edit the `PY=` absolute path in `run_gamepad.sh` for your environment.

## Files
```
control_gamepad.py       Xbox controller teleop + recording. X = stop recording (home-return segment is recorded)
control_keyboard.py      Mouse + keyboard teleop + recording
record_home_pose.py      Measures the home pose → home_pose.json
run_gamepad.sh           Resolves camera order (CAPSTONE_CAMERA_ORDER), then runs control_gamepad.py
home_pose.json           Home pose measured for the 0912 recordings
requirements.txt         Robot PC dependencies
src_control/
  robot_constants.py     Joint IDs (11,22,23,24,25) and limits
  robot_driver.py        Dynamixel read/write
  camera_setup.py        Auto-selects 3 cameras (Linux /dev/video*; DirectShow branch on Windows)
  save_data.py           DatasetRecorder: writes image_0~2/, joints.csv, metadata.csv at 10 Hz
  home_return.py         End-of-episode sequence (see below)
  keyboard_input.py      Input handling for control_keyboard.py
debug_camera*.py         Camera index / combination checks
save_data.md             Output format and environment variables in detail
```

## Output (one episode)
```
<dataset_root>/<person>/<task>/<YYYYMMDD_HHMMSS>/
  image_0/  image_1/  image_2/     320x240 PNG per frame (0 = left, 1 = right, 2 = top)
  joints.csv                       timestep,t_seconds,id11,id22,id23,id24,id25 (raw ticks, 10 Hz)
  metadata.csv                     key,value
```
Tick to radian: `rad = (tick - 2048) * 2π / 4096`, no sign flip.

## End-of-episode sequence (control_gamepad.py)
Pressing X does not stop immediately. The arm runs:
open gripper → return 4 arm joints to home → close gripper fully → hold for 1 s → save.

The whole sequence is recorded, so every episode ends in the same pose.
Rumble: 1× = recording started, 2× = return started, 3× = return done and saved.
If `home_pose.json` is missing, X stops immediately (old behavior).

## Environment variables
| Variable | Default | Purpose |
|---|---|---|
| `CAPSTONE_DATASET_META` | unset | `person,task,object` (used in folder names) |
| `CAPSTONE_DATASET_ROOT` | `dataset` | Output root |
| `CAPSTONE_CAMERA_ORDER` | auto | Force camera indices, e.g. `0,2,4` |
| `CAPSTONE_HOME_POSE` | `home_pose.json` | Home pose file |
| `CAPSTONE_HOME_ARM_SPEED` | `400` | Arm return speed limit (ticks/s) |
| `CAPSTONE_HOME_GRIP_SPEED` | `800` | Gripper speed limit (ticks/s) |
| `CAPSTONE_HOME_HOLD_SEC` | `1.0` | Hold time after arrival (s) |
| `CAPSTONE_HOME_TOL` | `20` | Step completion tolerance (ticks) |
