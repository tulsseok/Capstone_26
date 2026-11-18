# Example Episodes per Object (raw recording format, 0912 collection)

One per object, all tasks are "... on the right", **episodes belonging to the training split** (excluding the 18 held-out).
Source: `data/dataset_new0912/pick_place/<stamp>/` (copied as-is from the recorder output, no processing)

| object | instruction | episode_index after conversion | stamp | frames |
|---|---|---|---|---|
| small tape | place the small tape on the right | 9 | 20260912_162208 | 215 |
| shot glass | place the shot glass on the right | 36 | 20260912_163949 | 228 |
| big tape | place the big tape on the right | 63 | 20260912_165813 | 218 |
| pen | place the pen on the right | 90 | 20260912_172047 | 226 |
| eraser | place the eraser on the right | 117 | 20260912_174242 | 225 |
| utility knife | place the knife on the right | 144 | 20260912_182613 | 242 |

Episode folder contents:
- `image_0/ image_1/ image_2/` : 320x240 PNG, one per frame → `camera1/2/3` after conversion
- `joints.csv` : `timestep,t_seconds,id11,id22,id23,id24,id25` (10 Hz, raw Dynamixel tick)
- `metadata.csv` : key,value (person_name, task_name, etc.). **Do not trust task_name; use the folder name** (there have been mislabeled cases in the past).

Labels: `robot_arm_tasks_positioned_0908.csv` (243 rows, original instruction CSV) → `build_view0912.py` → `reasoning_0912.json` (English instructions).
If you run `convert_h5_to_lerobot_v3.py --input-format raw --input data/raw_0912 ...` on only these 6,
the labels will be misaligned: `episode_N` in `reasoning_0912.json` is based on the sorted order of all 243, but with only 6 present they get N=0..5.
The converter's validation compares only the verb (place), so **it passes while the object labels are wrong** (inferred from reading the code, not verified by running). **For format illustration only**.
