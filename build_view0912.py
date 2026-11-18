#!/usr/bin/env python3
"""dataset_new0912 -> data/_view0912 (symlink view) + reasoning_0912.json (English labels).

243 episodes, one recorder (metadata person_name=SEOK3), all pick_place. The label file is
`robot_arm_tasks_positioned_0908.csv` (243 rows, 6 objects):

    소주잔 27, 큰테이프 27, 작은테이프 27, 볼펜 54, 일반지우개 54, 커터칼 54

The three long objects carry 54 rows because each is shot twice, 세로 then 가로. Orientation
is NOT in the instruction (the 54-task vocabulary has no word for it), so those two halves
produce the same sentence -- deliberately.

BINDING, and how it was checked before this script was written
--------------------------------------------------------------
The user's account: recorded in CSV order EXCEPT that 작은테이프 was shot first, so its
27-row block moves to the front:

    disk = [작은테이프 27] + [소주잔 27, 큰테이프 27, 볼펜 54, 일반지우개 54, 커터칼 54]

Direction CANNOT test this. Every 27-row block of the CSV has the identical 9-left /
9-right / 9-me pattern, so moving a whole block leaves the direction sequence untouched:
both orderings score the same 92.6% against the base yaw (200/243, thresholds taken from
the data at 1416 / 2587 ticks). The gripper cannot test it either -- id25 runs 2697 (open)
down to 1000 and SATURATES at 1000 in nearly every episode (within-object SD 0.4 ticks),
so the demonstrator squeezes fully shut whatever the object is, and the 0905 grip-width
trick does not transfer.

What did settle it: one frame from disk block 0 and one from disk block 1. Block 0 shows a
small roll of yellow tape, block 1 a dark cup -- i.e. 작은테이프 then 소주잔, the user's
order, not the plain CSV order. Corroborating: the recording gaps longer than four minutes
fall at disk index 81 (exactly the 27x3 boundary) and 136 (next to 135).

NOTE FOR THE RECORD: both frames show a SINGLE object on the board. This recording changed
the camera placement and unified the start/end pose; it did not make the scene multi-object,
so it adds no supervision for telling nouns apart.
"""
import csv, json, sys
from pathlib import Path

SRC = Path("/home/vcl/Desktop/minjun/vla/data/dataset_new0912")
VIEW = Path("/home/vcl/Desktop/minjun/vla/data/_view0912")
LABELS = SRC / "robot_arm_tasks_positioned_0908.csv"
PERSON = "SEOK3"            # must equal metadata.csv person_name: the converter refuses an
                            # episode whose metadata disagrees with the folder
                            # (convert_h5_to_lerobot_v3.py:882)
TASK = "pick_place"
N = 243

# Korean -> the exact English noun already used by the 54 tasks of
# data/lerobot_v3_d8_merged/meta/tasks.parquet. A new string here would mint a duplicate
# task index for a task the policy has already seen.
OBJ = {"테니스공": "tennis ball", "소주잔": "shot glass", "큰테이프": "big tape",
       "작은테이프": "small tape", "휴지": "tissue", "볼펜": "pen",
       "칠판지우개": "blackboard eraser", "일반지우개": "eraser", "커터칼": "knife"}
TEMPLATE = {("pick_place", "left"):  "place the {o} on the left",
            ("pick_place", "right"): "place the {o} on the right",
            ("pick_place", "me"):    "place the {o} in front of me"}

SHOT_FIRST = "작은테이프"

# Dead takes: recorded, labelled, but nothing happens in them. Dropped from the view TOGETHER
# WITH THEIR LABEL ROW -- removing the folder alone would shift every later label by one.
#   20260912_183413 (raw index 157, "place the knife in front of me"): T=76 frames vs a median
#   of 215, base yaw moves 39 ticks vs a median of 714, total arm travel 1929 vs 3992, and the
#   gripper only opens and closes once with no grasp and no release. It does not start from the
#   home pose either. Found by the verifier on 2026-09-13.
EXCLUDE = {"20260912_183413"}


def main() -> int:
    rows = list(csv.DictReader(LABELS.open(encoding="utf-8-sig")))
    if len(rows) != N:
        sys.exit(f"{LABELS.name}: expected {N} rows, got {len(rows)}")

    # reorder: the block shot first moves to the front, everything else keeps CSV order
    first = [r for r in rows if r["object_primary"] == SHOT_FIRST]
    rest = [r for r in rows if r["object_primary"] != SHOT_FIRST]
    if len(first) != 27:
        sys.exit(f"expected 27 {SHOT_FIRST} rows, got {len(first)}")
    ordered = first + rest

    strings = []
    for r in ordered:
        obj = OBJ.get(r["object_primary"])
        tpl = TEMPLATE.get((r["task_subtype"], r["direction"]))
        if obj is None or tpl is None:
            sys.exit(f"row {r['episode_id']}: unmapped {r['object_primary']!r} / "
                     f"{r['task_subtype']!r}+{r['direction']!r}")
        strings.append(tpl.format(o=obj))

    if VIEW.exists():
        sys.exit(f"{VIEW} already exists -- remove it by hand first")
    src_eps = sorted(p for p in (SRC / TASK).iterdir() if p.is_dir())
    if len(src_eps) != N:
        sys.exit(f"{SRC/TASK}: {len(src_eps)} episodes, expected {N}")

    pairs = list(zip(src_eps, strings, strict=True))
    dropped = [(e.name, lab) for e, lab in pairs if e.name in EXCLUDE]
    pairs = [(e, lab) for e, lab in pairs if e.name not in EXCLUDE]
    if len(dropped) != len(EXCLUDE):
        sys.exit(f"EXCLUDE names not all found on disk: {EXCLUDE} vs {[d[0] for d in dropped]}")

    out = VIEW / PERSON / TASK
    out.mkdir(parents=True)
    entries = []
    for e, s in pairs:
        # two folders carry an "_sm" suffix the recorder did not put in metadata.csv
        # (started_at=20260912_161544 vs folder 20260912_161544_sm). The converter compares
        # the two and skips on a mismatch, so the link is named by the metadata stamp.
        # Checked: stripping the suffix collides with nothing and does not reorder the scan.
        name = e.name[:-3] if e.name.endswith("_sm") else e.name
        (out / name).symlink_to(e)
        entries.append({"filename": f"episode_{len(entries)}.hdf5", "language_raw": s,
                        "src": f"{TASK}/{e.name}", "view": f"{PERSON}/{TASK}/{name}"})

    # the converter binds labels[i] to sorted(person, task, timestamp)[i] -- assert the order
    # this script wrote is exactly that order.
    scan = [f"{p.parent.parent.name}/{p.parent.name}/{p.name}"
            for pp in sorted(q for q in VIEW.iterdir() if q.is_dir())
            for tt in sorted(q for q in pp.iterdir() if q.is_dir())
            for p in sorted(q for q in tt.iterdir())]
    if scan != [e["view"] for e in entries]:
        sys.exit("view order != sorted(person, task, timestamp); labels would be misbound")

    (VIEW / "reasoning_0912.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, lab in dropped:
        print(f"  dropped dead take {name} ({lab})")
    print(f"{len(entries)} symlinks, {len(set(s for _, s in pairs))} distinct instructions")
    from collections import Counter
    for s, n in sorted(Counter(lab for _, lab in pairs).items()):
        print(f"  {n:3d}  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
