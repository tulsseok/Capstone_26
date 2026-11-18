#!/usr/bin/env python
"""Read-only check: does each episode's language label agree with what the arm did?

The language JSON is matched to episodes *by position* (sorted person/task/timestamp).
That mapping cannot fail loudly -- a wrong offset silently trains every episode on the
wrong sentence -- so this cross-checks it against the joints, which are independent of
the JSON entirely.

Two signals, both from joints.csv only:
  direction  id11 is the base yaw. Use its FINAL ABSOLUTE value, not the change over
             the episode: the object starts somewhere on a 3x3 grid, so "push the pen
             to the right" can rotate the base either way depending on where the pen
             began, while the place it ENDS at is fixed by the target zone. Measured
             on dataset_new0822, end-start agreed with the label only 78% of the time
             while the final angle separates the three zones with no p10-p90 overlap.
             We do not assume which end is left; we only require the three groups to
             separate, and report the convention we inferred.
  task       "place" grips (id25 travels), "push" does not. This already caught a
             mislabelled folder in the previous dataset.

Nothing is written and no image is opened. Exit 1 if a check fails.
"""
import argparse, csv, json, os, sys
import numpy as np

def episodes(root):
    out = []
    for p in sorted(os.listdir(root)):
        pd = os.path.join(root, p)
        if not os.path.isdir(pd):
            continue
        for t in sorted(os.listdir(pd)):
            td = os.path.join(pd, t)
            if not os.path.isdir(td):
                continue
            for ts in sorted(os.listdir(td)):
                out.append((p, t, ts, os.path.join(td, ts)))
    return out

def joints(path):
    with open(os.path.join(path, "joints.csv")) as f:
        rows = list(csv.DictReader(f))
    a = np.array([[int(r[k]) for k in ("id11", "id22", "id23", "id24", "id25")] for r in rows],
                 dtype=np.int64)
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--min-dir-acc", type=float, default=0.90,
                    help="minimum nearest-median accuracy of final base yaw -> direction word")
    ap.add_argument("--grip-travel-min", type=float, default=200.0,
                    help="ticks of id25 travel below which an episode counts as 'no grasp'")
    a = ap.parse_args()

    labels = [x["language_raw"] for x in json.load(open(a.labels))]
    eps = episodes(a.root)
    if len(eps) != len(labels):
        print(f"[FAIL] {len(eps)} episodes vs {len(labels)} labels -- positional mapping impossible")
        return 1
    print(f"episodes={len(eps)}  labels={len(labels)}")

    fyaw, griptravel, dirs, verbs = [], [], [], []
    for (_p, _t, _ts, path), lang in zip(eps, labels):
        j = joints(path)
        fyaw.append(float(j[-1, 0]))
        griptravel.append(float(j[:, 4].max() - j[:, 4].min()))
        low = lang.lower()
        dirs.append("left" if "left" in low else "right" if "right" in low
                    else "front" if ("front" in low or "toward" in low) else "?")
        verbs.append("place" if low.startswith("place") else "push" if low.startswith("push") else "?")
    fyaw, griptravel = np.array(fyaw), np.array(griptravel)
    dirs, verbs = np.array(dirs), np.array(verbs)

    ok = True
    print("\n== 1. final base yaw (id11 at the last frame), by direction word ==")
    for v in ("place", "push"):
        cen = {}
        print(f"  -- {v} --")
        for d in ("left", "right", "front"):
            m = (dirs == d) & (verbs == v)
            if not m.any():
                continue
            cen[d] = float(np.median(fyaw[m]))
            print(f"    {d:<6s} n={m.sum():4d}  median {cen[d]:8.1f}  "
                  f"p10 {np.percentile(fyaw[m],10):8.1f}  p90 {np.percentile(fyaw[m],90):8.1f}")
        if len(cen) < 3:
            continue
        # nearest-median classification: if the labels were offset, this collapses to chance
        m = verbs == v
        pred = np.array([min(cen, key=lambda d: abs(x - cen[d])) for x in fyaw[m]])
        acc = float((pred == dirs[m]).mean())
        # An offset that is a multiple of 9 keeps the direction<->angle map bijective, so the
        # separation test above passes on a mirrored dataset. Verified 2026-08-23: k=9/18/27
        # rotations of the label list all scored 97-99% here. Pin the absolute convention.
        #   lower id11 = the user's LEFT. Evidence: (i) in camera2, which sits at the user's
        #   position looking at the robot, "left" episodes put the arm on the image's left;
        #   (ii) the frame-difference blob is 2.5x larger in camera1 for "left" and 2x larger
        #   in camera3 for "right", i.e. those two cameras flank the workspace; (iii) the D2
        #   model, trained on this same label list, drove the real arm the way the operator
        #   asked -- a global mirror would have been obvious there.
        if cen["left"] >= cen["right"]:
            print(f"    [FAIL] left ({cen['left']:.0f}) is not at a LOWER id11 than right "
                  f"({cen['right']:.0f}) -- the labels are mirrored. A separation-only check "
                  f"cannot see this; that is why the convention is pinned here.")
            ok = False
        if acc < a.min_dir_acc:
            print(f"    [FAIL] the final angle predicts the label only {acc:.1%} of the time "
                  f"(chance 33.3%) -- the label offset is wrong")
            ok = False
        else:
            side = "left = lower id11" if cen["left"] < cen["right"] else "left = higher id11"
            print(f"    [PASS] final angle predicts the label {acc:.1%} of the time ({side})")
        if not (min(cen["left"], cen["right"]) < cen["front"] < max(cen["left"], cen["right"])):
            print(f"    [WARN] 'front' ({cen['front']:.0f}) does not sit between left and right")

    print("\n== 2. gripper travel (id25 max-min), by verb ==")
    for v in ("place", "push"):
        m = verbs == v
        print(f"  {v:<6s} n={m.sum():4d}  median {np.median(griptravel[m]):8.1f}  "
              f"frac>{a.grip_travel_min:.0f}t = {(griptravel[m] > a.grip_travel_min).mean():.1%}")
    pl, pu = verbs == "place", verbs == "push"
    if np.median(griptravel[pl]) <= np.median(griptravel[pu]) + a.grip_travel_min:
        print("  [FAIL] 'place' episodes do not grip noticeably more than 'push' ones")
        ok = False
    else:
        print("  [PASS] 'place' grips, 'push' does not")

    print(f"\n== {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} ==")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
