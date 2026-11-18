#!/usr/bin/env python3
"""Decide whether a finished training run is trustworthy, from its LOG -- not its exit code.

`conda run` has returned 0 over a dead child more than once in this project, and several of the
failures that have actually cost us time were silent: a flag that parsed but never reached the
model (camera dropout with no image tensors in the batch), a policy path that fell back to a
randomly initialised VLM, an eval split that quietly came from one half of a merged dataset.
None of those raise. All of them leave -- or fail to leave -- a specific line in the log.

So this checks that the markers the run was SUPPOSED to print are present and agree with the
manifest written at launch, then reports the eval minimum. Read-only.

    python verify_run.py --run-dir runs/d5_G
    python verify_run.py --run-dir runs/d5_G --json-out runs/d5_G/verify.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def eval_curve(log_text: str) -> list[tuple[int, float]]:
    """[(step, eval_loss)]. Uses the eval line's own step: the metric line rounds ("step:5K")
    and reading the step from there is what made the pruner delete the real minima once."""
    out = []
    for line in log_text.replace("\r", "\n").splitlines():
        m = re.search(r"step\s+(\d+)\s*:\s*eval_loss=([\d.]+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None,
                    help="default: <run-dir>.log, then <run-dir>/train.log")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: <run-dir>/run_manifest.json (written by train_d5.sh)")
    ap.add_argument("--require-manifest", action=argparse.BooleanOptionalAction, default=True,
                    help="fail when no run manifest is found (default). The manifest is what "
                         "the flag and eval-split checks compare the log against; without it "
                         "only the five generic checks run and a PASS is nearly meaningless.")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    run = a.run_dir
    # with_name, not with_suffix: with_suffix replaces the last dotted component, so a run
    # directory called "d5_G.v2" would be validated against d5_G.log -- a DIFFERENT run's log.
    _sib = run.with_name(run.name + ".log")
    log = a.log or (_sib if _sib.is_file() else run / "train.log")
    # the manifest is written next to the run dir before training (lerobot refuses a
    # pre-existing output dir) and copied inside afterwards; accept either location.
    cands = [a.manifest] if a.manifest else [run / "run_manifest.json",
                                             run.with_name(run.name + ".manifest.json")]
    man_path = next((c for c in cands if c and c.is_file()), None)
    man = json.loads(man_path.read_text()) if man_path else {}
    # Without a manifest the flag and eval-split checks below are all skipped and the run
    # self-certifies on five generic checks. That is the same "trust the wrapper" failure this
    # script exists to prevent, so a missing manifest is a FAILURE, not a quiet downgrade.
    # --no-require-manifest is for auditing older runs that predate manifests.
    if man_path is None and a.require_manifest:
        print(f"FAIL: no manifest for {run.name} (looked for "
              + ", ".join(str(c) for c in cands if c) + ").\n"
              "      Without it the flag / eval-split checks cannot run and a PASS would mean "
              "only that the log has eval lines and no traceback.\n"
              "      Pass --no-require-manifest to audit a run that predates manifests.")
        return 1

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    if not log.is_file():
        print(f"FAIL: no log at {log}")
        return 1
    text = log.read_text(errors="ignore")

    # 1. the run got far enough to evaluate at all
    curve = eval_curve(text)
    check("eval lines present", len(curve) > 0, f"{len(curve)} evals")
    exp_evals = man.get("expected_evals")
    if exp_evals:
        check("eval count as configured", len(curve) == exp_evals,
              f"{len(curve)} of {exp_evals}")

    # 2. nothing blew up
    for pat, name in [(r"Traceback \(most recent call last\)", "no python traceback"),
                      (r"CUDA out of memory", "no CUDA OOM"),
                      (r"Killed|Out of memory: Killed process", "not OOM-killed"),
                      (r"NaN|nan\b", "no NaN in the log")]:
        hits = len(re.findall(pat, text))
        check(name, hits == 0, f"{hits} hit(s)" if hits else "")

    # 3. the flags actually reached the model
    if man.get("camera_dropout_p", 0) > 0:
        m = re.search(r"\[aug\] camera dropout p=([\d.]+) fill=(\w+)\s+over (\d+) views", text)
        check("camera dropout attached", m is not None,
              f"p={m.group(1)} fill={m.group(2)} views={m.group(3)}" if m else "marker missing")
        if m:
            check("dropout saw all 3 cameras", m.group(3) == "3", f"{m.group(3)} views")
        n = re.search(r"\[aug\] camera dropout: (\d+) views blanked", text)
        check("dropout fired during the run", bool(n) and int(n.group(1)) > 0,
              f"{n.group(1)} views blanked" if n else "no end-of-run count")
        check("dropout is not a no-op", "no observation.images.* tensors" not in text)

    if man.get("smooth_loss_weight", 0) > 0:
        m = re.search(r"\[smooth\].*weight=([\d.]+) order=(\d+)", text)
        check("smoothness penalty attached", m is not None,
              f"weight={m.group(1)} order={m.group(2)}" if m else "marker missing")

    # 3b. smolvla_ext (PVI / Coda / ROCKET): the attached modules are the ones the manifest names
    ext = man.get("ext")
    if ext:
        m = re.search(r"\[ext\] on: modules=(\S+) main_ae=(\w+)(.*)", text)
        check("ext attached", m is not None, m.group(0)[:160] if m else "marker missing")
        if m:
            want = "+".join(n for n in ("pvi", "coda", "rocket") if ext.get(n)) or "none"
            check("ext modules as configured", m.group(1) == want, f"{m.group(1)} vs {want}")
            check("ext main_ae as configured", m.group(2) == ext.get("main_ae", "train"),
                  f"{m.group(2)} vs {ext.get('main_ae', 'train')}")
            rest = m.group(3)
            if ext.get("main_ae") == "lr":
                lr = re.search(r"main_ae_lr=(\S+)", rest)
                check("ext main_ae_lr as configured",
                      bool(lr) and float(lr.group(1)) == float(ext.get("main_ae_lr", 1e-5)),
                      lr.group(1) if lr else "missing")
            if ext.get("coda"):
                cs = re.search(r"coda_steps=(\d+)\+(\d+)", rest)
                check("ext coda steps as configured", bool(cs) and int(cs.group(2)) ==
                      int(ext.get("coda_num_steps", 5)) and int(cs.group(1)) == int(ext.get("main_num_steps", 5)),
                      cs.group(0) if cs else "missing")
            if ext.get("rocket"):
                rp = re.search(r"rocket_pairs=(\S+) alpha=(\S+)", rest)
                check("ext rocket pairs/alpha as configured", bool(rp)
                      and rp.group(1) == ext.get("rocket_pairs") and float(rp.group(2)) == float(ext.get("rocket_alpha")),
                      rp.group(0) if rp else "missing")
                check("ext rocket teacher reader attached",
                      "[ext] rocket teacher targets read in dataloader workers" in text)
        if ext.get("attn_impl"):
            fa = re.search(r"attn=(\w+) compile=(\w+)", m.group(3)) if m else None
            check("ext attn/compile as configured", bool(fa) and fa.group(1) == ext["attn_impl"]
                  and fa.group(2) == ext.get("compile_mode", "off"), fa.group(0) if fa else "missing")
            if ext["attn_impl"] != "eager" or ext.get("compile_mode", "off") != "off":
                check("ext fast options applied", "[ext] fast:" in text)
        check("ext param groups logged", "[ext] trainable groups:" in text)
        loss_lines = re.findall(r"\[ext\] losses main=[\d.]+[^\n(]*", text)
        n_l = len(loss_lines)
        check("ext loss lines present", n_l > 0, f"{n_l} lines")
        # 2026-10-11 (87-run matrix): each loss term appears iff its module is on
        if loss_lines:
            has_c = all("corr=" in l for l in loss_lines); no_c = not any("corr=" in l for l in loss_lines)
            has_a = all("align=" in l for l in loss_lines); no_a = not any("align=" in l for l in loss_lines)
            check("ext corr loss iff Coda", has_c if ext.get("coda") else no_c, loss_lines[-1][:80])
            check("ext align loss iff ROCKET", has_a if ext.get("rocket") else no_a, loss_lines[-1][:80])
        g = re.search(r"\[ext\] trainable groups: rest=\d+@(\S+)", text)
        if g and man.get("optimizer_lr") is not None:
            check("ext rest-group LR as configured", float(g.group(1)) == float(man["optimizer_lr"]),
                  f"{g.group(1)} vs {man['optimizer_lr']}")
        if man.get("base_policy"):
            pp = re.search(r"'pretrained_path': (?:PosixPath\()?'([^']+)'", text)
            check("start checkpoint as configured",
                  bool(pp) and Path(pp.group(1)).as_posix().rstrip("/").endswith(
                      Path(man["base_policy"]).as_posix().rstrip("/")),
                  pp.group(1) if pp else "pretrained_path not in log")
        for key in ("scheduler_decay_lr", "num_steps"):
            if man.get(key) is not None:
                v = re.search(rf"'{key}': ([\d.e+-]+)", text)
                check(f"{key} as configured", bool(v) and float(v.group(1)) == float(man[key]),
                      f"{v.group(1) if v else 'missing'} vs {man[key]}")

    # 4. the eval split is the one we asked for (merged datasets make this load-bearing)
    if man.get("eval_split_mode") or man.get("eval_episodes"):
        m = re.search(r"\[eval-split\] mode=(\w+) seed=(\d+) .*?: (\d+) train / (\d+) eval "
                      r"episodes, (\d+) of (\d+) tasks", text)
        check("eval split hook ran", m is not None,
              f"{m.group(3)} train / {m.group(4)} eval over {m.group(5)}/{m.group(6)} tasks"
              if m else "marker missing")
        if m:
            if man.get("eval_split_mode"):
                check("eval split mode as configured", m.group(1) == man["eval_split_mode"],
                      f"{m.group(1)} vs {man['eval_split_mode']}")
            if man.get("eval_episodes"):
                check("eval episode count as configured",
                      int(m.group(4)) == int(man["eval_episodes"]),
                      f"{m.group(4)} vs {man['eval_episodes']}")
        idx = re.search(r"\[eval-split\] held-out episode indices: \[([^\]]*)\]", text)
        held = [int(x) for x in idx.group(1).split(",") if x.strip()] if idx else []
        check("held-out indices logged", len(held) > 0, f"{len(held)} episodes")
        # on a merged dataset, report how the held-out set splits across the sources
        b = man.get("source_boundary")
        if b and held:
            new = sum(1 for e in held if e >= b)
            check("eval not dominated by one source", new <= 0.5 * len(held),
                  f"{new}/{len(held)} from the appended source")

    # 5. the checkpoint at the eval minimum exists
    ck = run / "checkpoints"
    saved = sorted(int(d.name) for d in ck.iterdir() if d.is_dir() and d.name.isdigit()) \
        if ck.is_dir() else []
    best_step, best_loss = min(curve, key=lambda x: x[1]) if curve else (None, None)
    check("checkpoints on disk", len(saved) > 0, f"{len(saved)}: {saved}")
    if best_step is not None:
        check("eval-minimum checkpoint kept", best_step in saved,
              f"min at step {best_step}" + ("" if best_step in saved else " -- NOT on disk"))
        pm = ck / f"{best_step:06d}" / "pretrained_model"
        check("eval-minimum checkpoint loadable", pm.is_dir()
              and (pm / "config.json").is_file(), str(pm))

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"=== verify {run.name} ({log}) ===")
    for name, ok, detail in checks:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if curve:
        print(f"  eval minimum: {best_loss:.4f} @ step {best_step}   "
              f"(last {curve[-1][1]:.4f} @ {curve[-1][0]})")
    print(f"=== {'PASS' if n_fail == 0 else f'FAIL ({n_fail} check(s))'} ===")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps({
            "run": run.name, "log": str(log), "passed": n_fail == 0,
            "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks],
            "eval_curve": curve,
            "eval_min": {"step": best_step, "loss": best_loss} if curve else None,
            "checkpoints": saved, "manifest": man,
        }, indent=2))
        print(f"wrote {a.json_out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
