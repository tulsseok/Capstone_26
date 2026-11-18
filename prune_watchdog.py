#!/usr/bin/env python3
"""Delete superseded checkpoints WHILE a run is training, and time the run as it goes.

`prune_checkpoints.py` runs after a run finishes, so the peak disk cost is the full set:
saving every 250 steps for 6000 steps is ~58 GB. On a root filesystem at 93% that is the
difference between a run finishing and a run dying at step 4000. This keeps only the
eval-minimum checkpoint plus the newest one, from the outside -- it never touches the
training process, so it can be attached to a run that is ALREADY going.

Two rules make deleting-while-writing safe:
  * only steps strictly below the highest step present are considered; the newest directory
    may still be mid-write, and it is also what `checkpoints/last` points at.
  * the current eval minimum is never deleted, and the minimum is read from the eval line's
    own step -- the metric line rounds it ("step:5K"), which is what made the offline pruner
    delete the real minima of the d3 runs once.

It also reports step time and swap, because the answer to "is this run slower than the last
one" changes when the machine is swapping, and 14 GB of swap was in use when this was written.

    python prune_watchdog.py --run-dir runs/d5_G --dry-run       # watch, delete nothing
    python prune_watchdog.py --run-dir runs/d5_G                 # watch and delete
"""
from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path


def eval_curve(text: str) -> list[tuple[int, float]]:
    return [(int(m.group(1)), float(m.group(2)))
            for m in re.finditer(r"step\s+(\d+)\s*:\s*eval_loss=([\d.]+)", text)]


def step_times(text: str) -> tuple[list[float], list[float]]:
    return ([float(m.group(1)) for m in re.finditer(r"step_s:([\d.]+)", text)],
            [float(m.group(1)) for m in re.finditer(r"data_s:([\d.]+)", text)])


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def swap_used_gb() -> tuple[float, float]:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        mem[k] = float(v.strip().split()[0]) / 1024 / 1024
    return (mem["SwapTotal"] - mem["SwapFree"], mem["MemAvailable"])


def alive(pattern: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None, help="default: <run-dir>.log")
    ap.add_argument("--interval", type=int, default=120, help="poll seconds (default 120)")
    ap.add_argument("--proc-pattern", default=None,
                    help="pgrep pattern that means the run is still alive; default is the "
                         "job name derived from --run-dir. The watchdog exits one poll after "
                         "it stops matching.")
    ap.add_argument("--dry-run", action="store_true", help="report, delete nothing")
    a = ap.parse_args()

    run = a.run_dir
    log = a.log or run.with_name(run.name + ".log")
    ck = run / "checkpoints"
    # bracket the first character so this pattern cannot match the watchdog's own command line
    name = run.name
    if len(name) < 2 and a.proc_pattern is None:
        raise SystemExit(f"run directory name {name!r} is too short to build a self-excluding "
                         f"pgrep pattern from; pass --proc-pattern explicitly.")
    pat = a.proc_pattern or f"job_name={name[0]}[{name[1]}]{name[2:]}"

    print(f"[watchdog] run={run} log={log} interval={a.interval}s "
          f"{'DRY RUN' if a.dry_run else 'deleting'}")
    print(f"[watchdog] liveness pattern: {pat!r} -> alive={alive(pat)}")

    freed = 0
    last_report = 0.0
    max_seen_len = 0
    log_trusted = True
    while True:
        time.sleep(a.interval)
        try:
            running = alive(pat)
            text = log.read_text(errors="ignore") if log.is_file() else ""
        except OSError as e:
            print(f"[watchdog] poll failed ({type(e).__name__}: {e}); retrying")
            continue
        curve = eval_curve(text)
        st, dt = step_times(text)

        # A log that SHRANK cannot be trusted to still contain the global minimum: the curve
        # is rebuilt from the whole file each poll, so a truncated or rotated log would move
        # `best_step` onto the surviving tail and the real minimum would be deleted.
        if len(text) < max_seen_len:
            print(f"[watchdog] log shrank ({max_seen_len} -> {len(text)} bytes): it was "
                  f"truncated or rotated, so the eval curve is no longer complete. "
                  f"Deleting nothing from here on.", flush=True)
            log_trusted = False
        max_seen_len = max(max_seen_len, len(text))

        have = sorted(int(d.name) for d in ck.iterdir()
                      if d.is_dir() and d.name.isdigit()) if ck.is_dir() else []
        best_step = best_loss = newest = None
        if curve:
            best_step, best_loss = min(curve, key=lambda x: x[1])
            newest = max(have) if have else None
            # `checkpoints/last` is repointed AFTER the new directory is written
            # (lerobot_train.py:801 writes, :824 relinks), so for the length of one save it
            # still targets the PREVIOUS step. Deleting that leaves a dangling `last` and
            # breaks --resume, so resolve the link and keep whatever it points at.
            linked = None
            try:
                lt = (ck / "last").resolve()
                if lt.name.isdigit():
                    linked = int(lt.name)
            except OSError:
                linked = None
            keep = {best_step, newest, linked}
            drop = [s for s in have if s not in keep]
            for s in drop:
                d = ck / f"{s:06d}"
                if not log_trusted:
                    print(f"[watchdog] would drop {d.name} but the log is no longer trusted")
                    continue
                try:
                    n = dir_bytes(d)
                    if a.dry_run:
                        print(f"[watchdog] would drop {d.name} ({n/2**30:.2f} GB)")
                    else:
                        shutil.rmtree(d)
                        freed += n
                        print(f"[watchdog] dropped {d.name} ({n/2**30:.2f} GB), "
                              f"freed {freed/2**30:.1f} GB total")
                except OSError as e:
                    # a checkpoint being written, or already gone: never a reason to die and
                    # leave the run accumulating 58 GB of checkpoints unattended
                    print(f"[watchdog] could not drop {d.name}: {type(e).__name__}: {e}")

        now = time.time()
        if now - last_report > max(a.interval, 300) - 1:
            last_report = now
            sw, avail = swap_used_gb()
            tail = st[-20:]
            msg = (f"[watchdog] evals={len(curve)} step={curve[-1][0] if curve else 0} "
                   f"best={best_loss:.4f}@{best_step} " if curve else "[watchdog] no evals yet ")
            if tail:
                msg += (f"| step_s median {statistics.median(tail):.3f} "
                        f"(data_s {statistics.median(dt[-20:]):.3f}) ")
            msg += f"| kept {sorted(have)} | swap {sw:.1f}GB avail {avail:.1f}GB"
            print(msg, flush=True)

        if not running:
            print(f"[watchdog] training process is gone; final kept={sorted(have)}, "
                  f"freed {freed/2**30:.1f} GB")
            if st:
                print(f"[watchdog] step_s over the whole run: median "
                      f"{statistics.median(st):.3f} mean {statistics.fmean(st):.3f} "
                      f"max {max(st):.3f} over {len(st)} logged steps")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
