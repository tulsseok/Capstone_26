"""Make an int8 copy of a ROCKET teacher file so it fits the NVMe (user decision 2026-10-11).

Processing (all fixed here, recorded in the output meta.json):
  1. drop the 8x8 grid cells that are padding for the dataset's camera size
     (valid_token_mask(h, w): 16 of 64 cells for 240x320). The alignment loss already masks
     exactly these cells (AlignProjector.valid uses the same function), so nothing the loss reads
     is removed.
  2. per token (one 2048-d vector per frame / layer / camera / cell) symmetric int8:
     scale = max|x| / 127 (scale 1 if the vector is all zero), q = round(x / scale) in [-127, 127].
     Stored: targets_q.npy int8 [N, L, C, K, D], scales.npy float16 [N, L, C, K].
  3. quality, measured on every SAMPLE_EVERY-th frame: cosine(original, dequantised) per token
     (the loss is 1 - cosine), and relative L2 error. Written to meta.json["quantization"].
Read back by smolvla_ext.rocket.TargetFile (dequantised to float16 [L, C, 64, D], zeros in the
dropped cells). The source file is only read.

usage: python -m smolvla_ext.quantize_targets --src /media/vcl/DATA2/vla_vggt/d9_0912_L4-11-17-23 \
           --out data/vla_vggt/d9_0912_L4-11-17-23_int8 [--chunk 128]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from smolvla_ext.rocket import valid_token_mask

SAMPLE_EVERY = 97   # frames between quality samples (arbitrary; ~560 sampled frames for d9)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=128, help="frames per read")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (the math is identical; cuda is faster)")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    meta = json.loads((src / "meta.json").read_text())
    if meta.get("format", "fp16") != "fp16" or not meta.get("complete", False):
        raise SystemExit(f"{src}: need a complete fp16 teacher file")
    filled = np.load(src / "filled.npy")
    if not filled.all():
        raise SystemExit(f"{src}: {int((~filled).sum())} frames unfilled")
    info = json.loads((Path(meta["root"]) / "meta/info.json").read_text())
    hws = {tuple(info["features"][k]["shape"][:2]) for k in meta["camera_keys"]}
    if len(hws) != 1:
        raise SystemExit(f"cameras differ in size: {hws}")
    h, w = hws.pop()
    cells = np.flatnonzero(valid_token_mask(h, w, canvas=int(meta["canvas"]), grid=int(meta["grid"])).numpy())
    t = np.load(src / "targets.npy", mmap_mode="r")
    dev = torch.device(a.device)
    n, L, C, G, D = t.shape
    K = len(cells)
    if out.exists():
        raise SystemExit(f"{out} exists -- not overwriting")
    out.mkdir(parents=True)
    q = np.lib.format.open_memmap(out / "targets_q.npy", mode="w+", dtype=np.int8, shape=(n, L, C, K, D))
    s = np.lib.format.open_memmap(out / "scales.npy", mode="w+", dtype=np.float16, shape=(n, L, C, K))
    print(f"[q8] {src} {t.shape} fp16 -> {out} int8 {q.shape} (cells {K}/{G} for {h}x{w}), "
          f"{q.nbytes / 1e9:.1f} GB", flush=True)
    cos_all, rel_all = [], []
    t0 = time.time()
    for a0 in range(0, n, a.chunk):
        b0 = min(n, a0 + a.chunk)
        xt = torch.from_numpy(np.ascontiguousarray(np.asarray(t[a0:b0])[:, :, :, cells, :])).to(dev).float()
        amax = xt.abs().amax(dim=-1)                                           # [b, L, C, K]
        sct = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).half()
        scft = sct.float()[..., None]                                          # quantise with the STORED scale
        qqt = torch.clamp(torch.round(xt / scft), -127, 127).to(torch.int8)
        qq, sc = qqt.cpu().numpy(), sct.cpu().numpy()
        q[a0:b0] = qq
        s[a0:b0] = sc
        scf = sc.astype(np.float32)[..., None]
        x = None
        for i in range(a0, b0):
            if i % SAMPLE_EVERY == 0:
                xo = xt[i - a0].cpu().numpy().reshape(-1, D)
                xd = (qq[i - a0].astype(np.float32) * scf[i - a0]).reshape(-1, D)
                no, nd = np.linalg.norm(xo, axis=1), np.linalg.norm(xd, axis=1)
                ok = no > 0
                cos_all.append((xo[ok] * xd[ok]).sum(1) / (no[ok] * nd[ok]))
                rel_all.append(np.linalg.norm(xo[ok] - xd[ok], axis=1) / no[ok])
        if (a0 // a.chunk) % 40 == 0:
            el = time.time() - t0
            print(f"[q8] {b0}/{n} frames {el:.0f}s ETA {el / b0 * (n - b0) / 60:.1f} min", flush=True)
    q.flush(); s.flush()
    del q, s
    cos = np.concatenate(cos_all); rel = np.concatenate(rel_all)
    quality = {"sampled_frames": len(cos_all), "sampled_tokens": int(cos.size),
               "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
               "one_minus_cos_p99": float(np.quantile(1 - cos, 0.99)),
               "rel_l2_mean": float(rel.mean()), "rel_l2_max": float(rel.max())}
    shutil.copy2(src / "filled.npy", out / "filled.npy")
    m = dict(meta)
    m.update(format="int8_tokenwise", cells=cells.tolist(), camera_hw=[h, w], source=str(src),
             layout="targets_q [N, n_layers, n_cams, K valid cells, dim] int8; scales [N, n_layers, n_cams, K] fp16",
             quantization={"method": "per-token symmetric absmax int8, scale = max|x|/127 stored fp16",
                           "dropped_cells": f"{G - K} padding cells (loss-masked)",
                           "sample_every": SAMPLE_EVERY, **quality},
             complete=True)
    (out / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"[q8] done {time.time() - t0:.0f}s  quality {json.dumps(quality)}", flush=True)


if __name__ == "__main__":
    main()
