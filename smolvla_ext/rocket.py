"""ROCKET: align VLM image tokens with precomputed VGGT aggregator features (SmolVLA_ROCKET_구현.md).

Plan overrides (2026-10-11): student = INPUT of VLM layers (3, 7, 11, 15) paired 1:1 with VGGT
aggregator layers (4, 11, 17, 23); ONE shared AlignProjector; teacher file
`targets.npy` float16 [N, n_pairs, n_cams, 64, 2048] where slot k = VGGT layer pairs[k][1].
2026-10-11: the file may instead be the int8 copy written by `smolvla_ext.quantize_targets`
(meta "format": "int8_tokenwise": `targets_q.npy` int8 [N, L, C, K, D] for the K valid cells only,
`scales.npy` float16 [N, L, C, K], value = q * scale). Readers dequantise to the same float16
[L, C, 64, D] layout with zeros in the dropped (padding, loss-masked) cells.

Teacher targets are read in the dataloader workers (`TeacherDataset`, key `ext.vggt_targets`) and
carried across lerobot's preprocessor by `CarryKeysPre` (the preprocessor's batch->transition
conversion drops unknown top-level keys). `TeacherTargets` (main-process read by batch["index"])
is kept as the fallback described in the spec.

Precompute (needs the vggt package; not installed in the lerobot env, use the symlink):
    PYTHONPATH=third_party python -m smolvla_ext.rocket --root data/lerobot_v3_d9_0912 \
        --out /media/vcl/DATA2/vla_vggt/d9_0912_L4-11-17-23 --batch-size 8 --num-workers 2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

GRID = 8           # prefix image tokens per camera: 8 x 8
CANVAS = 512       # SmolVLA resize_imgs_with_padding
VGGT_SIZE = 518    # VGGT patch 14 x 37
PE_RATIO = 0.1     # ROCKET heads/utils.py::_apply_pos_embed default
VGGT_KEY = "ext.vggt_targets"


class AlignProjector(nn.Module):
    """ROCKET AlignProjector (use_vlm_norm=True, no Matryoshka): LN -> fc1 -> GELU -> fc2. Shared."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        for m in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


def valid_token_mask(h, w, canvas=CANVAS, grid=GRID):
    """[64] bool: 8x8 cell is valid unless its top-left pixel is padding after resize_with_pad
    (left/top padding) of an h x w image into canvas x canvas."""
    ratio = max(w / canvas, h / canvas)
    rh, rw = int(h / ratio), int(w / ratio)
    cell = canvas // grid
    top = math.ceil((canvas - rh) / cell)
    left = math.ceil((canvas - rw) / cell)
    m = torch.zeros(grid, grid, dtype=torch.bool)
    m[top:, left:] = True
    return m.flatten()


def apply_vggt_pos_embed(x):
    """x [n, C, 37, 37] float32 -> x + 0.1 * VGGT uv sin/cos embedding (square input, aspect 1)."""
    from vggt.heads.utils import create_uv_grid, position_grid_to_embed

    h, w = x.shape[-2:]
    grid = create_uv_grid(w, h, aspect_ratio=1.0, dtype=x.dtype, device=x.device)
    pe = position_grid_to_embed(grid, x.shape[1]) * PE_RATIO
    return x + pe.permute(2, 0, 1)[None]


def pool_to_grid(f):
    """f [n, C, 37, 37] -> [n, 64, C]; positional embedding then bilinear (align_corners) to 8 x 8,
    token t = row t // 8, col t % 8 (same order as the SmolVLA prefix image tokens)."""
    f = apply_vggt_pos_embed(f)
    f = F.interpolate(f, size=(GRID, GRID), mode="bilinear", align_corners=True)
    return f.flatten(2).transpose(1, 2)


@torch.no_grad()
def teacher_targets(aggregator, images01, layers):
    """images01: per camera [B, 3, H, W] in [0, 1], un-augmented. -> [B, len(layers), S, 64, C] float32."""
    from lerobot.policies.common.vla_utils import resize_with_pad

    x = [resize_with_pad(img, CANVAS, CANVAS, pad_value=0) for img in images01]
    x = [F.interpolate(img, size=(VGGT_SIZE, VGGT_SIZE), mode="bicubic", align_corners=False) for img in x]
    x = torch.stack(x, dim=1)                                  # [B, S, 3, 518, 518], cameras as one scene
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        tokens, patch_start_idx = aggregator(x)
    out = []
    for li in layers:
        t = tokens[li]
        if t is None:
            raise RuntimeError(f"VGGT aggregator did not return layer {li} (cached_layer_indices)")
        f = t[:, :, patch_start_idx:].float()                  # [B, S, 1369, 2048]
        B, S, P, C = f.shape
        side = int(round(P ** 0.5))
        f = f.reshape(B * S, side, side, C).permute(0, 3, 1, 2)
        out.append(pool_to_grid(f).reshape(B, S, GRID * GRID, C))
    return torch.stack(out, dim=1)


def _read_meta(target_dir, camera_keys, vggt_layers, dim):
    d = Path(target_dir)
    meta = json.loads((d / "meta.json").read_text())
    if list(meta["camera_keys"]) != list(camera_keys):
        raise ValueError(f"camera order mismatch: file {meta['camera_keys']} / policy {list(camera_keys)}")
    if int(meta["dim"]) != int(dim):
        raise ValueError(f"teacher dim mismatch: {meta['dim']} / {dim}")
    if [int(x) for x in meta["vggt_layers"]] != [int(x) for x in vggt_layers]:
        raise ValueError(f"VGGT layer mismatch: file {meta['vggt_layers']} / pairs {list(vggt_layers)}")
    return meta


class TargetFile:
    """Teacher file reader: float16 `targets.npy`, or the int8 token-wise copy (see module doc).
    `get(idx)` -> float16 ndarray [..., L, C, 64, D] for an int or an int array."""

    def __init__(self, target_dir):
        d = Path(target_dir)
        meta = json.loads((d / "meta.json").read_text())
        self.fmt = meta.get("format", "fp16")
        if self.fmt == "int8_tokenwise":
            self.q = np.load(d / "targets_q.npy", mmap_mode="r")
            self.s = np.load(d / "scales.npy", mmap_mode="r")
            self.cells = np.asarray(meta["cells"], dtype=np.int64)
            self.n_cells = int(meta.get("grid", GRID)) ** 2
        elif self.fmt == "fp16":
            self.t = np.load(d / "targets.npy", mmap_mode="r")
        else:
            raise ValueError(f"{d}: unknown teacher format {self.fmt!r}")

    def get(self, idx):
        if self.fmt == "fp16":
            return np.array(self.t[idx], copy=True)
        q = np.asarray(self.q[idx]).astype(np.float32)
        v = (q * np.asarray(self.s[idx]).astype(np.float32)[..., None]).astype(np.float16)
        out = np.zeros(v.shape[:-2] + (self.n_cells, v.shape[-1]), dtype=np.float16)
        out[..., self.cells, :] = v
        return out


class TeacherTargets:
    """Main-process reader by global frame index (batch["index"]). Not an nn.Module."""

    def __init__(self, target_dir, camera_keys, vggt_layers, dim):
        d = Path(target_dir)
        self.meta = _read_meta(d, camera_keys, vggt_layers, dim)
        self.filled = np.load(d / "filled.npy")
        self.targets = TargetFile(d)

    def __call__(self, index):
        idx = index.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if not self.filled[idx].all():
            raise KeyError("a frame in the batch has no teacher target")
        return torch.from_numpy(self.targets.get(idx))


class TeacherDataset(torch.utils.data.Dataset):
    """Wraps the TRAIN dataset: every item gets `ext.vggt_targets` [n_pairs, n_cams, 64, dim] fp16,
    read from the memmap by the item's global `index`. Picklable (the memmap is opened lazily in
    each dataloader worker, spawn-safe). Every other attribute is delegated to the dataset."""

    _OWN = ("dataset", "target_dir", "camera_keys", "vggt_layers", "dim", "_tt", "_filled")

    def __init__(self, dataset, target_dir, camera_keys, vggt_layers, dim):
        self.dataset = dataset
        self.target_dir = str(target_dir)
        self.camera_keys = list(camera_keys)
        self.vggt_layers = [int(x) for x in vggt_layers]
        self.dim = int(dim)
        _read_meta(self.target_dir, self.camera_keys, self.vggt_layers, self.dim)
        filled = np.load(Path(self.target_dir) / "filled.npy")
        if not filled.all():
            raise ValueError(f"{self.target_dir}: {int((~filled).sum())} frames have no teacher target")
        self._tt = None
        self._filled = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        if self._tt is None:
            self._tt = TargetFile(self.target_dir)
            self._filled = np.load(Path(self.target_dir) / "filled.npy")
        gi = int(item["index"])
        if not self._filled[gi]:
            raise KeyError(f"frame {gi} has no teacher target")
        item[VGGT_KEY] = torch.from_numpy(self._tt.get(gi))
        return item

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_tt"] = None
        d["_filled"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, name):
        if name.startswith("__") or name in TeacherDataset._OWN:
            raise AttributeError(name)
        return getattr(self.dataset, name)


class CarryKeysPre:
    """Wraps a lerobot preprocessor: pops `keys` before the pipeline (which would drop them) and puts
    them back afterwards on the batch's device. Everything else is delegated (save_pretrained ...)."""

    def __init__(self, pre, keys=(VGGT_KEY,)):
        self._pre, self._keys = pre, tuple(keys)
        self.carried = 0

    def __call__(self, batch, *a, **kw):
        carry = {}
        if isinstance(batch, dict):
            carry = {k: batch.pop(k) for k in self._keys if k in batch}
        out = self._pre(batch, *a, **kw)
        if carry:
            ref = out.get("observation.state")
            dev = ref.device if torch.is_tensor(ref) else None
            for k, v in carry.items():
                out[k] = v.to(dev, non_blocking=True) if dev is not None else v
            self.carried += 1
        return out

    def __getattr__(self, name):
        if name in ("_pre", "_keys"):
            raise AttributeError(name)
        return getattr(self._pre, name)


def align_loss(proj, hiddens, targets, valid, img_masks):
    """Shared-projector cosine loss.

    hiddens  list (one per pair) of [B, n_cams*64, d_vlm]  (INPUT of the student VLM layer)
    targets  [B, n_pairs, n_cams, 64, dim]                  (slot k = teacher of pair k)
    valid    [n_cams, 64] bool (padding cells False);  img_masks list of [B] bool
    -> per-sample loss [B] (mean over pairs of 1 - masked mean cos), per-pair batch means (floats)
    """
    dev = hiddens[0].device
    t_all = targets.to(dev, torch.float32)
    v = (valid.to(dev)[None] & torch.stack(img_masks, dim=1).to(dev)[:, :, None]).flatten(1).float()
    layer_losses = []
    for k, h in enumerate(hiddens):
        t = F.normalize(t_all[:, k].flatten(1, 2), dim=-1)
        z = F.normalize(proj(h.float()), dim=-1)
        cos = (z * t).sum(-1)
        mean_cos = (cos * v).sum(1) / v.sum(1).clamp_min(1)
        layer_losses.append(1 - mean_cos)
    per_sample = torch.stack(layer_losses).mean(0)
    return per_sample, [float(x.detach().mean()) for x in layer_losses]


# ---------------------------------------------------------------- precompute (GPU, needs vggt) --
def main(argv=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from smolvla_ext.config import DEFAULT_ROCKET_PAIRS

    p = argparse.ArgumentParser(description="Precompute VGGT teacher targets for ROCKET.")
    p.add_argument("--root", required=True, help="LeRobot v3 dataset root")
    p.add_argument("--repo-id", default="local/vggt_precompute")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--camera-keys", default=None, help="comma list; default dataset.meta.camera_keys")
    p.add_argument("--vggt-layers", default=",".join(str(b) for _, b in DEFAULT_ROCKET_PAIRS))
    p.add_argument("--model", default="facebook/VGGT-1B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=0, help="0 = all (timing / smoke)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run (skip filled)")
    p.add_argument("--flush-every", type=int, default=200, help="batches between filled.npy saves")
    args = p.parse_args(argv)

    from vggt.models.vggt import VGGT

    layers = [int(x) for x in args.vggt_layers.split(",") if x.strip()]
    ds = LeRobotDataset(args.repo_id, root=args.root)  # no image_transforms (un-augmented), one frame
    cams = args.camera_keys.split(",") if args.camera_keys else list(ds.meta.camera_keys)
    n = ds.meta.total_frames
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = (n, len(layers), len(cams), GRID * GRID, 2048)
    if args.resume and (out / "targets.npy").is_file():
        targets = np.load(out / "targets.npy", mmap_mode="r+")
        if tuple(targets.shape) != shape:
            raise SystemExit(f"existing targets.npy {targets.shape} != {shape}")
        filled = np.load(out / "filled.npy") if (out / "filled.npy").is_file() else np.zeros(n, bool)
    else:
        if (out / "targets.npy").exists():
            raise SystemExit(f"{out}/targets.npy exists; pass --resume or choose another --out")
        targets = np.lib.format.open_memmap(out / "targets.npy", mode="w+", dtype=np.float16, shape=shape)
        filled = np.zeros(n, dtype=bool)
    meta = {"repo_id": args.repo_id, "root": str(args.root), "camera_keys": cams, "num_frames": n,
            "vggt": args.model, "vggt_layers": layers, "layout": "[N, n_layers, n_cams, 64, dim]",
            "grid": GRID, "canvas": CANVAS, "vggt_size": VGGT_SIZE, "pos_embed_ratio": PE_RATIO,
            "dim": 2048, "dtype": "float16", "complete": False}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # heads are not needed: build the aggregator only, load its weights (unexpected head keys skipped)
    model = VGGT.from_pretrained(args.model, enable_camera=False, enable_point=False,
                                 enable_depth=False, enable_track=False)
    agg = model.aggregator
    agg.cached_layer_indices = set(layers) | {agg.depth - 1}   # make sure every requested layer is kept
    agg = agg.to(args.device).eval()
    del model

    todo = np.nonzero(~filled)[0]
    sub = torch.utils.data.Subset(ds, todo.tolist())
    loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    print(f"[vggt] {len(todo)}/{n} frames to do, layers {layers}, cams {cams}, out {out}", flush=True)
    t0 = time.time()
    done = 0
    for bi, batch in enumerate(loader):
        t = teacher_targets(agg, [batch[k].to(args.device) for k in cams], layers)
        if not torch.isfinite(t).all() or t.abs().max() > 6e4:
            raise ValueError(f"values that do not fit float16 in batch {bi}")
        idx = batch["index"].numpy().astype(np.int64)
        targets[idx] = t.cpu().numpy().astype(np.float16)
        filled[idx] = True
        done += len(idx)
        if bi in (0, 9, 99) or (bi + 1) % args.flush_every == 0:
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"[vggt] batch {bi + 1}: {done} frames in {el:.0f}s ({rate:.1f} fr/s), "
                  f"ETA {(len(todo) - done) / max(rate, 1e-9) / 3600:.2f} h", flush=True)
        if (bi + 1) % args.flush_every == 0:
            targets.flush()
            np.save(out / "filled.npy", filled)
        if args.max_batches and bi + 1 >= args.max_batches:
            break
    targets.flush()
    np.save(out / "filled.npy", filled)
    meta["complete"] = bool(filled.all())
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[vggt] done: {int(filled.sum())}/{n} filled, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
