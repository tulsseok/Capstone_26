#!/usr/bin/env python
"""Static pre-training checks for the SmolVLA (lerobot) stack.

Run this immediately BEFORE a training command. It prints a PASS/WARN/FAIL table
and exits 1 if anything FAILed.

It exists because every failure this project has hit so far was *silent*: the loss
curve looked fine and the robot did not work. The checks below target the six
paths that were confirmed to break without any warning, each grounded in the
installed source:

1. ``load_vlm_weights`` defaults to **False**
   (``policies/smolvla/configuration_smolvla.py``). With ``--policy.type=smolvla``
   (no ``--policy.path``) you train a *randomly initialised* VLM that is also
   frozen. Loss falls normally; vision is noise.
2. Camera-key mismatch. ``modeling_smolvla.py:prepare_images`` builds its list from
   ``config.image_features``; any camera in the batch that is not in the config is
   silently dropped, and any camera in the config that is missing from the batch is
   only substituted when ``empty_cameras > 0``.
3. Silent normalization skip. ``processor/normalize_processor.py:_apply_transform``
   returns the tensor unchanged when ``key not in self._tensor_stats`` -- no log, no
   error. A ``rename_map`` applied to the state key removes normalization entirely.
4. Motor / channel order. ``SmolVLAConfig`` has no ``action_feature_names`` field,
   so ``policies/factory.py:315`` never populates it and
   ``rollout/context.py:_align_state_feature_order`` is a no-op for SmolVLA. If the
   robot's joint order differs from the dataset's, the channels are silently
   permuted at deployment.
5. Chunk padding. With median episode length 34 and ``chunk_size=50`` most chunk
   entries are padding. The loss masks them (``action_is_pad``) so it still looks
   healthy while effective supervision is a fraction of the nominal amount.
6. ``SmolVLAPolicy.from_pretrained(path, chunk_size=16)`` silently keeps 50.
   The working recipe is ``SmolVLAConfig.from_pretrained(path)`` -> set the field ->
   ``SmolVLAPolicy.from_pretrained(path, config=cfg)``.

Example::

    python -m diag.preflight \
        --dataset-root ~/.cache/huggingface/lerobot/me/openmani \
        --policy-path lerobot/smolvla_base \
        --chunk-size 16 --gripper-index -1

    python -m diag.preflight --dataset-root ... --policy-path ... --no-build-policy
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

# ---------------------------------------------------------------------------
# Arbitrary thresholds (hand-chosen, not measured). They only affect WARN/FAIL
# labelling; nothing is modified.
# ---------------------------------------------------------------------------
PAD_RATIO_WARN = 0.30   # arbitrary: warn when >30% of chunk entries are padding
PAD_RATIO_FAIL = 0.60   # arbitrary: fail when >60% are padding
STD_TINY = 1e-6         # arbitrary: a normalization std below this makes MEAN_STD explode
TRAINABLE_FRAC_WARN = 0.05  # arbitrary: warn when <5% of parameters are trainable


@dataclass
class Result:
    name: str
    status: str
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)


class Report:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def add(self, name: str, status: str, detail: str, **extra: Any) -> Result:
        r = Result(name, status, detail, extra)
        self.results.append(r)
        return r

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if r.status == FAIL]

    def render(self) -> str:
        w = max((len(r.name) for r in self.results), default=10)
        lines = ["", "=" * 100, f"{'CHECK'.ljust(w)}  STATUS  DETAIL", "-" * 100]
        for r in self.results:
            first, *rest = (r.detail or "").split("\n")
            lines.append(f"{r.name.ljust(w)}  {r.status:<6}  {first}")
            for extra_line in rest:
                lines.append(f"{' ' * w}          {extra_line}")
        n = {s: sum(1 for r in self.results if r.status == s) for s in (PASS, WARN, FAIL, SKIP)}
        lines += ["-" * 100,
                  f"{n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail, {n[SKIP]} skipped",
                  "=" * 100, ""]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset metadata access.
#
# Preferred path is lerobot's own LeRobotDatasetMetadata. It pulls in `datasets`
# / `pandas` / `pyarrow`, which are an optional extra ('lerobot[dataset]') and are
# NOT installed in every env, so there is a pure-json fallback that reads
# meta/info.json + meta/stats.json directly (the two files preflight needs most).
# Which path was used is always reported.
# ---------------------------------------------------------------------------


class DatasetMeta:
    def __init__(self, root: Path, repo_id: str | None = None):
        self.root = Path(root).expanduser()
        self.repo_id = repo_id or "local/dataset"
        self.backend = "none"
        self.info: dict[str, Any] = {}
        self.stats: dict[str, Any] | None = None
        self.episode_lengths: list[int] | None = None
        self.episode_length_source = "unavailable"
        self.load_errors: list[str] = []
        self._load()

    def _load(self) -> None:
        # Only try the real loader when the files it needs are already on disk. Its
        # constructor falls back to snapshot_download() on any missing file, and a
        # preflight check must never reach for the network.
        complete = ((self.root / "meta" / "info.json").exists()
                    and (self.root / "meta" / "episodes").exists())
        try:
            if not complete:
                raise FileNotFoundError(
                    f"incomplete local meta/ under {self.root} "
                    "(need meta/info.json and meta/episodes/); not going to the Hub")
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

            m = LeRobotDatasetMetadata(self.repo_id, root=self.root)
            self.backend = "LeRobotDatasetMetadata"
            self.info = json.loads((self.root / "meta" / "info.json").read_text())
            self.stats = _numpyless(m.stats)
            if m.episodes is not None:
                cols = set(m.episodes.column_names)
                if "length" in cols:
                    self.episode_lengths = [int(x) for x in m.episodes["length"]]
                    self.episode_length_source = "meta/episodes[length]"
                elif {"dataset_from_index", "dataset_to_index"} <= cols:
                    self.episode_lengths = [
                        int(b) - int(a)
                        for a, b in zip(m.episodes["dataset_from_index"],
                                        m.episodes["dataset_to_index"], strict=True)
                    ]
                    self.episode_length_source = "meta/episodes[dataset_to-from_index]"
            return
        except Exception as e:  # noqa: BLE001 - optional dependency or older layout
            self.load_errors.append(f"LeRobotDatasetMetadata unavailable: {e!r}")

        # ---- fallback: plain json ----
        self.backend = "json-fallback"
        info_p = self.root / "meta" / "info.json"
        if not info_p.exists():
            raise FileNotFoundError(f"{info_p} not found")
        self.info = json.loads(info_p.read_text())
        stats_p = self.root / "meta" / "stats.json"
        self.stats = json.loads(stats_p.read_text()) if stats_p.exists() else None
        # v2.1 layout kept episode lengths in a jsonl; v3.0 uses parquet (needs pyarrow)
        legacy = self.root / "meta" / "episodes.jsonl"
        if legacy.exists():
            lens = []
            for line in legacy.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    if "length" in d:
                        lens.append(int(d["length"]))
            if lens:
                self.episode_lengths = lens
                self.episode_length_source = "meta/episodes.jsonl[length]"
                return
        pq = sorted(glob.glob(str(self.root / "meta" / "episodes" / "**" / "*.parquet"),
                              recursive=True))
        if pq:
            try:
                import pyarrow.parquet as pq_mod

                lens: list[int] = []
                for f in pq:
                    t = pq_mod.read_table(f)
                    if "length" in t.column_names:
                        lens += [int(x) for x in t.column("length").to_pylist()]
                    elif {"dataset_from_index", "dataset_to_index"} <= set(t.column_names):
                        a = t.column("dataset_from_index").to_pylist()
                        b = t.column("dataset_to_index").to_pylist()
                        lens += [int(y) - int(x) for x, y in zip(a, b, strict=True)]
                if lens:
                    self.episode_lengths = lens
                    self.episode_length_source = "meta/episodes/*.parquet"
            except Exception as e:  # noqa: BLE001
                self.load_errors.append(f"episode parquet unreadable: {e!r}")

    # -- convenience ---------------------------------------------------------
    @property
    def features(self) -> dict[str, Any]:
        return self.info.get("features", {})

    @property
    def camera_keys(self) -> list[str]:
        return sorted(k for k, v in self.features.items()
                      if v.get("dtype") in ("video", "image"))

    def feature_names(self, key: str) -> list[str] | None:
        n = self.features.get(key, {}).get("names")
        if isinstance(n, list):
            return [str(x) for x in n]
        if isinstance(n, dict):  # v2 nested {"motors": [...]} style
            for v in n.values():
                if isinstance(v, list):
                    return [str(x) for x in v]
        return None


def _numpyless(obj):
    """Convert numpy scalars/arrays inside a nested dict to plain python."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _numpyless(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpyless(v) for v in obj]
    tolist = getattr(obj, "tolist", None)
    return tolist() if callable(tolist) else obj


# ---------------------------------------------------------------------------
# Padding maths
# ---------------------------------------------------------------------------


def padding_ratio(lengths: list[int], chunk: int) -> tuple[float, int, int]:
    """Exact fraction of action-chunk entries that are padding.

    lerobot samples one chunk per frame: frame i of an episode of length L covers
    steps i..i+chunk-1, and the entries past the end of the episode are padded and
    masked out via ``action_is_pad``. Padded entries for one episode:
    ``sum_{j=1..L} max(0, chunk - j)``.
    """
    total = padded = 0
    for L in lengths:
        total += L * chunk
        padded += sum(chunk - j for j in range(1, min(L, chunk - 1) + 1)) if chunk > 1 else 0
    return (padded / total if total else 0.0), padded, total


def _quantiles(xs: list[int]) -> dict[str, float]:
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return float(s[0])
        i = p * (n - 1)
        lo, hi = math.floor(i), math.ceil(i)
        return float(s[lo] + (s[hi] - s[lo]) * (i - lo))

    return {"min": float(s[0]), "q10": q(0.10), "median": q(0.5), "q90": q(0.90),
            "max": float(s[-1]), "mean": sum(s) / n}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_env(rep: Report) -> None:
    import torch

    rep.add("env.torch", PASS, f"torch {torch.__version__}, cuda_available={torch.cuda.is_available()}, "
                               f"n_gpu={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    # Reproduce lerobot's OWN gate (importlib.metadata based) rather than a plain import:
    # lerobot/datasets/__init__.py calls require_package("datasets", extra="dataset"), which
    # fails on a package that imports fine but has no distribution metadata.
    from lerobot.utils.import_utils import is_package_available

    missing = [m for m in ("datasets", "pandas", "pyarrow") if not is_package_available(m)]
    if missing:
        rep.add("env.dataset_extra", FAIL,
                f"missing {missing} -- lerobot.datasets refuses to import without them, so "
                f"`lerobot-train` cannot load a LeRobotDataset.\n"
                f"fix: pip install 'lerobot[dataset]'")
    else:
        rep.add("env.dataset_extra", PASS, "datasets/pandas/pyarrow present")


def check_policy_source(rep: Report, args) -> None:
    if not args.policy_path:
        rep.add("policy.source", FAIL,
                "no --policy-path given. Training with `--policy.type=smolvla` builds the VLM "
                "from scratch (load_vlm_weights defaults to False) AND freezes it. "
                "Use --policy.path=lerobot/smolvla_base.")
        return
    rep.add("policy.source", PASS, f"pretrained path = {args.policy_path}")


def load_policy_config(args):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    cfg = SmolVLAConfig.from_pretrained(args.policy_path)
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
        cfg.n_action_steps = min(args.n_action_steps or args.chunk_size, args.chunk_size)
    elif args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    for name in ("freeze_vision_encoder", "train_expert_only", "train_state_proj"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    return cfg


def check_vlm_weights(rep: Report, cfg) -> None:
    if getattr(cfg, "load_vlm_weights", False):
        rep.add("policy.load_vlm_weights", PASS, "True (pretrained SmolVLM backbone is loaded)")
    else:
        rep.add("policy.load_vlm_weights", FAIL,
                "False -- the VLM will be RANDOMLY INITIALISED. Combined with "
                f"freeze_vision_encoder={cfg.freeze_vision_encoder}/train_expert_only="
                f"{cfg.train_expert_only} this trains an action expert on noise features "
                "while the loss still decreases. Set --policy.path=lerobot/smolvla_base.")


def check_chunk_kwargs_trap(rep: Report, args, cfg) -> None:
    """`from_pretrained(path, chunk_size=N)` is silently ignored -- verify + document."""
    if args.chunk_size is None:
        rep.add("policy.chunk_override", SKIP, "no --chunk-size requested")
        return
    ok = cfg.chunk_size == args.chunk_size
    rep.add("policy.chunk_override", PASS if ok else FAIL,
            f"chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps} "
            f"(requested {args.chunk_size}).\n"
            "Reminder: SmolVLAPolicy.from_pretrained(path, chunk_size=N) is SILENTLY IGNORED. "
            "The working recipe is SmolVLAConfig.from_pretrained(path) -> cfg.chunk_size=N -> "
            "SmolVLAPolicy.from_pretrained(path, config=cfg).")


def check_processor_files(rep: Report, args) -> None:
    """`lerobot-train` loads policy_preprocessor.json from the checkpoint and hard-fails without it.

    Measured on this machine: ``lerobot/smolvla_base`` DOES ship
    ``policy_preprocessor.json`` / ``policy_postprocessor.json`` on the Hub, but they are
    not in the local HF cache until something downloads them. With ``HF_HUB_OFFLINE=1``
    (or no network) ``make_pre_post_processors`` therefore raises
    ``ProcessorMigrationError``, and the migration command it suggests
    (``migrate_policy_normalization.py``) then fails too, with a 404 on
    ``train_config.json`` which that repo really does not have. Net effect: the error
    message points at a dead end. This check separates "missing on the Hub" from
    "just not cached locally".
    """
    name = "policy.processor_files"
    if not args.policy_path:
        rep.add(name, SKIP, "no --policy-path")
        return
    needed = ["policy_preprocessor.json", "policy_postprocessor.json"]
    p = Path(args.policy_path).expanduser()
    if p.is_dir():
        missing = [f for f in needed if not (p / f).exists()]
        rep.add(name, PASS if not missing else FAIL,
                f"local checkpoint {p}: " +
                ("all processor files present" if not missing else f"missing {missing}"))
        return
    # hub id: look in the local HF cache first, never force a download
    cache_hits = glob.glob(str(Path.home() / ".cache/huggingface/hub"
                               / f"models--{args.policy_path.replace('/', '--')}"
                               / "snapshots/*/policy_preprocessor.json"))
    if cache_hits:
        rep.add(name, PASS, f"found in local HF cache: {cache_hits[0]}")
        return
    if os.environ.get("HF_HUB_OFFLINE"):
        rep.add(name, FAIL,
                f"'{needed[0]}' is NOT in the local HF cache and HF_HUB_OFFLINE is set. "
                "make_pre_post_processors will raise ProcessorMigrationError, and the migration "
                "command it suggests fails too (lerobot/smolvla_base has no train_config.json). "
                "Fetch the processor files once with the network enabled, or unset HF_HUB_OFFLINE.")
        return
    try:
        from huggingface_hub import file_exists

        ok = file_exists(args.policy_path, needed[0])
    except Exception as e:  # noqa: BLE001
        rep.add(name, WARN, f"could not query the Hub for '{needed[0]}': {e!r}")
        return
    rep.add(name, PASS if ok else FAIL,
            f"Hub repo '{args.policy_path}' {'has' if ok else 'does NOT have'} {needed[0]}." +
            ("" if ok else
             "\nlerobot-train will raise ProcessorMigrationError. "
             "migrate_policy_normalization.py also fails on this repo (no train_config.json). "
             "Workaround: build the checkpoint locally -- SmolVLAConfig.from_pretrained(...) -> "
             "SmolVLAPolicy.from_pretrained(..., config=cfg) -> make_pre_post_processors("
             "policy_cfg=cfg, dataset_stats=meta.stats) -> save_pretrained() all three into one "
             "directory, then point --policy.path at that directory."))


def check_features(rep: Report, cfg, meta: DatasetMeta, rename_map: dict[str, str]) -> None:
    cfg_imgs = set(getattr(cfg, "image_features", {}) or {})
    ds_imgs = set(meta.camera_keys)
    ds_renamed = {rename_map.get(k, k) for k in ds_imgs}
    used = sorted(cfg_imgs & ds_renamed)
    only_ds = sorted(ds_renamed - cfg_imgs)
    only_cfg = sorted(cfg_imgs - ds_renamed)
    detail = (f"used by policy ({len(used)}/{len(cfg_imgs)} config cams): {used}\n"
              f"in dataset but NOT in policy config (silently dropped by prepare_images): {only_ds}\n"
              f"in policy config but NOT in dataset (empty_cameras={getattr(cfg, 'empty_cameras', 0)}): "
              f"{only_cfg}")
    detail += ("\nnote: factory.py:311 only fills input_features when they are EMPTY, so a "
               "pretrained checkpoint's camera list WINS over the dataset's. "
               "policies/utils.py:validate_visual_features_consistency only raises when neither "
               "set is a subset of the other -- a dataset that is a strict SUBSET of the "
               "checkpoint's cameras passes validation and trains with fewer cameras silently.")
    if not used:
        rep.add("features.cameras", FAIL, "NO camera key overlaps.\n" + detail)
    elif only_cfg and getattr(cfg, "empty_cameras", 0) < len(only_cfg):
        rep.add("features.cameras", FAIL,
                "config expects cameras the dataset does not have and empty_cameras is too "
                "small to substitute them -> those image tokens are simply absent.\n" + detail)
    elif only_ds or only_cfg:
        rep.add("features.cameras", WARN, detail)
    else:
        rep.add("features.cameras", PASS, detail)

    # state / action dimensionality
    for key, cfg_feat in (("observation.state",
                           (getattr(cfg, "input_features", {}) or {}).get("observation.state")),
                          ("action", getattr(cfg, "action_feature", None))):
        ds_key = {v: k for k, v in rename_map.items()}.get(key, key)
        ds_feat = meta.features.get(ds_key) or meta.features.get(key)
        if ds_feat is None:
            rep.add(f"features.{key}", FAIL, f"dataset has no feature '{ds_key}'")
            continue
        ds_dim = int(ds_feat["shape"][0])
        if cfg_feat is None:
            rep.add(f"features.{key}", FAIL,
                    f"policy config has no '{key}' feature (dataset dim {ds_dim})")
            continue
        cfg_dim = int(cfg_feat.shape[0])
        maxdim = cfg.max_state_dim if key.endswith("state") else cfg.max_action_dim
        msg = (f"dataset dim {ds_dim} vs checkpoint config dim {cfg_dim}; padded to {maxdim} "
               f"(slots {ds_dim}..{maxdim} must stay zero)")
        if ds_dim == cfg_dim:
            rep.add(f"features.{key}", PASS, msg)
        elif key == "action":
            # factory.py:310 unconditionally rebuilds output_features from the dataset
            rep.add(f"features.{key}", WARN, "DIM MISMATCH (auto-corrected): " + msg +
                    "\nfactory.py:310 overwrites output_features from the dataset, so training "
                    "will use the dataset dim. action_out_proj (720->32) is reused as-is.")
        else:
            # factory.py:311 does NOT overwrite a non-empty input_features, so the stale dim
            # survives in the config. Whether that MATTERS depends on the policy: some build
            # layers from config.robot_state_feature.shape[0] (act/diffusion/vqbet/xvla/
            # multi_task_dit -- verified by grep over lerobot/src/lerobot/policies/), while
            # SmolVLA reads batch[OBS_STATE] directly and pads to max_state_dim
            # (modeling_smolvla.py:411 prepare_state), and the normalizer builds its buffers
            # from meta/stats.json rather than from feature shapes.
            consumes_state_shape = getattr(cfg, "type", "") not in ("smolvla", "pi0", "pi05")
            common = "DIM MISMATCH: " + msg + (
                "\nfactory.py:311 only fills input_features when empty, so the checkpoint's "
                f"declared dim {cfg_dim} survives while the data supplies {ds_dim}.")
            if consumes_state_shape:
                rep.add(f"features.{key}", FAIL, common +
                        "\nThis policy BUILDS LAYERS from config.robot_state_feature.shape[0], "
                        "so the stale dim will size a Linear wrongly.")
            else:
                rep.add(f"features.{key}", WARN, common +
                        f"\nHARMLESS AT RUNTIME for policy type '{getattr(cfg, 'type', '?')}': it "
                        "reads batch[OBS_STATE] directly and pads to max_state_dim; the normalizer "
                        "builds buffers from meta/stats.json, not from feature shapes. "
                        "No lerobot code path consumes robot_state_feature for this policy "
                        "(grep: policies/{act,diffusion,vqbet,xvla,multi_task_dit} only).\n"
                        "STILL A HAZARD LATER: the saved checkpoint config will keep the wrong "
                        f"declared dim ({cfg_dim}), so anything at deploy time that trusts it -- "
                        "including this preflight and probe pad-tail bounds -- is off by "
                        f"{abs(cfg_dim - ds_dim)} channel(s).")


def check_channel_order(rep: Report, meta: DatasetMeta, gripper_index: int) -> None:
    """Print the channel order so a human can eyeball it -- nothing can verify it for us."""
    lines = []
    for key in ("observation.state", "action"):
        names = meta.feature_names(key)
        if names is None:
            lines.append(f"{key}: NO 'names' in info.json features -> order is undocumented")
        else:
            gi = gripper_index % len(names)
            lines.append(f"{key}: " + ", ".join(
                f"[{i}]{n}" + ("  <-- gripper" if i == gi else "") for i, n in enumerate(names)))
    lines.append("SmolVLAConfig has NO 'action_feature_names' field, so policies/factory.py:315 "
                 "never records this order in the checkpoint and rollout/context.py cannot "
                 "re-align it at deployment. The robot's motor order MUST match this list "
                 "by construction. VERIFY BY EYE.")
    undocumented = any("NO 'names'" in x for x in lines)
    # Always at least WARN: no automated check can confirm the robot's wiring order.
    rep.add("features.channel_order", FAIL if undocumented else WARN, "\n".join(lines))


def check_stats(rep: Report, cfg, meta: DatasetMeta, rename_map: dict[str, str]) -> None:
    if meta.stats is None:
        rep.add("stats.present", FAIL,
                f"{meta.root / 'meta' / 'stats.json'} is missing. NormalizerProcessor will then "
                "silently pass tensors through unchanged (normalize_processor.py: "
                "`key not in self._tensor_stats -> return tensor`).")
        return
    if not meta.stats:
        rep.add("stats.present", FAIL, "meta/stats.json exists but is EMPTY")
        return
    rep.add("stats.present", PASS, f"{len(meta.stats)} keys in meta/stats.json: "
                                   f"{sorted(meta.stats)[:8]}{' ...' if len(meta.stats) > 8 else ''}")

    from lerobot.configs.types import NormalizationMode

    norm_map = getattr(cfg, "normalization_mapping", {})
    feats = {**(getattr(cfg, "input_features", {}) or {}),
             **(getattr(cfg, "output_features", {}) or {})}
    rows, bad = [], []
    for key, feat in sorted(feats.items()):
        ftype = feat.type
        mode = norm_map.get(ftype if isinstance(ftype, str) else ftype.value,
                            norm_map.get(ftype, NormalizationMode.IDENTITY))
        mode_name = getattr(mode, "value", str(mode))
        # the stats dict is keyed by the *dataset* name; rename_map rewrites the batch keys
        ds_key = {v: k for k, v in rename_map.items()}.get(key, key)
        present = key in meta.stats or ds_key in meta.stats
        effective = "normalized" if (present and mode_name != "IDENTITY") else "PASSED THROUGH"
        rows.append(f"{key:<34} type={ftype if isinstance(ftype, str) else ftype.value:<7} "
                    f"mode={mode_name:<9} stats={'yes' if present else 'NO'} -> {effective}")
        if mode_name != "IDENTITY" and not present:
            bad.append(key)
    detail = "\n".join(rows)
    if bad:
        rep.add("stats.coverage", FAIL,
                f"keys with a non-IDENTITY norm mode but NO stats entry -> normalization is "
                f"silently skipped: {bad}\n" + detail)
    else:
        rep.add("stats.coverage", PASS, detail)

    # degenerate std / non-finite stats
    problems = []
    for key in ("observation.state", "action"):
        ds_key = {v: k for k, v in rename_map.items()}.get(key, key)
        s = meta.stats.get(key) or meta.stats.get(ds_key)
        if not s:
            continue
        for stat_name in ("mean", "std", "min", "max"):
            vals = s.get(stat_name)
            if vals is None:
                problems.append(f"{key}.{stat_name} absent")
                continue
            vals = vals if isinstance(vals, list) else [vals]
            flat = [float(v) for v in _flatten(vals)]
            if any(not math.isfinite(v) for v in flat):
                problems.append(f"{key}.{stat_name} has NaN/Inf")
            if stat_name == "std" and any(abs(v) < STD_TINY for v in flat):
                idx = [i for i, v in enumerate(flat) if abs(v) < STD_TINY]
                problems.append(f"{key}.std has near-zero entries at channels {idx} "
                                f"(constant channel -> MEAN_STD blows the value up)")
    rep.add("stats.sanity", FAIL if problems else PASS,
            "\n".join(problems) if problems else "state/action stats finite, no degenerate std")


def _flatten(x):
    for v in x:
        if isinstance(v, (list, tuple)):
            yield from _flatten(v)
        else:
            yield v


def check_rename_map(rep: Report, rename_map: dict[str, str]) -> None:
    if not rename_map:
        rep.add("processor.rename_map", PASS, "empty")
        return
    risky = {k: v for k, v in rename_map.items()
             if k in ("observation.state", "action") or v in ("observation.state", "action")}
    detail = json.dumps(rename_map)
    if risky:
        rep.add("processor.rename_map", FAIL,
                f"renames a state/action key: {risky}. The stats dict stays keyed by the ORIGINAL "
                "name, so NormalizerProcessor finds no entry for the renamed key and returns the "
                "tensor untouched -- silently unnormalized training.\n" + detail)
    else:
        rep.add("processor.rename_map", WARN,
                "camera-only renames (verify against stats): " + detail +
                "\nfactory.py:395 SKIPS validate_visual_features_consistency entirely when a "
                "rename_map is set, so a camera mismatch will no longer be raised at all.")


def check_chunking(rep: Report, cfg, meta: DatasetMeta) -> None:
    chunk = int(cfg.chunk_size)
    n_act = int(cfg.n_action_steps)
    if meta.episode_lengths:
        q = _quantiles(meta.episode_lengths)
        ratio, padded, total = padding_ratio(meta.episode_lengths, chunk)
        status = FAIL if ratio > PAD_RATIO_FAIL else (WARN if ratio > PAD_RATIO_WARN else PASS)
        rep.add("data.chunk_padding", status,
                f"chunk_size={chunk} n_action_steps={n_act}; episode length "
                f"min/q10/median/q90/max = {q['min']:.0f}/{q['q10']:.0f}/{q['median']:.0f}/"
                f"{q['q90']:.0f}/{q['max']:.0f} (n={len(meta.episode_lengths)}, "
                f"source={meta.episode_length_source})\n"
                f"padding = {padded}/{total} chunk entries = {ratio:.1%} "
                f"(masked by action_is_pad, so the loss hides it)\n"
                f"thresholds are arbitrary: warn>{PAD_RATIO_WARN:.0%}, fail>{PAD_RATIO_FAIL:.0%}")
    else:
        te, tf = meta.info.get("total_episodes"), meta.info.get("total_frames")
        if te and tf:
            mean_len = tf / te
            ratio, _, _ = padding_ratio([round(mean_len)] * min(te, 2000), chunk)
            rep.add("data.chunk_padding", WARN,
                    f"episode lengths unreadable ({meta.episode_length_source}; "
                    f"{'; '.join(meta.load_errors) or 'no reader'}).\n"
                    f"APPROXIMATION from mean length {mean_len:.1f} only: padding ~= {ratio:.1%} "
                    f"at chunk_size={chunk}. Install pyarrow for the exact distribution.")
        else:
            rep.add("data.chunk_padding", SKIP, "no episode length information")


def check_dataset_integrity(rep: Report, meta: DatasetMeta) -> None:
    info = meta.info
    lines = [f"codebase_version={info.get('codebase_version')} robot_type={info.get('robot_type')} "
             f"fps={info.get('fps')}",
             f"total_episodes={info.get('total_episodes')} total_frames={info.get('total_frames')} "
             f"total_tasks={info.get('total_tasks')}",
             f"metadata backend={meta.backend}"]
    by_shape: dict[tuple, list[str]] = {}
    for k in meta.camera_keys:
        by_shape.setdefault(tuple(meta.features[k].get("shape", ())), []).append(k)
    lines.append("camera shapes: " + "; ".join(f"{list(s)} <- {v}" for s, v in by_shape.items()))
    status = PASS
    if len(by_shape) > 1:
        status = WARN
        lines.append("cameras have DIFFERENT resolutions -- SmolVLA resize_with_pad handles it, "
                     "but check the aspect-ratio padding is what you want")
    if not info.get("total_frames") or not info.get("total_episodes"):
        status = FAIL
        lines.append("total_frames/total_episodes missing or zero")
    if meta.load_errors:
        lines += [f"loader note: {e}" for e in meta.load_errors]
    rep.add("data.integrity", status, "\n".join(lines))


def check_trainable_params(rep: Report, args, cfg) -> None:
    """Actually build the policy and count parameters -- the flags alone are not enough."""
    if not args.build_policy:
        rep.add("policy.trainable_params", SKIP, "--no-build-policy")
        return
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=cfg)
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frac = trainable / total if total else 0.0
    groups = {}
    for n, p in policy.named_parameters():
        top = ".".join(n.split(".")[:3])
        g = groups.setdefault(top, [0, 0])
        g[0] += p.numel()
        g[1] += p.numel() if p.requires_grad else 0
    biggest = sorted(groups.items(), key=lambda kv: -kv[1][0])[:6]
    detail = (f"freeze_vision_encoder={cfg.freeze_vision_encoder} "
              f"train_expert_only={cfg.train_expert_only} "
              f"train_state_proj={cfg.train_state_proj}\n"
              f"trainable {trainable:,} / {total:,} = {frac:.2%}\n"
              + "\n".join(f"  {k:<44} {v[1]:>12,} / {v[0]:>12,} trainable" for k, v in biggest))
    rep.add("policy.trainable_params", WARN if frac < TRAINABLE_FRAC_WARN else PASS, detail)

    sp = [p for n, p in policy.named_parameters() if n.startswith("model.state_proj")]
    if sp and not any(p.requires_grad for p in sp):
        rep.add("policy.state_proj_grad", FAIL,
                "state_proj is frozen (train_state_proj=False): proprioception cannot be "
                "learned at all; the expert only sees the VLM stream")
    elif sp:
        rep.add("policy.state_proj_grad", PASS,
                f"state_proj trainable ({sum(p.numel() for p in sp):,} params, "
                f"weight shape {tuple(policy.get_submodule('model.state_proj').weight.shape)})")
    del policy
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True,
                    help="local dataset directory containing meta/info.json")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--policy-path", default=None,
                    help="pretrained policy path/repo (e.g. lerobot/smolvla_base). "
                         "Omitting it is itself a FAIL: it means from-scratch VLM.")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--freeze-vision-encoder", type=_bool, default=None)
    ap.add_argument("--train-expert-only", type=_bool, default=None)
    ap.add_argument("--train-state-proj", type=_bool, default=None)
    ap.add_argument("--rename-map", default=None,
                    help="JSON dict, same value as lerobot-train's --rename_map")
    ap.add_argument("--gripper-index", type=int, default=-1)
    ap.add_argument("--build-policy", action=argparse.BooleanOptionalAction, default=True,
                    help="build the policy on CPU to count trainable params (needs ~2 GB RAM)")
    ap.add_argument("--json-out", default=None, help="also write the full report as JSON")
    args = ap.parse_args()

    rename_map = json.loads(args.rename_map) if args.rename_map else {}
    rep = Report()
    check_env(rep)

    try:
        meta = DatasetMeta(Path(args.dataset_root), args.repo_id)
    except Exception as e:  # noqa: BLE001
        rep.add("data.load", FAIL, f"cannot read dataset metadata at {args.dataset_root}: {e!r}")
        print(rep.render())
        return 1
    check_dataset_integrity(rep, meta)

    check_policy_source(rep, args)
    cfg = None
    if args.policy_path:
        try:
            cfg = load_policy_config(args)
        except Exception as e:  # noqa: BLE001
            rep.add("policy.load_config", FAIL, f"could not load policy config: {e!r}")
    if cfg is not None:
        check_vlm_weights(rep, cfg)
        check_processor_files(rep, args)
        check_chunk_kwargs_trap(rep, args, cfg)
        check_features(rep, cfg, meta, rename_map)
        check_stats(rep, cfg, meta, rename_map)
        check_chunking(rep, cfg, meta)
        try:
            check_trainable_params(rep, args, cfg)
        except Exception as e:  # noqa: BLE001
            rep.add("policy.trainable_params", FAIL, f"could not build policy: {e!r}")
    check_rename_map(rep, rename_map)
    check_channel_order(rep, meta, args.gripper_index)

    print(rep.render())
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            [{"name": r.name, "status": r.status, "detail": r.detail} for r in rep.results],
            indent=2))
        print(f"wrote {args.json_out}")
    if rep.failed:
        print("PREFLIGHT FAILED: " + ", ".join(r.name for r in rep.failed))
        return 1
    print("PREFLIGHT OK (warnings are not blocking -- read them)")
    return 0


def _bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y")


if __name__ == "__main__":
    raise SystemExit(main())
