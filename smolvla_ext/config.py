"""Extension settings. Saved as a sidecar `smolvla_ext.json` next to `config.json` in every
saved `pretrained_model/` directory. `SmolVLAConfig` / `config.json` are never touched.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

SIDECAR_NAME = "smolvla_ext.json"
SIDECAR_VERSION = 1

# where RTC guidance may act on a Coda checkpoint (inference only, see ExtConfig.coda_rtc)
CODA_RTC_MODES = ("off", "main", "correction", "both")

# The main action expert ("main AE"): the expert transformer and its four in/out projections.
# Names relative to VLAFlowMatching (policy.model).
MAIN_AE_MODULES = (
    "vlm_with_expert.lm_expert",
    "action_in_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
    "action_out_proj",
)

# Checkpoint key prefix of each new module (relative to the policy).
EXT_PREFIX = {"pvi": "model.bypass.", "coda": "model.correction.", "rocket": "model.align_proj."}

DEFAULT_ROCKET_PAIRS = ((3, 4), (7, 11), (11, 17), (15, 23))


def parse_pairs(text: str) -> list[tuple[int, int]]:
    """"3:4,7:11" -> [(3, 4), (7, 11)]  (VLM layer whose INPUT is aligned : VGGT aggregator layer)."""
    out = []
    for tok in text.replace(" ", "").split(","):
        if not tok:
            continue
        a, b = tok.split(":")
        out.append((int(a), int(b)))
    if not out:
        raise ValueError(f"no layer pairs in {text!r}")
    return out


@dataclass
class ExtConfig:
    pvi: bool = False
    coda: bool = False
    rocket: bool = False
    main_ae: str = "train"          # train | freeze | lr
    main_ae_lr: float = 1e-5        # used only when main_ae == "lr"
    new_lr: float = 1e-4            # lr of every new module
    coda_num_steps: int = 5         # correction AE Euler steps (main AE uses policy.config.num_steps)
    # WHERE the RTC guidance acts when this checkpoint is served with --rtc (2026-10-17).
    # INFERENCE ONLY: it changes no training numerics and no weight. "off" is today's behaviour
    # (Coda + RTC refuses). main = guidance on the main AE only, so the correction AE then moves
    # the overlap freely; correction = the final word is guided; both = guided twice.
    # The sidecar value is NEVER trusted at inference: load_policy() forces it back to "off"
    # unless the caller passes coda_rtc=..., and the server overwrites it on every start from
    # --coda-rtc (same pattern as compile_mode, attach.py).
    coda_rtc: str = "off"           # off | main | correction | both
    rocket_pairs: list = field(default_factory=lambda: [list(p) for p in DEFAULT_ROCKET_PAIRS])
    rocket_alpha: float = 0.125
    rocket_target_dir: str | None = None
    rocket_dim: int = 2048
    # (H, W) of the RAW camera images per image key, BEFORE SmolVLA's resize_with_pad. Needed for
    # the padding masks (PVI patches, ROCKET 8x8 cells). config.image_features cannot be used: the
    # d10_new checkpoint says (3, 256, 256) while the d9 frames are 240 x 320 (checked 2026-10-11).
    camera_hw: dict | None = None
    # informational, filled by attach(): main AE Euler steps at attach time (Coda = 5 + 5)
    main_num_steps: int | None = None
    # speed options (smolvla_ext/fast.py, 2026-10-11). attn_impl changes numerics slightly and is
    # kept in the sidecar so inference matches training; compile_mode only changes speed.
    attn_impl: str = "eager"        # eager | sdpa
    compile_mode: str = "off"       # off | layers | full
    version: int = SIDECAR_VERSION

    def __post_init__(self):
        if self.main_ae not in ("train", "freeze", "lr"):
            raise ValueError(f"main_ae must be train|freeze|lr, got {self.main_ae!r}")
        self.rocket_pairs = [[int(a), int(b)] for a, b in self.rocket_pairs]
        vl = [a for a, _ in self.rocket_pairs]
        if vl != sorted(set(vl)):  # student hiddens are captured in ascending layer order
            raise ValueError(f"rocket_pairs VLM layers must be strictly ascending, got {vl}")
        if self.coda_num_steps < 1:
            raise ValueError("coda_num_steps must be >= 1")
        if self.coda_rtc not in CODA_RTC_MODES:
            raise ValueError(f"coda_rtc must be {'|'.join(CODA_RTC_MODES)}, got {self.coda_rtc!r}")
        if self.attn_impl not in ("eager", "sdpa"):
            raise ValueError(f"attn_impl must be eager|sdpa, got {self.attn_impl!r}")
        if self.compile_mode not in ("off", "layers", "full"):
            raise ValueError(f"compile_mode must be off|layers|full, got {self.compile_mode!r}")

    @property
    def any_module(self) -> bool:
        return self.pvi or self.coda or self.rocket

    @property
    def enabled(self) -> list[str]:
        return [n for n in ("pvi", "coda", "rocket") if getattr(self, n)]

    @property
    def student_layers(self) -> list[int]:
        return [a for a, _ in self.rocket_pairs]

    @property
    def vggt_layers(self) -> list[int]:
        return [b for _, b in self.rocket_pairs]

    def summary(self) -> str:
        parts = [f"modules={'+'.join(self.enabled) or 'none'}", f"main_ae={self.main_ae}"]
        if self.main_ae == "lr":
            parts.append(f"main_ae_lr={self.main_ae_lr:g}")
        parts.append(f"new_lr={self.new_lr:g}")
        if self.coda:
            parts.append(f"coda_steps={self.main_num_steps}+{self.coda_num_steps}")
            if self.coda_rtc != "off":   # omitted when off so the pre-2026-10-17 log line is unchanged
                parts.append(f"coda_rtc={self.coda_rtc}")
        if self.rocket:
            parts.append("rocket_pairs=" + ",".join(f"{a}:{b}" for a, b in self.rocket_pairs))
            parts.append(f"alpha={self.rocket_alpha:g}")
            parts.append(f"targets={self.rocket_target_dir}")
        parts.append(f"attn={self.attn_impl} compile={self.compile_mode}")
        if self.camera_hw:
            parts.append("camera_hw=" + ",".join(f"{h}x{w}" for h, w in self.camera_hw.values()))
        return " ".join(parts)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExtConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown keys in {SIDECAR_NAME}: {sorted(unknown)}")
        return cls(**d)

    def save(self, directory: str | Path) -> Path:
        p = Path(directory) / SIDECAR_NAME
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, directory: str | Path) -> "ExtConfig | None":
        p = Path(directory) / SIDECAR_NAME
        if not p.is_file():
            return None
        return cls.from_dict(json.loads(p.read_text()))


def has_sidecar(directory: str | Path) -> bool:
    return (Path(directory) / SIDECAR_NAME).is_file()
