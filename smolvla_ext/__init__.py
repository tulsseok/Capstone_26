"""SmolVLA extensions (PVI / Coda / ROCKET) attached to a stock lerobot SmolVLAPolicy at runtime.

    from smolvla_ext import ExtConfig, attach, load_policy, has_sidecar

lerobot/ is not modified. Settings travel as `smolvla_ext.json` next to config.json.
"""

from .attach import attach, load_ext_weights, load_policy
from .config import SIDECAR_NAME, ExtConfig, has_sidecar, parse_pairs

__all__ = ["ExtConfig", "attach", "load_policy", "load_ext_weights", "has_sidecar", "parse_pairs",
           "SIDECAR_NAME"]
