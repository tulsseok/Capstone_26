"""Coda: correction AE = copy of the main AE, output projection zero-initialised
(SmolVLA_Coda_구현.md §4-2). Attached as `policy.model.correction` -> keys `model.correction.*`.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

PROJ_NAMES = ("action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj")


class CorrectionExpert(nn.Module):
    """Copy of the action expert + its four projections. Only the output projection starts at 0."""

    def __init__(self, flow):
        super().__init__()
        self.lm_expert = copy.deepcopy(flow.vlm_with_expert.lm_expert)
        for name in PROJ_NAMES:
            setattr(self, name, copy.deepcopy(getattr(flow, name)))
        for p in self.parameters():
            p.requires_grad_(True)
        self.zero_output()

    @torch.no_grad()
    def init_from_main(self, flow):
        self.lm_expert.load_state_dict(flow.vlm_with_expert.lm_expert.state_dict())
        for name in PROJ_NAMES:
            getattr(self, name).load_state_dict(getattr(flow, name).state_dict())
        self.zero_output()

    @torch.no_grad()
    def zero_output(self):
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)
