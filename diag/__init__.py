"""Anomaly / silent-failure diagnostics for the SmolVLA (lerobot) training + inference stack.

Two entry points:

* ``diag.preflight``  -- static checks run *before* a training command. Prints a
  PASS/FAIL table and exits non-zero if anything failed.
* ``diag.probe``      -- runtime monitor. Forward hooks on the projection layers +
  the action expert, per-group grad norms, channel-wise output diagnostics,
  and an in-process resource record. One JSONL line per record.

Both are deliberately written so that *nothing in the lerobot repo has to be
modified*: ``diag.probe.Probe`` only registers ``nn.Module`` forward hooks and an
optimizer step pre-hook on objects that are handed to it.

The checks target the six failure modes that were observed to break **silently**
(loss looks fine, robot does not work). See ``diag/preflight.py`` docstring.
"""

from diag.probe import Probe, ProbeConfig  # noqa: F401

__all__ = ["Probe", "ProbeConfig"]
