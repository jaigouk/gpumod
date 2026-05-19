"""Shipped driver venv compatibility profiles (gpumod-ng7).

Each profile lives under ``profiles/<name>.yaml`` and declares a
known-good set of PEP 440 version specifiers for a driver release.
Profiles ship as package data so ``importlib.resources`` resolves
them in installed wheels.

See ``gpumod compat list`` and ``gpumod compat show <name>``.
"""

from __future__ import annotations
