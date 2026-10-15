"""Compatibility package for REPA utility helpers.

This package keeps the existing ``from utils import ...`` imports working
while also hosting new utility modules such as ``utils.lr_warmup``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_legacy_utils_module():
    legacy_utils_path = Path(__file__).resolve().parent.parent / "utils.py"
    spec = importlib.util.spec_from_file_location("_repa_legacy_utils", legacy_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy utils module from {legacy_utils_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy_utils = _load_legacy_utils_module()

for _name in dir(_legacy_utils):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_legacy_utils, _name)

from .lr_warmup import apply_scheduled_lr, get_scheduled_lr

__all__ = [
    name for name in globals()
    if not name.startswith("_")
]
