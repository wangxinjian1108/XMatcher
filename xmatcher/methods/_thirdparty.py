"""Inject thirdparty/<repo> roots into sys.path.

Imported as a side effect by `xmatcher.methods.__init__`. Order matters:
must run before any adapter does `from lightglue import ...` or
`from src.loftr import ...`.
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_THIRDPARTY = _ROOT / "thirdparty"

_TARGETS = [
    "LightGlue",
    "EfficientLoFTR",
]

for _sub in _TARGETS:
    _path = _THIRDPARTY / _sub
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
