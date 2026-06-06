"""Inject thirdparty/<repo> roots into sys.path.

Imported as a side effect by `xmatcher.methods.__init__`. Order matters:
must run before any adapter does `from lightglue import ...`.

EfficientLoFTR is loaded via HuggingFace transformers
(`AutoModelForKeypointMatching`) and does not need its upstream code on
sys.path — the `thirdparty/EfficientLoFTR/` checkout is kept only as
historical reference and for users who want to run upstream's training
scripts directly.
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_THIRDPARTY = _ROOT / "thirdparty"

_TARGETS = [
    "LightGlue",
]

for _sub in _TARGETS:
    _path = _THIRDPARTY / _sub
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
