# Unified Keypoints Matching Interface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a YAML-configured, pluggable matcher framework that runs LightGlue and EfficientLoFTR through a single CLI, in a self-contained Docker image.

**Architecture:** Single `BaseMatcher` template owns unproject + mask filtering; per-method adapters in `xmatcher/methods/*.py` registered via decorator; thirdparty submodules wired via `sys.path` injection; pydantic two-stage config validation; outputs land as `.npz` + `manifest.jsonl`.

**Tech Stack:** Python 3.10 · PyTorch 2.4.1+cu121 · pydantic v2 · pytest · Docker (CUDA 12.1) · GitHub Actions (GHCR).

**Spec:** [`docs/superpowers/specs/2026-06-06-unified-kpts-matching-design.md`](../specs/2026-06-06-unified-kpts-matching-design.md)

**Process notes:**
- Each task is one TDD cycle ending in a commit.
- File paths are absolute relative to repo root.
- All tests live under `tests/`; `tests/unit` and `tests/contract` run on CI; `tests/smoke` is local-only.
- Two refinements from spec: (a) `BaseMatcher.__call__` filters in **processed** coords *before* unproject (mask is in processed coords); (b) `valid_mask` is `(H_proc, W_proc)` bool with row-major `[v, u]` indexing.

---

## File Layout (Locked-In)

```
xmatcher/
  __init__.py                       — Package version
  core/
    __init__.py                     — Re-export public types
    types.py                        — PreprocessMeta, ImagePair, MatchResult, DenseField, _RawOutput
    preprocess.py                   — unproject, filter_by_mask, _to_gray_align32
    registry.py                     — register, get_matcher_cls, list_methods
    base.py                         — BaseMatcher (template __call__)
    config.py                       — RunConfig, build_matcher
    io.py                           — save_npz, result_to_manifest, snapshot_config, load_npz
  methods/
    __init__.py                     — Triggers _thirdparty + adapter imports
    _thirdparty.py                  — sys.path injection
    lightglue.py                    — LightGlueParams, LightGlueMatcher
    efficient_loftr.py              — EfficientLoFTRParams, EfficientLoFTRMatcher
  dataset/
    __init__.py
    protocol.py                     — PairDataset (Protocol)
    from_pair_list.py               — FromPairListDataset
    config.py                       — DatasetConfig, build_dataset
  cli/
    __init__.py
    run.py                          — argparse + main()

configs/
  lightglue.yaml
  efficient_loftr.yaml
  dataset/sample_pairs.yaml

docker/
  Dockerfile
  requirements.common.txt
  requirements.lightglue.txt
  requirements.eloftr.txt
  build.sh
  .dockerignore                     — At repo root, not docker/

scripts/
  download_weights.sh
  _download.py

weights/
  README.md
  WEIGHTS.lock

tests/
  conftest.py                       — gpu / requires_weights markers
  unit/
    test_preprocess_meta.py
    test_unproject.py
    test_filter_by_mask.py
    test_preprocess_helpers.py
    test_match_result.py
    test_registry.py
    test_config.py
    test_dataset.py
    test_io.py
  contract/
    test_base_matcher_template.py
  smoke/
    test_lightglue_smoke.py
    test_efficient_loftr_smoke.py
  fixtures/
    sample_a.jpg
    sample_b.jpg
    sample_pairs.txt

.github/workflows/
  test.yml
  docker-build.yml

pyproject.toml
README.md
```

---

## Task 1: Project skeleton — `pyproject.toml`, `.gitignore`, package roots

**Files:**
- Create: `pyproject.toml`
- Create: `xmatcher/__init__.py`
- Create: `xmatcher/core/__init__.py`
- Create: `xmatcher/methods/__init__.py`
- Create: `xmatcher/dataset/__init__.py`
- Create: `xmatcher/cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/contract/__init__.py`
- Create: `tests/smoke/__init__.py`
- Modify: `.gitignore` (append weights / outputs / pycache rules)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "xmatcher"
version = "0.1.0"
description = "Unified keypoints matching interface"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "pyyaml",
  "pydantic>=2.0",
  "opencv-python",
  "einops",
  "gdown",
]

[project.optional-dependencies]
test = ["pytest>=7.0", "pytest-mock"]

[project.scripts]
xmatcher = "xmatcher.cli.run:main"

[tool.setuptools.packages.find]
include = ["xmatcher*"]

[tool.pytest.ini_options]
markers = [
  "gpu: requires CUDA",
  "requires_weights(method): requires weights for the named method to be present",
]
```

- [ ] **Step 2: Empty package init files**

```python
# xmatcher/__init__.py
__version__ = "0.1.0"
```

```python
# xmatcher/core/__init__.py
# (empty for now; re-exports added in Task 2)
```

`xmatcher/methods/__init__.py`, `xmatcher/dataset/__init__.py`, `xmatcher/cli/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/contract/__init__.py`, `tests/smoke/__init__.py`: each is a single empty file (just create with no content).

- [ ] **Step 3: Append to `.gitignore`**

Append these lines (preserve existing content):

```
# xmatcher
weights/*
!weights/README.md
!weights/WEIGHTS.lock
outputs/
.env
```

- [ ] **Step 4: Verify install works**

Run: `pip install -e .[test]`
Expected: installs without error; `python -c "import xmatcher; print(xmatcher.__version__)"` prints `0.1.0`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml xmatcher/ tests/__init__.py tests/unit/__init__.py tests/contract/__init__.py tests/smoke/__init__.py .gitignore
git commit -m "chore: project skeleton (pyproject, package roots, gitignore)"
```

---

## Task 2: `PreprocessMeta` data class with derived affine

**Files:**
- Create: `xmatcher/core/types.py`
- Modify: `xmatcher/core/__init__.py`
- Create: `tests/unit/test_preprocess_meta.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_preprocess_meta.py`:

```python
import torch
import pytest
from xmatcher.core.types import PreprocessMeta


def test_meta_no_op_yields_identity_affine():
    """No crop, scale=1, no pad → affine should be identity (in 2x3 form)."""
    meta = PreprocessMeta(
        original_size=(720, 1280),
        processed_size=(720, 1280),
        crop_box=None,
        scale=(1.0, 1.0),
        pad=(0, 0, 0, 0),
        valid_mask=None,
    )
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(meta.affine_proc_to_orig, expected)


def test_meta_pure_resize_recovers_scale_inverse():
    """Pure 2x downsample: proc point (u, v) maps back to (2u, 2v)."""
    meta = PreprocessMeta(
        original_size=(720, 1280),
        processed_size=(360, 640),
        crop_box=None,
        scale=(0.5, 0.5),
        pad=(0, 0, 0, 0),
        valid_mask=None,
    )
    # proc → orig: divide by scale
    expected = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert torch.allclose(meta.affine_proc_to_orig, expected)


def test_meta_crop_then_resize_then_pad():
    """Realistic chain: crop (100,50) of size 1000x720, resize to 640x640, pad 0.
    proc (0, 0) should map to original (100, 50);
    proc (640, 640) should map to original (1100, 770).
    """
    meta = PreprocessMeta(
        original_size=(720, 1280),
        processed_size=(640, 640),
        crop_box=(100, 50, 1100, 770),
        scale=(640.0 / 1000.0, 640.0 / 720.0),
        pad=(0, 0, 0, 0),
        valid_mask=None,
    )
    aff = meta.affine_proc_to_orig
    # (u, v) = (0, 0)
    pt = aff @ torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(pt, torch.tensor([100.0, 50.0]), atol=1e-4)
    # (u, v) = (640, 640)
    pt = aff @ torch.tensor([640.0, 640.0, 1.0])
    assert torch.allclose(pt, torch.tensor([1100.0, 770.0]), atol=1e-4)


def test_meta_with_pad_subtracts_pad_first():
    """Pad (left=20, top=10): proc (20, 10) should map to crop_origin (no scale/crop).
    """
    meta = PreprocessMeta(
        original_size=(100, 100),
        processed_size=(110, 120),  # H=100+10+0 (top pad), W=100+20+0 (left pad)
        crop_box=None,
        scale=(1.0, 1.0),
        pad=(20, 10, 0, 0),
        valid_mask=None,
    )
    aff = meta.affine_proc_to_orig
    pt = aff @ torch.tensor([20.0, 10.0, 1.0])
    assert torch.allclose(pt, torch.tensor([0.0, 0.0]), atol=1e-4)


def test_meta_valid_mask_shape_must_match_processed_size():
    bad = torch.ones(640, 640, dtype=torch.bool)
    with pytest.raises(ValueError, match="valid_mask shape"):
        PreprocessMeta(
            original_size=(720, 1280),
            processed_size=(360, 640),  # mismatch with mask
            crop_box=None,
            scale=(1.0, 1.0),
            pad=(0, 0, 0, 0),
            valid_mask=bad,
        )


def test_meta_zero_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        PreprocessMeta(
            original_size=(100, 100),
            processed_size=(100, 100),
            crop_box=None,
            scale=(0.0, 1.0),
            pad=(0, 0, 0, 0),
            valid_mask=None,
        )
```

- [ ] **Step 2: Run test to verify they fail**

Run: `pytest tests/unit/test_preprocess_meta.py -v`
Expected: All 6 tests FAIL with `ModuleNotFoundError: xmatcher.core.types` or `ImportError`.

- [ ] **Step 3: Implement `PreprocessMeta`**

`xmatcher/core/types.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class DenseField:
    warp: torch.Tensor
    certainty: torch.Tensor
    coord_space: Literal["processed", "original"] = "processed"


@dataclass
class PreprocessMeta:
    """Describes the geometric chain from original image to processed image.

    Apply order (original → processed): crop → scale → pad.
    Inverse (processed → original) is exposed as `affine_proc_to_orig`.
    """
    original_size: tuple[int, int]            # (H_orig, W_orig)
    processed_size: tuple[int, int]           # (H_proc, W_proc)
    crop_box: tuple[int, int, int, int] | None  # (x0, y0, x1, y1) on original
    scale: tuple[float, float]                # (sx, sy)
    pad: tuple[int, int, int, int]            # (left, top, right, bottom)
    valid_mask: torch.Tensor | None           # (H_proc, W_proc) bool
    affine_proc_to_orig: torch.Tensor = field(init=False)

    def __post_init__(self):
        sx, sy = self.scale
        if sx <= 0 or sy <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.valid_mask is not None:
            if tuple(self.valid_mask.shape) != self.processed_size:
                raise ValueError(
                    f"valid_mask shape {tuple(self.valid_mask.shape)} "
                    f"!= processed_size {self.processed_size}"
                )
        cx0, cy0 = (self.crop_box[0], self.crop_box[1]) if self.crop_box else (0, 0)
        pad_l, pad_t, _, _ = self.pad
        # proc → cropped-original (after subtracting pad, dividing by scale):
        # u_orig_in_crop = (u_proc - pad_l) / sx
        # v_orig_in_crop = (v_proc - pad_t) / sy
        # u_orig = u_orig_in_crop + cx0
        # → affine: [[1/sx, 0, cx0 - pad_l/sx], [0, 1/sy, cy0 - pad_t/sy]]
        self.affine_proc_to_orig = torch.tensor(
            [
                [1.0 / sx, 0.0, cx0 - pad_l / sx],
                [0.0, 1.0 / sy, cy0 - pad_t / sy],
            ],
            dtype=torch.float32,
        )


@dataclass
class ImagePair:
    image0: torch.Tensor
    image1: torch.Tensor
    meta0: PreprocessMeta
    meta1: PreprocessMeta
    pair_id: str
    extras: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    mkpts0: torch.Tensor
    mkpts1: torch.Tensor
    mconf: torch.Tensor
    method: str
    pair_id: str
    runtime_ms: float
    dense: DenseField | None = None


@dataclass
class _RawOutput:
    """Internal: what `BaseMatcher._forward` returns. Coords in processed space."""
    mkpts0: torch.Tensor
    mkpts1: torch.Tensor
    mconf: torch.Tensor
    dense: DenseField | None = None
```

- [ ] **Step 4: Re-export from `xmatcher/core/__init__.py`**

```python
from xmatcher.core.types import (
    PreprocessMeta,
    ImagePair,
    MatchResult,
    DenseField,
    _RawOutput,
)

__all__ = ["PreprocessMeta", "ImagePair", "MatchResult", "DenseField"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_preprocess_meta.py -v`
Expected: 6 PASSED.

- [ ] **Step 6: Commit**

```bash
git add xmatcher/core/types.py xmatcher/core/__init__.py tests/unit/test_preprocess_meta.py
git commit -m "feat(core): add PreprocessMeta + ImagePair/MatchResult dataclasses"
```

---

## Task 3: `unproject` and `filter_by_mask`

**Files:**
- Create: `xmatcher/core/preprocess.py`
- Create: `tests/unit/test_unproject.py`
- Create: `tests/unit/test_filter_by_mask.py`

- [ ] **Step 1: Write `test_unproject.py`**

```python
import torch
import pytest
from xmatcher.core.types import PreprocessMeta
from xmatcher.core.preprocess import unproject


def _make_meta(orig_hw, proc_hw, crop_box, scale, pad):
    return PreprocessMeta(
        original_size=orig_hw,
        processed_size=proc_hw,
        crop_box=crop_box,
        scale=scale,
        pad=pad,
        valid_mask=None,
    )


@pytest.mark.parametrize(
    "orig_hw,proc_hw,crop,scale,pad",
    [
        ((720, 1280), (720, 1280), None, (1.0, 1.0), (0, 0, 0, 0)),         # identity
        ((720, 1280), (360, 640),  None, (0.5, 0.5), (0, 0, 0, 0)),         # pure resize
        ((720, 1280), (640, 640),  (100, 50, 1100, 770), (0.64, 0.8888), (0, 0, 0, 0)),  # crop+resize
        ((100, 100),  (110, 120),  None, (1.0, 1.0), (20, 10, 0, 0)),       # left/top pad → H=100+10, W=100+20
        ((100, 100),  (120, 110),  None, (1.0, 1.0), (0, 0, 10, 20)),       # right/bottom pad → H=100+20, W=100+10
        ((720, 1280), (700, 1260), (10, 10, 1270, 710), (1.0, 1.0), (0, 0, 0, 0)),       # crop only
        ((720, 1280), (740, 1300), (10, 10, 1270, 710), (1.0, 1.0), (20, 20, 20, 20)),   # crop + 4-side pad
        ((720, 1280), (320, 320),  (200, 200, 1000, 700), (0.4, 0.64), (0, 0, 0, 0)),    # full chain
    ],
)
def test_unproject_roundtrip(orig_hw, proc_hw, crop, scale, pad):
    """Pick random points in processed coords; unproject to original; project back; should match."""
    meta = _make_meta(orig_hw, proc_hw, crop, scale, pad)
    H_proc, W_proc = proc_hw
    pts_proc = torch.tensor(
        [[0.0, 0.0],
         [W_proc - 1.0, H_proc - 1.0],
         [W_proc / 2, H_proc / 2],
         [10.0, 5.0]],
        dtype=torch.float32,
    )
    pts_orig = unproject(pts_proc, meta)
    # Manual forward (orig → proc): proc = (orig - crop_origin) * scale + pad
    cx0, cy0 = (crop[0], crop[1]) if crop else (0, 0)
    sx, sy = scale
    pad_l, pad_t, _, _ = pad
    expected_proc = torch.stack(
        [
            (pts_orig[:, 0] - cx0) * sx + pad_l,
            (pts_orig[:, 1] - cy0) * sy + pad_t,
        ],
        dim=1,
    )
    assert torch.allclose(expected_proc, pts_proc, atol=1e-3)


def test_unproject_empty_input():
    meta = _make_meta((100, 100), (100, 100), None, (1.0, 1.0), (0, 0, 0, 0))
    pts = torch.zeros((0, 2), dtype=torch.float32)
    out = unproject(pts, meta)
    assert out.shape == (0, 2)


def test_unproject_preserves_device():
    meta = _make_meta((100, 100), (100, 100), None, (1.0, 1.0), (0, 0, 0, 0))
    pts = torch.zeros((3, 2), dtype=torch.float32)
    out = unproject(pts, meta)
    assert out.device == pts.device
```

- [ ] **Step 2: Write `test_filter_by_mask.py`**

```python
import torch
import pytest
from xmatcher.core.types import PreprocessMeta
from xmatcher.core.preprocess import filter_by_mask


def _meta_with_mask(mask):
    H, W = mask.shape
    return PreprocessMeta(
        original_size=(H, W),
        processed_size=(H, W),
        crop_box=None,
        scale=(1.0, 1.0),
        pad=(0, 0, 0, 0),
        valid_mask=mask,
    )


def test_filter_no_mask_keeps_everything():
    meta_no = PreprocessMeta(
        original_size=(10, 10), processed_size=(10, 10),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    pts = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
    keep = filter_by_mask(pts, pts, meta_no, meta_no)
    assert keep.all()


def test_filter_drops_points_outside_valid_region():
    """Mask: top-left 5x5 valid, rest invalid."""
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[:5, :5] = True
    meta = _meta_with_mask(mask)
    pts0 = torch.tensor([[2.0, 2.0], [7.0, 7.0]])  # second is outside
    pts1 = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    keep = filter_by_mask(pts0, pts1, meta, meta)
    assert keep.tolist() == [True, False]


def test_filter_requires_both_sides_valid():
    """Point valid on side 0 but not side 1 → drop."""
    mask0 = torch.ones(10, 10, dtype=torch.bool)
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[:5, :5] = True
    pts0 = torch.tensor([[5.0, 5.0]])
    pts1 = torch.tensor([[7.0, 7.0]])  # outside mask1
    keep = filter_by_mask(pts0, pts1, _meta_with_mask(mask0), _meta_with_mask(mask1))
    assert keep.tolist() == [False]


def test_filter_drops_points_outside_processed_bounds():
    """Negative or out-of-bounds coords → drop, even with no mask."""
    meta = PreprocessMeta(
        original_size=(10, 10), processed_size=(10, 10),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    pts = torch.tensor([[-1.0, 5.0], [5.0, 12.0], [5.0, 5.0]])
    keep = filter_by_mask(pts, pts, meta, meta)
    assert keep.tolist() == [False, False, True]


def test_filter_empty_input():
    meta = PreprocessMeta(
        original_size=(10, 10), processed_size=(10, 10),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    pts = torch.zeros((0, 2))
    keep = filter_by_mask(pts, pts, meta, meta)
    assert keep.shape == (0,) and keep.dtype == torch.bool
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_unproject.py tests/unit/test_filter_by_mask.py -v`
Expected: All FAIL with `ModuleNotFoundError: xmatcher.core.preprocess`.

- [ ] **Step 4: Implement `preprocess.py`**

`xmatcher/core/preprocess.py`:

```python
from __future__ import annotations
import torch
from xmatcher.core.types import PreprocessMeta


def unproject(pts_proc: torch.Tensor, meta: PreprocessMeta) -> torch.Tensor:
    """Map (K, 2) points from processed coords to original coords.

    Uses meta.affine_proc_to_orig (a 2x3 tensor). Empty inputs are passed through.
    Output device matches input.
    """
    if pts_proc.numel() == 0:
        return pts_proc.clone()
    aff = meta.affine_proc_to_orig.to(pts_proc.device, dtype=pts_proc.dtype)
    ones = torch.ones((pts_proc.shape[0], 1), dtype=pts_proc.dtype, device=pts_proc.device)
    homog = torch.cat([pts_proc, ones], dim=1)        # (K, 3)
    return homog @ aff.T                              # (K, 2)


def filter_by_mask(
    pts0_proc: torch.Tensor,
    pts1_proc: torch.Tensor,
    meta0: PreprocessMeta,
    meta1: PreprocessMeta,
) -> torch.Tensor:
    """Bool keep-mask. Coordinates are in *processed* space for both inputs.

    Drops points that fall outside [0, W_proc) x [0, H_proc) on either side,
    plus points whose nearest-pixel in processed valid_mask is False (when mask provided).
    """
    K = pts0_proc.shape[0]
    if K == 0:
        return torch.zeros((0,), dtype=torch.bool, device=pts0_proc.device)

    keep = torch.ones(K, dtype=torch.bool, device=pts0_proc.device)
    for pts, meta in ((pts0_proc, meta0), (pts1_proc, meta1)):
        H_proc, W_proc = meta.processed_size
        u, v = pts[:, 0], pts[:, 1]
        in_bounds = (u >= 0) & (u < W_proc) & (v >= 0) & (v < H_proc)
        keep = keep & in_bounds
        if meta.valid_mask is not None:
            uu = u.clamp(0, W_proc - 1).round().long()
            vv = v.clamp(0, H_proc - 1).round().long()
            mask_val = meta.valid_mask.to(pts.device)[vv, uu]
            keep = keep & mask_val
    return keep
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_unproject.py tests/unit/test_filter_by_mask.py -v`
Expected: 11 PASSED (8 unproject roundtrip + 2 unproject misc + 5 filter_by_mask).

- [ ] **Step 6: Commit**

```bash
git add xmatcher/core/preprocess.py tests/unit/test_unproject.py tests/unit/test_filter_by_mask.py
git commit -m "feat(core): add unproject and filter_by_mask"
```

---

## Task 4: `_to_gray_align32` helper

**Files:**
- Modify: `xmatcher/core/preprocess.py`
- Create: `tests/unit/test_preprocess_helpers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_preprocess_helpers.py
import torch
import pytest
from xmatcher.core.preprocess import _to_gray_align32


def test_grayscale_conversion_keeps_dtype_and_range():
    img = torch.rand(3, 64, 64)            # already 32-aligned
    out, pad = _to_gray_align32(img)
    assert out.shape == (1, 64, 64)
    assert pad == (0, 0)
    assert out.dtype == img.dtype
    assert out.min() >= 0 and out.max() <= 1


def test_align32_pads_to_next_multiple():
    img = torch.rand(3, 50, 70)
    out, pad = _to_gray_align32(img)
    # H: 50 → 64 (pad 14 bottom); W: 70 → 96 (pad 26 right). pad order (left, top).
    assert out.shape == (1, 64, 96)
    assert pad == (0, 0)


def test_align32_pads_only_right_and_bottom_not_left_top():
    """`pad` returned is (left, top). Right/bottom padding shifts nothing in coords."""
    img = torch.rand(3, 50, 70)
    out, pad = _to_gray_align32(img)
    # The returned pad tuple is what adapters subtract from output keypoints.
    # Right/bottom padding does not shift origin, so left=top=0.
    assert pad == (0, 0)


def test_grayscale_shape_when_input_already_gray():
    img = torch.rand(1, 32, 32)
    out, pad = _to_gray_align32(img)
    assert out.shape == (1, 32, 32)


def test_align32_already_aligned_is_noop():
    img = torch.rand(3, 32, 64)
    out, pad = _to_gray_align32(img)
    assert out.shape == (1, 32, 64)
    assert pad == (0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_preprocess_helpers.py -v`
Expected: All FAIL with `ImportError: cannot import name '_to_gray_align32'`.

- [ ] **Step 3: Add `_to_gray_align32` to `preprocess.py`**

Append to `xmatcher/core/preprocess.py`:

```python
import torch.nn.functional as F


def _to_gray_align32(img: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """RGB or gray (C, H, W) in [0,1] → gray (1, H', W') padded right/bottom to multiples of 32.

    Returns (img_padded, (pad_left, pad_top)). Right/bottom padding does not shift origin,
    so the returned pad tuple is always (0, 0). Adapters subtract this from output
    keypoints to get back to dataset's processed coordinate space.
    """
    if img.dim() != 3:
        raise ValueError(f"expected (C, H, W), got shape {tuple(img.shape)}")
    C, H, W = img.shape
    if C == 3:
        # Standard ITU-R BT.601 luminance weights.
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=img.dtype, device=img.device)
        gray = (img * weights[:, None, None]).sum(dim=0, keepdim=True)
    elif C == 1:
        gray = img
    else:
        raise ValueError(f"expected 1 or 3 channels, got {C}")

    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h or pad_w:
        # F.pad layout for (C, H, W): (left, right, top, bottom)
        gray = F.pad(gray, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return gray, (0, 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_preprocess_helpers.py -v`
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add xmatcher/core/preprocess.py tests/unit/test_preprocess_helpers.py
git commit -m "feat(core): add _to_gray_align32 helper"
```

---

## Task 5: Matcher registry

**Files:**
- Create: `xmatcher/core/registry.py`
- Create: `tests/unit/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_registry.py
import pytest
from xmatcher.core.registry import register, get_matcher_cls, list_methods, _REGISTRY


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Registry is global; snapshot/restore around each test."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def test_register_decorator_assigns_method_name():
    @register("foo")
    class Foo:
        pass
    assert Foo.method_name == "foo"
    assert get_matcher_cls("foo") is Foo


def test_register_duplicate_raises():
    @register("bar")
    class Bar:
        pass
    with pytest.raises(KeyError, match="already registered"):
        @register("bar")
        class Bar2:
            pass


def test_get_unknown_raises_with_available_list():
    @register("baz")
    class Baz:
        pass
    with pytest.raises(KeyError, match=r"Unknown matcher 'qux'.*Available.*baz"):
        get_matcher_cls("qux")


def test_list_methods_returns_sorted_names():
    @register("zeta")
    class Z: pass
    @register("alpha")
    class A: pass
    assert list_methods() == ["alpha", "zeta"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: xmatcher.core.registry`.

- [ ] **Step 3: Implement registry**

`xmatcher/core/registry.py`:

```python
from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise KeyError(f"Matcher '{name}' already registered")
        cls.method_name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_matcher_cls(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown matcher '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_methods() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/unit/test_registry.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add xmatcher/core/registry.py tests/unit/test_registry.py
git commit -m "feat(core): matcher registry"
```

---

## Task 6: `BaseMatcher` template + contract tests

**Files:**
- Create: `xmatcher/core/base.py`
- Create: `tests/contract/test_base_matcher_template.py`

- [ ] **Step 1: Write contract tests**

```python
# tests/contract/test_base_matcher_template.py
import torch
import pytest
from pydantic import BaseModel
from xmatcher.core.base import BaseMatcher
from xmatcher.core.types import (
    PreprocessMeta, ImagePair, MatchResult, _RawOutput,
)


class _MockParams(BaseModel):
    keypoints_proc: list[list[float]] = [[10.0, 10.0], [50.0, 50.0]]
    keypoints_proc_other: list[list[float]] = [[20.0, 20.0], [60.0, 60.0]]
    confs: list[float] = [0.9, 0.5]


class _MockMatcher(BaseMatcher):
    method_name = "mock"
    Params = _MockParams

    def _setup(self):
        self.calls = 0

    def _forward(self, pair):
        self.calls += 1
        p = self.params
        return _RawOutput(
            mkpts0=torch.tensor(p.keypoints_proc, dtype=torch.float32),
            mkpts1=torch.tensor(p.keypoints_proc_other, dtype=torch.float32),
            mconf=torch.tensor(p.confs, dtype=torch.float32),
            dense=None,
        )


def _pair_with_2x_resize() -> ImagePair:
    """Original 200x200, processed 100x100 (sx=sy=0.5). Identity images."""
    meta = PreprocessMeta(
        original_size=(200, 200),
        processed_size=(100, 100),
        crop_box=None,
        scale=(0.5, 0.5),
        pad=(0, 0, 0, 0),
        valid_mask=None,
    )
    img = torch.zeros(3, 100, 100)
    return ImagePair(image0=img, image1=img.clone(), meta0=meta, meta1=meta,
                     pair_id="test")


def test_default_call_applies_unproject_to_original_coords():
    pair = _pair_with_2x_resize()
    m = _MockMatcher(device="cpu", params=_MockParams())
    res = m(pair)
    # processed (10,10) → original (20,20); processed (50,50) → original (100,100)
    assert torch.allclose(res.mkpts0, torch.tensor([[20., 20.], [100., 100.]]))
    assert torch.allclose(res.mkpts1, torch.tensor([[40., 40.], [120., 120.]]))


def test_return_processed_coords_skips_unproject_and_filter():
    pair = _pair_with_2x_resize()
    m = _MockMatcher(device="cpu", params=_MockParams())
    res = m(pair, return_processed_coords=True)
    # Unchanged from _forward output.
    assert torch.allclose(res.mkpts0, torch.tensor([[10., 10.], [50., 50.]]))


def test_mask_filtering_drops_outside_valid_region():
    """Mask: only top-left quadrant of processed image is valid."""
    mask = torch.zeros(100, 100, dtype=torch.bool)
    mask[:30, :30] = True   # only proc (u<30, v<30) valid
    meta = PreprocessMeta(
        original_size=(200, 200), processed_size=(100, 100),
        crop_box=None, scale=(0.5, 0.5), pad=(0, 0, 0, 0),
        valid_mask=mask,
    )
    img = torch.zeros(3, 100, 100)
    pair = ImagePair(image0=img, image1=img.clone(), meta0=meta, meta1=meta, pair_id="t")
    m = _MockMatcher(device="cpu", params=_MockParams())
    res = m(pair)
    # First point (10,10) inside; second (50,50) outside → only one survives.
    assert res.mkpts0.shape == (1, 2)
    assert torch.allclose(res.mkpts0, torch.tensor([[20., 20.]]))


def test_call_records_method_pair_id_and_runtime():
    pair = _pair_with_2x_resize()
    m = _MockMatcher(device="cpu", params=_MockParams())
    res = m(pair)
    assert res.method == "mock"
    assert res.pair_id == "test"
    assert res.runtime_ms >= 0


def test_call_invokes_setup_and_forward():
    pair = _pair_with_2x_resize()
    m = _MockMatcher(device="cpu", params=_MockParams())
    assert m.calls == 0
    _ = m(pair)
    assert m.calls == 1


def test_dense_field_passes_through():
    from xmatcher.core.types import DenseField

    class _MockDense(_MockMatcher):
        def _forward(self, pair):
            base = super()._forward(pair)
            base.dense = DenseField(
                warp=torch.zeros(10, 10, 4),
                certainty=torch.zeros(10, 10),
                coord_space="processed",
            )
            return base

    pair = _pair_with_2x_resize()
    m = _MockDense(device="cpu", params=_MockParams())
    res = m(pair)
    assert res.dense is not None
    assert res.dense.coord_space == "processed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/contract/test_base_matcher_template.py -v`
Expected: FAIL with `ModuleNotFoundError: xmatcher.core.base`.

- [ ] **Step 3: Implement `BaseMatcher`**

`xmatcher/core/base.py`:

```python
from __future__ import annotations
import abc
import time
from typing import ClassVar
import torch
from pydantic import BaseModel
from xmatcher.core.types import ImagePair, MatchResult, _RawOutput
from xmatcher.core.preprocess import unproject, filter_by_mask


class BaseMatcher(abc.ABC):
    """Template Method base. Subclasses fill in `_setup` + `_forward`.

    `__call__` is the only public entry; it owns:
      - timing
      - mask-based filtering (in processed coords)
      - unproject to original coords
    """

    method_name: ClassVar[str]            # set by @register("name")
    Params: ClassVar[type[BaseModel]]

    def __init__(self, *, device: str = "cuda", params: BaseModel):
        self.device = device
        self.params = params
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None: ...

    @abc.abstractmethod
    def _forward(self, pair: ImagePair) -> _RawOutput: ...

    @torch.inference_mode()
    def __call__(
        self,
        pair: ImagePair,
        *,
        return_processed_coords: bool = False,
    ) -> MatchResult:
        t0 = time.perf_counter()
        raw = self._forward(pair)

        if return_processed_coords:
            mk0, mk1 = raw.mkpts0, raw.mkpts1
            mconf = raw.mconf
        else:
            keep = filter_by_mask(raw.mkpts0, raw.mkpts1, pair.meta0, pair.meta1)
            mk0 = unproject(raw.mkpts0[keep], pair.meta0)
            mk1 = unproject(raw.mkpts1[keep], pair.meta1)
            mconf = raw.mconf[keep]

        return MatchResult(
            mkpts0=mk0,
            mkpts1=mk1,
            mconf=mconf,
            method=self.method_name,
            pair_id=pair.pair_id,
            runtime_ms=(time.perf_counter() - t0) * 1000,
            dense=raw.dense,
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/contract/test_base_matcher_template.py -v`
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add xmatcher/core/base.py tests/contract/test_base_matcher_template.py
git commit -m "feat(core): BaseMatcher template (filter then unproject)"
```

---

## Task 7: `RunConfig` and `build_matcher`

**Files:**
- Create: `xmatcher/core/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
import pytest
from pydantic import BaseModel, ValidationError
from xmatcher.core.config import RunConfig, build_matcher
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register, _REGISTRY
from xmatcher.core.types import _RawOutput
import torch


@pytest.fixture(autouse=True)
def _isolated_registry():
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


class _ToyParams(BaseModel):
    threshold: float = 0.1
    label: str


@register("toy")
class _ToyMatcher(BaseMatcher):
    Params = _ToyParams
    def _setup(self): pass
    def _forward(self, pair):
        return _RawOutput(
            mkpts0=torch.zeros(0, 2), mkpts1=torch.zeros(0, 2),
            mconf=torch.zeros(0), dense=None,
        )


def test_runconfig_accepts_minimum_fields():
    cfg = RunConfig.model_validate({
        "method": "toy",
        "params": {"label": "x"},
    })
    assert cfg.method == "toy"
    assert cfg.device == "cuda"
    assert cfg.seed == 0


def test_build_matcher_routes_params_to_typed_model():
    cfg = RunConfig.model_validate({
        "method": "toy",
        "device": "cpu",
        "params": {"threshold": 0.5, "label": "x"},
    })
    m = build_matcher(cfg)
    assert isinstance(m, _ToyMatcher)
    assert m.params.threshold == 0.5
    assert m.params.label == "x"
    assert m.device == "cpu"


def test_build_matcher_rejects_unknown_method():
    cfg = RunConfig.model_validate({"method": "nope", "params": {}})
    with pytest.raises(KeyError, match="Unknown matcher"):
        build_matcher(cfg)


def test_build_matcher_rejects_invalid_params():
    cfg = RunConfig.model_validate({
        "method": "toy",
        "params": {"threshold": 0.5},   # missing required `label`
    })
    with pytest.raises(ValidationError):
        build_matcher(cfg)


def test_runconfig_rejects_unknown_device():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"method": "toy", "device": "tpu", "params": {}})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: xmatcher.core.config`.

- [ ] **Step 3: Implement `config.py`**

`xmatcher/core/config.py`:

```python
from __future__ import annotations
import random
from typing import Literal
import numpy as np
import torch
from pydantic import BaseModel, Field
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import get_matcher_cls


class RunConfig(BaseModel):
    method: str
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    seed: int = 0
    params: dict = Field(default_factory=dict)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_matcher(cfg: RunConfig) -> BaseMatcher:
    cls = get_matcher_cls(cfg.method)
    typed_params = cls.Params(**cfg.params)
    _set_seed(cfg.seed)
    return cls(device=cfg.device, params=typed_params)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_config.py -v`
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add xmatcher/core/config.py tests/unit/test_config.py
git commit -m "feat(core): RunConfig and build_matcher"
```

---

## Task 8: `MatchResult` IO — `save_npz` / `load_npz` / `result_to_manifest`

**Files:**
- Create: `xmatcher/core/io.py`
- Create: `tests/unit/test_io.py`
- Create: `tests/unit/test_match_result.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_io.py
import json
from pathlib import Path
import torch
from xmatcher.core.types import MatchResult
from xmatcher.core.io import save_npz, load_npz, result_to_manifest


def _make_result(tmp_path):
    return MatchResult(
        mkpts0=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        mkpts1=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        mconf=torch.tensor([0.9, 0.8]),
        method="lightglue",
        pair_id="scene01__0001__0002",
        runtime_ms=12.34,
        dense=None,
    )


def test_save_npz_roundtrip(tmp_path):
    r = _make_result(tmp_path)
    p = tmp_path / "out.npz"
    save_npz(r, p)
    loaded = load_npz(p)
    assert torch.allclose(loaded.mkpts0, r.mkpts0)
    assert torch.allclose(loaded.mkpts1, r.mkpts1)
    assert torch.allclose(loaded.mconf, r.mconf)
    assert loaded.method == "lightglue"
    assert loaded.pair_id == "scene01__0001__0002"
    assert loaded.runtime_ms == 12.34


def test_result_to_manifest_serializes_to_json(tmp_path):
    r = _make_result(tmp_path)
    p = tmp_path / "matches" / "x.npz"
    p.parent.mkdir()
    p.write_bytes(b"")  # placeholder; manifest uses path string
    entry = result_to_manifest(r, p)
    payload = json.dumps(entry)            # must be JSON-serializable
    parsed = json.loads(payload)
    assert parsed["pair_id"] == "scene01__0001__0002"
    assert parsed["method"] == "lightglue"
    assert parsed["num_matches"] == 2
    assert parsed["runtime_ms"] == 12.34
    assert parsed["npz_path"].endswith("x.npz")


def test_save_npz_creates_parent_dir(tmp_path):
    r = _make_result(tmp_path)
    p = tmp_path / "deep" / "nested" / "file.npz"
    save_npz(r, p)
    assert p.exists()


def test_save_npz_handles_empty_matches(tmp_path):
    r = MatchResult(
        mkpts0=torch.zeros(0, 2), mkpts1=torch.zeros(0, 2), mconf=torch.zeros(0),
        method="lightglue", pair_id="x", runtime_ms=1.0, dense=None,
    )
    p = tmp_path / "out.npz"
    save_npz(r, p)
    loaded = load_npz(p)
    assert loaded.mkpts0.shape == (0, 2)
```

- [ ] **Step 2: Run test to verify they fail**

Run: `pytest tests/unit/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `io.py`**

`xmatcher/core/io.py`:

```python
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from xmatcher.core.types import MatchResult


def save_npz(result: MatchResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mkpts0=result.mkpts0.detach().cpu().numpy(),
        mkpts1=result.mkpts1.detach().cpu().numpy(),
        mconf=result.mconf.detach().cpu().numpy(),
        method=np.array(result.method),
        pair_id=np.array(result.pair_id),
        runtime_ms=np.array(result.runtime_ms),
    )


def load_npz(path: str | Path) -> MatchResult:
    z = np.load(Path(path), allow_pickle=False)
    return MatchResult(
        mkpts0=torch.from_numpy(z["mkpts0"]),
        mkpts1=torch.from_numpy(z["mkpts1"]),
        mconf=torch.from_numpy(z["mconf"]),
        method=str(z["method"]),
        pair_id=str(z["pair_id"]),
        runtime_ms=float(z["runtime_ms"]),
        dense=None,
    )


def result_to_manifest(result: MatchResult, npz_path: str | Path) -> dict:
    return {
        "pair_id": result.pair_id,
        "method": result.method,
        "num_matches": int(result.mkpts0.shape[0]),
        "runtime_ms": float(result.runtime_ms),
        "npz_path": str(npz_path),
    }


def snapshot_config(method_cfg, dataset_cfg, out_path: str | Path) -> None:
    """Dump method+dataset config plus git commit hash to a single YAML for replay."""
    import subprocess
    import yaml
    from datetime import datetime

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"

    snap = {
        "method_cfg": method_cfg.model_dump(),
        "dataset_cfg": dataset_cfg.model_dump(),
        "git_commit": commit,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(snap, sort_keys=False))
```

- [ ] **Step 4: Add minimal `test_match_result.py` (smoke for dataclass)**

```python
# tests/unit/test_match_result.py
import torch
from xmatcher.core.types import MatchResult, DenseField


def test_match_result_default_dense_is_none():
    r = MatchResult(
        mkpts0=torch.zeros(0, 2), mkpts1=torch.zeros(0, 2), mconf=torch.zeros(0),
        method="x", pair_id="y", runtime_ms=0.0,
    )
    assert r.dense is None


def test_dense_field_default_coord_space_is_processed():
    d = DenseField(warp=torch.zeros(2, 2, 4), certainty=torch.zeros(2, 2))
    assert d.coord_space == "processed"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_io.py tests/unit/test_match_result.py -v`
Expected: 6 PASSED.

- [ ] **Step 6: Commit**

```bash
git add xmatcher/core/io.py tests/unit/test_io.py tests/unit/test_match_result.py
git commit -m "feat(core): npz IO + manifest entry + config snapshot"
```

---

## Task 9: `conftest.py` — `gpu` and `requires_weights` markers

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Implement `conftest.py`**

```python
# tests/conftest.py
import os
from pathlib import Path
import pytest
import torch


def pytest_collection_modifyitems(config, items):
    has_cuda = torch.cuda.is_available()
    weights_dir = Path(os.environ.get(
        "XMATCHER_WEIGHTS_DIR",
        Path.home() / ".cache" / "xmatcher",
    ))

    skip_gpu = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)

        rw = item.get_closest_marker("requires_weights")
        if rw is None:
            continue
        method = rw.args[0]
        method_dir = weights_dir / method
        if not method_dir.exists() or not any(method_dir.iterdir()):
            item.add_marker(pytest.mark.skip(
                reason=f"weights for '{method}' not found at {method_dir}"
            ))
```

- [ ] **Step 2: Verify conftest doesn't break existing tests**

Run: `pytest tests/unit tests/contract -v`
Expected: All previously-passing tests still PASS (count = 27 currently). No new tests added — this just registers skipif behaviour for later smoke tests.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: gpu and requires_weights markers via conftest"
```

---

## Task 10: `_thirdparty.py` sys.path injection + `methods/__init__.py` wiring

**Files:**
- Create: `xmatcher/methods/_thirdparty.py`
- Modify: `xmatcher/methods/__init__.py`

- [ ] **Step 1: Implement `_thirdparty.py`**

`xmatcher/methods/_thirdparty.py`:

```python
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
```

- [ ] **Step 2: Wire `methods/__init__.py`**

`xmatcher/methods/__init__.py`:

```python
from xmatcher.methods import _thirdparty  # noqa: F401  (must be first)

# Adapter modules will be added in subsequent tasks. Each one self-registers
# via @register("<name>") on import.
```

- [ ] **Step 3: Verify nothing breaks**

Run: `pytest tests/unit tests/contract -v`
Expected: All PASS, no import errors.

- [ ] **Step 4: Commit**

```bash
git add xmatcher/methods/_thirdparty.py xmatcher/methods/__init__.py
git commit -m "feat(methods): sys.path injection scaffold"
```

---

## Task 11: LightGlue adapter

**Files:**
- Create: `xmatcher/methods/lightglue.py`
- Modify: `xmatcher/methods/__init__.py`
- Create: `configs/lightglue.yaml`

- [ ] **Step 1: Pre-flight — confirm submodule populated**

Run: `ls thirdparty/LightGlue/lightglue/__init__.py`
Expected: file exists. If not, run `git submodule update --init thirdparty/LightGlue` first.

- [ ] **Step 2: Implement `lightglue.py`**

`xmatcher/methods/lightglue.py`:

```python
from __future__ import annotations
from typing import Literal
import torch
from pydantic import BaseModel, Field
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register
from xmatcher.core.types import _RawOutput, ImagePair


class LightGlueParams(BaseModel):
    features: Literal["superpoint", "disk", "aliked", "sift"] = "superpoint"
    max_num_keypoints: int = Field(default=2048, gt=0)
    detection_threshold: float = 0.0005
    match_threshold: float = 0.1


@register("lightglue")
class LightGlueMatcher(BaseMatcher):
    Params = LightGlueParams

    def _setup(self):
        from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
        EXTRACTORS = {
            "superpoint": SuperPoint, "disk": DISK,
            "aliked": ALIKED, "sift": SIFT,
        }
        p = self.params
        self.extractor = EXTRACTORS[p.features](
            max_num_keypoints=p.max_num_keypoints,
            detection_threshold=p.detection_threshold,
        ).eval().to(self.device)
        self.matcher = LightGlue(
            features=p.features,
            filter_threshold=p.match_threshold,
        ).eval().to(self.device)

    def _forward(self, pair: ImagePair) -> _RawOutput:
        from lightglue.utils import rbd
        img0 = pair.image0.to(self.device)
        img1 = pair.image1.to(self.device)
        # Critical: resize=None disables LightGlue's internal resize.
        # Dataset already shaped the image; if extractor resizes again,
        # output keypoints will be in its internal grid → unproject breaks.
        feats0 = self.extractor.extract(img0, resize=None)
        feats1 = self.extractor.extract(img1, resize=None)
        out = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, out = [rbd(x) for x in (feats0, feats1, out)]
        m = out["matches"]
        return _RawOutput(
            mkpts0=feats0["keypoints"][m[:, 0]],
            mkpts1=feats1["keypoints"][m[:, 1]],
            mconf=out["scores"],
            dense=None,
        )
```

- [ ] **Step 3: Add to `methods/__init__.py`**

Replace `xmatcher/methods/__init__.py` body so it now reads:

```python
from xmatcher.methods import _thirdparty  # noqa: F401

from xmatcher.methods import lightglue  # noqa: F401  (registers "lightglue")
```

- [ ] **Step 4: Add a tiny import test (no GPU needed)**

Append to `tests/unit/test_registry.py` a single test outside the autouse-isolated registry:

```python
def test_lightglue_registers_on_methods_import():
    """Importing xmatcher.methods triggers @register('lightglue')."""
    from xmatcher.core.registry import _REGISTRY
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        # Force reimport of the methods package.
        import importlib, sys
        if "xmatcher.methods" in sys.modules:
            del sys.modules["xmatcher.methods"]
        if "xmatcher.methods.lightglue" in sys.modules:
            del sys.modules["xmatcher.methods.lightglue"]
        importlib.import_module("xmatcher.methods")
        assert "lightglue" in _REGISTRY
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)
```

- [ ] **Step 5: Run import test**

Run: `pytest tests/unit/test_registry.py::test_lightglue_registers_on_methods_import -v`
Expected: PASS. (If the import of `lightglue` itself fails — e.g., kornia missing — install: `pip install kornia` and retry.)

- [ ] **Step 6: Sample config**

`configs/lightglue.yaml`:

```yaml
method: lightglue
device: cuda
seed: 0
params:
  features: superpoint
  max_num_keypoints: 2048
  detection_threshold: 0.0005
  match_threshold: 0.1
```

- [ ] **Step 7: Commit**

```bash
git add xmatcher/methods/lightglue.py xmatcher/methods/__init__.py configs/lightglue.yaml tests/unit/test_registry.py
git commit -m "feat(methods): LightGlue adapter (resize=None, sparse output)"
```

---

## Task 12: EfficientLoFTR adapter

**Files:**
- Create: `xmatcher/methods/efficient_loftr.py`
- Modify: `xmatcher/methods/__init__.py`
- Create: `configs/efficient_loftr.yaml`

- [ ] **Step 1: Pre-flight**

Run: `ls thirdparty/EfficientLoFTR/src/loftr/__init__.py`
Expected: exists.

- [ ] **Step 2: Implement adapter**

`xmatcher/methods/efficient_loftr.py`:

```python
from __future__ import annotations
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Literal
import torch
from pydantic import BaseModel, Field, field_validator
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register
from xmatcher.core.types import _RawOutput, ImagePair
from xmatcher.core.preprocess import _to_gray_align32


class EfficientLoFTRParams(BaseModel):
    weights: Path
    precision: Literal["fp32", "fp16", "mp"] = "mp"
    match_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    border_rm: int = Field(default=2, ge=0)

    @field_validator("weights", mode="after")
    @classmethod
    def _resolve_weights(cls, v: Path) -> Path:
        if v.is_absolute():
            if not v.exists():
                raise FileNotFoundError(
                    f"Weight not found: {v}\n"
                    f"Run scripts/download_weights.sh efficient_loftr to fetch it."
                )
            return v
        base = Path(os.environ.get(
            "XMATCHER_WEIGHTS_DIR",
            Path.home() / ".cache" / "xmatcher",
        ))
        resolved = base / v
        if not resolved.exists():
            raise FileNotFoundError(
                f"Weight not found: {resolved}\n"
                f"Run scripts/download_weights.sh efficient_loftr to fetch it."
            )
        return resolved


@register("efficient_loftr")
class EfficientLoFTRMatcher(BaseMatcher):
    Params = EfficientLoFTRParams

    def _setup(self):
        from src.loftr import LoFTR, reparameter
        from src.config.default import get_cfg_defaults
        cfg = get_cfg_defaults()
        cfg.LOFTR.MATCH_COARSE.THR = self.params.match_threshold
        cfg.LOFTR.MATCH_COARSE.BORDER_RM = self.params.border_rm
        if self.params.precision == "fp16":
            cfg.LOFTR.HALF = True
            cfg.LOFTR.MP = False
        elif self.params.precision == "fp32":
            cfg.LOFTR.HALF = False
            cfg.LOFTR.MP = False
        else:  # "mp"
            cfg.LOFTR.HALF = False
            cfg.LOFTR.MP = True

        self.matcher = LoFTR(config=cfg.LOFTR)
        state = torch.load(self.params.weights, map_location="cpu")
        self.matcher.load_state_dict(state["state_dict"])
        self.matcher = reparameter(self.matcher).eval().to(self.device)
        self._cfg = cfg

    def _forward(self, pair: ImagePair) -> _RawOutput:
        # Adapter-internal preprocessing: gray + pad to 32-multiple.
        img0_pad, _pad0 = _to_gray_align32(pair.image0)
        img1_pad, _pad1 = _to_gray_align32(pair.image1)
        # _to_gray_align32 returns (0, 0) — right/bottom padding does not shift origin,
        # so kpts in proc-padded coords already equal kpts in dataset's processed coords.
        data = {
            "image0": img0_pad.to(self.device).unsqueeze(0),
            "image1": img1_pad.to(self.device).unsqueeze(0),
        }
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._cfg.LOFTR.MP and "cuda" in str(self.device)
            else nullcontext()
        )
        with ctx:
            self.matcher(data)
        return _RawOutput(
            mkpts0=data["mkpts0_f"],
            mkpts1=data["mkpts1_f"],
            mconf=data["mconf"],
            dense=None,
        )
```

- [ ] **Step 3: Wire adapter into `methods/__init__.py`**

Replace contents:

```python
from xmatcher.methods import _thirdparty  # noqa: F401

from xmatcher.methods import lightglue  # noqa: F401
from xmatcher.methods import efficient_loftr  # noqa: F401
```

- [ ] **Step 4: Sample config**

`configs/efficient_loftr.yaml`:

```yaml
method: efficient_loftr
device: cuda
seed: 0
params:
  weights: efficient_loftr/eloftr_outdoor.ckpt
  precision: mp
  match_threshold: 0.2
  border_rm: 2
```

- [ ] **Step 5: Add validator test (no model load)**

Append to `tests/unit/test_config.py`:

```python
def test_efficient_loftr_params_rejects_missing_weights(tmp_path, monkeypatch):
    from xmatcher.methods.efficient_loftr import EfficientLoFTRParams
    monkeypatch.setenv("XMATCHER_WEIGHTS_DIR", str(tmp_path))
    import pytest
    with pytest.raises(FileNotFoundError, match="Weight not found"):
        EfficientLoFTRParams(weights="efficient_loftr/missing.ckpt")


def test_efficient_loftr_params_accepts_existing_weights(tmp_path, monkeypatch):
    from xmatcher.methods.efficient_loftr import EfficientLoFTRParams
    monkeypatch.setenv("XMATCHER_WEIGHTS_DIR", str(tmp_path))
    fake = tmp_path / "efficient_loftr" / "x.ckpt"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"")
    p = EfficientLoFTRParams(weights="efficient_loftr/x.ckpt")
    assert p.weights == fake.resolve() or p.weights == fake
```

- [ ] **Step 6: Run validator tests**

Run: `pytest tests/unit/test_config.py -v`
Expected: 7 PASSED (5 prior + 2 new).

- [ ] **Step 7: Commit**

```bash
git add xmatcher/methods/efficient_loftr.py xmatcher/methods/__init__.py configs/efficient_loftr.yaml tests/unit/test_config.py
git commit -m "feat(methods): EfficientLoFTR adapter"
```

---

## Task 13: Smoke tests for both adapters (local-only)

**Files:**
- Create: `tests/smoke/test_lightglue_smoke.py`
- Create: `tests/smoke/test_efficient_loftr_smoke.py`

- [ ] **Step 1: Implement LightGlue smoke**

`tests/smoke/test_lightglue_smoke.py`:

```python
from pathlib import Path
import yaml
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from xmatcher.core.config import RunConfig, build_matcher
from xmatcher.core.types import ImagePair, PreprocessMeta
import xmatcher.methods  # noqa: F401  (triggers registration)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_pair():
    img0 = to_tensor(Image.open(FIXTURES / "sample_a.jpg").convert("RGB"))
    img1 = to_tensor(Image.open(FIXTURES / "sample_b.jpg").convert("RGB"))
    H0, W0 = img0.shape[-2:]
    H1, W1 = img1.shape[-2:]
    meta0 = PreprocessMeta(
        original_size=(H0, W0), processed_size=(H0, W0),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    meta1 = PreprocessMeta(
        original_size=(H1, W1), processed_size=(H1, W1),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    return ImagePair(image0=img0, image1=img1, meta0=meta0, meta1=meta1, pair_id="smoke")


@pytest.mark.gpu
def test_lightglue_runs_on_sample_pair():
    cfg = RunConfig.model_validate(yaml.safe_load(
        Path("configs/lightglue.yaml").read_text()
    ))
    matcher = build_matcher(cfg)
    pair = _load_pair()
    res = matcher(pair)
    assert res.method == "lightglue"
    assert res.mkpts0.shape == res.mkpts1.shape
    assert res.mkpts0.shape[1] == 2
    assert len(res.mkpts0) >= 50, f"too few matches: {len(res.mkpts0)}"
    H, W = pair.meta0.original_size
    assert (res.mkpts0[:, 0] >= 0).all() and (res.mkpts0[:, 0] < W).all()
    assert (res.mkpts0[:, 1] >= 0).all() and (res.mkpts0[:, 1] < H).all()
```

- [ ] **Step 2: Implement EfficientLoFTR smoke**

`tests/smoke/test_efficient_loftr_smoke.py`:

```python
from pathlib import Path
import yaml
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from xmatcher.core.config import RunConfig, build_matcher
from xmatcher.core.types import ImagePair, PreprocessMeta
import xmatcher.methods  # noqa: F401

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_pair():
    """Load + resize so H/W are 32-aligned (so the adapter's pad is a no-op,
    making coord checks easier)."""
    img0 = to_tensor(Image.open(FIXTURES / "sample_a.jpg").convert("RGB"))
    img1 = to_tensor(Image.open(FIXTURES / "sample_b.jpg").convert("RGB"))
    # Crop to 32-aligned size for clean smoke test (no padding inside adapter).
    def _crop32(t):
        _, H, W = t.shape
        H32 = (H // 32) * 32
        W32 = (W // 32) * 32
        return t[:, :H32, :W32]
    img0 = _crop32(img0); img1 = _crop32(img1)
    H0, W0 = img0.shape[-2:]
    H1, W1 = img1.shape[-2:]
    meta0 = PreprocessMeta(
        original_size=(H0, W0), processed_size=(H0, W0),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    meta1 = PreprocessMeta(
        original_size=(H1, W1), processed_size=(H1, W1),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    return ImagePair(image0=img0, image1=img1, meta0=meta0, meta1=meta1, pair_id="smoke")


@pytest.mark.gpu
@pytest.mark.requires_weights("efficient_loftr")
def test_efficient_loftr_runs_on_sample_pair():
    cfg = RunConfig.model_validate(yaml.safe_load(
        Path("configs/efficient_loftr.yaml").read_text()
    ))
    matcher = build_matcher(cfg)
    pair = _load_pair()
    res = matcher(pair)
    assert res.method == "efficient_loftr"
    assert res.mkpts0.shape == res.mkpts1.shape
    assert res.mkpts0.shape[1] == 2
    assert len(res.mkpts0) >= 50, f"too few matches: {len(res.mkpts0)}"
    H, W = pair.meta0.original_size
    assert (res.mkpts0[:, 0] >= 0).all() and (res.mkpts0[:, 0] < W).all()
    assert (res.mkpts0[:, 1] >= 0).all() and (res.mkpts0[:, 1] < H).all()
```

- [ ] **Step 3: Verify test collection (skipped without GPU)**

Run: `pytest tests/smoke -v --collect-only`
Expected: 2 tests collected.

Run: `pytest tests/smoke -v`
Expected: On non-GPU host, 2 SKIPPED ("requires CUDA").

- [ ] **Step 4: Commit**

```bash
git add tests/smoke/test_lightglue_smoke.py tests/smoke/test_efficient_loftr_smoke.py
git commit -m "test(smoke): LightGlue and EfficientLoFTR smoke tests"
```

---

## Task 14: `PairDataset` Protocol + `FromPairListDataset`

**Files:**
- Create: `xmatcher/dataset/protocol.py`
- Create: `xmatcher/dataset/from_pair_list.py`
- Modify: `xmatcher/dataset/__init__.py`
- Create: `tests/unit/test_dataset.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_dataset.py
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from xmatcher.dataset.from_pair_list import (
    FromPairListDataset, FromPairListConfig, PreprocessConfig,
)


def _make_jpg(path: Path, h: int, w: int):
    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _setup_imgs(tmp_path):
    (tmp_path / "imgs").mkdir()
    _make_jpg(tmp_path / "imgs" / "a.jpg", 480, 640)
    _make_jpg(tmp_path / "imgs" / "b.jpg", 480, 640)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a.jpg b.jpg\n")
    return pairs


def test_dataset_iterates_and_yields_imagepair(tmp_path):
    pairs = _setup_imgs(tmp_path)
    ds = FromPairListDataset(FromPairListConfig(
        pairs_file=str(pairs),
        image_root=str(tmp_path / "imgs"),
        preprocess=PreprocessConfig(),
    ))
    items = list(ds)
    assert len(ds) == 1
    assert len(items) == 1
    p = items[0]
    assert p.image0.shape == (3, 480, 640)
    assert p.image1.shape == (3, 480, 640)
    assert p.meta0.original_size == (480, 640)
    assert p.meta0.processed_size == (480, 640)
    assert p.pair_id  # non-empty


def test_dataset_resize_long_side(tmp_path):
    pairs = _setup_imgs(tmp_path)
    ds = FromPairListDataset(FromPairListConfig(
        pairs_file=str(pairs),
        image_root=str(tmp_path / "imgs"),
        preprocess=PreprocessConfig(resize_long_side=320),
    ))
    p = next(iter(ds))
    # original 480x640, long side 640 → scale 320/640=0.5; processed 240x320
    assert p.image0.shape == (3, 240, 320)
    assert p.meta0.original_size == (480, 640)
    assert p.meta0.processed_size == (240, 320)
    assert p.meta0.scale == (0.5, 0.5)


def test_dataset_image_in_zero_one_range(tmp_path):
    pairs = _setup_imgs(tmp_path)
    ds = FromPairListDataset(FromPairListConfig(
        pairs_file=str(pairs),
        image_root=str(tmp_path / "imgs"),
        preprocess=PreprocessConfig(),
    ))
    p = next(iter(ds))
    assert p.image0.min() >= 0.0 and p.image0.max() <= 1.0
    assert p.image0.dtype == torch.float32


def test_dataset_pair_id_distinct_for_different_pairs(tmp_path):
    _make_jpg(tmp_path / "a.jpg", 100, 100)
    _make_jpg(tmp_path / "b.jpg", 100, 100)
    _make_jpg(tmp_path / "c.jpg", 100, 100)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a.jpg b.jpg\na.jpg c.jpg\n")
    ds = FromPairListDataset(FromPairListConfig(
        pairs_file=str(pairs), image_root=str(tmp_path),
        preprocess=PreprocessConfig(),
    ))
    ids = [p.pair_id for p in ds]
    assert len(ids) == 2 and ids[0] != ids[1]
```

- [ ] **Step 2: Run test to verify they fail**

Run: `pytest tests/unit/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement Protocol**

`xmatcher/dataset/protocol.py`:

```python
from __future__ import annotations
from typing import Iterator, Protocol, runtime_checkable
from xmatcher.core.types import ImagePair


@runtime_checkable
class PairDataset(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ImagePair]: ...
```

- [ ] **Step 4: Implement `FromPairListDataset`**

`xmatcher/dataset/from_pair_list.py`:

```python
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from xmatcher.core.types import ImagePair, PreprocessMeta


class PreprocessConfig(BaseModel):
    resize_long_side: int | None = None
    # crop / pad fields kept for forward-compat; FromPairList honors only resize_long_side.
    crop: tuple[int, int, int, int] | None = None
    pad_to_multiple: int | None = None


class FromPairListConfig(BaseModel):
    pairs_file: str
    image_root: str
    preprocess: PreprocessConfig = PreprocessConfig()


class FromPairListDataset:
    """Reads `pair_file`. Each line: '<rel_path0> <rel_path1>'.

    Loads images, applies optional resize by long side, returns ImagePair
    with PreprocessMeta. Minimal reference implementation, not production-grade.
    """

    def __init__(self, cfg: FromPairListConfig):
        self.cfg = cfg
        lines = [
            ln.strip() for ln in Path(cfg.pairs_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        self._pairs: list[tuple[str, str]] = [tuple(ln.split()) for ln in lines]
        self._root = Path(cfg.image_root)

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[ImagePair]:
        for rel0, rel1 in self._pairs:
            img0, meta0 = self._load(rel0)
            img1, meta1 = self._load(rel1)
            pair_id = f"{Path(rel0).stem}__{Path(rel1).stem}"
            yield ImagePair(
                image0=img0, image1=img1,
                meta0=meta0, meta1=meta1,
                pair_id=pair_id,
            )

    def _load(self, rel_path: str) -> tuple[torch.Tensor, PreprocessMeta]:
        img = Image.open(self._root / rel_path).convert("RGB")
        W_orig, H_orig = img.size           # PIL: (W, H)
        long_side = self.cfg.preprocess.resize_long_side
        if long_side is None:
            scale = 1.0
            new_W, new_H = W_orig, H_orig
        else:
            longest = max(W_orig, H_orig)
            scale = long_side / longest
            new_W = round(W_orig * scale)
            new_H = round(H_orig * scale)
            img = img.resize((new_W, new_H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0      # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        meta = PreprocessMeta(
            original_size=(H_orig, W_orig),
            processed_size=(new_H, new_W),
            crop_box=None,
            scale=(scale, scale),
            pad=(0, 0, 0, 0),
            valid_mask=None,
        )
        return tensor, meta
```

- [ ] **Step 5: Wire `dataset/__init__.py`**

```python
from xmatcher.dataset.protocol import PairDataset
from xmatcher.dataset.from_pair_list import (
    FromPairListDataset, FromPairListConfig, PreprocessConfig,
)

__all__ = [
    "PairDataset",
    "FromPairListDataset", "FromPairListConfig", "PreprocessConfig",
]
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_dataset.py -v`
Expected: 4 PASSED.

- [ ] **Step 7: Commit**

```bash
git add xmatcher/dataset/ tests/unit/test_dataset.py
git commit -m "feat(dataset): PairDataset Protocol + FromPairListDataset"
```

---

## Task 15: `DatasetConfig` and `build_dataset`

**Files:**
- Create: `xmatcher/dataset/config.py`
- Modify: `xmatcher/dataset/__init__.py`
- Create: `configs/dataset/sample_pairs.yaml`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_dataset.py`:

```python
def test_build_dataset_from_yaml_dict(tmp_path):
    from xmatcher.dataset.config import DatasetConfig, build_dataset
    pairs = _setup_imgs(tmp_path)
    cfg = DatasetConfig.model_validate({
        "type": "from_pair_list",
        "params": {
            "pairs_file": str(pairs),
            "image_root": str(tmp_path / "imgs"),
            "preprocess": {"resize_long_side": 320},
        },
    })
    ds = build_dataset(cfg)
    p = next(iter(ds))
    assert p.image0.shape[-1] == 320 or p.image0.shape[-2] == 320


def test_build_dataset_unknown_type_raises():
    import pytest
    from xmatcher.dataset.config import DatasetConfig, build_dataset
    cfg = DatasetConfig.model_validate({"type": "nope", "params": {}})
    with pytest.raises(KeyError, match="Unknown dataset"):
        build_dataset(cfg)
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/unit/test_dataset.py -v`
Expected: 2 new tests FAIL.

- [ ] **Step 3: Implement `dataset/config.py`**

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from xmatcher.dataset.protocol import PairDataset
from xmatcher.dataset.from_pair_list import FromPairListDataset, FromPairListConfig


class DatasetConfig(BaseModel):
    type: str
    params: dict = Field(default_factory=dict)


_BUILDERS = {
    "from_pair_list": (FromPairListDataset, FromPairListConfig),
}


def build_dataset(cfg: DatasetConfig) -> PairDataset:
    if cfg.type not in _BUILDERS:
        raise KeyError(
            f"Unknown dataset '{cfg.type}'. Available: {sorted(_BUILDERS)}"
        )
    cls, ParamModel = _BUILDERS[cfg.type]
    typed = ParamModel(**cfg.params)
    return cls(typed)
```

- [ ] **Step 4: Wire into `dataset/__init__.py`**

Replace contents:

```python
from xmatcher.dataset.protocol import PairDataset
from xmatcher.dataset.from_pair_list import (
    FromPairListDataset, FromPairListConfig, PreprocessConfig,
)
from xmatcher.dataset.config import DatasetConfig, build_dataset

__all__ = [
    "PairDataset",
    "FromPairListDataset", "FromPairListConfig", "PreprocessConfig",
    "DatasetConfig", "build_dataset",
]
```

- [ ] **Step 5: Sample dataset config**

`configs/dataset/sample_pairs.yaml`:

```yaml
type: from_pair_list
params:
  pairs_file: tests/fixtures/sample_pairs.txt
  image_root: tests/fixtures
  preprocess:
    resize_long_side: 1024
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_dataset.py -v`
Expected: 6 PASSED.

- [ ] **Step 7: Commit**

```bash
git add xmatcher/dataset/config.py xmatcher/dataset/__init__.py configs/dataset/sample_pairs.yaml tests/unit/test_dataset.py
git commit -m "feat(dataset): DatasetConfig and build_dataset"
```

---

## Task 16: CLI — `xmatcher.cli.run`

**Files:**
- Create: `xmatcher/cli/run.py`

- [ ] **Step 1: Implement CLI**

`xmatcher/cli/run.py`:

```python
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from xmatcher.core.config import RunConfig, build_matcher
from xmatcher.core.io import save_npz, result_to_manifest, snapshot_config
from xmatcher.dataset.config import DatasetConfig, build_dataset
import xmatcher.methods  # noqa: F401  (registers all adapters)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xmatcher.cli.run")
    p.add_argument("--method-cfg", required=True, type=Path)
    p.add_argument("--dataset-cfg", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None)
    p.add_argument("--no-postprocess", action="store_true",
                   help="Return processed-coord keypoints (skip unproject + mask filter).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    method_cfg = RunConfig.model_validate(yaml.safe_load(args.method_cfg.read_text()))
    dataset_cfg = DatasetConfig.model_validate(yaml.safe_load(args.dataset_cfg.read_text()))
    if args.device:
        method_cfg.device = args.device

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "matches").mkdir(exist_ok=True)
    snapshot_config(method_cfg, dataset_cfg, out / "config.snapshot.yaml")

    matcher = build_matcher(method_cfg)
    dataset = build_dataset(dataset_cfg)

    n_done = 0
    with (out / "manifest.jsonl").open("w") as mf:
        for i, pair in enumerate(dataset):
            if args.limit is not None and i >= args.limit:
                break
            res = matcher(pair, return_processed_coords=args.no_postprocess)
            npz_path = out / "matches" / f"{pair.pair_id}.npz"
            save_npz(res, npz_path)
            mf.write(json.dumps(result_to_manifest(res, npz_path)) + "\n")
            n_done += 1

    print(f"[xmatcher] {n_done} pair(s) processed → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify CLI help works**

Run: `python -m xmatcher.cli.run --help`
Expected: prints argparse help with all flags.

- [ ] **Step 3: Add a CLI integration test using the toy registered matcher (no GPU)**

Create `tests/contract/test_cli_smoke.py`:

```python
from pathlib import Path
import yaml
import json
import numpy as np
from PIL import Image
import torch
from pydantic import BaseModel
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register, _REGISTRY
from xmatcher.core.types import _RawOutput
from xmatcher.cli.run import main


class _CLIToyParams(BaseModel):
    pass


def _ensure_toy_registered():
    if "cli_toy" not in _REGISTRY:
        @register("cli_toy")
        class _CLIToy(BaseMatcher):
            Params = _CLIToyParams
            def _setup(self): pass
            def _forward(self, pair):
                return _RawOutput(
                    mkpts0=torch.tensor([[10.0, 10.0]]),
                    mkpts1=torch.tensor([[20.0, 20.0]]),
                    mconf=torch.tensor([0.9]),
                    dense=None,
                )


def test_cli_runs_end_to_end(tmp_path):
    _ensure_toy_registered()
    # Build minimal dataset
    img_dir = tmp_path / "imgs"; img_dir.mkdir()
    for n in ("a.jpg", "b.jpg"):
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(img_dir / n)
    (tmp_path / "pairs.txt").write_text("a.jpg b.jpg\n")

    method_cfg = tmp_path / "m.yaml"
    method_cfg.write_text(yaml.safe_dump({
        "method": "cli_toy", "device": "cpu", "params": {},
    }))
    dataset_cfg = tmp_path / "d.yaml"
    dataset_cfg.write_text(yaml.safe_dump({
        "type": "from_pair_list",
        "params": {
            "pairs_file": str(tmp_path / "pairs.txt"),
            "image_root": str(img_dir),
            "preprocess": {},
        },
    }))
    out = tmp_path / "out"

    rc = main([
        "--method-cfg", str(method_cfg),
        "--dataset-cfg", str(dataset_cfg),
        "--out", str(out),
    ])
    assert rc == 0
    assert (out / "manifest.jsonl").exists()
    assert (out / "config.snapshot.yaml").exists()
    npzs = list((out / "matches").glob("*.npz"))
    assert len(npzs) == 1
    line = (out / "manifest.jsonl").read_text().splitlines()[0]
    entry = json.loads(line)
    assert entry["method"] == "cli_toy"
    assert entry["num_matches"] == 1
```

- [ ] **Step 4: Run CLI test**

Run: `pytest tests/contract/test_cli_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xmatcher/cli/run.py tests/contract/test_cli_smoke.py
git commit -m "feat(cli): xmatcher run end-to-end"
```

---

## Task 17: Test fixtures (sample images + pairs.txt)

**Files:**
- Create: `tests/fixtures/sample_a.jpg`
- Create: `tests/fixtures/sample_b.jpg`
- Create: `tests/fixtures/sample_pairs.txt`
- Create: `tests/fixtures/README.md`

- [ ] **Step 1: Source two images**

Use the LightGlue assets (Apache 2.0):

```bash
cp thirdparty/LightGlue/assets/DSC_0411.JPG tests/fixtures/sample_a.jpg
cp thirdparty/LightGlue/assets/DSC_0410.JPG tests/fixtures/sample_b.jpg
```

If those filenames don't exist, run `ls thirdparty/LightGlue/assets/` and pick any two co-scene images and copy them.

- [ ] **Step 2: Write `sample_pairs.txt`**

```
sample_a.jpg sample_b.jpg
```

- [ ] **Step 3: Write `tests/fixtures/README.md`**

```markdown
# Test fixtures

`sample_a.jpg`, `sample_b.jpg` — two views of the same scene, sourced from the
LightGlue repo (`thirdparty/LightGlue/assets/`). License: Apache 2.0.
Used by `tests/smoke/` and any contract tests that need a real image pair.
```

- [ ] **Step 4: Verify load**

Run: `python -c "from PIL import Image; Image.open('tests/fixtures/sample_a.jpg').load(); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/
git commit -m "test: add sample image pair fixtures"
```

---

## Task 18: Run full unit + contract suite green

**Files:**
- (no edits; verification gate)

- [ ] **Step 1: Run full non-smoke suite**

Run: `pytest tests/unit tests/contract -v`
Expected: All PASS. Tally should be ≥ 35 tests (6 + 11 + 5 + 5 + 4 + 6 + 2 + 2 + 6 + 2 + 6 + 1 + others).

- [ ] **Step 2: If any test fails, fix root cause; do not skip**

If a test depends on `kornia` being installed (LightGlue's `dog_hardnet`), ensure `pip install kornia` ran. If `pytorch-lightning` import side-effects break test collection, mark the import inside the adapter `_setup` (already done) — failures here mean you may also need to gate `xmatcher.methods.efficient_loftr`'s top-level imports behind `_setup()` (already done) or skip that test path.

- [ ] **Step 3: Commit (no-op or fixup)**

If any fixes were needed:

```bash
git add -p           # review carefully
git commit -m "fix: <specific issue>"
```

---

## Task 19: `WEIGHTS.lock` + `download_weights.sh`

**Files:**
- Create: `weights/README.md`
- Create: `weights/WEIGHTS.lock`
- Create: `scripts/download_weights.sh`
- Create: `scripts/_download.py`

- [ ] **Step 1: `weights/README.md`**

```markdown
# Weights

This directory holds the lock file (`WEIGHTS.lock`). Actual weights are NOT in git
and NOT in the Docker image. They live under `$XMATCHER_WEIGHTS_DIR`
(default `~/.cache/xmatcher/`), populated by `scripts/download_weights.sh`.

| Method            | Source                                       | License        |
|-------------------|----------------------------------------------|----------------|
| LightGlue         | Auto-downloaded by upstream (`torch.hub`)    | Apache-2.0     |
| EfficientLoFTR    | Google Drive (see WEIGHTS.lock)              | Apache-2.0     |
```

- [ ] **Step 2: `weights/WEIGHTS.lock`**

> Note: `<TO_FILL_DURING_FIRST_DOWNLOAD>` placeholders MUST be replaced with
> real values during Task 19 Step 5 below. They are not allowed to ship.

```yaml
# Schema: <method> -> <variant> -> {source, sha256, target}
# source is one of: gdrive_id, http_url
efficient_loftr:
  outdoor:
    gdrive_id: "<TO_FILL_DURING_FIRST_DOWNLOAD>"
    sha256:    "<TO_FILL_DURING_FIRST_DOWNLOAD>"
    target:    "efficient_loftr/eloftr_outdoor.ckpt"
```

- [ ] **Step 3: `scripts/download_weights.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
TARGET="${XMATCHER_WEIGHTS_DIR:-$HOME/.cache/xmatcher}"
mkdir -p "$TARGET"
METHODS="${1:-efficient_loftr}"
for m in $METHODS; do
    python scripts/_download.py --method "$m" --target "$TARGET"
done
```

Make it executable: `chmod +x scripts/download_weights.sh`.

- [ ] **Step 4: `scripts/_download.py`**

```python
#!/usr/bin/env python
"""Download weights described in weights/WEIGHTS.lock.

Reads the lock file, fetches missing/changed entries for the given method,
and verifies sha256. Hard error on mismatch — never silent success.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
import yaml


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_gdrive(file_id: str, dest: Path) -> None:
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False)


def _fetch_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _process_entry(name: str, entry: dict, target_root: Path) -> None:
    target = target_root / entry["target"]
    expected_sha = entry.get("sha256", "")
    if expected_sha.startswith("<"):
        sys.exit(
            f"[download] '{name}': sha256 placeholder not filled in WEIGHTS.lock. "
            f"Run a one-time manual download, then update the lock file."
        )
    if target.exists() and _sha256_of(target) == expected_sha:
        print(f"[download] {name}: up to date ({target})")
        return
    if "gdrive_id" in entry:
        gid = entry["gdrive_id"]
        if gid.startswith("<"):
            sys.exit(f"[download] '{name}': gdrive_id placeholder not filled.")
        _fetch_gdrive(gid, target)
    elif "http_url" in entry:
        _fetch_http(entry["http_url"], target)
    else:
        sys.exit(f"[download] '{name}': missing 'gdrive_id' or 'http_url'.")
    actual = _sha256_of(target)
    if actual != expected_sha:
        target.unlink(missing_ok=True)
        sys.exit(
            f"[download] '{name}' sha256 mismatch: expected {expected_sha}, got {actual}"
        )
    print(f"[download] {name}: ok → {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()

    lock_path = Path(__file__).resolve().parents[1] / "weights" / "WEIGHTS.lock"
    lock = yaml.safe_load(lock_path.read_text()) or {}
    if args.method not in lock:
        sys.exit(f"[download] no entries for method '{args.method}' in {lock_path}")
    for variant, entry in lock[args.method].items():
        _process_entry(f"{args.method}.{variant}", entry, args.target)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: First-download bootstrap (manual one-shot)**

This step replaces the `<TO_FILL...>` placeholders in `WEIGHTS.lock`.

- Manually download `eloftr_outdoor.ckpt` from the upstream Google Drive link in `thirdparty/EfficientLoFTR/README.md`.
- Run: `sha256sum eloftr_outdoor.ckpt`
- Open the Google Drive share URL; the ID is the long alphanumeric string in the URL.
- Edit `weights/WEIGHTS.lock`: replace `gdrive_id` and `sha256` placeholders.

- [ ] **Step 6: Verify download script works**

```bash
rm -rf ~/.cache/xmatcher_test
XMATCHER_WEIGHTS_DIR=~/.cache/xmatcher_test scripts/download_weights.sh efficient_loftr
ls ~/.cache/xmatcher_test/efficient_loftr/eloftr_outdoor.ckpt
```

Expected: file present; second run reports "up to date".

- [ ] **Step 7: Commit**

```bash
git add weights/ scripts/
git commit -m "feat(weights): WEIGHTS.lock + download_weights.sh"
```

---

## Task 20: Docker — Dockerfile + requirements + build.sh + .dockerignore

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/requirements.common.txt`
- Create: `docker/requirements.lightglue.txt`
- Create: `docker/requirements.eloftr.txt`
- Create: `docker/build.sh`
- Create: `.dockerignore`

- [ ] **Step 1: `.dockerignore`** (at repo root)

```
.git
.github
.vscode
outputs/
weights/*
!weights/README.md
!weights/WEIGHTS.lock
**/__pycache__
**/*.pyc
.env
VRecHub
docs/
*.md
!README.md
```

- [ ] **Step 2: `docker/requirements.common.txt`**

```
numpy
opencv-python
pyyaml
pydantic>=2.0
einops
gdown
pillow
```

- [ ] **Step 3: `docker/requirements.lightglue.txt`**

```
kornia
```

- [ ] **Step 4: `docker/requirements.eloftr.txt`**

```
pytorch-lightning>=2.0,<2.5
yacs
loguru
h5py
```

- [ ] **Step 5: `docker/Dockerfile`**

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git wget curl ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Layer 2: PyTorch (separate so common deps cache is preserved)
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Layer 3: common deps
COPY docker/requirements.common.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.common.txt

# Layer 4: per-method deps
COPY docker/requirements.lightglue.txt docker/requirements.eloftr.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.lightglue.txt && \
    pip install --no-cache-dir -r /tmp/requirements.eloftr.txt

# Layer 5: app code
WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app
ENV XMATCHER_WEIGHTS_DIR=/root/.cache/xmatcher

ENTRYPOINT ["python", "-m", "xmatcher.cli.run"]
CMD ["--help"]
```

- [ ] **Step 6: `docker/build.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
git submodule update --init --recursive
docker build -t xmatcher:dev -f docker/Dockerfile .
```

`chmod +x docker/build.sh`.

- [ ] **Step 7: Build the image (sanity check)**

Run: `docker/build.sh`
Expected: builds successfully. (Actual GPU runtime check is done by smoke tests later, not here.)

If build fails because pip-installed `pytorch-lightning` requires a newer torch: relax the upper bound or pin a lower lightning. Don't ignore the error.

- [ ] **Step 8: Verify ENTRYPOINT works**

Run: `docker run --rm xmatcher:dev`
Expected: prints argparse `--help` output.

- [ ] **Step 9: Commit**

```bash
git add docker/ .dockerignore
git commit -m "build(docker): CUDA 12.1 + PyTorch 2.4 image, ENTRYPOINT=cli.run"
```

---

## Task 21: GitHub Actions — test workflow

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Implement test workflow**

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Configure git for private submodules
        run: |
          git config --global url."https://${{ secrets.GH_PAT }}@github.com/".insteadOf "git@github.com:"
          git config --global url."https://${{ secrets.GH_PAT }}@github.com/".insteadOf "https://github.com/"

      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GH_PAT }}
          submodules: recursive

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install CPU PyTorch
        run: |
          pip install --upgrade pip
          pip install torch==2.4.1 torchvision==0.19.1 \
              --index-url https://download.pytorch.org/whl/cpu

      - name: Install package
        run: pip install -e .[test] kornia pytorch-lightning yacs loguru h5py

      - name: Run unit + contract tests
        run: pytest tests/unit tests/contract -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: run unit + contract tests on push/PR"
```

- [ ] **Step 3: (Manual) Add `GH_PAT` to repo secrets**

Open `https://github.com/<owner>/XMatcher/settings/secrets/actions` and add `GH_PAT` = a fresh PAT with `repo` scope. The token previously pasted in `.env` should be revoked first.

This step is documented for the operator; the workflow file commit itself is sufficient for plan completion.

---

## Task 22: GitHub Actions — Docker build & GHCR push

**Files:**
- Create: `.github/workflows/docker-build.yml`

- [ ] **Step 1: Implement workflow**

```yaml
name: Build & Push Image

on:
  push:
    branches: [main]
    tags: ["v*"]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Configure git for private submodules
        run: |
          git config --global url."https://${{ secrets.GH_PAT }}@github.com/".insteadOf "git@github.com:"
          git config --global url."https://${{ secrets.GH_PAT }}@github.com/".insteadOf "https://github.com/"

      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GH_PAT }}
          submodules: recursive

      - uses: docker/setup-buildx-action@v3

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Lowercase repo owner
        id: lower
        run: echo "owner=${OWNER,,}" >> "$GITHUB_OUTPUT"
        env:
          OWNER: ${{ github.repository_owner }}

      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ steps.lower.outputs.owner }}/xmatcher:latest
            ghcr.io/${{ steps.lower.outputs.owner }}/xmatcher:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/docker-build.yml
git commit -m "ci: build and push image to GHCR on main + tags"
```

---

## Task 23: README.md (project-level)

**Files:**
- Create: `README.md` (replaces the existing single-line README if any)

- [ ] **Step 1: Read existing README to avoid losing content**

Run: `cat README.md` (if it exists, preserve any non-trivial bits).

- [ ] **Step 2: Write `README.md`**

```markdown
# XMatcher

Unified keypoints matching interface — a single CLI and Python API for running
multiple keypoint matchers (LightGlue, EfficientLoFTR; RoMa v2 / XFeat planned)
through a YAML-configured pipeline.

## Quick start

```bash
# 1. Submodules
git submodule update --init --recursive

# 2. Install
pip install -e .[test] kornia pytorch-lightning yacs loguru h5py

# 3. Weights (EfficientLoFTR — LightGlue auto-downloads)
scripts/download_weights.sh efficient_loftr

# 4. Run
python -m xmatcher.cli.run \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/lg_demo/
```

## Docker

```bash
docker/build.sh   # build xmatcher:dev locally
docker run --rm --gpus all \
    -v $HOME/.cache/xmatcher:/root/.cache/xmatcher \
    -v $PWD/outputs:/app/outputs \
    xmatcher:dev \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/lg/
```

## Layout

- `xmatcher/core/` — shared types, base matcher, registry, config, IO.
- `xmatcher/methods/` — per-method adapters (each registers via `@register("name")`).
- `xmatcher/dataset/` — `PairDataset` Protocol + `FromPairListDataset`.
- `xmatcher/cli/run.py` — `--method-cfg / --dataset-cfg / --out` CLI.
- `thirdparty/` — upstream method repos as git submodules.
- `configs/` — sample method and dataset YAMLs.
- `tests/{unit,contract,smoke,fixtures}/` — three-layer test suite.

## Tests

```bash
pytest tests/unit tests/contract -v          # CI-safe (no GPU, no weights)
pytest tests/smoke -v                        # local-only (GPU + weights required)
```

## Design

See [`docs/superpowers/specs/2026-06-06-unified-kpts-matching-design.md`](docs/superpowers/specs/2026-06-06-unified-kpts-matching-design.md).
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: project README"
```

---

## Task 24: Final verification — full local suite + CLI smoke

**Files:**
- (no edits; gate)

- [ ] **Step 1: Full non-smoke**

Run: `pytest tests/unit tests/contract -v`
Expected: all PASS.

- [ ] **Step 2: Local smoke (if GPU + weights present)**

Run: `pytest tests/smoke -v`
Expected: PASS, or SKIPPED (no GPU / no weights). No FAIL.

- [ ] **Step 3: Live CLI dry-run**

```bash
mkdir -p tests/fixtures
python -m xmatcher.cli.run \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/_smoke_/ \
    --device cpu \
    --limit 1
```

Expected (on CPU): runs to completion, prints `[xmatcher] 1 pair(s) processed`. If LightGlue is too slow on CPU and times out, replace `--device cpu` with `--device cuda` on a GPU host. Output dir contains `manifest.jsonl`, `config.snapshot.yaml`, `matches/<pair>.npz`.

- [ ] **Step 4: Tag**

```bash
git tag v0.1.0-mvp
```

- [ ] **Step 5: Push and watch GHA**

```bash
git push origin main --tags
```

Then verify `Tests` workflow goes green and the image lands in GHCR
`ghcr.io/<owner>/xmatcher:v0.1.0-mvp`.

- [ ] **Step 6: Final commit (if any cleanup)**

If any small fixes were needed in step 3 (e.g., a typo, a missing fixture path),
commit them with a focused message.

---

