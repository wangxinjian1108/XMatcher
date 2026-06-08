"""Unit tests for apply_input_mask + filter_kpts_by_mask.

These cover the two new mask-driven helpers used by adapters to enforce
'keypoints only inside the user's mask' at two stages:
  1. Image preprocessing (zero out pixels outside the mask).
  2. Post-detection (drop keypoints landing outside the mask).
"""
from __future__ import annotations
import torch
import pytest
from xmatcher.core.types import PreprocessMeta
from xmatcher.core.preprocess import apply_input_mask, filter_kpts_by_mask


def _meta_no_mask():
    return PreprocessMeta(
        original_size=(10, 10), processed_size=(10, 10),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )


def _meta_with_mask(mask):
    H, W = mask.shape
    return PreprocessMeta(
        original_size=(H, W), processed_size=(H, W),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=mask,
    )


# ===== apply_input_mask =====


def test_apply_input_mask_no_op_when_mask_none():
    img = torch.rand(3, 10, 10)
    out = apply_input_mask(img, _meta_no_mask())
    assert torch.equal(out, img)


def test_apply_input_mask_zeros_outside_mask():
    """Mask: top-left 4x4 valid, rest invalid."""
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[:4, :4] = True
    img = torch.ones(3, 10, 10)
    out = apply_input_mask(img, _meta_with_mask(mask))
    # Inside mask region → unchanged (1.0).
    assert (out[:, :4, :4] == 1.0).all()
    # Outside mask region → zeroed.
    assert (out[:, 4:, :] == 0.0).all()
    assert (out[:, :, 4:] == 0.0).all()


def test_apply_input_mask_does_not_mutate_input():
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[:4, :4] = True
    img = torch.ones(3, 10, 10)
    img_copy = img.clone()
    _ = apply_input_mask(img, _meta_with_mask(mask))
    assert torch.equal(img, img_copy)


def test_apply_input_mask_preserves_dtype_and_device():
    mask = torch.ones(10, 10, dtype=torch.bool)
    img = torch.rand(3, 10, 10, dtype=torch.float64)
    out = apply_input_mask(img, _meta_with_mask(mask))
    assert out.dtype == img.dtype
    assert out.device == img.device


def test_apply_input_mask_supports_grayscale():
    """Works for both 3-channel RGB and 1-channel gray inputs."""
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:4, :4] = True
    img = torch.ones(1, 8, 8)
    out = apply_input_mask(img, _meta_with_mask(mask))
    assert out.shape == (1, 8, 8)
    assert (out[:, :4, :4] == 1.0).all()
    assert (out[:, 4:, :] == 0.0).all()


# ===== filter_kpts_by_mask =====


def test_filter_kpts_no_mask_keeps_in_bounds():
    pts = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
    keep = filter_kpts_by_mask(pts, _meta_no_mask())
    assert keep.tolist() == [True, True]


def test_filter_kpts_no_mask_drops_out_of_bounds():
    pts = torch.tensor([[-1.0, 5.0], [5.0, 12.0], [5.0, 5.0]])
    keep = filter_kpts_by_mask(pts, _meta_no_mask())
    assert keep.tolist() == [False, False, True]


def test_filter_kpts_drops_points_outside_mask():
    """Mask: top-left 5x5 valid."""
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[:5, :5] = True
    pts = torch.tensor([[2.0, 2.0], [7.0, 7.0]])
    keep = filter_kpts_by_mask(pts, _meta_with_mask(mask))
    assert keep.tolist() == [True, False]


def test_filter_kpts_uses_row_col_indexing():
    """Mask is (H, W); kpts are (u=col, v=row). Verify indexing order."""
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[2, 7] = True   # row=2, col=7
    pts = torch.tensor([[7.0, 2.0], [2.0, 7.0]])  # u=7,v=2 valid; u=2,v=7 invalid
    keep = filter_kpts_by_mask(pts, _meta_with_mask(mask))
    assert keep.tolist() == [True, False]


def test_filter_kpts_empty_input():
    pts = torch.zeros((0, 2))
    keep = filter_kpts_by_mask(pts, _meta_no_mask())
    assert keep.shape == (0,) and keep.dtype == torch.bool
