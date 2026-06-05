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
