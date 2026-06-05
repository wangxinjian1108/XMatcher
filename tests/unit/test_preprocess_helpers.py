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
