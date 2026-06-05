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
        processed_size=(110, 120),  # H=100+10 (top pad), W=100+20 (left pad)
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
