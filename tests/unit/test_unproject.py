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
