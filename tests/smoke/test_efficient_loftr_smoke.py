from pathlib import Path
import yaml
import pytest
import torch
import numpy as np
from PIL import Image
from xmatcher.core.config import RunConfig, build_matcher
from xmatcher.core.types import ImagePair, PreprocessMeta
import xmatcher.methods  # noqa: F401

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _pil_to_tensor(pil_img):
    """Convert PIL image to CHW float tensor [0, 1]."""
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)  # grayscale: HW -> 1HW
    else:
        arr = np.transpose(arr, (2, 0, 1))  # RGB: HWC -> CHW
    return torch.from_numpy(arr)


def _load_pair():
    """Load + crop so H/W are 32-aligned (so the adapter's pad is a no-op,
    making coord checks easier)."""
    img0 = _pil_to_tensor(Image.open(FIXTURES / "sample_a.jpg").convert("RGB"))
    img1 = _pil_to_tensor(Image.open(FIXTURES / "sample_b.jpg").convert("RGB"))
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
