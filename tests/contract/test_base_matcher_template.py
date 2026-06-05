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
