"""Contract tests for `BaseMatcher.batch_call`.

These verify the protocol — order preservation, length, no-pair / single-pair
edge cases — without needing a real GPU model. We use a Mock matcher that
records what it was called with.
"""
from __future__ import annotations
import torch
import pytest
from pydantic import BaseModel
from xmatcher.core.base import BaseMatcher
from xmatcher.core.types import (
    PreprocessMeta, ImagePair, MatchResult, _RawOutput,
)


class _MockParams(BaseModel):
    pass


class _MockMatcher(BaseMatcher):
    method_name = "mock"
    Params = _MockParams

    def _setup(self):
        self.single_calls = 0
        self.batch_calls: list[int] = []

    def _forward(self, pair):
        self.single_calls += 1
        # Per-pair output reflects pair_id so we can verify order preservation.
        seed = int(pair.pair_id.split("-")[-1])
        return _RawOutput(
            mkpts0=torch.tensor([[float(seed), 0.0]]),
            mkpts1=torch.tensor([[float(seed) * 2, 0.0]]),
            mconf=torch.tensor([1.0]),
            dense=None,
        )


class _MockBatchedMatcher(_MockMatcher):
    """Subclass that overrides `_forward_batch` to a single batched call."""

    def _forward_batch(self, pairs):
        self.batch_calls.append(len(pairs))
        return [self._forward(p) for p in pairs]


def _make_pair(pair_id: str) -> ImagePair:
    meta = PreprocessMeta(
        original_size=(100, 100), processed_size=(100, 100),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    img = torch.zeros(3, 100, 100)
    return ImagePair(image0=img, image1=img.clone(), meta0=meta, meta1=meta,
                     pair_id=pair_id)


def test_batch_call_empty_list_returns_empty():
    m = _MockMatcher(device="cpu", params=_MockParams())
    assert m.batch_call([]) == []
    assert m.single_calls == 0


def test_batch_call_default_falls_back_to_per_pair_forward():
    m = _MockMatcher(device="cpu", params=_MockParams())
    pairs = [_make_pair(f"p-{i}") for i in range(3)]
    results = m.batch_call(pairs)
    assert len(results) == 3
    # _forward called once per pair via the default _forward_batch fallback
    assert m.single_calls == 3


def test_batch_call_preserves_pair_order():
    m = _MockMatcher(device="cpu", params=_MockParams())
    pairs = [_make_pair("p-7"), _make_pair("p-2"), _make_pair("p-5")]
    results = m.batch_call(pairs)
    assert [r.pair_id for r in results] == ["p-7", "p-2", "p-5"]
    # The seed-encoded coordinate should match the original pair order.
    assert results[0].mkpts0[0, 0].item() == 7.0
    assert results[1].mkpts0[0, 0].item() == 2.0
    assert results[2].mkpts0[0, 0].item() == 5.0


def test_batch_call_invokes_forward_batch_when_overridden():
    m = _MockBatchedMatcher(device="cpu", params=_MockParams())
    pairs = [_make_pair(f"p-{i}") for i in range(4)]
    m.batch_call(pairs)
    assert m.batch_calls == [4]


def test_batch_call_validates_output_length():
    """If a subclass returns the wrong number of outputs, raise loudly."""
    class _BrokenMatcher(_MockMatcher):
        def _forward_batch(self, pairs):
            return [self._forward(pairs[0])]  # always 1, regardless of input
    m = _BrokenMatcher(device="cpu", params=_MockParams())
    pairs = [_make_pair(f"p-{i}") for i in range(3)]
    with pytest.raises(RuntimeError, match="returned 1 outputs for 3 pairs"):
        m.batch_call(pairs)


def test_batch_call_single_pair_matches_per_pair_call():
    """For B=1 the results should match `__call__(pair)` numerically."""
    m = _MockMatcher(device="cpu", params=_MockParams())
    pair = _make_pair("p-42")
    single = m(pair)
    batched = m.batch_call([pair])
    assert len(batched) == 1
    assert torch.equal(single.mkpts0, batched[0].mkpts0)
    assert torch.equal(single.mkpts1, batched[0].mkpts1)
    assert torch.equal(single.mconf, batched[0].mconf)
    assert single.method == batched[0].method
    assert single.pair_id == batched[0].pair_id


def test_batch_call_runs_postprocess_per_pair():
    """unproject must be applied with each pair's own meta — not the first
    pair's, not pooled."""
    class _ConstMatcher(_MockMatcher):
        def _forward(self, pair):
            return _RawOutput(
                mkpts0=torch.tensor([[10.0, 10.0]]),
                mkpts1=torch.tensor([[10.0, 10.0]]),
                mconf=torch.tensor([1.0]),
                dense=None,
            )

    # Pair A: identity meta (proc==orig). Pair B: 2x downsample (proc=100,orig=200).
    metaA = PreprocessMeta(
        original_size=(100, 100), processed_size=(100, 100),
        crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0), valid_mask=None,
    )
    metaB = PreprocessMeta(
        original_size=(200, 200), processed_size=(100, 100),
        crop_box=None, scale=(0.5, 0.5), pad=(0, 0, 0, 0), valid_mask=None,
    )
    img = torch.zeros(3, 100, 100)
    pairA = ImagePair(image0=img, image1=img.clone(), meta0=metaA, meta1=metaA, pair_id="A")
    pairB = ImagePair(image0=img, image1=img.clone(), meta0=metaB, meta1=metaB, pair_id="B")

    m = _ConstMatcher(device="cpu", params=_MockParams())
    rA, rB = m.batch_call([pairA, pairB])
    # A: identity → (10, 10) stays (10, 10).
    assert torch.allclose(rA.mkpts0, torch.tensor([[10.0, 10.0]]))
    # B: 0.5x → (10, 10) maps back to original (20, 20).
    assert torch.allclose(rB.mkpts0, torch.tensor([[20.0, 20.0]]))
