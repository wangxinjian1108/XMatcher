from __future__ import annotations
import abc
import time
from dataclasses import replace
from typing import ClassVar
import torch
from pydantic import BaseModel
from xmatcher.core.types import ImagePair, MatchResult, _RawOutput
from xmatcher.core.preprocess import unproject, filter_by_mask, apply_input_mask


class BaseMatcher(abc.ABC):
    """Template Method base. Subclasses fill in `_setup` + `_forward`.

    Two public entry points:
      * `__call__(pair)` — single-pair, the original contract. Subclasses MUST
        implement `_forward`; this is the only required override.
      * `batch_call(pairs)` — multi-pair. By default it loops over `__call__`,
        which gives correct results without any GPU batching. Subclasses with
        a real batched forward path override `_forward_batch` to take
        advantage of it.

    Both paths run a uniform pre/post pipeline around the subclass forward:

      pre  (template, applied to inputs before `_forward`):
        1. zero out pixels of `image0/image1` that fall outside `valid_mask`
           (`apply_input_mask`). The subclass receives a copy of the pair
           with masked images; the original `pair` object is not mutated.

      post (template, applied to subclass outputs):
        1. filter raw matches by `valid_mask` (in processed coords)
        2. unproject surviving matches to original coords

    The template also owns timing.
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

    def _forward_batch(self, pairs: list[ImagePair]) -> list[_RawOutput]:
        """Optional: batched forward. Default falls back to per-pair `_forward`.

        Subclasses override this when the underlying model has a real batched
        path. Output length must equal `len(pairs)` and order must be preserved.
        """
        return [self._forward(p) for p in pairs]

    @torch.inference_mode()
    def __call__(
        self,
        pair: ImagePair,
        *,
        return_processed_coords: bool = False,
    ) -> MatchResult:
        t0 = time.perf_counter()
        raw = self._forward(_apply_pair_input_mask(pair))
        return self._postprocess(
            raw, pair, t0, return_processed_coords=return_processed_coords
        )

    @torch.inference_mode()
    def batch_call(
        self,
        pairs: list[ImagePair],
        *,
        return_processed_coords: bool = False,
    ) -> list[MatchResult]:
        """Process multiple pairs. Order of returned `MatchResult` matches `pairs`.

        Default implementation calls `_forward_batch` (which itself defaults to
        looping `_forward`). Each result records the same `runtime_ms` value:
        the wall-clock for the whole batch divided by len(pairs). Per-pair
        timing is not preserved in batch mode.
        """
        if not pairs:
            return []
        t0 = time.perf_counter()
        masked_pairs = [_apply_pair_input_mask(p) for p in pairs]
        raws = self._forward_batch(masked_pairs)
        if len(raws) != len(pairs):
            raise RuntimeError(
                f"_forward_batch returned {len(raws)} outputs for {len(pairs)} "
                f"pairs; lengths must match"
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_pair_ms = elapsed_ms / len(pairs)
        return [
            self._postprocess(
                raw,
                pair,
                # Synthetic t0 such that `now - t0_synth == per_pair_ms`.
                time.perf_counter() - per_pair_ms / 1000,
                return_processed_coords=return_processed_coords,
            )
            for raw, pair in zip(raws, pairs)
        ]

    def _postprocess(
        self,
        raw: _RawOutput,
        pair: ImagePair,
        t0: float,
        *,
        return_processed_coords: bool,
    ) -> MatchResult:
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


def _apply_pair_input_mask(pair: ImagePair) -> ImagePair:
    """Return a new ImagePair whose images are zeroed outside their valid_mask.

    No-op (returns the original pair) when neither side has a valid_mask set.
    The metas, pair_id and extras are passed through unchanged. The original
    `pair.image0/1` tensors are never mutated.
    """
    has_mask = pair.meta0.valid_mask is not None or pair.meta1.valid_mask is not None
    if not has_mask:
        return pair
    return replace(
        pair,
        image0=apply_input_mask(pair.image0, pair.meta0),
        image1=apply_input_mask(pair.image1, pair.meta1),
    )
