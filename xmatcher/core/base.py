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
