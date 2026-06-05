from __future__ import annotations
import random
from typing import Literal
import numpy as np
import torch
from pydantic import BaseModel, Field
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import get_matcher_cls


class RunConfig(BaseModel):
    method: str
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    seed: int = 0
    params: dict = Field(default_factory=dict)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_matcher(cfg: RunConfig) -> BaseMatcher:
    cls = get_matcher_cls(cfg.method)
    typed_params = cls.Params(**cfg.params)
    _set_seed(cfg.seed)
    return cls(device=cfg.device, params=typed_params)
