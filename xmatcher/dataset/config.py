from __future__ import annotations
from pydantic import BaseModel, Field
from xmatcher.dataset.protocol import PairDataset
from xmatcher.dataset.from_pair_list import FromPairListDataset, FromPairListConfig


class DatasetConfig(BaseModel):
    type: str
    params: dict = Field(default_factory=dict)


_BUILDERS = {
    "from_pair_list": (FromPairListDataset, FromPairListConfig),
}


def build_dataset(cfg: DatasetConfig) -> PairDataset:
    if cfg.type not in _BUILDERS:
        raise KeyError(
            f"Unknown dataset '{cfg.type}'. Available: {sorted(_BUILDERS)}"
        )
    cls, ParamModel = _BUILDERS[cfg.type]
    typed = ParamModel(**cfg.params)
    return cls(typed)
