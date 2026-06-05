from __future__ import annotations
from typing import Iterator, Protocol, runtime_checkable
from xmatcher.core.types import ImagePair


@runtime_checkable
class PairDataset(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ImagePair]: ...
