from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from xmatcher.core.types import ImagePair, PreprocessMeta


class PreprocessConfig(BaseModel):
    resize_long_side: int | None = None
    # crop / pad fields kept for forward-compat; FromPairList honors only resize_long_side.
    crop: tuple[int, int, int, int] | None = None
    pad_to_multiple: int | None = None


class FromPairListConfig(BaseModel):
    pairs_file: str
    image_root: str
    preprocess: PreprocessConfig = PreprocessConfig()


class FromPairListDataset:
    """Reads `pair_file`. Each line: '<rel_path0> <rel_path1>'.

    Loads images, applies optional resize by long side, returns ImagePair
    with PreprocessMeta. Minimal reference implementation, not production-grade.
    """

    def __init__(self, cfg: FromPairListConfig):
        self.cfg = cfg
        lines = [
            ln.strip() for ln in Path(cfg.pairs_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        self._pairs: list[tuple[str, str]] = [tuple(ln.split()) for ln in lines]
        self._root = Path(cfg.image_root)

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[ImagePair]:
        for rel0, rel1 in self._pairs:
            img0, meta0 = self._load(rel0)
            img1, meta1 = self._load(rel1)
            pair_id = f"{Path(rel0).stem}__{Path(rel1).stem}"
            yield ImagePair(
                image0=img0, image1=img1,
                meta0=meta0, meta1=meta1,
                pair_id=pair_id,
            )

    def _load(self, rel_path: str) -> tuple[torch.Tensor, PreprocessMeta]:
        img = Image.open(self._root / rel_path).convert("RGB")
        W_orig, H_orig = img.size           # PIL: (W, H)
        long_side = self.cfg.preprocess.resize_long_side
        if long_side is None:
            scale = 1.0
            new_W, new_H = W_orig, H_orig
        else:
            longest = max(W_orig, H_orig)
            scale = long_side / longest
            new_W = round(W_orig * scale)
            new_H = round(H_orig * scale)
            img = img.resize((new_W, new_H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0      # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        meta = PreprocessMeta(
            original_size=(H_orig, W_orig),
            processed_size=(new_H, new_W),
            crop_box=None,
            scale=(scale, scale),
            pad=(0, 0, 0, 0),
            valid_mask=None,
        )
        return tensor, meta
