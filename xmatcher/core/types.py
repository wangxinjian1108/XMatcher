from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import torch


@dataclass
class DenseField:
    warp: torch.Tensor
    certainty: torch.Tensor
    coord_space: Literal["processed", "original"] = "processed"


@dataclass
class PreprocessMeta:
    """Describes the geometric chain from original image to processed image.

    Apply order (original → processed): crop → scale → pad.
    Inverse (processed → original) is exposed as `affine_proc_to_orig`.
    """
    original_size: tuple[int, int]              # (H_orig, W_orig)
    processed_size: tuple[int, int]             # (H_proc, W_proc)
    crop_box: tuple[int, int, int, int] | None  # (x0, y0, x1, y1) on original
    scale: tuple[float, float]                  # (sx, sy)
    pad: tuple[int, int, int, int]              # (left, top, right, bottom)
    valid_mask: torch.Tensor | None             # (H_proc, W_proc) bool
    affine_proc_to_orig: torch.Tensor = field(init=False)

    def __post_init__(self):
        sx, sy = self.scale
        if sx <= 0 or sy <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.valid_mask is not None:
            if tuple(self.valid_mask.shape) != self.processed_size:
                raise ValueError(
                    f"valid_mask shape {tuple(self.valid_mask.shape)} "
                    f"!= processed_size {self.processed_size}"
                )
        cx0, cy0 = (self.crop_box[0], self.crop_box[1]) if self.crop_box else (0, 0)
        pad_l, pad_t, _, _ = self.pad
        # proc → cropped-original (subtract pad, divide by scale, add crop origin):
        # u_orig = (u_proc - pad_l) / sx + cx0
        # v_orig = (v_proc - pad_t) / sy + cy0
        # → affine 2×3: [[1/sx, 0, cx0 - pad_l/sx], [0, 1/sy, cy0 - pad_t/sy]]
        self.affine_proc_to_orig = torch.tensor(
            [
                [1.0 / sx, 0.0, cx0 - pad_l / sx],
                [0.0, 1.0 / sy, cy0 - pad_t / sy],
            ],
            dtype=torch.float32,
        )


@dataclass
class ImagePair:
    image0: torch.Tensor
    image1: torch.Tensor
    meta0: PreprocessMeta
    meta1: PreprocessMeta
    pair_id: str
    extras: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    mkpts0: torch.Tensor
    mkpts1: torch.Tensor
    mconf: torch.Tensor
    method: str
    pair_id: str
    runtime_ms: float
    dense: DenseField | None = None


@dataclass
class _RawOutput:
    """Internal: what `BaseMatcher._forward` returns. Coords in processed space."""
    mkpts0: torch.Tensor
    mkpts1: torch.Tensor
    mconf: torch.Tensor
    dense: DenseField | None = None
