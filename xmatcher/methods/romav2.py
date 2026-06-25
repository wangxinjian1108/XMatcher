"""RoMA v2 dense matcher adapter for XMatcher.

RoMA v2 is a dense feature matcher (predicts a warp + confidence map for the
entire image, then samples sparse correspondences). Compared to LightGlue
(sparse keypoint detection + transformer matcher) it tends to produce more
matches in low-texture regions but is slower per pair.

Settings (see RoMaV2.apply_setting):
  - "turbo":    320×320, mono-direction, fastest
  - "fast":     512×512
  - "base":     640×640
  - "precise":  800×800 + 1280×1280 refine, bidirectional, slowest
  - "mega1500": 800×800 + 1024×1024 refine, bidirectional, threshold=0.05
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field

from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register
from xmatcher.core.types import _RawOutput, ImagePair
from xmatcher.core.preprocess import filter_kpts_by_mask


# Ensure the vendored RoMA v2 package is importable.
_THIRDPARTY = (
    Path(__file__).resolve().parents[2] / "thirdparty" / "RoMaV2" / "src"
)
if str(_THIRDPARTY) not in sys.path:
    sys.path.insert(0, str(_THIRDPARTY))


class RomaV2Params(BaseModel):
    setting: Literal["turbo", "fast", "base", "precise", "mega1500"] = "base"
    num_correspondences: int = Field(default=2048, gt=0)
    cert_threshold: float = 0.0
    # Local checkpoint path. None -> RoMaV2 resolves $ROMAV2_WEIGHTS then the
    # default /opt/var/models/romav2/romav2.0.1.pt. Runtime downloads are
    # disabled (air-gapped release image).
    weights: str | None = None


@register("romav2")
class RomaV2Matcher(BaseMatcher):
    Params = RomaV2Params

    def _setup(self):
        from romav2 import RoMaV2

        self.model = (
            RoMaV2(weights_path=self.params.weights).eval().to(self.device)
        )
        self.model.apply_setting(self.params.setting)

    def _image_to_tensor(self, img: torch.Tensor) -> torch.Tensor:
        """ImagePair gives image as (3, H, W) float [0, 1] on some device.

        RoMaV2._load_image expects (B, 3, H, W) or HxWx3 numpy/PIL. We pass a
        4D float tensor on self.device which is the cleanest path through
        _load_image's torch.Tensor branch.
        """
        if img.dim() == 3:
            img = img[None]  # (1, 3, H, W)
        return img.to(self.device).contiguous()

    def _forward(self, pair: ImagePair) -> _RawOutput:
        img_A = self._image_to_tensor(pair.image0)
        img_B = self._image_to_tensor(pair.image1)
        H_A, W_A = img_A.shape[-2], img_A.shape[-1]
        H_B, W_B = img_B.shape[-2], img_B.shape[-1]

        with torch.no_grad():
            preds = self.model.match(img_A, img_B)
            try:
                sampled = self.model.sample(preds, self.params.num_correspondences)
            except RuntimeError:
                # No reliable correspondences (e.g. low overlap region).
                return _RawOutput(
                    mkpts0=torch.empty(0, 2, device=self.device),
                    mkpts1=torch.empty(0, 2, device=self.device),
                    mconf=torch.empty(0, device=self.device),
                    dense=None,
                )

        # sample() returns a tuple where the first element is (N, 4)
        # normalized [-1, 1] coordinates (x_A, y_A, x_B, y_B) and the
        # second is confidence.
        sampled_matches, sampled_confidence = sampled[0], sampled[1]

        if self.params.cert_threshold > 0:
            keep = sampled_confidence >= self.params.cert_threshold
            sampled_matches = sampled_matches[keep]
            sampled_confidence = sampled_confidence[keep]

        # Convert normalized → pixel coordinates.
        norm_A = sampled_matches[:, :2]   # (N, 2)
        norm_B = sampled_matches[:, 2:]   # (N, 2)
        mkpts0 = self._normalized_to_pixel(norm_A, H_A, W_A)
        mkpts1 = self._normalized_to_pixel(norm_B, H_B, W_B)

        # Filter by valid_mask (foreground / ROI) — same contract as other
        # matchers: no keypoint leaves this adapter outside the mask.
        keep0 = self._mask_keep(mkpts0, pair.meta0)
        keep1 = self._mask_keep(mkpts1, pair.meta1)
        keep = keep0 & keep1
        mkpts0 = mkpts0[keep]
        mkpts1 = mkpts1[keep]
        mconf = sampled_confidence[keep]

        return _RawOutput(
            mkpts0=mkpts0, mkpts1=mkpts1, mconf=mconf, dense=None,
        )

    @staticmethod
    def _normalized_to_pixel(
        norm: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Convert normalized coords in [-1, 1] (x, y) to pixel coords."""
        x = (norm[:, 0] + 1.0) * 0.5 * (W - 1)
        y = (norm[:, 1] + 1.0) * 0.5 * (H - 1)
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _mask_keep(kpts: torch.Tensor, meta) -> torch.Tensor:
        if meta.valid_mask is None:
            return torch.ones(kpts.shape[0], dtype=torch.bool, device=kpts.device)
        return filter_kpts_by_mask(kpts, meta)
