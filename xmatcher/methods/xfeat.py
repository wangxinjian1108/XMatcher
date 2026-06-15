"""XFeat (Accelerated Features, CVPR 2024) adapter for XMatcher.

XFeat is a lightweight sparse matcher (~smaller than SuperPoint) with two
match modes:
  - "sparse"  : detectAndCompute + MNN matching (match_xfeat)
  - "semidense": batched semi-dense matching (match_xfeat_star)

Weights ship with the thirdparty repo under
`thirdparty/accelerated_features/weights/xfeat.pt` (no download needed).
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


# Make the vendored XFeat package importable. The repo lays out its code
# at thirdparty/accelerated_features/modules/xfeat.py and the upstream
# scripts add the repo root to sys.path before doing `from modules.xfeat`.
# We mirror that here.
_REPO = (
    Path(__file__).resolve().parents[2] / "thirdparty" / "accelerated_features"
)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class XFeatParams(BaseModel):
    mode: Literal["sparse", "semidense"] = "sparse"
    top_k: int = Field(default=4096, gt=0)
    detection_threshold: float = 0.05
    # Sparse mode: minimum cosine similarity for MNN matching (-1 = no
    # filtering). Semidense mode ignores this.
    min_cossim: float = -1.0


@register("xfeat")
class XFeatMatcher(BaseMatcher):
    Params = XFeatParams

    def _setup(self):
        # XFeat does many `from modules.X import ...` calls inside its
        # own files. That collides with vision-label-hub's top-level
        # `modules/` package. Strategy: temporarily evict any `modules*`
        # entries from sys.modules and put XFeat's repo at sys.path[0],
        # do all the imports so XFeat's whole import graph caches into
        # sys.modules, save those entries under a private prefix, then
        # restore the host's `modules.*` and use the cached entries on
        # demand (see `_forward_with_xfeat_modules`).
        import sys

        host_modules = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "modules" or k.startswith("modules.")
        }
        sys.path.insert(0, str(_REPO))
        try:
            import modules.xfeat as _xfeat_mod  # noqa: F401
            # Pre-load the descriptor / matching head sibling modules
            # so that XFeat's runtime lazy imports also hit the cache.
            import modules.model  # noqa: F401
            import modules.interpolator  # noqa: F401
            self._xfeat_modules = {
                k: sys.modules[k]
                for k in list(sys.modules)
                if k == "modules" or k.startswith("modules.")
            }
            XFeat = _xfeat_mod.XFeat
        finally:
            # Remove XFeat's modules.* so vision-label-hub's own modules
            # work again outside this matcher.
            for k in list(sys.modules):
                if k == "modules" or k.startswith("modules."):
                    del sys.modules[k]
            for k, v in host_modules.items():
                sys.modules[k] = v
            if str(_REPO) in sys.path:
                sys.path.remove(str(_REPO))

        p = self.params
        weights = str(_REPO / "weights" / "xfeat.pt")
        self.model = XFeat(
            weights=weights,
            top_k=p.top_k,
            detection_threshold=p.detection_threshold,
        )
        # XFeat's XFeat() init reads self.dev = "cuda" if available;
        # override to honour our configured device.
        self.model.dev = self.device
        self.model.net = self.model.net.to(self.device).eval()

    def _normalise_image(self, img: torch.Tensor) -> torch.Tensor:
        """ImagePair stores (3, H, W) floats in [0, 1] on some device.
        XFeat expects (1, C, H, W); the parse_input it does internally is
        fine for torch tensors. Convert to expected layout + device.
        """
        if img.dim() == 3:
            img = img[None]
        return img.to(self.device).contiguous()

    def _swap_modules_in(self):
        """Install XFeat's cached modules.* into sys.modules; return host."""
        import sys
        host = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "modules" or k.startswith("modules.")
        }
        sys.modules.update(self._xfeat_modules)
        return host

    def _swap_modules_out(self, host):
        import sys
        for k in list(sys.modules):
            if k == "modules" or k.startswith("modules."):
                del sys.modules[k]
        sys.modules.update(host)

    def _forward(self, pair: ImagePair) -> _RawOutput:
        img_a = self._normalise_image(pair.image0)
        img_b = self._normalise_image(pair.image1)

        host = self._swap_modules_in()
        try:
            return self._forward_inner(pair, img_a, img_b)
        finally:
            self._swap_modules_out(host)

    def _forward_inner(self, pair, img_a, img_b) -> _RawOutput:
        with torch.no_grad():
            if self.params.mode == "sparse":
                mkpts_0_np, mkpts_1_np = self.model.match_xfeat(
                    img_a, img_b,
                    top_k=self.params.top_k,
                    min_cossim=self.params.min_cossim,
                )
                mkpts_0 = torch.as_tensor(
                    mkpts_0_np, dtype=torch.float32, device=self.device,
                )
                mkpts_1 = torch.as_tensor(
                    mkpts_1_np, dtype=torch.float32, device=self.device,
                )
                # match_xfeat doesn't expose per-pair confidences;
                # default to 1.0 (downstream uses score>0 as a validity flag).
                mconf = torch.ones(mkpts_0.shape[0], device=self.device)
            else:  # semidense
                ret = self.model.match_xfeat_star(
                    img_a, img_b, top_k=self.params.top_k,
                )
                # For B=1, match_xfeat_star returns a tuple
                # (mkpts_0_np, mkpts_1_np). For B>1, a list of (N, 4) tensors.
                if isinstance(ret, tuple):
                    mkpts_0 = torch.as_tensor(
                        ret[0], dtype=torch.float32, device=self.device,
                    )
                    mkpts_1 = torch.as_tensor(
                        ret[1], dtype=torch.float32, device=self.device,
                    )
                else:
                    m = ret[0]
                    m = torch.as_tensor(m, dtype=torch.float32, device=self.device)
                    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] < 4:
                        mkpts_0 = torch.empty(0, 2, device=self.device)
                        mkpts_1 = torch.empty(0, 2, device=self.device)
                    else:
                        mkpts_0 = m[:, :2]
                        mkpts_1 = m[:, 2:4]
                if mkpts_0.ndim != 2 or mkpts_0.shape[0] == 0:
                    mkpts_0 = torch.empty(0, 2, device=self.device)
                    mkpts_1 = torch.empty(0, 2, device=self.device)
                mconf = torch.ones(mkpts_0.shape[0], device=self.device)

        # Mask filter — preserve the "no kpt leaves the adapter outside
        # the foreground mask" contract.
        keep0 = self._mask_keep(mkpts_0, pair.meta0)
        keep1 = self._mask_keep(mkpts_1, pair.meta1)
        keep = keep0 & keep1
        return _RawOutput(
            mkpts0=mkpts_0[keep],
            mkpts1=mkpts_1[keep],
            mconf=mconf[keep],
            dense=None,
        )

    @staticmethod
    def _mask_keep(kpts: torch.Tensor, meta) -> torch.Tensor:
        if meta.valid_mask is None:
            return torch.ones(kpts.shape[0], dtype=torch.bool, device=kpts.device)
        return filter_kpts_by_mask(kpts, meta)
