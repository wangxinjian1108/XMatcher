from __future__ import annotations
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Literal
import torch
from pydantic import BaseModel, Field, field_validator
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register
from xmatcher.core.types import _RawOutput, ImagePair
from xmatcher.core.preprocess import _to_gray_align32


class EfficientLoFTRParams(BaseModel):
    weights: Path
    precision: Literal["fp32", "fp16", "mp"] = "mp"
    match_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    border_rm: int = Field(default=2, ge=0)

    @field_validator("weights", mode="after")
    @classmethod
    def _resolve_weights(cls, v: Path) -> Path:
        if v.is_absolute():
            if not v.exists():
                raise FileNotFoundError(
                    f"Weight not found: {v}\n"
                    f"Run scripts/download_weights.sh efficient_loftr to fetch it."
                )
            return v
        base = Path(os.environ.get(
            "XMATCHER_WEIGHTS_DIR",
            Path.home() / ".cache" / "xmatcher",
        ))
        resolved = base / v
        if not resolved.exists():
            raise FileNotFoundError(
                f"Weight not found: {resolved}\n"
                f"Run scripts/download_weights.sh efficient_loftr to fetch it."
            )
        return resolved


@register("efficient_loftr")
class EfficientLoFTRMatcher(BaseMatcher):
    Params = EfficientLoFTRParams

    def _setup(self):
        from src.loftr import LoFTR, reparameter
        from src.config.default import get_cfg_defaults
        cfg = get_cfg_defaults()
        cfg.LOFTR.MATCH_COARSE.THR = self.params.match_threshold
        cfg.LOFTR.MATCH_COARSE.BORDER_RM = self.params.border_rm
        if self.params.precision == "fp16":
            cfg.LOFTR.HALF = True
            cfg.LOFTR.MP = False
        elif self.params.precision == "fp32":
            cfg.LOFTR.HALF = False
            cfg.LOFTR.MP = False
        else:  # "mp"
            cfg.LOFTR.HALF = False
            cfg.LOFTR.MP = True

        self.matcher = LoFTR(config=cfg.LOFTR)
        state = torch.load(self.params.weights, map_location="cpu")
        self.matcher.load_state_dict(state["state_dict"])
        self.matcher = reparameter(self.matcher).eval().to(self.device)
        self._cfg = cfg

    def _forward(self, pair: ImagePair) -> _RawOutput:
        # Adapter-internal preprocessing: gray + pad to 32-multiple.
        img0_pad, _pad0 = _to_gray_align32(pair.image0)
        img1_pad, _pad1 = _to_gray_align32(pair.image1)
        # _to_gray_align32 returns (0, 0) — right/bottom padding does not shift origin,
        # so kpts in proc-padded coords already equal kpts in dataset's processed coords.
        data = {
            "image0": img0_pad.to(self.device).unsqueeze(0),
            "image1": img1_pad.to(self.device).unsqueeze(0),
        }
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._cfg.LOFTR.MP and "cuda" in str(self.device)
            else nullcontext()
        )
        with ctx:
            self.matcher(data)
        return _RawOutput(
            mkpts0=data["mkpts0_f"],
            mkpts1=data["mkpts1_f"],
            mconf=data["mconf"],
            dense=None,
        )
