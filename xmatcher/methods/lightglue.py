from __future__ import annotations
from typing import Literal
import torch
from pydantic import BaseModel, Field
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register
from xmatcher.core.types import _RawOutput, ImagePair


class LightGlueParams(BaseModel):
    features: Literal["superpoint", "disk", "aliked", "sift"] = "superpoint"
    max_num_keypoints: int = Field(default=2048, gt=0)
    detection_threshold: float = 0.0005
    match_threshold: float = 0.1


@register("lightglue")
class LightGlueMatcher(BaseMatcher):
    Params = LightGlueParams

    def _setup(self):
        from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
        EXTRACTORS = {
            "superpoint": SuperPoint, "disk": DISK,
            "aliked": ALIKED, "sift": SIFT,
        }
        p = self.params
        self.extractor = EXTRACTORS[p.features](
            max_num_keypoints=p.max_num_keypoints,
            detection_threshold=p.detection_threshold,
        ).eval().to(self.device)
        self.matcher = LightGlue(
            features=p.features,
            filter_threshold=p.match_threshold,
        ).eval().to(self.device)

    def _forward(self, pair: ImagePair) -> _RawOutput:
        from lightglue.utils import rbd
        img0 = pair.image0.to(self.device)
        img1 = pair.image1.to(self.device)
        # Critical: resize=None disables LightGlue's internal resize.
        # Dataset already shaped the image; if extractor resizes again,
        # output keypoints will be in its internal grid → unproject breaks.
        feats0 = self.extractor.extract(img0, resize=None)
        feats1 = self.extractor.extract(img1, resize=None)
        out = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, out = [rbd(x) for x in (feats0, feats1, out)]
        m = out["matches"]
        return _RawOutput(
            mkpts0=feats0["keypoints"][m[:, 0]],
            mkpts1=feats1["keypoints"][m[:, 1]],
            mconf=out["scores"],
            dense=None,
        )
