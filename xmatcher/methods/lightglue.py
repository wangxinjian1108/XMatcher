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

    def _forward_batch(self, pairs: list[ImagePair]) -> list[_RawOutput]:
        """Batched LightGlue inference.

        Approach (mirrors what vauto4d's End2EndLightGluer does):
          1. Run SuperPoint on each image of each pair via the per-image
             `extract` helper. This is N forward passes through SuperPoint;
             we don't try to batch SuperPoint itself because upstream's
             SuperPoint.forward only batches when every image happens to
             yield exactly `max_num_keypoints` detections (it does
             `torch.stack` on the per-image keypoint lists).
          2. Pad each image's keypoints/descriptors/scores to the same K
             (max count across the batch, capped at max_num_keypoints).
             Padded slots are zeroed out and the score=0 effectively
             excludes them from matcher attention.
          3. Stack into (B, K, ...) batches for image0 and image1.
          4. Run LightGlue once. Adaptive **depth** pruning still operates
             at the batch granularity (early-stop kicks in when the batch
             as a whole has converged). Adaptive **width** pruning (point
             pruning) is **disabled** during batch inference: upstream
             LightGlue's point-pruning bookkeeping initializes `ind0`/`ind1`
             with batch dim 1 (`torch.arange(...)[None]`) and crashes for
             any B > 1. Disabling width_confidence is the user-facing
             documented fallback for this case.
          5. Per-pair, slice `out["matches"][i]` and `out["scores"][i]`,
             which are already lists indexed by pair.
        """
        if not pairs:
            return []
        if len(pairs) == 1:
            # Single-pair: take the per-pair fast path so we don't need to
            # touch LightGlue's adaptive-width settings.
            return [self._forward(pairs[0])]

        # 1+2: extract per-image, pad to common K.
        per_image_feats: list[dict] = []
        for pair in pairs:
            for img in (pair.image0, pair.image1):
                feats = self.extractor.extract(img.to(self.device), resize=None)
                # extract() returns batched (1, *, ...); strip batch.
                per_image_feats.append({
                    "keypoints": feats["keypoints"][0],         # (N, 2)
                    "keypoint_scores": feats["keypoint_scores"][0],  # (N,)
                    "descriptors": feats["descriptors"][0],     # (N, D)
                    "image_size": feats["image_size"][0],       # (2,)
                })

        K = max(f["keypoints"].shape[0] for f in per_image_feats)
        K = min(K, self.params.max_num_keypoints)

        kpts_b, scores_b, desc_b, sizes_b = [], [], [], []
        for f in per_image_feats:
            n = f["keypoints"].shape[0]
            if n >= K:
                kpts_b.append(f["keypoints"][:K])
                scores_b.append(f["keypoint_scores"][:K])
                desc_b.append(f["descriptors"][:K])
            else:
                pad = K - n
                kpts_b.append(torch.cat([
                    f["keypoints"],
                    torch.zeros(pad, 2, device=self.device, dtype=f["keypoints"].dtype),
                ]))
                scores_b.append(torch.cat([
                    f["keypoint_scores"],
                    torch.zeros(pad, device=self.device, dtype=f["keypoint_scores"].dtype),
                ]))
                desc_b.append(torch.cat([
                    f["descriptors"],
                    torch.zeros(
                        pad, f["descriptors"].shape[-1],
                        device=self.device, dtype=f["descriptors"].dtype,
                    ),
                ]))
            sizes_b.append(f["image_size"])

        # 3: stack and split into image0 / image1 batches.
        all_kpts = torch.stack(kpts_b)        # (2B, K, 2)
        all_scores = torch.stack(scores_b)    # (2B, K)
        all_desc = torch.stack(desc_b)        # (2B, K, D)
        all_sizes = torch.stack(sizes_b)      # (2B, 2)

        feats0_b = {
            "keypoints": all_kpts[0::2],
            "keypoint_scores": all_scores[0::2],
            "descriptors": all_desc[0::2],
            "image_size": all_sizes[0::2],
        }
        feats1_b = {
            "keypoints": all_kpts[1::2],
            "keypoint_scores": all_scores[1::2],
            "descriptors": all_desc[1::2],
            "image_size": all_sizes[1::2],
        }

        # 4: temporarily disable adaptive point pruning; run matcher.
        # See docstring: upstream's width-pruning code is not batch-safe.
        # Depth pruning (early stop) is left on; it works at the batch
        # granularity and saves the most compute in practice.
        original_width = self.matcher.conf.width_confidence
        self.matcher.conf.width_confidence = -1
        try:
            out = self.matcher({"image0": feats0_b, "image1": feats1_b})
        finally:
            self.matcher.conf.width_confidence = original_width

        # 5: split per-pair. matches/scores are returned as lists of length B
        # because each pair yields a different number of matches.
        matches_list = out["matches"]
        scores_list = out["scores"]
        results: list[_RawOutput] = []
        for i in range(len(pairs)):
            m = matches_list[i]   # (Mi, 2)
            s = scores_list[i]    # (Mi,)
            kp0 = feats0_b["keypoints"][i]   # (K, 2)
            kp1 = feats1_b["keypoints"][i]   # (K, 2)
            results.append(_RawOutput(
                mkpts0=kp0[m[:, 0]],
                mkpts1=kp1[m[:, 1]],
                mconf=s,
                dense=None,
            ))
        return results

