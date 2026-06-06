from __future__ import annotations
from pathlib import Path
from typing import Literal
import torch
from pydantic import BaseModel, Field
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register
from xmatcher.core.types import _RawOutput, ImagePair


class EfficientLoFTRParams(BaseModel):
    """Parameters for the transformers-based EfficientLoFTR adapter.

    Weights are loaded via `from_pretrained(repo_id)`; transformers handles
    download + cache (~/.cache/huggingface/) + integrity. To use a local
    snapshot or a private repo, set `repo_id` to the path or set HF_TOKEN.
    To route through a HuggingFace mirror, set HF_ENDPOINT in the env.
    """
    repo_id: str = "zju-community/efficientloftr"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    match_threshold: float = Field(default=0.2, ge=0.0, le=1.0)


@register("efficient_loftr")
class EfficientLoFTRMatcher(BaseMatcher):
    Params = EfficientLoFTRParams

    def _setup(self):
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching
        self.processor = AutoImageProcessor.from_pretrained(self.params.repo_id)
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.params.precision]
        # transformers 4.51-4.54 takes `torch_dtype`; 5.x deprecated it in favor
        # of `dtype` but still accepts it. We're pinned to <4.55 due to a
        # PyTorch 2.7-only symbol in transformers.integrations.finegrained_fp8.
        self.model = (
            AutoModelForKeypointMatching
            .from_pretrained(self.params.repo_id, torch_dtype=torch_dtype)
            .eval()
            .to(self.device)
        )

    def _forward(self, pair: ImagePair) -> _RawOutput:
        # We feed the dataset's already-processed images directly so target_sizes
        # equals processed_size; the post-processor maps back to those same
        # coords, which is the contract _forward must satisfy (BaseMatcher's
        # __call__ then runs unproject + mask filtering).
        # do_rescale=False because our ImagePair tensors are already in [0,1];
        # without this the processor divides by 255 again, giving a ~max=0.004
        # input that produces only border-noise matches.
        H0, W0 = pair.meta0.processed_size
        H1, W1 = pair.meta1.processed_size
        inputs = self.processor(
            images=[[pair.image0, pair.image1]],
            return_tensors="pt",
            do_rescale=False,
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_keypoint_matching(
            outputs,
            target_sizes=[[(H0, W0), (H1, W1)]],
            threshold=self.params.match_threshold,
        )
        r = results[0]
        return _RawOutput(
            mkpts0=r["keypoints0"].to(self.device, dtype=torch.float32),
            mkpts1=r["keypoints1"].to(self.device, dtype=torch.float32),
            mconf=r["matching_scores"].to(self.device, dtype=torch.float32),
            dense=None,
        )
