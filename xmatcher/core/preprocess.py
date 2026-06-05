from __future__ import annotations
import torch
from xmatcher.core.types import PreprocessMeta


def unproject(pts_proc: torch.Tensor, meta: PreprocessMeta) -> torch.Tensor:
    """Map (K, 2) points from processed coords to original coords.

    Uses meta.affine_proc_to_orig (a 2x3 tensor). Empty inputs are passed through.
    Output device matches input.
    """
    if pts_proc.numel() == 0:
        return pts_proc.clone()
    aff = meta.affine_proc_to_orig.to(pts_proc.device, dtype=pts_proc.dtype)
    ones = torch.ones((pts_proc.shape[0], 1), dtype=pts_proc.dtype, device=pts_proc.device)
    homog = torch.cat([pts_proc, ones], dim=1)        # (K, 3)
    return homog @ aff.T                              # (K, 2)


def filter_by_mask(
    pts0_proc: torch.Tensor,
    pts1_proc: torch.Tensor,
    meta0: PreprocessMeta,
    meta1: PreprocessMeta,
) -> torch.Tensor:
    """Bool keep-mask. Coordinates are in *processed* space for both inputs.

    Drops points that fall outside [0, W_proc) x [0, H_proc) on either side,
    plus points whose nearest-pixel in processed valid_mask is False (when mask provided).
    """
    K = pts0_proc.shape[0]
    if K == 0:
        return torch.zeros((0,), dtype=torch.bool, device=pts0_proc.device)

    keep = torch.ones(K, dtype=torch.bool, device=pts0_proc.device)
    for pts, meta in ((pts0_proc, meta0), (pts1_proc, meta1)):
        H_proc, W_proc = meta.processed_size
        u, v = pts[:, 0], pts[:, 1]
        in_bounds = (u >= 0) & (u < W_proc) & (v >= 0) & (v < H_proc)
        keep = keep & in_bounds
        if meta.valid_mask is not None:
            uu = u.clamp(0, W_proc - 1).round().long()
            vv = v.clamp(0, H_proc - 1).round().long()
            mask_val = meta.valid_mask.to(pts.device)[vv, uu]
            keep = keep & mask_val
    return keep
