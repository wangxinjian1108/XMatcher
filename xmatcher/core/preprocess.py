from __future__ import annotations
import torch
import torch.nn.functional as F
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


def _to_gray_align32(img: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """RGB or gray (C, H, W) in [0,1] → gray (1, H', W') padded right/bottom to multiples of 32.

    Returns (img_padded, (pad_left, pad_top)). Right/bottom padding does not shift origin,
    so the returned pad tuple is always (0, 0). Adapters subtract this from output
    keypoints to get back to dataset's processed coordinate space.
    """
    if img.dim() != 3:
        raise ValueError(f"expected (C, H, W), got shape {tuple(img.shape)}")
    C, H, W = img.shape
    if C == 3:
        # Standard ITU-R BT.601 luminance weights.
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=img.dtype, device=img.device)
        gray = (img * weights[:, None, None]).sum(dim=0, keepdim=True)
    elif C == 1:
        gray = img
    else:
        raise ValueError(f"expected 1 or 3 channels, got {C}")

    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h or pad_w:
        # F.pad layout for (C, H, W): (left, right, top, bottom)
        gray = F.pad(gray, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return gray, (0, 0)


def apply_input_mask(image: torch.Tensor, meta: PreprocessMeta) -> torch.Tensor:
    """Zero out pixels of `image` that fall outside `meta.valid_mask`.

    The image is (C, H_proc, W_proc); the mask is (H_proc, W_proc) bool.
    No-op if `meta.valid_mask is None`. The original tensor is left intact;
    a new tensor is returned. Padding pixels (mask=False from preprocessing)
    AND user-supplied foreground masks (mask=False outside ROI) are both
    handled by this single mechanism — callers AND them together at dataset
    construction time.
    """
    if meta.valid_mask is None:
        return image
    mask = meta.valid_mask.to(image.device)
    # Broadcast (H, W) → (1, H, W) → (C, H, W)
    return image * mask.unsqueeze(0)


def filter_kpts_by_mask(
    keypoints: torch.Tensor, meta: PreprocessMeta
) -> torch.Tensor:
    """Bool keep-mask for `keypoints` (K, 2) in processed coords.

    Drops points that fall outside [0, W_proc) x [0, H_proc) OR outside
    `meta.valid_mask` (when provided). Returns a (K,) bool tensor on the
    same device as `keypoints`. No-op-equivalent (all True, modulo bounds)
    when `meta.valid_mask is None`.
    """
    K = keypoints.shape[0]
    if K == 0:
        return torch.zeros((0,), dtype=torch.bool, device=keypoints.device)
    H_proc, W_proc = meta.processed_size
    u, v = keypoints[:, 0], keypoints[:, 1]
    keep = (u >= 0) & (u < W_proc) & (v >= 0) & (v < H_proc)
    if meta.valid_mask is not None:
        uu = u.clamp(0, W_proc - 1).round().long()
        vv = v.clamp(0, H_proc - 1).round().long()
        mask_val = meta.valid_mask.to(keypoints.device)[vv, uu]
        keep = keep & mask_val
    return keep
