from __future__ import annotations

import torch
import torch.nn.functional as F


def downsample_segmentation_to_patch_grid(seg: torch.Tensor, grid_size: int = 6) -> torch.Tensor:
    if seg.ndim != 4:
        raise ValueError(f"Expected segmentation [B,T,H,W], got {tuple(seg.shape)}")
    seg_small = F.interpolate(seg.float(), size=(grid_size, grid_size), mode="nearest")
    return seg_small.long()


def region_masks_from_lax_seg(seg: torch.Tensor, grid_size: int = 6) -> torch.Tensor:
    seg_small = downsample_segmentation_to_patch_grid(seg, grid_size=grid_size)
    lv = (seg_small == 1).float().mean(dim=1)
    rv = (seg_small == 3).float().mean(dim=1)
    myo = (seg_small == 2).float().mean(dim=1)
    foreground = (seg_small > 0).float().mean(dim=1)
    other = (foreground - torch.maximum(torch.maximum(lv, rv), myo)).clamp_min(0.0)
    masks = torch.stack([lv, rv, myo, other], dim=1)
    masks = masks.flatten(2)
    return masks


def mask_pool_tokens(tokens: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [B,P,D], got {tuple(tokens.shape)}")
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [B,K,P], got {tuple(masks.shape)}")
    weights = masks / masks.sum(dim=-1, keepdim=True).clamp_min(eps)
    pooled = torch.einsum("bkp,bpd->bkd", weights, tokens)
    fallback = tokens.mean(dim=1, keepdim=True).expand_as(pooled)
    valid = (masks.sum(dim=-1, keepdim=True) > eps).float()
    return pooled * valid + fallback * (1.0 - valid)
