from __future__ import annotations

import torch
import torch.nn.functional as F


def saturation_mask(
    frame: torch.Tensor,
    threshold: float = 0.98,
) -> torch.Tensor:
    """Return a binary saturation mask [B,1,H,W].

    frame is expected in [0, 1], shape [B,C,H,W].
    """
    if frame.ndim != 4:
        raise ValueError(f"frame must be [B,C,H,W], got {frame.shape}")
    max_ch = frame.max(dim=1, keepdim=True).values
    return (max_ch >= threshold).float()


def soft_saturation_mask(
    frame: torch.Tensor,
    tau: float = 0.92,
    eta: float = 0.02,
) -> torch.Tensor:
    """Soft saturation map using sigmoid((S - tau) / eta)."""
    max_ch = frame.max(dim=1, keepdim=True).values
    return torch.sigmoid((max_ch - tau) / max(eta, 1e-6))


def reliability_map(soft_mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - soft_mask.clamp(0, 1)


def reliable_normalize(frame: torch.Tensor, reliability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize frame statistics using reliable pixels only.

    This avoids saturated pixels dominating mean/std.
    """
    if reliability.shape[1] == 1 and frame.shape[1] != 1:
        reliability = reliability.expand(-1, frame.shape[1], -1, -1)
    denom = reliability.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    mean = (frame * reliability).sum(dim=(2, 3), keepdim=True) / denom
    var = (((frame - mean) ** 2) * reliability).sum(dim=(2, 3), keepdim=True) / denom
    return (frame - mean) / torch.sqrt(var + eps)


def smooth_mask(mask: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """Boundary-smooth a mask with average filtering."""
    pad = kernel_size // 2
    return F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad).clamp(0, 1)
