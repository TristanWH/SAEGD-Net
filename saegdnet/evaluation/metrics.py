from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from saegdnet.training.losses import sobel_edges


def mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (x - y) ** 2
    if mask is not None:
        err = err * mask
        return err.sum() / (mask.sum() * x.shape[1] + 1e-6)
    return err.mean()


def psnr(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None, max_val: float = 1.0) -> torch.Tensor:
    return 10.0 * torch.log10(torch.tensor(max_val ** 2, device=x.device) / mse(x, y, mask).clamp_min(1e-10))


def rmse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return torch.sqrt(mse(x, y, mask).clamp_min(1e-10))


def ssim_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Compact global SSIM for monitoring. For publication, use skimage or a windowed implementation.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux = x.mean(dim=(-1, -2), keepdim=True)
    muy = y.mean(dim=(-1, -2), keepdim=True)
    vx = ((x - mux) ** 2).mean(dim=(-1, -2), keepdim=True)
    vy = ((y - muy) ** 2).mean(dim=(-1, -2), keepdim=True)
    cov = ((x - mux) * (y - muy)).mean(dim=(-1, -2), keepdim=True)
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2))
    return score.mean()


def event_aware_sharpness(pred: torch.Tensor, activity: torch.Tensor, threshold_edge: float = 0.1, threshold_event: float = 0.1) -> torch.Tensor:
    edge = sobel_edges(pred)
    if activity.shape[-2:] != edge.shape[-2:]:
        activity = F.interpolate(activity, size=edge.shape[-2:], mode="bilinear", align_corners=False)
    e1 = edge > threshold_edge
    e2 = activity > threshold_event
    inter = (e1 & e2).float().sum()
    return 2 * inter / (e1.float().sum() + e2.float().sum() + 1e-6)


def full_reference_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, activity: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        return {
            "psnr": float(psnr(pred, target).detach().cpu()),
            "ssim": float(ssim_simple(pred, target).detach().cpu()),
            "rmse": float(rmse(pred, target).detach().cpu()),
            "psnr_sat": float(psnr(pred, target, mask).detach().cpu()),
            "eas": float(event_aware_sharpness(pred, activity).detach().cpu()),
        }
