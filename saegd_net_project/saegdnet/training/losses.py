from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] > 1:
        gray = x.mean(dim=1, keepdim=True)
    else:
        gray = x
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


class SAEGDLoss(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        loss_cfg = cfg.get("loss", {})
        self.l_sat = float(loss_cfg.get("lambda_sat", 1.0))
        self.l_valid = float(loss_cfg.get("lambda_valid", 0.5))
        self.l_edge = float(loss_cfg.get("lambda_edge", 0.2))
        self.l_event = float(loss_cfg.get("lambda_event", 0.3))
        self.l_perc = float(loss_cfg.get("lambda_perc", 0.05))
        self.l_temp = float(loss_cfg.get("lambda_temp", 0.1))

    def forward(self, diffusion_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x0_pred = diffusion_out["x0_pred"]
        target = batch["target"]
        mask = batch["soft_mask"]
        reliability = batch["reliability"]
        over = batch["over"]
        activity = batch["activity"]

        losses: Dict[str, torch.Tensor] = {}
        losses["diff"] = diffusion_out["diffusion"]

        losses["sat"] = (mask * (x0_pred - target).abs()).sum() / (mask.sum() + 1e-6)
        losses["valid"] = (reliability * (x0_pred - over).abs()).sum() / (reliability.sum() + 1e-6)

        pred_edge = sobel_edges(x0_pred)
        target_edge = sobel_edges(target)
        losses["edge"] = (mask * (pred_edge - target_edge).abs()).sum() / (mask.sum() + 1e-6)

        act = activity
        if act.shape[-2:] != pred_edge.shape[-2:]:
            act = F.interpolate(act, size=pred_edge.shape[-2:], mode="bilinear", align_corners=False)
        pred_norm = pred_edge / (pred_edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1) + 1e-6)
        act_norm = act / (act.flatten(1).amax(dim=1).view(-1, 1, 1, 1) + 1e-6)
        num = (mask * pred_norm * act_norm).sum(dim=(1, 2, 3))
        den = torch.sqrt((mask * pred_norm ** 2).sum(dim=(1, 2, 3)) * (mask * act_norm ** 2).sum(dim=(1, 2, 3)) + 1e-6)
        losses["event"] = (1.0 - num / den.clamp_min(1e-6)).mean()

        # Lightweight perceptual proxy. Replace with VGG/LPIPS if desired.
        losses["perc"] = F.l1_loss(F.avg_pool2d(x0_pred, 4), F.avg_pool2d(target, 4))

        losses["temp"] = torch.zeros((), device=x0_pred.device)

        total = (
            losses["diff"]
            + self.l_sat * losses["sat"]
            + self.l_valid * losses["valid"]
            + self.l_edge * losses["edge"]
            + self.l_event * losses["event"]
            + self.l_perc * losses["perc"]
            + self.l_temp * losses["temp"]
        )
        losses["total"] = total
        return losses
