from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from saegdnet.training.losses import sobel_edges


def binary_edges(x: torch.Tensor, q: float = 0.85) -> torch.Tensor:
    edge = sobel_edges(x)
    flat = edge.flatten(1)
    th = torch.quantile(flat, q, dim=1).view(-1, 1, 1, 1)
    return edge >= th


def event_boundaries(activity: torch.Tensor, q: float = 0.85) -> torch.Tensor:
    flat = activity.flatten(1)
    th = torch.quantile(flat, q, dim=1).view(-1, 1, 1, 1)
    return activity >= th


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    return F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=radius) > 0


def event_supported_edge_prf(
    pred: torch.Tensor,
    activity: torch.Tensor,
    saturation_mask: torch.Tensor,
    tolerance_px: int = 2,
) -> Dict[str, float]:
    """Compute event-edge precision/recall/F-score.

    A predicted edge is matched when it lies within tolerance_px Euclidean-like
    dilation radius after homography alignment to the frame coordinate system.
    """
    pred_e = binary_edges(pred)
    ev_e = event_boundaries(activity)
    region = dilate(saturation_mask > 0.5, tolerance_px)

    pred_r = pred_e & region
    ev_r = ev_e & region

    ev_dil = dilate(ev_r, tolerance_px)
    pred_dil = dilate(pred_r, tolerance_px)

    tp_prec = (pred_r & ev_dil).float().sum()
    tp_rec = (ev_r & pred_dil).float().sum()
    precision = tp_prec / (pred_r.float().sum() + 1e-6)
    recall = tp_rec / (ev_r.float().sum() + 1e-6)
    f = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "event_edge_precision": float(precision.detach().cpu()),
        "event_edge_recall": float(recall.detach().cpu()),
        "event_edge_fscore": float(f.detach().cpu()),
    }
