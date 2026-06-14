from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class EventConfig:
    height: int
    width: int
    bins: int = 3
    filter_enabled: bool = True
    spatial_radius: int = 1
    temporal_window_us: float = 500.0
    min_neighbors: int = 2


def warp_events_homography(events: np.ndarray, H: np.ndarray, width: int, height: int) -> np.ndarray:
    """Warp event x,y using homography from event plane to frame plane."""
    if events.size == 0:
        return events
    xy1 = np.stack([events[:, 0], events[:, 1], np.ones(len(events))], axis=0)
    warped = H @ xy1
    warped = warped[:2] / np.maximum(warped[2:3], 1e-8)
    out = events.copy()
    out[:, 0] = warped[0]
    out[:, 1] = warped[1]
    valid = (
        (out[:, 0] >= 0) & (out[:, 0] < width) &
        (out[:, 1] >= 0) & (out[:, 1] < height)
    )
    return out[valid]


def filter_events_local_density(
    events: np.ndarray,
    spatial_radius: int = 1,
    temporal_window_us: float = 500.0,
    min_neighbors: int = 2,
) -> np.ndarray:
    """Simple local spatiotemporal consistency filter.

    This implementation is intentionally readable. For millions of events,
    replace with a grid/hash or GPU implementation.
    """
    if len(events) == 0:
        return events
    keep = np.zeros(len(events), dtype=bool)
    xs, ys, ts = events[:, 0], events[:, 1], events[:, 2]
    for i in range(len(events)):
        spatial = (np.abs(xs - xs[i]) <= spatial_radius) & (np.abs(ys - ys[i]) <= spatial_radius)
        temporal = np.abs(ts - ts[i]) <= temporal_window_us
        count = int(np.count_nonzero(spatial & temporal)) - 1
        keep[i] = count >= min_neighbors
    return events[keep]


def voxelize_events_np(
    events: np.ndarray,
    height: int,
    width: int,
    bins: int = 3,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create polarity-aware event voxel tensor [2K,H,W] and activity map [1,H,W].

    Events are columns x, y, t, p, with p in {-1,+1}.
    Temporal interpolation is used.
    """
    voxel = np.zeros((2 * bins, height, width), dtype=np.float32)
    if len(events) == 0:
        return voxel, np.zeros((1, height, width), dtype=np.float32)

    x = np.rint(events[:, 0]).astype(np.int64)
    y = np.rint(events[:, 1]).astype(np.int64)
    t = events[:, 2].astype(np.float64)
    p = events[:, 3]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    if len(t) == 0:
        return voxel, np.zeros((1, height, width), dtype=np.float32)

    if t0 is None:
        t0 = float(t.min())
    if t1 is None:
        t1 = float(t.max())
    denom = max(t1 - t0, 1e-6)
    tb = (bins - 1) * (t - t0) / denom

    for xi, yi, ti, pi in zip(x, y, tb, p):
        base = 0 if pi > 0 else bins
        lo = int(np.floor(ti))
        hi = min(lo + 1, bins - 1)
        for k in {lo, hi}:
            if 0 <= k < bins:
                w = max(0.0, 1.0 - abs(k - ti))
                voxel[base + k, yi, xi] += float(w)

    activity = np.sum(np.abs(voxel), axis=0, keepdims=True)
    if activity.max() > 0:
        activity = 1.0 - np.exp(-activity / (activity.mean() + 1e-6))
        activity = np.clip(activity, 0, 1)
    return voxel, activity.astype(np.float32)


def voxelize_events_torch(events: torch.Tensor, height: int, width: int, bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    events_np = events.detach().cpu().numpy()
    voxel, activity = voxelize_events_np(events_np, height, width, bins)
    return torch.from_numpy(voxel), torch.from_numpy(activity)
