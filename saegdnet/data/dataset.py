from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from saegdnet.data.events import filter_events_local_density, voxelize_events_np, warp_events_homography
from saegdnet.data.masks import saturation_mask, soft_saturation_mask, reliability_map, reliable_normalize
from saegdnet.utils.io import read_manifest, load_npz_sample


def _to_tensor_image(arr: np.ndarray) -> torch.Tensor:
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return torch.from_numpy(arr).float()


class SAEGDDataset(Dataset):
    """Dataset for paired UHSC frames and events.

    Each item is .npz containing:
      over: [H,W] or [H,W,C]
      target: [H,W] or [H,W,C], optional
      events: [N,4] columns x,y,t,p
      homography: [3,3], optional
      dense_reference: scalar optional, bool-like
    """

    def __init__(self, manifest: str | Path, cfg: Dict[str, Any], training: bool = True):
        self.paths = read_manifest(manifest)
        self.cfg = cfg
        self.training = training
        data_cfg = cfg.get("data", {})
        self.image_size = tuple(data_cfg.get("image_size", [256, 256]))
        self.event_bins = int(data_cfg.get("event_bins", 3))
        self.sat_threshold = float(data_cfg.get("saturation_threshold", 0.98))
        self.soft_tau = float(data_cfg.get("soft_mask_tau", 0.92))
        self.soft_eta = float(data_cfg.get("soft_mask_eta", 0.02))
        self.event_filter_cfg = data_cfg.get("event_filter", {})

    def __len__(self) -> int:
        return len(self.paths)

    def _resize_image(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.image_size
        x = torch.nn.functional.interpolate(x[None], size=(h, w), mode="bilinear", align_corners=False)[0]
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        sample = load_npz_sample(path)

        over = self._resize_image(_to_tensor_image(sample["over"]))
        _, h, w = over.shape

        if "target" in sample:
            target = self._resize_image(_to_tensor_image(sample["target"]))
            dense_reference = torch.tensor(float(sample.get("dense_reference", 1.0)), dtype=torch.float32)
        else:
            target = over.clone()
            dense_reference = torch.tensor(0.0, dtype=torch.float32)

        events = sample.get("events", np.zeros((0, 4), dtype=np.float32)).astype(np.float32)
        H = sample.get("homography", None)
        if H is not None:
            events = warp_events_homography(events, H.astype(np.float32), w, h)

        if self.event_filter_cfg.get("enabled", False):
            events = filter_events_local_density(
                events,
                spatial_radius=int(self.event_filter_cfg.get("spatial_radius", 1)),
                temporal_window_us=float(self.event_filter_cfg.get("temporal_window_us", 500.0)),
                min_neighbors=int(self.event_filter_cfg.get("min_neighbors", 2)),
            )

        voxel, activity = voxelize_events_np(events, h, w, bins=self.event_bins)
        event = torch.from_numpy(voxel).float()
        activity = torch.from_numpy(activity).float()

        bin_mask = saturation_mask(over[None], self.sat_threshold)[0]
        soft_mask = soft_saturation_mask(over[None], self.soft_tau, self.soft_eta)[0]
        reliability = reliability_map(soft_mask)
        over_norm = reliable_normalize(over[None], reliability[None])[0]

        return {
            "path": str(path),
            "over": over,
            "over_norm": over_norm,
            "target": target,
            "event": event,
            "activity": activity,
            "mask": bin_mask,
            "soft_mask": soft_mask,
            "reliability": reliability,
            "dense_reference": dense_reference,
        }


def collate_batch(batch):
    out = {}
    for k in batch[0]:
        if k == "path":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
