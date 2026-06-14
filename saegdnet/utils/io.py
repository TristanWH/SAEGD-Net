from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import numpy as np
import torch
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_manifest(path: str | Path) -> list[Path]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
    return [Path(x) for x in items]


def save_image_tensor(x: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    x = x.clamp(0, 1).numpy()
    if x.ndim == 2:
        arr = (x * 255).astype(np.uint8)
    elif x.shape[-1] == 1:
        arr = (x[..., 0] * 255).astype(np.uint8)
    else:
        arr = (x * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_npz_sample(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    sample: Dict[str, Any] = {}
    for k in data.files:
        value = data[k]
        if k == "meta":
            try:
                sample[k] = json.loads(str(value.item()))
            except Exception:
                sample[k] = value
        else:
            sample[k] = value
    return sample
