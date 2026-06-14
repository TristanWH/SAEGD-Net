from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from saegdnet.data.events import voxelize_events_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    events = np.load(args.events).astype(np.float32)
    voxel, activity = voxelize_events_np(events, args.height, args.width, args.bins)
    pos = voxel[:args.bins].sum(axis=0)
    neg = voxel[args.bins:].sum(axis=0)
    img = np.ones((args.height, args.width, 3), np.float32)
    if pos.max() > 0:
        img[..., 0] -= pos / pos.max()
        img[..., 1] -= pos / pos.max()
        img[..., 2] -= pos / pos.max()
    if neg.max() > 0:
        img[..., 0] -= 0.1 * neg / neg.max()
        img[..., 1] -= 0.2 * neg / neg.max()
        img[..., 2] -= 0.8 * neg / neg.max()
    img = np.clip(img, 0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((img * 255).astype(np.uint8)).save(args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
