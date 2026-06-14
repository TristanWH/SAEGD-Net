from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from saegdnet.data.events import filter_events_local_density, warp_events_homography, voxelize_events_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="Numpy .npy file with [N,4] x,y,t,p.")
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--homography", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", action="store_true")
    args = ap.parse_args()

    events = np.load(args.events).astype(np.float32)
    if args.homography:
        H = np.loadtxt(args.homography).astype(np.float32)
        events = warp_events_homography(events, H, args.width, args.height)
    if args.filter:
        events = filter_events_local_density(events)
    voxel, activity = voxelize_events_np(events, args.height, args.width, args.bins)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, event=voxel, activity=activity)
    print("Saved", out)


if __name__ == "__main__":
    main()
