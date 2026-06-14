from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frame(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= np.iinfo(np.uint16).max if img.max() > 255 else 255.0
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--events", required=True, help="A .npy [N,4] event file. For precise pairing, provide already time-windowed events or adapt this script.")
    ap.add_argument("--homography", default=None)
    ap.add_argument("--targets", default=None, help="Optional dense reference frame folder.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = sorted(Path(args.frames).glob("*"))
    events_all = np.load(args.events).astype(np.float32)
    H = np.loadtxt(args.homography).astype(np.float32) if args.homography else np.eye(3, dtype=np.float32)
    targets = sorted(Path(args.targets).glob("*")) if args.targets else [None] * len(frames)

    out = Path(args.out)
    sample_dir = out / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, frame_path in enumerate(frames):
        over = read_frame(frame_path)
        target = read_frame(targets[i]) if targets and targets[i] is not None else over
        dense = 1.0 if targets and targets[i] is not None else 0.0
        # Placeholder pairing: use all events. For real data, replace with timestamp slicing.
        path = sample_dir / f"{i:06d}.npz"
        np.savez_compressed(
            path,
            over=over.astype(np.float32),
            target=target.astype(np.float32),
            events=events_all,
            homography=H,
            dense_reference=np.array(dense, dtype=np.float32),
            meta=json.dumps({"frame": str(frame_path), "dense_reference": bool(dense)}),
        )
        manifest.append(path)

    (out / "all.txt").write_text("\n".join(str(p) for p in manifest) + "\n", encoding="utf-8")
    print("Built", len(manifest), "paired samples in", out)


if __name__ == "__main__":
    main()
