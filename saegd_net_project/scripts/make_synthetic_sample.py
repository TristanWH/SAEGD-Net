from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def make_one(h: int, w: int, idx: int):
    yy, xx = np.mgrid[0:h, 0:w]
    cx = int(w * (0.25 + 0.5 * np.random.rand()))
    cy = int(h * (0.25 + 0.5 * np.random.rand()))
    sigma = np.random.uniform(4, 12)
    bg = 0.1 + 0.2 * np.sin(xx / w * np.pi) + 0.1 * np.cos(yy / h * np.pi)
    target = bg + np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    # Draw a thin transient line.
    x2 = np.clip(cx + np.random.randint(-20, 21), 0, w - 1)
    y2 = np.clip(cy + np.random.randint(-20, 21), 0, h - 1)
    line = np.zeros((h, w), np.float32)
    cv2.line(line, (cx, cy), (x2, y2), 1.0, 1)
    target += 0.6 * line
    target = np.clip(target, 0, 1).astype(np.float32)

    over = np.clip(target * np.random.uniform(1.5, 2.5), 0, 1).astype(np.float32)

    # Synthetic events around gradients/line.
    gx = cv2.Sobel(target, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(target, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    ys, xs = np.where(mag > np.quantile(mag, 0.9))
    n = min(len(xs), 800)
    if n > 0:
        sel = np.random.choice(len(xs), n, replace=False)
        xs, ys = xs[sel], ys[sel]
    ts = np.random.uniform(0, 1000, size=len(xs)).astype(np.float32)
    ps = np.where(np.random.rand(len(xs)) > 0.5, 1, -1).astype(np.float32)
    events = np.stack([xs.astype(np.float32), ys.astype(np.float32), ts, ps], axis=1) if len(xs) else np.zeros((0, 4), np.float32)
    return over, target, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out)
    sample_dir = out / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(args.num_samples):
        over, target, events = make_one(args.height, args.width, i)
        path = sample_dir / f"{i:06d}.npz"
        np.savez_compressed(
            path,
            over=over,
            target=target,
            events=events,
            homography=np.eye(3, dtype=np.float32),
            dense_reference=np.array(1.0, dtype=np.float32),
            meta=json.dumps({"synthetic": True, "id": i}),
        )
        paths.append(path)

    n_train = int(0.7 * len(paths))
    n_val = int(0.15 * len(paths))
    splits = {
        "train": paths[:n_train],
        "val": paths[n_train:n_train+n_val],
        "test": paths[n_train+n_val:],
    }
    for name, ps in splits.items():
        (out / f"{name}.txt").write_text("\n".join(str(p) for p in ps) + "\n", encoding="utf-8")
    print(f"Created synthetic dataset at {out}")


if __name__ == "__main__":
    main()
