from __future__ import annotations

import argparse
from pathlib import Path

import torch

from saegdnet.config import load_config
from saegdnet.data.dataset import SAEGDDataset, collate_batch
from saegdnet.diffusion.gaussian import GaussianDiffusion
from saegdnet.models.saegd import build_model
from saegdnet.training.engine import make_cond, _move_batch
from saegdnet.utils.checkpoint import load_checkpoint
from saegdnet.utils.io import save_image_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tmp_manifest = Path(args.out).with_suffix(".manifest.txt")
    tmp_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest.write_text(str(Path(args.input).resolve()) + "\n", encoding="utf-8")

    device = torch.device(cfg.get("eval", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ds = SAEGDDataset(tmp_manifest, cfg, training=False)
    batch = collate_batch([ds[0]])
    batch = _move_batch(batch, device)

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    diffusion = GaussianDiffusion(
        model,
        timesteps=int(cfg["diffusion"].get("timesteps", 1000)),
        beta_schedule=cfg["diffusion"].get("beta_schedule", "cosine"),
    ).to(device)
    diffusion.eval()
    with torch.no_grad():
        pred = diffusion.reconstruct(make_cond(batch), sampling_steps=int(cfg["diffusion"].get("sampling_steps", 50)))
    save_image_tensor(pred, args.out)


if __name__ == "__main__":
    main()
