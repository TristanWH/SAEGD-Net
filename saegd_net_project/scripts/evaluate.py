from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from saegdnet.config import load_config
from saegdnet.data.dataset import SAEGDDataset, collate_batch
from saegdnet.diffusion.gaussian import GaussianDiffusion
from saegdnet.evaluation.metrics import full_reference_metrics
from saegdnet.evaluation.event_support import event_supported_edge_prf
from saegdnet.models.saegd import build_model
from saegdnet.training.engine import make_cond, _move_batch
from saegdnet.utils.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("eval", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    diffusion = GaussianDiffusion(
        model,
        timesteps=int(cfg["diffusion"].get("timesteps", 1000)),
        beta_schedule=cfg["diffusion"].get("beta_schedule", "cosine"),
    ).to(device)

    manifest = cfg["data"][f"{args.split}_manifest"]
    ds = SAEGDDataset(manifest, cfg, training=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)

    metrics_sum = {}
    count_full = 0
    count_event = 0
    tol = int(cfg.get("eval", {}).get("event_edge_tolerance_px", 2))

    for batch in tqdm(loader, desc="evaluate"):
        batch = _move_batch(batch, device)
        cond = make_cond(batch)
        pred = diffusion.reconstruct(cond, sampling_steps=int(cfg["diffusion"].get("sampling_steps", 50)))

        prf = event_supported_edge_prf(pred, batch["activity"], batch["mask"], tolerance_px=tol)
        for k, v in prf.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        count_event += 1

        if float(batch["dense_reference"].mean().item()) > 0.5:
            m = full_reference_metrics(pred, batch["target"], batch["mask"], batch["activity"])
            for k, v in m.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            count_full += 1

    print("Dense-reference frames:", count_full)
    print("All evaluated frames:", len(ds))
    print("Dense-reference ratio:", count_full / max(len(ds), 1))
    print("Hardware note:", cfg.get("eval", {}).get("runtime_hardware", "not specified"))
    for k, v in sorted(metrics_sum.items()):
        denom = count_full if k in {"psnr", "ssim", "rmse", "psnr_sat", "eas"} else count_event
        print(f"{k}: {v / max(denom, 1):.6f}")


if __name__ == "__main__":
    main()
