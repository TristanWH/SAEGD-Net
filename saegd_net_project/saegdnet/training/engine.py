from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from saegdnet.data.dataset import SAEGDDataset, collate_batch
from saegdnet.diffusion.gaussian import GaussianDiffusion
from saegdnet.models.saegd import build_model
from saegdnet.training.losses import SAEGDLoss
from saegdnet.evaluation.metrics import full_reference_metrics
from saegdnet.utils.checkpoint import save_checkpoint
from saegdnet.utils.io import ensure_dir
from saegdnet.utils.seed import seed_everything


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def make_cond(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "over": batch["over"],
        "over_norm": batch["over_norm"],
        "event": batch["event"],
        "activity": batch["activity"],
        "soft_mask": batch["soft_mask"],
        "reliability": batch["reliability"],
    }


def train_from_config(cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg.get("seed", 3407)))
    train_cfg = cfg.get("train", {})
    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    run_dir = ensure_dir(train_cfg.get("run_dir", "runs/saegd"))
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    train_ds = SAEGDDataset(cfg["data"]["train_manifest"], cfg, training=True)
    val_ds = SAEGDDataset(cfg["data"]["val_manifest"], cfg, training=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = build_model(cfg).to(device)
    diffusion = GaussianDiffusion(
        model,
        timesteps=int(cfg["diffusion"].get("timesteps", 1000)),
        beta_schedule=cfg["diffusion"].get("beta_schedule", "cosine"),
        objective=cfg["diffusion"].get("objective", "eps"),
    ).to(device)

    criterion = SAEGDLoss(cfg)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    best_psnr = -1.0
    epochs = int(train_cfg.get("epochs", 100))
    log_every = int(train_cfg.get("log_every", 20))

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            batch = _move_batch(batch, device)
            cond = make_cond(batch)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                diff_out = diffusion.p_losses(batch["target"], cond)
                losses = criterion(diff_out, batch)
            scaler.scale(losses["total"]).backward()
            if float(train_cfg.get("grad_clip", 0)) > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            scaler.step(optim)
            scaler.update()
            if step % log_every == 0:
                pbar.set_postfix({k: float(v.detach().cpu()) for k, v in losses.items() if k != "total"} | {"total": float(losses["total"].detach().cpu())})

        val_psnr = validate(diffusion, val_loader, device, int(cfg["diffusion"].get("sampling_steps", 50)))
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "cfg": cfg,
            "val_psnr": val_psnr,
        }
        save_checkpoint(ckpt_dir / "latest.pt", state)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(ckpt_dir / "best.pt", state)


@torch.no_grad()
def validate(diffusion: GaussianDiffusion, loader: DataLoader, device: torch.device, sampling_steps: int) -> float:
    diffusion.model.eval()
    scores = []
    for batch in loader:
        batch = _move_batch(batch, device)
        cond = make_cond(batch)
        pred = diffusion.reconstruct(cond, sampling_steps=sampling_steps)
        if float(batch["dense_reference"].mean().item()) > 0.5:
            metrics = full_reference_metrics(pred, batch["target"], batch["mask"], batch["activity"])
            scores.append(metrics["psnr"])
    return sum(scores) / max(len(scores), 1)
