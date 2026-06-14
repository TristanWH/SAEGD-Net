from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from saegdnet.diffusion.schedules import make_beta_schedule
from saegdnet.data.masks import smooth_mask


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_schedule: str = "cosine", objective: str = "eps"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.objective = objective

        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(x_t, t, cond)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps) / extract(self.sqrt_alphas_cumprod, t, x_t.shape).clamp_min(1e-8)

    def p_losses(self, x0: torch.Tensor, cond: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.predict_noise(x_t, t, cond)
        x0_pred = self.predict_x0_from_eps(x_t, t, pred_noise).clamp(0, 1)
        return {
            "diffusion": F.mse_loss(pred_noise, noise),
            "x0_pred": x0_pred,
            "noise": noise,
            "t": t,
        }

    @torch.no_grad()
    def ddim_sample(self, shape: tuple[int, ...], cond: Dict[str, torch.Tensor], steps: int = 50) -> torch.Tensor:
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for i, t_scalar in enumerate(times):
            t = torch.full((shape[0],), int(t_scalar.item()), device=device, dtype=torch.long)
            eps = self.predict_noise(x, t, cond)
            x0 = self.predict_x0_from_eps(x, t, eps).clamp(0, 1)
            if i == len(times) - 1:
                x = x0
            else:
                t_next = torch.full((shape[0],), int(times[i + 1].item()), device=device, dtype=torch.long)
                alpha_next = extract(self.alphas_cumprod, t_next, shape)
                x = torch.sqrt(alpha_next) * x0 + torch.sqrt(1 - alpha_next) * eps
        return x.clamp(0, 1)

    @torch.no_grad()
    def reconstruct(self, cond: Dict[str, torch.Tensor], sampling_steps: int = 50) -> torch.Tensor:
        over = cond["over"]
        pred = self.ddim_sample(over.shape, cond, steps=sampling_steps)
        mask = smooth_mask(cond["soft_mask"])
        return (mask * pred + (1 - mask) * over).clamp(0, 1)
