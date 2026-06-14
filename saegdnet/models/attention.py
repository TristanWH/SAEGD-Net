from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpatialSelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        head_dim = c // self.heads
        q = q.reshape(b, self.heads, head_dim, h * w).transpose(-1, -2)
        k = k.reshape(b, self.heads, head_dim, h * w)
        v = v.reshape(b, self.heads, head_dim, h * w).transpose(-1, -2)
        attn = torch.softmax(torch.matmul(q, k) / (head_dim ** 0.5), dim=-1)
        out = torch.matmul(attn, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class CrossModalStructuralAttention(nn.Module):
    """Cross-attention from diffusion features to event features with mask bias."""

    def __init__(self, ch: int, event_ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(event_ch, ch, 1)
        self.v = nn.Conv2d(event_ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.norm_x = nn.GroupNorm(min(8, ch), ch)
        self.norm_e = nn.GroupNorm(min(8, event_ch), event_ch)

    def forward(self, x: torch.Tensor, event_feat: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if event_feat.shape[-2:] != x.shape[-2:]:
            event_feat = F.interpolate(event_feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if mask is not None and mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)

        b, c, h, w = x.shape
        q = self.q(self.norm_x(x))
        k = self.k(self.norm_e(event_feat))
        v = self.v(self.norm_e(event_feat))

        head_dim = c // self.heads
        q = q.reshape(b, self.heads, head_dim, h * w).transpose(-1, -2)
        k = k.reshape(b, self.heads, head_dim, h * w)
        v = v.reshape(b, self.heads, head_dim, h * w).transpose(-1, -2)

        logits = torch.matmul(q, k) / (head_dim ** 0.5)
        if mask is not None:
            bias = mask.flatten(2).unsqueeze(1).unsqueeze(2)
            logits = logits + bias
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)
