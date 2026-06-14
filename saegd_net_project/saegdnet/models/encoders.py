from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from saegdnet.models.blocks import ConvGNAct


def laplacian_pyramid(x: torch.Tensor, levels: int) -> List[torch.Tensor]:
    pyr = []
    cur = x
    for _ in range(levels):
        down = F.avg_pool2d(cur, 2, ceil_mode=True)
        up = F.interpolate(down, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        pyr.append(cur - up)
        cur = down
    return pyr


class SaturationAwareFrameEncoder(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, channel_mults: list[int], use_laplacian: bool = True):
        super().__init__()
        self.use_laplacian = use_laplacian
        self.blocks = nn.ModuleList()
        prev = in_ch + 3  # normalized frame + soft mask + reliability + activity
        for mult in channel_mults:
            ch = base_ch * mult
            extra = 1 if use_laplacian else 0
            self.blocks.append(nn.Sequential(
                ConvGNAct(prev + extra, ch),
                ConvGNAct(ch, ch),
            ))
            prev = ch

    def forward(self, over_norm: torch.Tensor, soft_mask: torch.Tensor, reliability: torch.Tensor, activity: torch.Tensor) -> list[torch.Tensor]:
        x = torch.cat([over_norm, soft_mask, reliability, activity], dim=1)
        laps = laplacian_pyramid(over_norm, len(self.blocks)) if self.use_laplacian else [None] * len(self.blocks)
        feats = []
        cur = x
        for i, block in enumerate(self.blocks):
            if i > 0:
                cur = F.avg_pool2d(cur, 2, ceil_mode=True)
            inp = cur
            if self.use_laplacian:
                lap = F.interpolate(laps[i], size=inp.shape[-2:], mode="bilinear", align_corners=False)
                inp = torch.cat([inp, lap.mean(dim=1, keepdim=True)], dim=1)
            f = block(inp)
            r = F.interpolate(reliability, size=f.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(f * r)
            cur = f
        return feats


class EventEncoder(nn.Module):
    def __init__(self, event_ch: int, base_ch: int, channel_mults: list[int]):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.guides = nn.ModuleList()
        prev = event_ch
        for mult in channel_mults:
            ch = base_ch * mult
            self.blocks.append(nn.Sequential(
                ConvGNAct(prev, ch),
                ConvGNAct(ch, ch),
            ))
            self.guides.append(nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid()))
            prev = ch

    def forward(self, event: torch.Tensor, soft_mask: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        feats, weights = [], []
        cur = event
        for i, block in enumerate(self.blocks):
            if i > 0:
                cur = F.avg_pool2d(cur, 2, ceil_mode=True)
            f = block(cur)
            guide = self.guides[i](f)
            sm = F.interpolate(soft_mask, size=f.shape[-2:], mode="bilinear", align_corners=False)
            weights.append(guide * sm)
            feats.append(f)
            cur = f
        return feats, weights
