from __future__ import annotations

from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

from saegdnet.models.blocks import ResBlock, Downsample, Upsample, timestep_embedding, EventFiLM
from saegdnet.models.encoders import SaturationAwareFrameEncoder, EventEncoder
from saegdnet.models.attention import CrossModalStructuralAttention, SpatialSelfAttention


class SAEGDUNet(nn.Module):
    """Pixel-space U-Net diffusion backbone with saturation-aware event conditioning."""

    def __init__(
        self,
        image_ch: int = 1,
        event_bins: int = 3,
        base_ch: int = 64,
        channel_mults: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attention_resolutions: list[int] | None = None,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        use_event_film: bool = True,
        use_laplacian: bool = True,
        concat_only: bool = False,
    ):
        super().__init__()
        self.image_ch = image_ch
        self.event_ch = 2 * event_bins
        self.concat_only = concat_only
        self.use_event_film = use_event_film
        self.use_cross_attention = use_cross_attention
        self.attention_resolutions = set(attention_resolutions or [])

        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        if concat_only:
            cond_ch = image_ch + 1 + 1 + 1 + self.event_ch
            self.concat_condition = nn.Sequential(
                nn.Conv2d(cond_ch, base_ch, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
            )
            in_ch = image_ch + base_ch
            self.frame_encoder = None
            self.event_encoder = None
        else:
            self.frame_encoder = SaturationAwareFrameEncoder(
                image_ch, base_ch, channel_mults, use_laplacian=use_laplacian
            )
            self.event_encoder = EventEncoder(self.event_ch, base_ch, channel_mults)
            in_ch = image_ch

        self.input = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.films_down = nn.ModuleList()
        self.cross_down = nn.ModuleList()
        ch = base_ch
        self.skip_channels = []
        for level, mult in enumerate(channel_mults):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                self.skip_channels.append(ch)
            self.downs.append(blocks)
            self.films_down.append(EventFiLM(out_ch, out_ch))
            self.cross_down.append(CrossModalStructuralAttention(out_ch, out_ch) if use_cross_attention else nn.Identity())
            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SpatialSelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.films_up = nn.ModuleList()
        self.cross_up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                ch = out_ch
            self.ups.append(blocks)
            self.films_up.append(EventFiLM(out_ch, out_ch))
            self.cross_up.append(CrossModalStructuralAttention(out_ch, out_ch) if use_cross_attention else nn.Identity())
            if level != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, image_ch, 3, padding=1),
        )

    def _condition_features(self, cond: Dict[str, torch.Tensor]):
        if self.concat_only:
            cat = torch.cat([
                cond["over_norm"], cond["soft_mask"], cond["reliability"], cond["activity"], cond["event"]
            ], dim=1)
            c = self.concat_condition(cat)
            return None, None, None, c
        frame_feats = self.frame_encoder(cond["over_norm"], cond["soft_mask"], cond["reliability"], cond["activity"])
        event_feats, event_weights = self.event_encoder(cond["event"], cond["soft_mask"])
        return frame_feats, event_feats, event_weights, None

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, _, h, w = x_t.shape
        t_emb = timestep_embedding(timesteps, self.input.out_channels)
        t_emb = self.time_mlp(t_emb)

        frame_feats, event_feats, event_weights, concat_cond = self._condition_features(cond)
        if self.concat_only:
            if concat_cond.shape[-2:] != x_t.shape[-2:]:
                concat_cond = F.interpolate(concat_cond, size=x_t.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x_t, concat_cond], dim=1)
        else:
            x = x_t

        hfeat = self.input(x)
        skips = []
        for i, blocks in enumerate(self.downs):
            for block in blocks:
                hfeat = block(hfeat, t_emb)
                if not self.concat_only:
                    ff = F.interpolate(frame_feats[i], size=hfeat.shape[-2:], mode="bilinear", align_corners=False)
                    ef = F.interpolate(event_feats[i], size=hfeat.shape[-2:], mode="bilinear", align_corners=False)
                    ew = F.interpolate(event_weights[i], size=hfeat.shape[-2:], mode="bilinear", align_corners=False)
                    hfeat = hfeat + ff
                    if self.use_event_film:
                        hfeat = (1 - ew) * hfeat + ew * self.films_down[i](hfeat, ef)
                    if self.use_cross_attention and not isinstance(self.cross_down[i], nn.Identity):
                        hfeat = self.cross_down[i](hfeat, ef, cond["soft_mask"])
                skips.append(hfeat)
            hfeat = self.downsamples[i](hfeat)

        hfeat = self.mid1(hfeat, t_emb)
        hfeat = self.mid_attn(hfeat)
        hfeat = self.mid2(hfeat, t_emb)

        for i, blocks in enumerate(self.ups):
            level = len(self.ups) - 1 - i
            for block in blocks:
                skip = skips.pop()
                if hfeat.shape[-2:] != skip.shape[-2:]:
                    hfeat = F.interpolate(hfeat, size=skip.shape[-2:], mode="nearest")
                hfeat = torch.cat([hfeat, skip], dim=1)
                hfeat = block(hfeat, t_emb)
                if not self.concat_only:
                    ef = F.interpolate(event_feats[level], size=hfeat.shape[-2:], mode="bilinear", align_corners=False)
                    ew = F.interpolate(event_weights[level], size=hfeat.shape[-2:], mode="bilinear", align_corners=False)
                    if self.use_event_film:
                        hfeat = (1 - ew) * hfeat + ew * self.films_up[i](hfeat, ef)
                    if self.use_cross_attention and not isinstance(self.cross_up[i], nn.Identity):
                        hfeat = self.cross_up[i](hfeat, ef, cond["soft_mask"])
            hfeat = self.upsamples[i](hfeat)

        return self.out(hfeat)


def build_model(cfg: Dict[str, Any]) -> SAEGDUNet:
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    return SAEGDUNet(
        image_ch=int(data.get("in_channels", 1)),
        event_bins=int(data.get("event_bins", 3)),
        base_ch=int(model.get("base_channels", 64)),
        channel_mults=list(model.get("channel_mults", [1, 2, 4, 4])),
        num_res_blocks=int(model.get("num_res_blocks", 2)),
        attention_resolutions=list(model.get("attention_resolutions", [])),
        dropout=float(model.get("dropout", 0.0)),
        use_cross_attention=bool(model.get("use_cross_attention", True)),
        use_event_film=bool(model.get("use_event_film", True)),
        use_laplacian=bool(model.get("use_laplacian", True)),
        concat_only=bool(model.get("concat_only", False)),
    )
