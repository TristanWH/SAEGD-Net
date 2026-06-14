# Model Details

SAEGD-Net contains:

1. Saturation-aware frame encoder
2. Polarity-aware event encoder
3. Reliability-gated feature fusion
4. Event-FiLM denoising blocks
5. Cross-modal structural attention
6. Pixel-space diffusion U-Net
7. Mask-guided reconstruction fusion

## Key idea

- Valid frame pixels are treated as reliable appearance constraints.
- Saturated pixels are treated as locally unobservable.
- Events provide high-dynamic-range structural evidence where the frame is clipped.
- Diffusion fills the missing content, but is constrained by event-supported structures.

## Variants

- `frame-only diffusion`: removes event input and event modulation.
- `event-only diffusion`: removes frame branch and reliability-gated frame fusion.
- `concat-only diffusion`: concatenates frame/mask/reliability/activity/event tensors through a simple condition stem, without gated fusion, FiLM, or cross-attention.
