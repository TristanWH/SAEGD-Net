import torch
from saegdnet.models.saegd import SAEGDUNet


def test_model_forward_tiny():
    model = SAEGDUNet(image_ch=1, event_bins=2, base_ch=8, channel_mults=[1, 2], num_res_blocks=1)
    b, h, w = 1, 32, 32
    x = torch.randn(b, 1, h, w)
    cond = {
        "over": torch.rand(b, 1, h, w),
        "over_norm": torch.rand(b, 1, h, w),
        "event": torch.rand(b, 4, h, w),
        "activity": torch.rand(b, 1, h, w),
        "soft_mask": torch.rand(b, 1, h, w),
        "reliability": torch.rand(b, 1, h, w),
    }
    t = torch.randint(0, 10, (b,))
    y = model(x, t, cond)
    assert y.shape == x.shape
