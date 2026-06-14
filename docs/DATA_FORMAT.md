# Data Format

Each sample is an `.npz` file with the following keys.

## Required

- `over`: overexposed UHSC frame, `[H,W]` or `[H,W,C]`, float in `[0,1]`
- `events`: event array `[N,4]` with columns:
  - `x`
  - `y`
  - `t`
  - `p`, where polarity is `-1` or `+1`

## Optional but recommended

- `target`: dense reference frame, same shape as `over`
- `dense_reference`: `1.0` if `target` is a verified dense radiometric reference, otherwise `0.0`
- `homography`: 3x3 event-to-frame homography
- `meta`: JSON string

## Important evaluation rule

Full-reference metrics such as PSNR/SSIM/LPIPS/RMSE should only be computed on samples with:

```python
dense_reference == 1
```

Real severely saturated samples without dense references should be evaluated using event-supported structure verification and downstream measurement probes.
