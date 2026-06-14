# SAEGD-Net: Saturation-Aware Event-Guided Diffusion Reconstruction

Welcome to **SAEGD-Net**, a research-oriented PyTorch codebase for reconstructing severely overexposed ultra-high-speed camera (UHSC) frames using event-camera guidance.

It is designed for the paper-style pipeline:

> overexposed UHSC frame + event stream → saturation reliability → polarity-aware event tensor → event-guided diffusion → measurement-friendly reconstruction

The code is intentionally modular. It is not one mysterious `train_final_final_v9.py` that only works on the author's laptop during a full moon. You can inspect, replace, and debug each component.

## What is included?

- Saturation mask and reliability-map generation
- Event denoising, homography warping, polarity-aware voxelization, and activity-map generation
- Saturation-aware frame encoder
- Event encoder
- Reliability-gated frame-event fusion
- FiLM-style event-modulated denoising blocks
- Cross-modal structural attention
- Pixel-space diffusion U-Net
- DDPM/DDIM training and inference utilities
- Loss terms:
  - diffusion loss
  - saturation-weighted reconstruction loss
  - valid-region preservation loss
  - Sobel edge loss
  - event-structure consistency loss
  - perceptual loss placeholder / optional VGG perceptual loss
  - temporal consistency loss
- Full-reference metrics:
  - PSNR
  - SSIM
  - RMSE
  - masked saturated-region PSNR
  - event-aware sharpness
- Event-supported structure verification
- Training, evaluation, inference, preprocessing scripts
- Synthetic smoke-test data generator
- Clean configs, docs, tests, and CLI entry points

## Project layout

```text
saegd_net_project/
├── configs/
│   ├── default.yaml
│   ├── tiny_smoke.yaml
│   └── inference.yaml
├── docs/
│   ├── DATA_FORMAT.md
│   ├── MODEL_DETAILS.md
│   ├── TRAINING_PROTOCOL.md
│   └── TROUBLESHOOTING.md
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── preprocess_events.py
│   ├── build_pairs.py
│   ├── make_synthetic_sample.py
│   └── visualize_events.py
├── saegdnet/
│   ├── data/
│   ├── diffusion/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/
```

## Installation

```bash
conda create -n saegd python=3.10 -y
conda activate saegd

pip install -r requirements.txt
pip install -e .
```

For CUDA acceleration, install a PyTorch build matching your driver and CUDA version.

## Quick smoke test

Generate a tiny synthetic event-frame sample:

```bash
python scripts/make_synthetic_sample.py --out data/smoke --num-samples 16
```

Run a tiny training job:

```bash
python scripts/train.py --config configs/tiny_smoke.yaml
```

Run inference:

```bash
python scripts/infer.py \
  --config configs/inference.yaml \
  --checkpoint runs/tiny_smoke/checkpoints/latest.pt \
  --input data/smoke/samples/000000.npz \
  --out outputs/smoke_recon.png
```

If it runs, the codebase is alive. If it fails, check `docs/TROUBLESHOOTING.md`. If it still fails, make tea. Debugging diffusion models is a lifestyle.

## Expected dataset format

Each sample is stored as `.npz`:

```python
{
  "over":      float32 array [H, W] or [H, W, C], normalized to [0, 1],
  "target":    float32 array [H, W] or [H, W, C], optional for inference,
  "events":    float32 array [N, 4], columns: x, y, t, p,
  "homography":float32 array [3, 3], optional,
  "meta":      optional JSON string
}
```

For real UHSC--EVS data, the recommended preparation is:

```bash
python scripts/build_pairs.py \
  --frames path/to/uhsc_frames \
  --events path/to/events.npy \
  --homography path/to/H_event_to_frame.txt \
  --out data/my_dataset
```

More details: `docs/DATA_FORMAT.md`.

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

The default config includes:

- event bins `K=3`
- saturation threshold `0.98`
- soft mask transition `eta=0.02`
- loss weights:
  - `lambda_sat = 1.0`
  - `lambda_valid = 0.5`
  - `lambda_edge = 0.2`
  - `lambda_event = 0.3`
  - `lambda_perc = 0.05`
  - `lambda_temp = 0.1`

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint runs/saegd/checkpoints/best.pt \
  --split test
```

## Reproducibility notes

- Full-reference metrics should only be computed on samples with verified dense references.
- Real severe-saturation samples without dense radiometric references should be evaluated by:
  - qualitative inspection
  - event-supported edge precision/recall/F-score
  - downstream measurement probes
- Runtime reporting should include hardware. Example:
  - NVIDIA GeForce RTX 4090
  - PyTorch version
  - input resolution
  - diffusion steps
  - batch size

## License

This project is released for academic research. See `LICENSE`.

## Citation

If this codebase helps your research, cite the corresponding SAEGD-Net paper.

And remember: if your reconstruction looks too beautiful to be true, it might be hallucinating. Ask the events.
