# Troubleshooting

## CUDA out of memory

Reduce:

- image size
- base channels
- diffusion steps
- batch size

## Reconstructions look too pretty

That may be hallucination. Check event-edge F-score.

## Event images look empty

Check timestamp units. Microseconds and seconds are not the same beast.

## Saturation mask covers everything

Lower gain next time, or lower `soft_mask_tau` only if this is a visualization issue.

## Full-reference metrics look suspiciously high

Verify that synthetic saturation is not mixed into real full-reference test scores.
