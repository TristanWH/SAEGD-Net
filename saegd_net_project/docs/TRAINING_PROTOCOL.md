# Training Protocol

Recommended protocol:

1. Split by physical trial, not by frame.
2. Select hyperparameters on validation split only.
3. Freeze all hyperparameters before test evaluation.
4. Use verified dense references only for full-reference metrics.
5. Use real severe samples without dense references for:
   - qualitative evaluation
   - event-supported structure verification
   - robustness analysis
   - downstream measurement validation

## Default loss weights

```text
lambda_sat   = 1.0
lambda_valid = 0.5
lambda_edge  = 0.2
lambda_event = 0.3
lambda_perc  = 0.05
lambda_temp  = 0.1
```
