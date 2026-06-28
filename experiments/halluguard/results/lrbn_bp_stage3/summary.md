# Stage 3 LRBN + Optional Boundary Projection Summary

- Samples: 1536
- Test samples: 768
- Configs: 8
- Calibration: validation-only alpha/tau selection; test-only evaluation.
- Verdict: `strong_pass` / `enter_full_table`
- Test threshold leakage: `False`

## Test Overall

| Method | Mean MSE | Mean MAE | Delta % vs LRBN | Coverage | Harm vs LRBN |
|---|---:|---:|---:|---:|---:|
| HalluGuard-LRBN-BP-always | 4.643981 | 1.621852 | -5.111746 | 1.000000 | 0.423177 |
| HalluGuard-LRBN-BP-gated | 4.864131 | 1.674353 | -0.613520 | 0.042969 | 0.018229 |
| HalluGuard-LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 |
| HalluGuard-BP-global | 5.985373 | 1.843170 | 22.296291 | 1.000000 | 0.569010 |
| ema_smoothing | 6.060487 | 1.848221 | 23.831056 | 0.000000 | 0.438802 |
| naive_smoothing | 6.072069 | 1.849256 | 24.067707 | 0.000000 | 0.430990 |
| median_smoothing | 6.129929 | 1.860337 | 25.249936 | 0.000000 | 0.449219 |
| matched_sparse_smoothing | 6.222475 | 1.877730 | 27.140880 | 0.000000 | 0.584635 |
| raw_no_correction | 6.427221 | 1.914908 | 31.324348 | 0.000000 | 0.641927 |

## Boundary Slice: LRBN-BP-gated

| Bin | Method MSE | LRBN MSE | Delta % vs LRBN | Coverage | Harm |
|---|---:|---:|---:|---:|---:|
| q1_low | 5.480761 | 5.480761 | 0.000000 | 0.000000 | 0.000000 |
| q2 | 4.559878 | 4.559878 | 0.000000 | 0.000000 | 0.000000 |
| q3 | 5.136111 | 5.136111 | 0.000000 | 0.000000 | 0.000000 |
| q4_high | 4.279774 | 4.399880 | -2.729767 | 0.171875 | 0.072917 |

## Bootstrap vs LRBN

```json
{
  "HalluGuard-LRBN-BP-gated": {
    "mean_delta": -0.03002662419422079,
    "ci95_low": -0.06152689269364038,
    "ci95_high": -0.005077181252980037,
    "p_improve_bootstrap": 0.993
  },
  "HalluGuard-LRBN-BP-always": {
    "mean_delta": -0.2501768870054856,
    "ci95_low": -0.3242095428572702,
    "ci95_high": -0.1768667218643575,
    "p_improve_bootstrap": 1.0
  },
  "HalluGuard-BP-global": {
    "mean_delta": 1.0912155998794109,
    "ci95_low": 0.849475758625798,
    "ci95_high": 1.3351945157745686,
    "p_improve_bootstrap": 0.0
  }
}
```

## Decision

`HalluGuard-LRBN-BP-gated` status: `strong_pass`. Decision: `enter_full_table`.
