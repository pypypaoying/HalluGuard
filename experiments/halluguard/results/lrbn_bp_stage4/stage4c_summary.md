# Stage 4C BP Safe Controller Summary

- Verdict: `perf_only` / `keep_lrbn_main_report_bp_perf_ablation`
- Test threshold leakage: `False`

## Test Overall

| Method | MSE | MAE | Delta % vs LRBN | Harm | Coverage | q4 improvement | low-gap delta | config improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LRBN-BP-always | 4.643981 | 1.621852 | -5.111746 | 0.423177 | 1.000000 | 14.164335 | 0.162111 | 1.000000 |
| LRBN-BP-gap-strength | 4.782061 | 1.657199 | -2.290412 | 0.363281 | 0.873698 | 9.713605 | 0.000000 | 1.000000 |
| LRBN-BP-bounded | 4.806175 | 1.662432 | -1.797710 | 0.365885 | 1.000000 | 3.281651 | 0.050410 | 1.000000 |
| LRBN-BP-repair-gate | 4.808634 | 1.661626 | -1.747470 | 0.218750 | 0.610677 | 5.396815 | 0.048472 | 1.000000 |
| LRBN-BP-robust-anchor | 4.812579 | 1.663763 | -1.666865 | 0.364583 | 1.000000 | 4.477076 | -0.060525 | 1.000000 |
| LRBN-BP-conflict-filter | 4.822251 | 1.666692 | -1.469239 | 0.360677 | 1.000000 | 4.292585 | 0.015451 | 1.000000 |
| LRBN-BP-stage3-gated | 4.864131 | 1.674353 | -0.613520 | 0.018229 | 0.042969 | 2.021084 | 0.000000 | 0.750000 |
| LRBN-BP-short-bridge | 4.864551 | 1.673294 | -0.604945 | 0.173177 | 1.000000 | 1.695024 | -0.026224 | 1.000000 |
| LRBN-BP-safe-controller | 4.884224 | 1.679675 | -0.202964 | 0.092448 | 0.725260 | 0.607284 | -0.000199 | 1.000000 |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 1.000000 |

## Bootstrap CI

```json
{
  "LRBN": {
    "mean_delta": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "p_improve_bootstrap": 0.0
  },
  "LRBN-BP-always": {
    "mean_delta": -0.2501768870054856,
    "ci95_low": -0.32420954285727016,
    "ci95_high": -0.1768667218643575,
    "p_improve_bootstrap": 1.0
  },
  "LRBN-BP-gap-strength": {
    "mean_delta": -0.11209637619331488,
    "ci95_low": -0.14513378820411074,
    "ci95_high": -0.07947145544957496,
    "p_improve_bootstrap": 1.0
  },
  "LRBN-BP-bounded": {
    "mean_delta": -0.08798274341600294,
    "ci95_low": -0.10970741287091278,
    "ci95_high": -0.06621270109227745,
    "p_improve_bootstrap": 1.0
  },
  "LRBN-BP-robust-anchor": {
    "mean_delta": -0.08157898450708263,
    "ci95_low": -0.09821497371819356,
    "ci95_high": -0.06459553785342691,
    "p_improve_bootstrap": 1.0
  },
  "LRBN-BP-short-bridge": {
    "mean_delta": -0.029606938147181128,
    "ci95_low": -0.033658816109501,
    "ci95_high": -0.025727943210982696,
    "p_improve_bootstrap": 1.0
  },
  "LRBN-BP-conflict-filter": {
    "mean_delta": -0.07190686101377201,
    "ci95_low": -0.08702287134983447,
    "ci95_high": -0.05667296158617628,
    "p_improve_bootstrap": 1.0
  },
  "LRBN-BP-repair-gate": {
    "mean_delta": -0.08552394640923015,
    "ci95_low": -0.10947173430549288,
    "ci95_high": -0.061698074111621565,
    "p_improve_bootstrap": 1.0
  },
  "LRBN-BP-safe-controller": {
    "mean_delta": -0.009933364256515493,
    "ci95_low": -0.011730080240710414,
    "ci95_high": -0.008315421752094453,
    "p_improve_bootstrap": 1.0
  },
  "LRBN-BP-stage3-gated": {
    "mean_delta": -0.03002662419422079,
    "ci95_low": -0.06152689269364038,
    "ci95_high": -0.005077181252980037,
    "p_improve_bootstrap": 0.993
  }
}
```

## Verdict

```json
{
  "status": "perf_only",
  "decision": "keep_lrbn_main_report_bp_perf_ablation",
  "safe_delta_pct_vs_lrbn": -0.20296371829813556,
  "safe_delta_mse_vs_lrbn": -0.009933364256515493,
  "safe_harm_rate_vs_lrbn": 0.09244791666666667,
  "safe_q4_improvement_pct_vs_lrbn": 0.6072844989383837,
  "safe_low_delta_pct_vs_lrbn": -0.000199250876966675,
  "safe_config_improved_ratio": 1.0,
  "safe_ci95_low": -0.011730080240710414,
  "safe_ci95_high": -0.008315421752094453,
  "harm_reduction_vs_bp_always": 0.7815384615384615,
  "bp_always_delta_pct_vs_lrbn": -5.111745618870136,
  "bp_always_harm_rate_vs_lrbn": 0.4231770833333333,
  "stage3_gated_delta_pct_vs_lrbn": -0.6135197640016364,
  "stage3_gated_harm_rate_vs_lrbn": 0.018229166666666668,
  "performance_variant_pass": true,
  "test_threshold_leakage": false
}
```