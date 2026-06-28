# HalluGuard-SRA-BP Method Card

## Retained Main Line

The consolidated main line is **HalluGuard-SRA-BP**, with two retained variants:

- `Safe-SRA`
- `Balanced-SRA`

Both are inference-time residual/boundary repair modules applied after HalluGuard-LRBN. They do not change the backbone forecaster, training loss, or hidden states.

## Mechanism

The motivating failure mode is a forecast-start boundary discontinuity that survives LRBN normalization. Dense boundary projection (`BP-always`) can improve mean MSE, but it harms many samples. SRA-BP therefore makes boundary projection sparse and repair-aware.

For each sample:

1. Build a context anchor from the context tail.
2. Measure the raw prediction boundary gap `g_raw`.
3. Measure the LRBN prediction boundary gap `g_l`.
4. Estimate how much LRBN already repaired the boundary:

   ```text
   repair_ratio = 1 - g_l / (g_raw + eps)
   ```

5. Optionally measure whether the apparent jump is supported by local trend/volatility/smoothness.
6. Activate only when validation-calibrated gates say the residual boundary gap is large and under-repaired.
7. Apply a short bridge from the forecast start toward the context anchor:

   ```text
   y_sra = y_lrbn + alpha * gate * bridge * (anchor - y_lrbn[0])
   ```

The bridge decays over `K = H_div_4`, so SRA-BP edits the boundary region instead of dragging the whole forecast.

## Variant Definitions

### Safe-SRA

Conservative gate:

```json
{
  "method_family": "short",
  "anchor_mode": "last",
  "tail_len": 16,
  "tau_g": 5.265299801054961,
  "tau_r": 0.8,
  "tau_j": null,
  "alpha": 0.75,
  "K": "H_div_4",
  "continuous": false
}
```

Compact test metrics:

- MSE: `4.813149`
- MAE: `1.660950`
- MSE delta vs LRBN: `-1.655217%`
- MAE delta vs LRBN: `-1.260989%`
- harm rate: `0.035156`
- coverage: `0.187500`
- q4 boundary improvement: `6.928297%`
- test threshold leakage: `false`

### Balanced-SRA

Higher coverage gate with jump-support veto:

```json
{
  "method_family": "support",
  "anchor_mode": "last",
  "tail_len": 16,
  "tau_g": 2.4260872328869336,
  "tau_r": 0.8,
  "tau_j": 0.3,
  "alpha": 0.75,
  "K": "H_div_4",
  "continuous": false
}
```

Compact test metrics:

- MSE: `4.766983`
- MAE: `1.645627`
- MSE delta vs LRBN: `-2.598508%`
- MAE delta vs LRBN: `-2.171922%`
- harm rate: `0.104167`
- coverage: `0.436198`
- q4 boundary improvement: `8.587733%`
- test threshold leakage: `false`

## Calibration Protocol

Thresholds and policy selection use validation split only. Test split is used only for final reporting.

The included compact runner writes:

```text
stage5_calibration_grid.csv
stage5_selected_safe_params.json
stage5_selected_balanced_params.json
stage5_overall.csv
stage5_per_config.csv
stage5_boundary_gap_slices.csv
stage5_gap_repair_interaction.csv
stage5_horizon_segments.csv
stage5_selected_alignment.csv
stage5_verdict.json
summary.md
```

## Why Not Keep Later Atom/Selector Lines Here?

Later Stage18/19 performance atom diagnostics found strong oracle space, but deployable residual atom adapters failed compact gates. They are useful research leads, not the clean SRA-BP method.

This repository intentionally keeps SRA-BP small and auditable:

- one mechanism;
- two validation-calibrated variants;
- compact fixture and result artifacts;
- no hidden-state intervention;
- no test threshold tuning.

## Reproduction Command

```bash
python experiments/halluguard/run_stage5_sra_bp.py \
  --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv \
  --stage3-dir experiments/halluguard/results/lrbn_bp_stage3 \
  --stage4-dir experiments/halluguard/results/lrbn_bp_stage4 \
  --stage45-dir experiments/halluguard/results/lrbn_bp_attribution_stage45 \
  --output-dir experiments/halluguard/results/lrbn_sra_bp_stage5_repro \
  --n-bootstrap 2000
```
