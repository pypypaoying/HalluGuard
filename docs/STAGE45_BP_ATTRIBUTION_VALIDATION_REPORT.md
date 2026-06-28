# Stage 4.5 BP Attribution Validation Report

## What This Stage Tested

Stage 4.5 tested whether `LRBN-BP-always` is unsafe because of an identifiable dense boundary-projection mechanism, rather than merely because its alpha/anchor parameters were not tuned. This was an attribution stage, not a new method-search stage.

The run reused existing compact real-forecast assets:

- Datasets: `ETTm1`, `ETTh1`
- Backbones: `DLinear`, `PatchTST`
- Horizons: `96`, `192`
- Seed: `2026`
- Samples: `768` validation and `768` test
- Test configs: `8`
- Calibration: validation-only quantiles and thresholds
- Test threshold leakage: `False`

Command:

```bash
python experiments/halluguard/run_bp_attribution_stage45.py --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv --stage3-dir experiments/halluguard/results/lrbn_bp_stage3 --stage4-dir experiments/halluguard/results/lrbn_bp_stage4 --output-dir experiments/halluguard/results/lrbn_bp_attribution_stage45 --n-bootstrap 2000
```

## Output Package

Primary output directory:

`experiments/halluguard/results/lrbn_bp_attribution_stage45/`

Required artifacts were generated:

- `attribution_config.json`
- `attribution_sample_table.csv`
- `attribution_sample_table.parquet`
- `oracle_boundary_truth.csv`
- `residual_alignment_by_slice.csv`
- `gap_repair_interaction.csv`
- `win_loss_distribution.csv`
- `horizon_segment_attribution.csv`
- `anchor_reliability.csv`
- `anchor_disagreement_slices.csv`
- `per_config_attribution.csv`
- `bootstrap_ci.json`
- `failure_cases_topk.csv`
- `verdict.json`
- `summary.md`

## Headline Results

Compared with HalluGuard-LRBN:

| Method | MSE Delta % | Mean Delta | Harm Rate | Bootstrap Delta 95% CI |
|---|---:|---:|---:|---:|
| BP-always | -5.111746 | -0.250177 | 0.423177 | [-0.324210, -0.176867] |
| Stage3 gated BP | -0.613520 | -0.030027 | 0.018229 | [-0.060448, -0.004969] |
| Repair-gate BP | -1.747470 | -0.085524 | 0.218750 | [-0.111369, -0.061620] |

Interpretation: BP-always is a strong performance branch, but its harm rate is too high for a safe default. Stage3 gated is much safer and still has a statistically negative delta in the compact test.

## Mechanism Evidence

### Oracle Boundary Truth

`high_gL_low_gY` is the cleanest useful region:

- Count: `143`
- Mean delta vs LRBN: `-0.719912`
- Harm rate: `0.307692`
- BP-needed rate (`A_B > 1`): `0.692308`

`low_gL_high_gY` is a harmful true-jump region:

- Count: `35`
- Mean delta vs LRBN: `+0.011579`
- Harm rate: `0.628571`
- BP-needed rate: `0.371429`

This supports the claim that a large post-LRBN boundary gap can indicate a real forecast boundary error, but a real future jump should not be blindly pulled back to the context anchor.

### Gap x Repair Interaction

The high-gap / low-repair slice is strongly useful:

- Count: `163`
- Mean delta vs LRBN: `-0.627193`
- Harm rate: `0.361963`
- `A_B > 1` rate: `0.638037`
- Bootstrap CI for mean delta: `[-0.880570, -0.387852]`

The low-gap / high-repair slice is unsafe but not strongly significant in mean delta:

- Count: `90`
- Mean delta vs LRBN: `+0.012179`
- Harm rate: `0.500000`
- Bootstrap CI for mean delta: `[-0.019242, +0.044175]`

This is enough to support low-gap/high-repair as a harm-risk slice, but not enough to claim it alone explains all dense BP harm.

### Residual Alignment

The verdict file reports:

- Harmful slices have lower `A_B > 1` rate than useful slices.
- BP-always mechanism defect conditions passed `6/7`.
- Sparse repair-aware BP support conditions passed `3/4`.
- The only sparse-support condition that failed was `selected_samples_higher_A`; Stage3 selection is safe, but its selected samples were not higher than overall BP-always alignment under the current sample-level `A_B` statistic.

### Horizon Segment Attribution

BP-always gain is front-loaded:

| Segment | MSE Delta % vs LRBN | Harm Rate |
|---|---:|---:|
| early | -19.679989 | 0.363281 |
| mid | -3.232959 | 0.464844 |
| late | -0.747157 | 0.446615 |

This supports replacing full-horizon dense projection with a short-support or sparse boundary expert.

### Anchor Reliability

Changing the anchor does not solve harm:

| Anchor | MSE Delta % vs LRBN | Harm Rate |
|---|---:|---:|
| last | -5.111746 | 0.423177 |
| trend | -5.142234 | 0.424479 |
| robust | -4.949051 | 0.434896 |

Anchor choice changes details, but the main defect is dense always-on projection rather than the specific last-value anchor.

## Verdict

From `verdict.json`:

```json
{
  "bp_always_mechanism_defect_supported": "yes",
  "bp_always_defect_score": 6,
  "sparse_repair_aware_bp_supported": "yes",
  "sparse_repair_aware_score": 3,
  "test_threshold_leakage": false
}
```

Stage 4.5 supports this narrowed claim:

> BP-always validates boundary inconsistency as a real post-LRBN failure mode, but dense always-on boundary projection is not safe. The next BP line should be LRBN plus a sparse, repair-aware boundary expert.

It does not support making BP-always the clean default method. It also does not prove that anchor replacement alone can fix the harm.

## Caveats

- This is compact attribution only: 8 configs, seed 2026, horizons 96/192.
- Some normalized gap means are extremely large because a few context-difference scales are very small. The quantile-based bins are still validation-calibrated, but absolute `g_L` magnitude should not be overinterpreted.
- Low-gap/high-repair harm has high harm rate, but its mean-delta bootstrap CI crosses zero.
- The Stage3 gated line is safe and significant in compact attribution, but its coverage is low; the next method should improve sparse coverage without returning to dense BP.

## Next Step

Move to `LRBN + sparse repair-aware BP expert`: combine high-gap/low-repair detection, short-support boundary bridge, and validation-only harm control. Keep BP-always as a performance attribution / upper correction-power ablation, not as TableA default.
