# Stage 4 BP Harm Control Validation Plan

## Goal

Explain and control the sample-level harm of `LRBN-BP-always`, while preserving the boundary-repair gain discovered in Stage 3.

## Parent Evidence

- Parent method: `HalluGuard-LRBN unified_revin_rdn_hybrid`
- Stage 3 candidate: `HalluGuard-LRBN-BP-gated`
- Stage 3 result: MSE improvement vs LRBN `0.613520%`, harm `1.822917 pp`
- Stage 3 performance ablation: `LRBN-BP-always` improves MSE `5.111746%`, but harm is `42.3177%`

## Stage 4 Slices

1. Stage 4A: BP-always harm attribution.
2. Stage 4B: mechanism-level ablations:
   - gap strength
   - bounded BP
   - robust anchor
   - short bridge
   - conflict filter
   - repair gate
3. Stage 4C: `LRBN-BP-safe-controller` combination validation.

## Data Contract

- Input: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- Scope: compact validation assets only.
- Validation split selects all parameters.
- Test split only evaluates.
- All comparisons are against `HalluGuard-LRBN`.

## Outputs

`experiments/halluguard/results/lrbn_bp_stage4/`

- `stage4a_failure_attribution.csv`
- `stage4a_boundary_gap_bins.csv`
- `stage4a_repair_ratio_bins.csv`
- `stage4a_conflict_cosine_bins.csv`
- `stage4a_norm_ratio_bins.csv`
- `stage4a_anchor_reliability_bins.csv`
- `stage4a_horizon_segment_mse.csv`
- `stage4a_failure_cases_topk.csv`
- `stage4a_summary.md`
- `stage4b_calibration_grid.csv`
- `stage4c_overall.csv`
- `stage4c_boundary_slices.csv`
- `stage4c_per_config.csv`
- `stage4c_horizon_segments.csv`
- `stage4c_bootstrap_ci.json`
- `stage4c_direction_verdict.json`
- `stage4c_summary.md`

