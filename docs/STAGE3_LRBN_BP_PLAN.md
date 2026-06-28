# Stage 3 LRBN + Optional Boundary Projection Validation Plan

## Goal

Validate whether `HalluGuard-LRBN + optional Boundary Projection` improves over the frozen `HalluGuard-LRBN unified_revin_rdn_hybrid` line on the compact real-forecast assets before promoting it into TableA.

## Contract

- Parent method: `HalluGuard-LRBN unified_revin_rdn_hybrid`.
- Candidate: `HalluGuard-LRBN-BP-gated`.
- Scope: compact sanity check, not full TableA.
- Data: existing `research_direction_validation/forecast_inputs/combined_metrics.csv` predictions.
- Splits: validation selects `alpha/tau`; test only evaluates.
- Main comparison: candidate vs `HalluGuard-LRBN`, not only vs raw forecast.
- Safe fallback: `alpha=0` is always allowed and equals LRBN.

## Required Methods

- `raw_no_correction`
- `HalluGuard-LRBN`
- `HalluGuard-LRBN-BP-gated`
- `HalluGuard-LRBN-BP-always`
- `HalluGuard-BP-global`
- `matched_sparse_smoothing`
- `naive_smoothing`
- `ema_smoothing`
- `median_smoothing`

## Pass Criteria

Strong pass:

- overall MSE improvement vs LRBN >= 0.5%;
- q4 high-boundary-gap improvement vs LRBN >= 2.0%;
- harm rate vs LRBN <= 2 percentage points;
- q1/q2 low-gap degradation <= 0.5%;
- configs improved ratio >= 60%.

Weak pass:

- overall MSE delta vs LRBN within [-0.2%, +0.2%];
- q4 improvement >= 2.0%;
- harm rate vs LRBN <= 2 percentage points.

Fail:

- overall MSE worsens vs LRBN by > 0.5%;
- or harm rate exceeds LRBN +2 pp;
- or q1/q2 low-gap slice degrades;
- or q4 has no stable improvement.

## Outputs

`experiments/halluguard/results/lrbn_bp_stage3/`

- `selected_lrbn_bp_params.json`
- `lrbn_bp_calibration_grid.csv`
- `lrbn_bp_overall.csv`
- `lrbn_bp_boundary_slices.csv`
- `lrbn_bp_per_config.csv`
- `lrbn_bp_bootstrap_ci.json`
- `lrbn_bp_failure_cases.csv`
- `lrbn_bp_direction_verdict.json`
- `summary.md`

