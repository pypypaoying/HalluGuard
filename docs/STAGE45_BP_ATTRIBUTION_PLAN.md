# Stage 4.5 BP Attribution Validation Plan

## Selected Question

Stage 4 showed that `LRBN-BP-always` has strong compact-table MSE gain but high harm. Stage 4.5 is a mechanism attribution run, not a new method search: it tests whether dense always-on Boundary Projection is intrinsically unsafe and whether sparse repair-aware BP is better supported by oracle attribution.

## Fixed Contract

- Parent line: HalluGuard-LRBN `unified_revin_rdn_hybrid`.
- Compared branches: LRBN, LRBN-BP-always, Stage3 gated BP, Stage4 repair-gate BP.
- Data source: existing compact forecast assets listed in `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`.
- Scope: ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026.
- Calibration: validation split only for quantiles and gates.
- Evaluation: test split only for final attribution tables and verdict.
- Primary outputs: sample-level attribution table, A-H attribution CSVs, bootstrap CI, verdict, summary report.
- Non-goal: do not tune a new high-performance method in this stage.

## Experiment Slices

1. Oracle boundary truth: compare post-LRBN predicted gap `g_L` against true first-step gap `g_y`.
2. Residual alignment: report `A_B = 2 delta_B^T e_L / ||delta_B||^2` by slices.
3. Gap x repair interaction: test whether low-gap/high-repair explains harm and high-gap/low-repair explains gain.
4. Win/loss distribution: quantify whether average BP gain is carried by a small tail of large wins.
5. Horizon segment attribution: test whether gains are early and harm diffuses mid/late.
6. Anchor reliability: compare last/trend/robust anchors.
7. Per-config stability: verify mechanism consistency across compact configs.
8. Top failure export: save top harm/win cases for manual inspection.

## Commands

```bash
python experiments/halluguard/run_bp_attribution_stage45.py \
  --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv \
  --stage3-dir experiments/halluguard/results/lrbn_bp_stage3 \
  --stage4-dir experiments/halluguard/results/lrbn_bp_stage4 \
  --output-dir experiments/halluguard/results/lrbn_bp_attribution_stage45 \
  --n-bootstrap 2000
```

## Success Criteria

This stage succeeds if the full A-H attribution package is generated with `test_threshold_leakage=false`, and the report gives a clear verdict on:

- whether BP-always has a dense mechanism defect;
- whether Stage3 gated / repair-gate has oracle support as sparse repair-aware BP;
- whether anchor replacement alone explains the harm.

## Revision Log

- 2026-06-27: Created Stage 4.5 attribution validation contract from the user-provided validation document.
