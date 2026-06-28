# Stage 5 SRA-BP Validation Plan

## Selected Idea

Stage 5 validates `HalluGuard-SRA-BP`: HalluGuard-LRBN plus a sparse repair-aware Boundary Projection expert. The idea is to keep LRBN as the clean parent and activate BP only on validation-supported high post-LRBN boundary-gap / low LRBN-repair samples, with short-support boundary bridges and optional target-free true-jump suppression.

## Fixed Contract

- Parent: HalluGuard-LRBN `unified_revin_rdn_hybrid`.
- Upstream evidence: Stage 4.5 BP attribution.
- Scope: compact validation only, not full TableA.
- Input: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`.
- Datasets: `ETTm1`, `ETTh1`.
- Backbones: `DLinear`, `PatchTST`.
- Horizons: `96`, `192`.
- Seed: `2026`.
- Samples: `768` validation + `768` test.
- Calibration: validation-only.
- Evaluation: test-only.
- Bootstrap: `2000`.

## Candidate Families

- `LRBN-SRA-BP-basic`: hard high-gap / low-repair gate with full-horizon bridge.
- `LRBN-SRA-BP-short`: hard high-gap / low-repair gate with short bridge.
- `LRBN-SRA-BP-support`: short bridge plus target-free jump-support suppressor.
- `LRBN-SRA-BP-continuous`: continuous sparse strength after the same SRA features.

Required baselines:

- `LRBN`
- `raw_no_correction`
- `matched_sparse_smoothing`
- `ema_smoothing`
- `median_smoothing`
- `naive_smoothing`
- `LRBN-BP-always`
- `LRBN-BP-stage3-gated`
- `LRBN-BP-repair-gate`
- `LRBN-BP-short-bridge`

## Selection Rules

Safe-SRA is selected on validation among candidates satisfying:

- MSE improvement vs LRBN >= `0.5%`
- harm rate vs LRBN <= `0.05`
- q4 high-gap improvement >= `2.0%`
- low-gap/high-repair degradation <= `0.5%`
- config improved ratio >= `0.75`

Balanced-SRA is selected on validation among candidates satisfying:

- MSE improvement vs LRBN >= `1.5%`
- harm rate vs LRBN <= `0.15`
- q4 high-gap improvement >= `4.0%`
- config improved ratio >= `0.75`

Final pass gates are evaluated on test. No thresholds, gates, or method choices may use test targets.

## Command

```bash
python experiments/halluguard/run_stage5_sra_bp.py \
  --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv \
  --stage3-dir experiments/halluguard/results/lrbn_bp_stage3 \
  --stage4-dir experiments/halluguard/results/lrbn_bp_stage4 \
  --stage45-dir experiments/halluguard/results/lrbn_bp_attribution_stage45 \
  --output-dir experiments/halluguard/results/lrbn_sra_bp_stage5 \
  --n-bootstrap 2000
```

## Success Meaning

- Safe-SRA pass: can be a safe sparse boundary enhancement candidate.
- Balanced-SRA pass: can be a risk-performance tradeoff candidate.
- Both fail: keep BP as attribution/performance ablation and return to method design.

## Revision Log

- 2026-06-27: Created from user-provided Stage 5 SRA-BP validation document.
