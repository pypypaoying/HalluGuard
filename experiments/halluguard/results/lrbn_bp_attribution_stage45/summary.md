# Stage 4.5 BP-Always Failure Attribution

## Setup

- Input metrics: `experiments\halluguard\results\research_direction_validation\forecast_inputs\combined_metrics.csv`
- Output directory: `experiments\halluguard\results\lrbn_bp_attribution_stage45`
- Compact configs: `8` test configs
- Validation samples: `768`
- Test samples: `768`
- Test threshold leakage: `False`

## Headline

- BP-always MSE delta vs LRBN: `-5.111746%`; harm rate `0.423177`.
- Stage3 gated MSE delta vs LRBN: `-0.613520%`; harm rate `0.018229`.
- Repair-gate MSE delta vs LRBN: `-1.747470%`; harm rate `0.218750`.

## Verdict

- BP-always mechanism defect: `yes` (6/7 conditions).
- Sparse repair-aware BP support: `yes` (3/4 conditions).
- Recommendation: Use LRBN + sparse repair-aware boundary expert as the next BP line; keep BP-always as performance attribution only.

## Oracle Boundary Truth

| bin | count | mean_delta_vs_lrbn | harm_rate | mean_g_L | mean_g_y | true_jump_rate | bp_needed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low_gL_low_gY | 138 | -0.006267 | 0.478261 | 0.519837 | 0.534028 | 0.000000 | 0.521739 |
| high_gL_low_gY | 143 | -0.719912 | 0.307692 | 32205.315619 | 0.497982 | 0.000000 | 0.692308 |
| high_gL_high_gY | 51 | -0.433022 | 0.411765 | 7.842809 | 2.117936 | 1.000000 | 0.588235 |
| low_gL_high_gY | 35 | 0.011579 | 0.628571 | 0.666348 | 2.292293 | 1.000000 | 0.371429 |
| mid_or_mixed | 401 | -0.166196 | 0.428928 | 2.503483 | 0.860991 | 0.229426 | 0.571072 |

## Gap x Repair Interaction

| gap_group | repair_group | count | mean_delta_vs_lrbn | harm_rate | mean_A | A_gt_1_rate | win_loss_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low | low | 48 | 0.002880 | 0.500000 | 0.369912 | 0.500000 | 0.967212 |
| low | mid | 35 | -0.048399 | 0.542857 | 2.062825 | 0.457143 | 1.953445 |
| low | high | 90 | 0.012179 | 0.500000 | 66.708309 | 0.500000 | 0.762873 |
| mid | low | 258 | -0.119392 | 0.445736 | 1.322089 | 0.554264 | 1.214178 |
| mid | mid | 119 | -0.294199 | 0.378151 | 4.178195 | 0.621849 | 2.338051 |
| mid | high | 24 | -0.034659 | 0.500000 | 5.836665 | 0.500000 | 1.489681 |
| high | low | 163 | -0.627193 | 0.361963 | 2.458947 | 0.638037 | 1.902409 |
| high | mid | 29 | -0.786150 | 0.172414 | 2.865710 | 0.827586 | 1.293273 |
| high | high | 2 | -0.000372 | 0.500000 | -332.074062 | 0.500000 | 13.552819 |

## Win/Loss Distribution

| scope | count | win_rate | harm_rate | mean_win | median_win | mean_loss | median_loss | p95_loss | top1_gain_share | top5_gain_share | gini_positive_gain | total_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 768 | 0.576823 | 0.423177 | 0.735544 | 0.363234 | 0.411416 | 0.207571 | 1.385717 | 0.089289 | 0.292746 | 0.615972 | 192.135849 |
| low_gap | 173 | 0.491329 | 0.508671 | 0.140288 | 0.070988 | 0.130283 | 0.095039 | 0.335858 | 0.088595 | 0.295533 | 0.610091 | 0.459580 |
| high_gap | 194 | 0.664948 | 0.335052 | 1.339920 | 0.904285 | 0.735664 | 0.522995 | 2.528410 | 0.077045 | 0.214613 | 0.520642 | 125.031481 |
| high_repair | 116 | 0.500000 | 0.500000 | 0.104445 | 0.053557 | 0.108989 | 0.058317 | 0.325392 | 0.112568 | 0.283182 | 0.585228 | -0.263569 |
| low_gap_high_repair | 90 | 0.500000 | 0.500000 | 0.078366 | 0.042745 | 0.102724 | 0.059830 | 0.223795 | 0.162497 | 0.360633 | 0.565867 | -1.096140 |
| high_gap_low_repair | 163 | 0.638037 | 0.361963 | 1.400699 | 0.950571 | 0.736277 | 0.525215 | 2.201798 | 0.091419 | 0.223918 | 0.511987 | 102.232387 |

## Horizon Segment Attribution

| segment | count | mse_delta_pct_vs_lrbn | harm_rate | mean_A | A_gt_1_rate | win_loss_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| early | 768 | -19.679989 | 0.363281 | 7.480313 | 0.636719 | 2.152705 |
| mid | 768 | -3.232959 | 0.464844 | 10.375152 | 0.535156 | 1.372042 |
| late | 768 | -0.747157 | 0.446615 | 29.175965 | 0.553385 | 1.235780 |

## Anchor Reliability

| anchor | mean_oracle_anchor_error | mean_pred_anchor_gap | mean_delta_vs_lrbn | delta_pct_vs_lrbn | harm_rate | mean_A | A_gt_1_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| last | 0.883346 | 5998.514418 | -0.250177 | -5.111746 | 0.423177 | 8.973766 | 0.576823 |
| trend | 0.907441 | 5998.625651 | -0.251669 | -5.142234 | 0.424479 | -1.863833 | 0.575521 |
| robust | 0.964556 | 5998.403640 | -0.242214 | -4.949051 | 0.434896 | -27.480158 | 0.565104 |

## Per-Config Attribution (first rows)

| dataset | backbone | horizon | seed | method | count | mse_delta_pct_vs_lrbn | harm_rate | q4_gain | low_gap_harm | high_repair_harm | mean_A | A_gt_1_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | DLinear | 96 | 2026 | bp | 96 | -7.050909 | 0.458333 | 19.224495 | 0.500000 | 0.666667 | 17.576732 | 0.541667 |
| ETTh1 | DLinear | 192 | 2026 | bp | 96 | -4.001996 | 0.416667 | 10.561850 | 0.500000 | 0.485714 | 6.752899 | 0.583333 |
| ETTh1 | PatchTST | 96 | 2026 | bp | 96 | -5.065733 | 0.406250 | 9.907570 | 0.392857 | 0.375000 | 8.398263 | 0.593750 |
| ETTh1 | PatchTST | 192 | 2026 | bp | 96 | -3.178037 | 0.427083 | 11.342846 | 0.466667 | 0.400000 | -2.582606 | 0.572917 |
| ETTm1 | DLinear | 96 | 2026 | bp | 96 | -11.340649 | 0.354167 | 23.762084 | 0.555556 | 0.533333 | -2.777765 | 0.645833 |
| ETTm1 | DLinear | 192 | 2026 | bp | 96 | -3.288373 | 0.427083 | 10.713728 | 0.500000 | 0.600000 | 59.675765 | 0.572917 |
| ETTm1 | PatchTST | 96 | 2026 | bp | 96 | -9.236916 | 0.416667 | 21.677203 | 0.615385 | 0.428571 | -6.483121 | 0.583333 |
| ETTm1 | PatchTST | 192 | 2026 | bp | 96 | -2.459627 | 0.479167 | 6.797061 | 0.687500 | 0.750000 | -8.770038 | 0.520833 |
| ETTh1 | DLinear | 96 | 2026 | stage3 | 96 | -1.582620 | 0.000000 | 7.023419 | 0.000000 | 0.000000 | 0.134653 | 0.020833 |
| ETTh1 | DLinear | 192 | 2026 | stage3 | 96 | 0.163490 | 0.010417 | -3.105135 | 0.000000 | 0.000000 | 0.162734 | 0.010417 |
| ETTh1 | PatchTST | 96 | 2026 | stage3 | 96 | -0.218705 | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 0.046174 | 0.010417 |
| ETTh1 | PatchTST | 192 | 2026 | stage3 | 96 | -0.215954 | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 0.242429 | 0.010417 |

## Output Files

- `attribution_config.json`
- `attribution_sample_table.csv` and optional `attribution_sample_table.parquet`
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