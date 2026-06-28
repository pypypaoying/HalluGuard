# Stage 5 SRA-BP Validation Summary

## Setup

- Input metrics: `experiments\halluguard\results\research_direction_validation\forecast_inputs\combined_metrics.csv`
- Output directory: `experiments\halluguard\results\lrbn_sra_bp_stage5`
- Validation samples: `768`
- Test samples: `768`
- Test configs: `8`
- Test threshold leakage: `False`

## Verdict

- Status: `safe_and_balanced_pass`
- Decision: `promote_sra_bp_to_mini_extension`
- Safe-SRA pass: `True`
- Balanced-SRA pass: `True`

## Test Overall

| method | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | coverage | q4_improvement_pct | low_gap_high_repair_delta_pct | high_gap_low_repair_delta_pct | config_improved_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN-BP-always | 4.643981 | 1.621852 | -5.111746 | 0.423177 | 1.000000 | 13.923609 | 0.197510 | -12.864612 | 1.000000 |
| LRBN-SRA-BP-basic | 4.722565 | 1.640118 | -3.506063 | 0.113281 | 0.324219 | 13.923527 | 0.000000 | -12.864612 | 1.000000 |
| LRBN-SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.436198 | 8.587733 | 0.000000 | -7.917318 | 1.000000 |
| LRBN-SRA-BP-support | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.436198 | 8.587733 | 0.000000 | -7.917318 | 1.000000 |
| LRBN-SRA-BP-short | 4.773001 | 1.648444 | -2.475540 | 0.105469 | 0.428385 | 7.977599 | 0.000000 | -7.874603 | 1.000000 |
| LRBN-BP-repair-gate | 4.808634 | 1.661626 | -1.747470 | 0.218750 | 0.610677 | 5.275949 | 0.000000 | -5.961786 | 1.000000 |
| LRBN-SRA-BP-safe | 4.813149 | 1.660950 | -1.655217 | 0.035156 | 0.187500 | 6.928297 | 0.000000 | -6.197214 | 1.000000 |
| LRBN-SRA-BP-continuous | 4.859156 | 1.672857 | -0.715180 | 0.098958 | 0.360677 | 2.993552 | -0.000000 | -2.810568 | 0.875000 |
| LRBN-BP-stage3-gated | 4.864131 | 1.674353 | -0.613520 | 0.018229 | 0.042969 | 1.988351 | 0.000000 | -2.294853 | 0.750000 |
| LRBN-BP-short-bridge | 4.864551 | 1.673294 | -0.604945 | 0.173177 | 1.000000 | 1.678003 | -0.004229 | -1.649208 | 1.000000 |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 1.000000 |
| ema_smoothing | 6.060487 | 1.848221 | 23.831056 | 0.438802 | 0.000000 | -26.813839 | 30.911031 | 21.243180 | 0.500000 |
| naive_smoothing | 6.072069 | 1.849256 | 24.067707 | 0.430990 | 0.000000 | -27.326156 | 31.023079 | 21.979716 | 0.500000 |
| median_smoothing | 6.129929 | 1.860337 | 25.249936 | 0.449219 | 0.000000 | -28.248810 | 32.144491 | 22.657339 | 0.500000 |
| matched_sparse_smoothing | 6.222475 | 1.877730 | 27.140880 | 0.584635 | 0.000000 | -31.598240 | 33.058032 | 26.185858 | 0.375000 |
| raw_no_correction | 6.427221 | 1.914908 | 31.324348 | 0.641927 | 0.000000 | -35.038420 | 37.442939 | 28.979945 | 0.000000 |

## Gap x Repair Slices

| method | slice_type | slice_name | count | coverage | mse_delta_pct_vs_lrbn | mean_delta | harm_rate | win_rate | win_loss_ratio | A_gt_1_rate | mean_A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN-SRA-BP-safe | gap_repair | low_gap__low_repair | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-safe | gap_repair | low_gap__mid_repair | 35 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-safe | gap_repair | low_gap__high_repair | 90 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-safe | gap_repair | mid_gap__low_repair | 258 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-safe | gap_repair | mid_gap__mid_repair | 119 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-safe | gap_repair | mid_gap__high_repair | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-safe | gap_repair | high_gap__low_repair | 163 | 0.748466 | -6.197214 | -0.302135 | 0.159509 | 0.588957 | 2.314117 | 0.588957 | 1.996953 |
| LRBN-SRA-BP-safe | gap_repair | high_gap__mid_repair | 29 | 0.758621 | -12.635077 | -0.447134 | 0.034483 | 0.724138 | 1.985915 | 0.724138 | 2.659502 |
| LRBN-SRA-BP-safe | gap_repair | high_gap__high_repair | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-balanced | gap_repair | low_gap__low_repair | 48 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-balanced | gap_repair | low_gap__mid_repair | 35 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-balanced | gap_repair | low_gap__high_repair | 90 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-balanced | gap_repair | mid_gap__low_repair | 258 | 0.453488 | -1.276181 | -0.058421 | 0.147287 | 0.306202 | 1.462606 | 0.306202 | 0.921046 |
| LRBN-SRA-BP-balanced | gap_repair | mid_gap__mid_repair | 119 | 0.285714 | -0.945675 | -0.047236 | 0.075630 | 0.210084 | 2.964871 | 0.210084 | 0.907334 |
| LRBN-SRA-BP-balanced | gap_repair | mid_gap__high_repair | 24 | 0.083333 | 0.137916 | 0.005816 | 0.041667 | 0.041667 | 0.332862 | 0.041667 | 0.153410 |
| LRBN-SRA-BP-balanced | gap_repair | high_gap__low_repair | 163 | 0.944785 | -7.917318 | -0.385996 | 0.177914 | 0.766871 | 2.010605 | 0.766871 | 2.525260 |
| LRBN-SRA-BP-balanced | gap_repair | high_gap__mid_repair | 29 | 0.965517 | -13.835671 | -0.489621 | 0.103448 | 0.862069 | 2.591365 | 0.862069 | 3.124843 |
| LRBN-SRA-BP-balanced | gap_repair | high_gap__high_repair | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Selected Alignment

| method | selection_slice | count | A_gt_1_rate | mean_A | mean_alignment_cosine | harm_rate | mean_win_size | mean_loss_size | true_jump_rate | high_gap_low_repair_rate | low_gap_high_repair_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN-SRA-BP-safe | selected | 144 | 0.812500 | 2.796034 | 0.335685 | 0.187500 | 0.590266 | 0.253566 | 0.250000 | 0.847222 | 0.000000 |
| LRBN-SRA-BP-safe | unselected | 624 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.227564 | 0.065705 | 0.144231 |
| LRBN-SRA-BP-safe | overall | 768 | 0.152344 | 0.524256 | 0.062941 | 0.035156 | 0.590266 | 0.253566 | 0.231771 | 0.212240 | 0.117188 |
| LRBN-SRA-BP-balanced | selected | 335 | 0.761194 | 2.541858 | 0.279069 | 0.238806 | 0.450741 | 0.215857 | 0.223881 | 0.459701 | 0.000000 |
| LRBN-SRA-BP-balanced | unselected | 433 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.237875 | 0.020785 | 0.207852 |
| LRBN-SRA-BP-balanced | overall | 768 | 0.332031 | 1.108753 | 0.121729 | 0.104167 | 0.450741 | 0.215857 | 0.231771 | 0.212240 | 0.117188 |

## Horizon Segments

| method | segment | count | coverage | mse_delta_pct_vs_lrbn | mean_delta | harm_rate | win_loss_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN-SRA-BP-balanced | early | 768 | 0.436198 | -15.919673 | -0.508700 | 0.104167 | 2.088152 |
| LRBN-SRA-BP-balanced | late | 768 | 0.436198 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-balanced | mid | 768 | 0.436198 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-safe | early | 768 | 0.187500 | -10.140630 | -0.324036 | 0.035156 | 2.327858 |
| LRBN-SRA-BP-safe | late | 768 | 0.187500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LRBN-SRA-BP-safe | mid | 768 | 0.187500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

Current SRA-BP has enough compact evidence to enter mini-extension, but not full TableA.

## Output Files

- `stage5_config.json`
- `stage5_selected_safe_params.json`
- `stage5_selected_balanced_params.json`
- `stage5_calibration_grid.csv`
- `stage5_overall.csv`
- `stage5_boundary_gap_slices.csv`
- `stage5_gap_repair_interaction.csv`
- `stage5_oracle_boundary_truth_slices.csv`
- `stage5_horizon_segments.csv`
- `stage5_selected_alignment.csv`
- `stage5_per_config.csv`
- `stage5_bootstrap_ci.json`
- `stage5_failure_cases_topk.csv`
- `stage5_verdict.json`