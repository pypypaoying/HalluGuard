# Expanded BigTable Results

Latest published result directory:

```text
results/lrbn_clean_claim_bigtable_v2_expanded_latest/
```

The uploaded table was produced from the latest server result CSV supplied by
the project owner and summarized into repository-native artifacts:

- `combined_metrics.csv`
- `combined_metrics_summary.json`
- `summary_by_method.csv`
- `summary_by_dataset.csv`
- `summary_by_backbone.csv`
- `lrbn_pairwise_wins.csv`
- `best_method_counts.csv`
- `summary.md`

## Matrix

- 7 datasets
- 5 forecasting backbones
- 4 horizons
- 3 seeds
- 11 methods
- 4620 total rows
- 4620 completed rows
- 0 blocked rows
- `test_threshold_leakage=False` for all rows

## Method-Level Snapshot

| Method | Mean MSE Delta vs Raw | Median Delta | Improved / 420 | Max Harm |
|---|---:|---:|---:|---:|
| HalluGuard-LRBN | -3.7952% | -0.2853% | 237 / 420 | 12.8114% |
| RevIN | -3.6921% | -0.3383% | 238 / 420 | 21.9336% |
| NST | -3.6155% | -0.2863% | 232 / 420 | 18.3111% |
| DishTS | -2.4163% | 0.3261% | 184 / 420 | 66.6358% |
| SAN | 22.5883% | 2.9232% | 158 / 420 | 645.3047% |
| TAFAS | -0.0210% | -0.2337% | 233 / 420 | 19.3640% |
| median_smoothing | -0.0041% | -0.0810% | 288 / 420 | 3.0411% |
| naive_smoothing | 0.5509% | -0.0942% | 266 / 420 | 8.6217% |
| matched_sparse_smoothing | 0.5784% | -0.0486% | 268 / 420 | 8.6217% |
| ema_smoothing | 17.8904% | 0.7775% | 144 / 420 | 141.9579% |
| raw_no_correction | 0.0000% | 0.0000% | 0 / 420 | 0.0000% |

## Interpretation

The current clean claim should be conservative:

- HalluGuard-LRBN is competitive with RevIN and NST on the expanded table.
- The average gain is not explained by simple smoothing controls.
- LRBN is not a universal winner. ECL remains a major weak region, and max harm
  should be reported rather than hidden.
- SAN and EMA smoothing show large instability in this unified setting, so they
  are useful stress baselines but not reliable reference points for a main claim.
