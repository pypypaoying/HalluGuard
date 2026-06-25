# HalluGuard-LRBN Expanded Big Table Results

- Source CSV: latest expanded BigTable `combined_metrics.csv` imported into this directory
- Rows: `4620`
- Matrix: `7 datasets x 5 backbones x 4 horizons x 3 seeds x 11 methods`
- Completed: `4620 / 4620`
- Test threshold leakage: `False` for all rows

## Method Summary

| Method | Mean MSE Delta vs Raw | Median Delta | Improved / 420 | Max Harm | Best Gain |
|---|---:|---:|---:|---:|---:|
| HalluGuard-LRBN | -3.7952% | -0.2853% | 237 / 420 | 12.8114% | -79.2960% |
| RevIN | -3.6921% | -0.3383% | 238 / 420 | 21.9336% | -78.7599% |
| NST | -3.6155% | -0.2863% | 232 / 420 | 18.3111% | -78.7932% |
| DishTS | -2.4163% | 0.3261% | 184 / 420 | 66.6358% | -74.6821% |
| SAN | 22.5883% | 2.9232% | 158 / 420 | 645.3047% | -52.2611% |
| TAFAS | -0.0210% | -0.2337% | 233 / 420 | 19.3640% | -16.2856% |
| matched_sparse_smoothing | 0.5784% | -0.0486% | 268 / 420 | 8.6217% | -2.0670% |
| naive_smoothing | 0.5509% | -0.0942% | 266 / 420 | 8.6217% | -2.0670% |
| median_smoothing | -0.0041% | -0.0810% | 288 / 420 | 3.0411% | -1.4851% |
| ema_smoothing | 17.8904% | 0.7775% | 144 / 420 | 141.9579% | -2.5699% |
| raw_no_correction | 0.0000% | 0.0000% | 0 / 420 | 0.0000% | 0.0000% |

## LRBN Interpretation

- HalluGuard-LRBN macro mean MSE delta is `-3.7952%` over raw across 420 configs.
- LRBN is competitive with RevIN (`-3.6921%`) and NST (`-3.6155%`), but the margin is small.
- Strongest LRBN evidence is DLinear-like normalization benefit; deep backbones are near-neutral.
- ECL remains the main failure dataset; reports should avoid claiming universal improvement.
- Simple smoothing controls are not enough to explain LRBN gains.

## Files

- `combined_metrics.csv`: full 4620-row table
- `summary_by_method.csv`: method-level aggregate
- `summary_by_dataset.csv`: dataset x method aggregate
- `summary_by_backbone.csv`: backbone x method aggregate
- `lrbn_pairwise_wins.csv`: pairwise LRBN comparisons
