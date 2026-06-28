# Stage 4A BP-Always Harm Attribution

- Test samples: 768
- BP-always mean delta vs LRBN: -0.250177
- BP-always harm rate vs LRBN: 0.423177

## Boundary Gap Bins

| feature | bin | n | mean_feature | mean_delta_mse_vs_lrbn | harm_rate_vs_lrbn | mean_win_size | mean_loss_size | win_loss_ratio | true_boundary_jump |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| post_lrbn_gap | (0.012799999999999999, 1.132] | 192 | 0.600383 | 0.009167 | 0.520833 | 0.150342 | 0.155915 | 0.964255 | 0.907195 |
| post_lrbn_gap | (1.132, 2.406] | 192 | 1.779171 | -0.190302 | 0.427083 | 0.575387 | 0.326276 | 1.763496 | 0.904580 |
| post_lrbn_gap | (2.406, 4.512] | 192 | 3.391069 | -0.167838 | 0.411458 | 0.678887 | 0.563157 | 1.205502 | 0.788991 |
| post_lrbn_gap | (4.512, 1013864.0] | 192 | 23988.287049 | -0.651735 | 0.333333 | 1.343811 | 0.732416 | 1.834765 | 0.932618 |

## Conflict Cosine Bins

| feature | bin | n | mean_feature | mean_delta_mse_vs_lrbn | harm_rate_vs_lrbn | mean_win_size | mean_loss_size | win_loss_ratio | true_boundary_jump |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conflict_cosine | (-inf, -0.2] | 200 | -0.555613 | -0.103760 | 0.425000 | 0.491865 | 0.421322 | 1.167432 | 0.938765 |
| conflict_cosine | (-0.2, 0.2] | 381 | 0.008547 | -0.233604 | 0.427822 | 0.749201 | 0.455967 | 1.643102 | 0.876813 |
| conflict_cosine | (0.2, inf] | 187 | 0.604362 | -0.440538 | 0.411765 | 0.963235 | 0.306170 | 3.146075 | 0.837385 |

## Norm Ratio Bins

| feature | bin | n | mean_feature | mean_delta_mse_vs_lrbn | harm_rate_vs_lrbn | mean_win_size | mean_loss_size | win_loss_ratio | true_boundary_jump |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| norm_ratio | (-inf, 0.25] | 410 | 0.115637 | -0.051215 | 0.480488 | 0.318193 | 0.237446 | 1.340065 | 0.942568 |
| norm_ratio | (0.25, 0.5] | 188 | 0.360187 | -0.506160 | 0.319149 | 0.962911 | 0.468241 | 2.056442 | 0.804232 |
| norm_ratio | (0.5, 1.0] | 140 | 0.683790 | -0.401640 | 0.421429 | 1.319836 | 0.858935 | 1.536596 | 0.803659 |
| norm_ratio | (1.0, inf] | 30 | 1.206822 | -0.658328 | 0.300000 | 1.329117 | 0.906847 | 1.465646 | 0.941638 |

## Horizon Segments

| segment | rows | mean_delta_mse_vs_lrbn | harm_rate | lrbn_mse | method_mse | delta_pct_vs_lrbn |
| --- | --- | --- | --- | --- | --- | --- |
| early | 768 | -0.628858 | 0.363281 | 3.195419 | 2.566561 | -19.679989 |
| late | 768 | -0.047414 | 0.446615 | 6.345978 | 6.298563 | -0.747157 |
| mid | 768 | -0.162217 | 0.464844 | 5.017617 | 4.855399 | -3.232959 |
