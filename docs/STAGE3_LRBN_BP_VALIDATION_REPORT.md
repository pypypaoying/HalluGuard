# Stage 3 Report: LRBN + Optional Boundary Projection

## 结论

`HalluGuard-LRBN-BP-gated` 通过 compact sanity check 的 **strong pass**，可以进入完整主表 TableA 作为候选主方法行。它不是替代 LRBN 的新主线，而是在 frozen `HalluGuard-LRBN unified_revin_rdn_hybrid` 后面加一个 validation-only 的稀疏边界投影增强。

相较 `HalluGuard-LRBN`：

- Test MSE: `4.894158 -> 4.864131`
- MSE 绝对变化: `-0.030027`
- MSE 相对提升: `-0.613520%`
- Test MAE: `1.682162 -> 1.674353`
- MAE 绝对变化: `-0.007809`
- MAE 相对提升: `-0.464217%`
- Bootstrap 95% CI for MSE delta: `[-0.061527, -0.005077]`
- Bootstrap improvement probability: `0.993`

因此，进入 TableA 的推荐方法名是：

```text
HalluGuard-LRBN-BP-gated
```

同时保留：

```text
HalluGuard-LRBN
HalluGuard-LRBN-BP-always
HalluGuard-BP-global
```

作为必要 ablation / mechanism rows。

## 实验设置

- 输入: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- 输出: `experiments/halluguard/results/lrbn_bp_stage3/`
- Samples: 1536 total, 768 test
- Configs: 8 compact configs
- Dataset: `ETTm1`, `ETTh1`
- Backbone: `DLinear`, `PatchTST`
- Horizon: `96`, `192`
- Seed: `2026`
- Calibration: validation split only
- Evaluation: test split only
- Test threshold leakage: `False`

运行命令：

```bash
python experiments/halluguard/run_lrbn_bp_validation.py --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv --output-dir experiments/halluguard/results/lrbn_bp_stage3 --tail 24 --seed 2026 --n-bootstrap 2000
```

## Validation 选择结果

`HalluGuard-LRBN-BP-gated` 的 validation-only 参数：

```json
{
  "alpha": 0.5,
  "tau": 13.766675883152962,
  "coverage": 0.05078125,
  "delta_pct_vs_lrbn": -1.1650865777946104,
  "harm_rate_vs_lrbn": 0.01171875,
  "low_delta_pct_vs_lrbn": 0.0,
  "q4_improvement_pct_vs_lrbn": 3.6652999634598156
}
```

解释：validation 选择的是一个很稀疏的 high-boundary-gap gate，覆盖率约 `5.08%`，而不是全局平滑或全局 BP。

## Test Overall

| Method | Mean MSE | Mean MAE | MSE delta % vs LRBN | Coverage | Harm vs LRBN |
|---|---:|---:|---:|---:|---:|
| HalluGuard-LRBN-BP-always | 4.643981 | 1.621852 | -5.111746 | 1.000000 | 0.423177 |
| HalluGuard-LRBN-BP-gated | 4.864131 | 1.674353 | -0.613520 | 0.042969 | 0.018229 |
| HalluGuard-LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 |
| HalluGuard-BP-global | 5.985373 | 1.843170 | 22.296291 | 1.000000 | 0.569010 |
| ema_smoothing | 6.060487 | 1.848221 | 23.831056 | 0.000000 | 0.438802 |
| naive_smoothing | 6.072069 | 1.849256 | 24.067707 | 0.000000 | 0.430990 |
| median_smoothing | 6.129929 | 1.860337 | 25.249936 | 0.000000 | 0.449219 |
| matched_sparse_smoothing | 6.222475 | 1.877730 | 27.140880 | 0.000000 | 0.584635 |
| raw_no_correction | 6.427221 | 1.914908 | 31.324348 | 0.000000 | 0.641927 |

`BP-always` 的 MSE 更低，但 harm rate `42.3%`，不满足安全直觉，不应作为主方法 claim。它可以作为 “boundary projection has strong correction power but needs gating” 的 ablation。

`HalluGuard-BP-global` 在 raw 上仍有收益，但相对 LRBN 明显更差，说明 TableA 主方法必须是 LRBN 后增强，而不是 BP 替代 LRBN。

## Boundary-Gap Slice

`HalluGuard-LRBN-BP-gated` 只触发 high boundary-gap 区域：

| Boundary bin | LRBN MSE | LRBN-BP MSE | Delta % vs LRBN | Coverage | Harm |
|---|---:|---:|---:|---:|---:|
| q1_low | 5.480761 | 5.480761 | 0.000000 | 0.000000 | 0.000000 |
| q2 | 4.559878 | 4.559878 | 0.000000 | 0.000000 | 0.000000 |
| q3 | 5.136111 | 5.136111 | 0.000000 | 0.000000 | 0.000000 |
| q4_high | 4.399880 | 4.279774 | -2.729767 | 0.171875 | 0.072917 |

这正好验证了文档里的机制假设：LRBN 后仍残留的高 normalized boundary gap 样本，是 BP 额外收益的主要来源；低 gap 样本没有被改动，因此没有低 gap 误伤。

## Per-Config Stability

`HalluGuard-LRBN-BP-gated` 在 8 个 compact configs 中有 6 个 config MSE 不劣于 LRBN，比例 `0.75`，超过 strong-pass 要求的 `0.60`。

主要观察：

- DLinear 平均 delta pct: about `-0.7640%`
- PatchTST 平均 delta pct: about `-0.8493%`
- ETTh1 平均 delta pct: about `-0.4634%`
- ETTm1 平均 delta pct: about `-1.1499%`
- 两个轻微负配置：
  - `ETTh1 / DLinear / 192`: `+0.1635%`
  - `ETTm1 / DLinear / 192`: `+0.1383%`

这说明收益不是只发生在 PatchTST，也不是只发生在 DLinear；但 DLinear-192 有轻微 harm，需要在 TableA 里重点看。

## Pass Gate

| Gate | Required | Observed | Pass |
|---|---:|---:|---:|
| Overall MSE improvement vs LRBN | >= 0.5% | 0.613520% | yes |
| q4 high-boundary improvement | >= 2.0% | 2.729767% | yes |
| Harm rate vs LRBN | <= 2 pp | 1.822917 pp | yes |
| q1/q2 low-gap degradation | <= 0.5% | 0.000000% | yes |
| Improved config ratio | >= 0.60 | 0.75 | yes |
| Test threshold leakage | False | False | yes |

Verdict:

```json
{
  "status": "strong_pass",
  "decision": "enter_full_table",
  "overall_delta_pct_vs_lrbn": -0.6135197640016364,
  "q4_improvement_pct_vs_lrbn": 2.729767308058518,
  "harm_extra_pp_vs_lrbn": 0.018229166666666668,
  "configs_improved_ratio": 0.75,
  "test_threshold_leakage": false
}
```

## 是否进入 TableA

可以进入 TableA，但 claim 要写窄：

> HalluGuard-LRBN-BP-gated is a validation-calibrated sparse boundary repair layer on top of HalluGuard-LRBN. It improves compact real-forecast MSE by 0.61% over LRBN, with gains concentrated in high boundary-gap samples and no low-gap intervention.

不要 claim：

- 不要说 BP 替代 LRBN；
- 不要说它是 smoothing；
- 不要把 `BP-always` 当主方法，尽管它 MSE 更低；
- 不要跳过完整 TableA 再声称 paper-level 稳定提升。

TableA 推荐包含：

- `raw_no_correction`
- `HalluGuard-LRBN`
- `HalluGuard-LRBN-BP-gated`
- `HalluGuard-LRBN-BP-always` as ablation
- `HalluGuard-BP-global` as mechanism ablation
- smoothing controls
- RevIN / Dish-TS / SAN / NST / TAFAS / SoP / SOLID 等 test-time baselines

