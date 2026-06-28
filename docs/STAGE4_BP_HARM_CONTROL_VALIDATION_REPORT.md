# Stage 4 Report: BP Harm Attribution and Safe Controller

## 结论

Stage 4 完成了 `LRBN-BP-always` harm 归因、单机制控制 ablation、以及组合型 `LRBN-BP-safe-controller` compact 验证。结果是：

```text
status = perf_only
decision = keep_lrbn_main_report_bp_perf_ablation
```

也就是说，BP 的 performance branch 很强，但本轮 safe-controller 没有达到进入主表 TableA 的安全主方法门槛。当前 TableA 推荐仍应保留：

- `HalluGuard-LRBN` as clean main line
- `HalluGuard-LRBN-BP-gated` from Stage 3 as safe sparse boundary enhancement candidate
- `LRBN-BP-always` as performance ablation / upper correction-power variant
- Stage 4 safe-controller 暂不替换 Stage 3 gated

## 实验设置

- Input: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- Output: `experiments/halluguard/results/lrbn_bp_stage4/`
- Samples: 1536 total, 768 test
- Configs: 8 compact configs
- Dataset: `ETTm1`, `ETTh1`
- Backbone: `DLinear`, `PatchTST`
- Horizon: `96`, `192`
- Seed: `2026`
- Calibration: validation-only
- Evaluation: test-only
- Test threshold leakage: `False`

Command:

```bash
python experiments/halluguard/run_stage4_bp_harm_control.py --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv --stage3-dir experiments/halluguard/results/lrbn_bp_stage3 --output-dir experiments/halluguard/results/lrbn_bp_stage4 --seed 2026 --n-bootstrap 2000
```

## Stage 4A: Harm Attribution

`LRBN-BP-always` remains a strong performance variant:

- Mean MSE delta vs LRBN: `-0.250177`
- Relative MSE delta vs LRBN: `-5.111746%`
- Harm rate vs LRBN: `0.423177`

### H1: Boundary gap is not always boundary error

Supported. Low post-LRBN boundary-gap samples are the harmful region:

| post-LRBN gap bin | mean delta vs LRBN | harm rate | win/loss ratio |
|---|---:|---:|---:|
| q1 low | `+0.009167` | `0.520833` | `0.964255` |
| q2 | `-0.190302` | `0.427083` | `1.763496` |
| q3 | `-0.167838` | `0.411458` | `1.205502` |
| q4 high | `-0.651735` | `0.333333` | `1.834765` |

Interpretation: BP-always should not operate on low boundary-gap samples. This supports the Stage 3 sparse gate.

### H2: BP may repeat/undo LRBN repair

Partially supported. The high repair-ratio bin is clearly bad:

| repair-ratio bin | mean delta vs LRBN | harm rate | win/loss ratio |
|---|---:|---:|---:|
| `>0.7` | `+0.002272` | `0.500000` | `0.958305` |

This supports a repair-gate mechanism. Conflict cosine is less decisive: positive conflict cosine is best, but negative conflict is not uniquely harmful.

### H3: Last anchor can be unreliable

Partially supported. High anchor-disagreement has weak win/loss ratio:

- highest anchor-disagreement bin win/loss ratio: `0.970670`
- mean delta remains negative but much weaker than middle bins

Robust-anchor improves MSE vs LRBN by `-1.666865%`, but harm remains high at `0.364583`, so anchor replacement alone is not safe.

### H4: Full linear decay affects too much of the horizon

Supported in mechanism but not enough as a full solution. BP-always gains are concentrated early:

| segment | delta % vs LRBN | harm rate |
|---|---:|---:|
| early | `-19.679989%` | `0.363281` |
| mid | `-3.232959%` | `0.464844` |
| late | `-0.747157%` | `0.446615` |

Short bridge lowers harm to `0.173177`, but only improves MSE by `-0.604945%` and q4 improvement is `1.695024%`, below the safe-controller q4 gate.

### H5: Norm ratio / magnitude is not the main harm source

Not strongly supported. High norm-ratio bins still have negative mean deltas; norm clipping helps but does not solve harm. Bounded BP improves MSE by `-1.797710%`, but harm remains `0.365885`.

## Stage 4B/C: Mechanism Ablation Results

Test results vs LRBN:

| Method | MSE | MAE | MSE delta % vs LRBN | MAE delta % vs LRBN | Harm | Coverage | q4 improvement | Config improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LRBN-BP-always | 4.643981 | 1.621852 | -5.111746 | -3.585289 | 0.423177 | 1.000000 | 14.164335 | 1.000000 |
| LRBN-BP-gap-strength | 4.782061 | 1.657199 | -2.290412 | -1.483990 | 0.363281 | 0.873698 | 9.713605 | 1.000000 |
| LRBN-BP-bounded | 4.806175 | 1.662432 | -1.797710 | -1.172892 | 0.365885 | 1.000000 | 3.281651 | 1.000000 |
| LRBN-BP-repair-gate | 4.808634 | 1.661626 | -1.747470 | -1.220830 | 0.218750 | 0.610677 | 5.396815 | 1.000000 |
| LRBN-BP-robust-anchor | 4.812579 | 1.663763 | -1.666865 | -1.093773 | 0.364583 | 1.000000 | 4.477076 | 1.000000 |
| LRBN-BP-conflict-filter | 4.822251 | 1.666692 | -1.469239 | -0.919665 | 0.360677 | 1.000000 | 4.292585 | 1.000000 |
| LRBN-BP-stage3-gated | 4.864131 | 1.674353 | -0.613520 | -0.464217 | 0.018229 | 0.042969 | 2.021084 | 0.750000 |
| LRBN-BP-short-bridge | 4.864551 | 1.673294 | -0.604945 | -0.527213 | 0.173177 | 1.000000 | 1.695024 | 1.000000 |
| LRBN-BP-safe-controller | 4.884224 | 1.679675 | -0.202964 | -0.147893 | 0.092448 | 0.725260 | 0.607284 | 1.000000 |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |

## Safe-Controller Gate

Safe-controller failed the Stage 4 safe gate:

| Gate | Required | Observed | Pass |
|---|---:|---:|---:|
| MSE improvement vs LRBN | >= 1.0% | `0.202964%` | no |
| Harm control | <= 2 pp or >=30% reduction vs BP-always | `78.15%` reduction | yes |
| q4 high-gap improvement | >= 2.0% | `0.607284%` | no |
| low-gap degradation | <= 0.5% | `-0.000199%` | yes |
| config improved ratio | >= 0.60 | `1.0` | yes |
| bootstrap significant degradation | no | CI entirely negative | yes |

The safe-controller successfully suppresses harm relative to BP-always, but it suppresses the useful BP mechanism too much. It is safer than many single-factor variants, but weaker than Stage 3 gated on both MSE improvement and q4 mechanism improvement.

## Stage 4E: Learnable Alpha Adapter

The secondary learnable-alpha experiment was also run. It learns alpha on validation only and evaluates on test only.

| Method | Split | MSE | MAE | MSE delta % vs LRBN | Harm | Mean alpha |
|---|---|---:|---:|---:|---:|---:|
| adaptive-alpha-safe-loss | val | 8.435078 | 2.241743 | -4.116097 | 0.429688 | 0.267660 |
| adaptive-alpha-safe-loss | test | 4.704777 | 1.638125 | -3.869534 | 0.385417 | 0.257851 |
| global-alpha-safe-loss | val | 8.419233 | 2.230162 | -4.296210 | 0.453125 | 0.500000 |
| global-alpha-safe-loss | test | 4.643981 | 1.621852 | -5.111746 | 0.423177 | 0.500000 |

Adaptive alpha reduces harm relative to global alpha / BP-always (`38.54%` vs `42.32%`) while retaining a large MSE gain (`-3.87%`), but it is still a high-harm dense correction. It does not qualify as the safe default method. It should be treated as a performance-oriented appendix variant, not as the main TableA method.

## Best Interpretation

Stage 4 gives a clean mechanism story:

1. BP has real correction power after LRBN.
2. Most useful BP gain is in high post-LRBN boundary-gap samples.
3. Low boundary-gap and high repair-ratio samples explain much of the harm.
4. Repair-gate is the most promising single harm-control mechanism:
   - MSE improvement: `-1.747470%`
   - harm: `0.218750`
   - q4 improvement: `5.396815%`
5. Learnable alpha confirms that alpha adaptation can reduce harm slightly versus global alpha, but not enough to become safe.
6. The current combined safe-controller is over-conservative and does not beat Stage 3 gated.

## Decision

Do not replace Stage 3 `LRBN-BP-gated` with Stage 4 `LRBN-BP-safe-controller`.

Recommended TableA rows:

- `HalluGuard-LRBN`
- `HalluGuard-LRBN-BP-gated` as safe sparse enhancement candidate
- `LRBN-BP-always` as performance ablation
- `LRBN-BP-repair-gate` as harm-control ablation
- optionally `LRBN-BP-gap-strength` as high-gain/high-harm ablation
- optionally `adaptive-alpha-safe-loss` as a performance-oriented appendix variant

Do not claim Stage 4 safe-controller is the final method. Claim should be narrowed:

> Boundary Projection has strong post-LRBN correction power, but safe deployment requires sparse or repair-aware gating. Current mechanism-level safe controller reduces harm but over-suppresses the boundary repair signal.
