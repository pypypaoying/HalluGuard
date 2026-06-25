# HalluGuard MVP Agent Instructions

本仓库当前目标是实现并验证 **HalluGuard Trend-Frequency Test-Time Correction** 的最小可运行实验。

## Research Goal

HalluGuard 是一个 inference-time / offline post-processing 方法。它不改主模型、不改训练 loss、不做 hidden-state intervention，只根据 `context` 与 `prediction` 的趋势和频谱一致性触发保守校正。

核心实验问题：

- forecast 中是否存在 trend/frequency dynamics violation？
- trend/frequency trigger 是否比 random trigger 更有效？
- 保守校正是否能降低 violation，同时不恶化 MSE/MAE？
- 它是否会误伤真实 turning point？

## First-Stage Scope

第一阶段只做 synthetic / stress MVP，不接复杂 TSF 训练框架。

必须实现：

- 统一预测样本格式：`sample_id, dataset, model, context, prediction, target`
- synthetic/stress benchmark：clean、trend drift、high-frequency noise、local oscillation、real turning point
- 离线校正：trend-only、frequency-only、trend+frequency
- 对照：no correction、naive smoothing、random trigger
- 指标：MSE、MAE、HallucinationRate、TrendViolationRate、FreqViolationRate、SpectralConsistency、TurningPointFalseCorrectionRate、InferenceLatency
- 阈值只从 validation/calibration split 估计，禁止用 test split 调阈值

## Non-Negotiable Rules

- 不修改 `docs/aris_outputs/SELECTED_IDEA_MVP.md` 的研究目标。
- 不一开始接入 PatchTST / ETTm1 大工程。
- 不用 test split 调 threshold、lambda 或任何触发规则。
- 不把 synthetic stress test 结果夸大成真实 benchmark 结论。
- 不编造结果。所有结论必须来自实际输出文件。
- 每轮自动实验只改一个主要变量。

## Expected Files

实现代码放在：

```text
experiments/halluguard/
  README.md
  correction.py
  metrics.py
  stress.py
  evaluate_predictions.py
  run_mvp.py
  configs/halluguard_mvp.yaml
  results/
```

每轮实验结果追加到：

```text
results_halluguard.tsv
```

## Success Criteria

第一轮成功不要求“论文级提升”，但必须满足：

- `run_mvp.py` 可一键运行。
- ablation 表完整。
- rule trigger 明显优于 random trigger，或者给出机制无效的清晰失败结论。
- 没有 test threshold leakage。
- `summary.md` 明确说明是否值得接入真实 `ETTm1 + DLinear/PatchTST predictions`。

## Stop Conditions

停止继续优化 HalluGuard，并转向失败分析，如果：

- random trigger 与 rule trigger 表现接近。
- HallucinationRate 降低但 MSE/MAE 恶化超过 3%。
- TurningPointFalseCorrectionRate 明显高于 naive smoothing。
- 只在人工扰动有效，对 clean / real turning-point case 明显误伤。
- threshold 或 correction strength 极度敏感。
