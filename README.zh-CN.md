# HalluGuard

[English](README.md) | [中文](README.zh-CN.md)

HalluGuard 是一个面向时间序列预测的轻量级测试时修正模块。它在模型已经生成预测之后，检查预测曲线是否出现局部动态不一致，例如边界跳变、趋势衔接错误、过度平滑真实变化，或者生成近期上下文不支持的高频波动。

当前版本是研究原型。它接收基础模型的 forecast 和 recent context，通过验证集校准过的规则或 router 判断是否需要修正，并只在预计有帮助时执行小幅 correction。

## 研究动机

现代时间序列预测模型在平均指标上可以很强，但仍可能产生局部动态不可信的预测输出。典型失败模式包括：

- 突变或 level shift 被过度平滑；
- forecast 起点和历史窗口末端衔接不自然；
- 趋势、斜率或曲率变化没有跟上；
- 生成上下文不支持的高频波动；
- 对已经稳定的预测做过度修正。

HalluGuard 研究的是：能否用一个模型无关的 post-processing layer，在不重训基础模型、不访问模型梯度的情况下，减少这些局部动态错误。

## 当前方法

目前最有价值的机制由三部分组成：

- **边界感知修复**：检测 forecast 起点与历史窗口末端之间的局部不连续。
- **选择性残差平滑**：把平滑限制在 context 不支持的局部 residual spike 上，保留预测 horizon 中的主要形态。
- **验证集校准路由**：在 no correction、boundary repair、smoothing、selective repair 等动作之间选择，并持续与 random-action、matched-smoothing 等控制组比较。

这一设计的目标是保留 smoothing 对 MSE 的收益，同时避免退化成通用平滑器。

## 初步实验

当前实验使用 DLinear 和 PatchTST 的预测输出，覆盖多个 horizon。评估包括：

- clean benchmark table；
- boundary discontinuity、trend drift、slope break、delayed level shift、high-frequency perturbation、variance shift 等 stress tests；
- compact external fixture，主要用于检查外部预测表上的 harm 风险。

主指标是相对 uncorrected forecast 的 MSE delta percentage。负数表示修正后 MSE 更低。

| Method | Role | Clean MSE delta | Clean PatchTST delta | Stress MSE delta | External PatchTST delta | Notes |
|---|---|---:|---:|---:|---:|---|
| Adaptive router baseline | Previous baseline | -1.289% | -0.298% | -1.391% | +0.004% | Strong internal baseline, weak PatchTST external harm profile |
| Boundary-selective adaptive router | Strong selective action | -2.164% | -0.553% | -2.488% | -0.046% | Boundary plus selective median substantially improves clean/stress |
| Smoothing-cap selective router | Clean/stress leader | -2.193% | -0.617% | -2.509% | -0.065% | Best clean and stress result so far, but external PatchTST harm is only partially improved |
| Stable smoothing-cap guard | External-harm guard | -2.135% | -0.571% | -2.463% | -0.366% | Best PatchTST harm reduction on the external fixture, with 0/8 harmed PatchTST configurations |
| Conditional stable-cap guard | Compromise candidate | -2.181% | -0.609% | -2.505% | -0.171% | Preserves most clean/stress gains while improving external behavior, but does not fully remove PatchTST harm |

更多结果见：[preliminary results](docs/preliminary_results.md)、[研究叙事与架构说明](docs/research_narrative.zh-CN.md) 和 [CSV result table](results/preliminary_results.csv)。

## 当前结论

初步证据显示，HalluGuard 作为时间序列 forecast 的测试时后处理模块是有价值的。当前最有前景的机制由局部边界修复、选择性残差平滑和验证集校准 router 组成。

外部泛化仍需更大规模验证。当前 external fixture 更适合作为 harm diagnostic，尤其用于观察 PatchTST-like forecasts 是否会被过度修正。stable-forecast guard 能在这个 fixture 中移除观察到的 PatchTST harm，同时会牺牲一部分 clean/stress 表现，因此还需要更大规模的外部 benchmark。

## 未来工作与期望

- 冻结当前 clean/stress leader 作为公开研究快照，并把它和 safety-oriented diagnostic variants 区分清楚。
- 扩展到更多数据集、forecasting backbone 和 horizon，重点判断 HalluGuard 何时应该修正、何时应该 abstain。
- 将 correction module 包装成清晰 API，支持输入外部框架导出的 forecast table，并输出 corrected forecast、selected action、confidence score 和 diagnostics。
- 增加可复现实验脚本，用于 benchmark generation、correction、result aggregation 和 action-level case study。
- 继续研究 safety-aware routing，尤其是对已经稳定的 forecast 避免过度修正，降低 PatchTST-like external cases 上的 harm 风险。

项目的期望形态是一个轻量、模型无关、可解释的 forecast correction package：它可以接在现有时间序列预测模型之后，不需要重训基础模型，同时提供 performance-oriented mode 和 conservative mode，分别服务于 clean/stress 收益最大化和外部未知预测表上的低伤害部署。

## 仓库状态

本仓库目前存放初步公开研究说明、阶段性结果表和方法文档。代码与复现实验脚本会在原型进一步稳定后继续整理进来。
