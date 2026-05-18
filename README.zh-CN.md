# HalluGuard

[English](README.md) | [中文](README.zh-CN.md)

HalluGuard 是一个面向时间序列预测的轻量级测试时修正模块。它关注的问题是：预测模型的输出在整体指标上可能不错，但在局部动态上仍然会出现不合理现象，例如突变边界被过度平滑、趋势变化没有跟上，或者生成了近期上下文并不支持的高频波动。

当前版本是一个研究原型。它不重新训练基础预测模型，而是在模型已经生成预测之后，结合近期历史上下文和预测形态，判断预测是否存在局部动态风险，并在验证集校准的条件下执行小幅修正。

## 研究动机

时间序列预测模型经常会遇到一种“看起来合理但局部不可信”的输出：曲线平滑、误差不一定极端，但在边界、趋势或频率结构上违背了序列本身的动态规律。

HalluGuard 希望研究一个更通用的问题：是否可以在不改变原模型训练过程的前提下，通过一个模型无关的后处理模块，降低这类预测幻觉带来的误差和风险。

## 方法概览

当前最有希望的路线由三部分组成：

- **边界感知修复**：针对预测起点附近的局部不连续和突变误差进行修正。
- **选择性残差平滑**：只处理局部残差尖峰，而不是对整个预测区间做统一平滑。
- **验证集校准路由**：根据验证集上的行为，在不修正、边界修复、平滑修正和选择性修复之间做选择，并用随机动作、匹配平滑和泄漏检查作为对照。

这种设计的目标不是简单追求更强的平滑效果，而是避免退化成“所有情况都平滑”的后处理器。

## 初步实验

目前的实验基于 DLinear 和 PatchTST 等时间序列预测模型，在多个预测长度上进行评估。实验包括：

- 标准 clean benchmark；
- 面向边界突变、趋势漂移、斜率变化、延迟 level shift、高频扰动和方差变化的 stress benchmark；
- 用于检查外部风险的 compact external fixture。

主要指标是相对于未修正预测的 MSE delta 百分比，数值越负表示修正后误差越低。

| 方法 | 作用 | Clean MSE delta | Clean PatchTST delta | Stress MSE delta | External PatchTST delta | 说明 |
|---|---|---:|---:|---:|---:|---|
| Adaptive router baseline | 前一版基线 | -1.289% | -0.298% | -1.391% | +0.004% | 内部结果较稳，但 PatchTST 外部 harm 风险较高 |
| Capped logistic router | 学习式路由诊断 | -1.658% | -0.483% | -1.920% | -0.135% | clean/stress 提升明显，但外部动作集中问题仍存在 |
| Boundary-selective adaptive router | 已完整验证的 clean/stress 主结果 | -2.164% | -0.553% | -2.488% | -0.046% | 当前最可靠的 clean/stress 完整结果 |
| Stable selective fallback router | 外部 harm 诊断 | -2.158% | -0.579% | -2.468% | -0.368% | 当前对 PatchTST 外部 harm 改善最明显的版本 |
| Smoothing-cap selective router | clean benchmark 领先候选 | -2.193% | -0.617% | pending | pending | clean 表现目前最好，stress/external 仍需补完 |

更完整的结果见 [preliminary results](docs/preliminary_results.md) 和 [CSV result table](results/preliminary_results.csv)。

## 当前结论

初步结果表明，HalluGuard 作为时间序列预测后处理模块具有可行性，尤其是在 clean benchmark 和 stress benchmark 上能稳定降低误差。当前较有价值的机制并不是全局平滑，而是局部边界修复、选择性残差平滑和验证集校准路由的组合。

外部泛化仍然需要谨慎表述。目前的 external fixture 更适合作为 harm diagnostic，用来检查某些修正策略是否会伤害 PatchTST 等模型；它还不足以支撑广泛外部泛化的结论。

## 后续工作

- 补完当前 clean benchmark 领先候选的 stress 和 external 评估。
- 扩展到更多数据集、模型和预测长度。
- 将修正模块整理成可接收外部预测结果的简单 API。
- 增加可复现实验脚本，包括 benchmark 生成、预测修正和结果汇总。
- 继续研究更稳健的路由模块，减少对已经稳定预测的误修正。

## 仓库状态

本仓库目前保存的是 HalluGuard 的初步公开快照，包括项目动机、方法简介和阶段性实验结果。代码、API 和复现实验脚本会随着原型稳定逐步整理进来。
