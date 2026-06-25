# HalluGuard

本仓库发布当前收束后的 HalluGuard clean-claim 主线方法和最新扩展大表结果。

当前主方法：

```text
HalluGuard-LRBN unified_revin_rdn_hybrid
```

HalluGuard-LRBN 是一个可学习的可逆边界归一化包装器。它在输入窗口上学习
boundary/instance center 与 robust/instance scale 的混合归一化，在归一化空间
训练/运行预测骨干模型，再把输出反归一化回原空间。当前主张是 clean forecasting
robustness，不是基于 test threshold 的后处理调参。

## 服务器一键运行

首次运行：

```bash
git clone https://github.com/pypypaoying/HalluGuard.git
cd HalluGuard

conda env create -f environment.yml
conda activate halluguard-run

FETCH_DATA=1 \
DEVICE=cuda \
EPOCHS=10 \
MAX_TRAIN_WINDOWS=8192 \
MAX_EVAL_WINDOWS=1024 \
OUTPUT_DIR=experiments/halluguard/results/lrbn_clean_claim_bigtable_v1 \
bash scripts/run_clean_claim_bigtable.sh
```

已有仓库时：

```bash
git pull
conda activate halluguard-run

FETCH_DATA=1 \
DEVICE=cuda \
EPOCHS=10 \
MAX_TRAIN_WINDOWS=8192 \
MAX_EVAL_WINDOWS=1024 \
OUTPUT_DIR=experiments/halluguard/results/lrbn_clean_claim_bigtable_v1 \
bash scripts/run_clean_claim_bigtable.sh
```

快速 smoke：

```bash
DATASETS=ETTm1 \
BACKBONES=DLinear \
HORIZONS=96 \
SEEDS=2026 \
METHODS=raw_no_correction,HalluGuard-LRBN,RevIN,NST,median_smoothing \
FETCH_DATA=1 \
DEVICE=cpu \
EPOCHS=1 \
MAX_TRAIN_WINDOWS=128 \
MAX_EVAL_WINDOWS=32 \
OUTPUT_DIR=experiments/halluguard/results/lrbn_smoke \
bash scripts/run_clean_claim_bigtable.sh
```

## 默认大表

- 数据集：`ETTm1`, `ETTm2`, `ETTh1`, `ETTh2`, `Weather`, `ECL`, `Traffic`
- 预测骨干：`DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `TimeMixer`
- 预测长度：`96`, `192`, `336`, `720`
- 种子：`2026`, `2027`, `2028`
- 方法：
  - `raw_no_correction`
  - `HalluGuard-LRBN`
  - `matched_sparse_smoothing`
  - `naive_smoothing`
  - `ema_smoothing`
  - `median_smoothing`
  - `RevIN`
  - `DishTS`
  - `SAN`
  - `NST`
  - `TAFAS`

最新上传结果位于：

```text
results/lrbn_clean_claim_bigtable_v2_expanded_latest/
```

其中 `combined_metrics.csv` 是完整 4620 行大表，`summary.md` 是聚合总结。
这次结果 4620/4620 行全部完成，`test_threshold_leakage=False`。

## 最新结论简述

- HalluGuard-LRBN mean MSE delta vs raw: `-3.7952%`
- RevIN: `-3.6921%`
- NST: `-3.6155%`
- LRBN 与 RevIN/NST 属于同一量级，略优但差距不大。
- ECL 是主要失败数据集，不能声称 universal improvement。
- simple smoothing controls 不能解释 LRBN 的主要收益。
