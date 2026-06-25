# HalluGuard-LRBN Clean Claim BigTable Server Run

This is the one-command server entry for the clean-claim method:

```text
HalluGuard-LRBN unified_revin_rdn_hybrid
```

The exploratory NST/future-center lines are not part of the main claim table.

## One Command

```bash
bash scripts/run_clean_claim_bigtable.sh
```

The repository already includes the public baseline code needed by the unified
runner, so `FETCH_PLUGIN_REPOS=0` is the default. Set `FETCH_PLUGIN_REPOS=1`
only when you intentionally want to refresh missing/upstream baseline repos.

First run on a fresh server should fetch data explicitly:

```bash
FETCH_DATA=1 bash scripts/run_clean_claim_bigtable.sh
```

Later reruns should normally skip re-downloading data:

```bash
EXTRA_FLAGS="--skip-existing" bash scripts/run_clean_claim_bigtable.sh
```

Outputs:

```text
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/combined_metrics.csv
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/combined_metrics.json
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/summary.md
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/summary_by_method.csv
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/summary_by_backbone.csv
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/summary_by_dataset.csv
```

## Default Matrix

- Datasets: `ETTm1, ETTm2, ETTh1, ETTh2, Weather, ECL, Traffic`
- Backbones: `DLinear, PatchTST, iTransformer, TimesNet, TimeMixer`
- Horizons: `96,192,336,720`
- Seeds: `2026,2027,2028`
- Methods:
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

Every requested row is written as `completed` or `blocked`. The current unified
exporter completes the default matrix using one shared training/evaluation
contract:

- data: single target series, `target=OT` when available, otherwise the last
  numeric column
- split: ETT official 12/4/4-month split; Weather/ECL/Traffic use 70/10/20
- `seq_len=96`, shared horizons, seeds, optimizer, epoch budget, batch size,
  and train/eval window caps
- backbones: local DLinear/PatchTST plus public Time-Series-Library
  `iTransformer`, `TimesNet`, and `TimeMixer` classes wrapped into the same
  `context -> prediction` exporter
- SAN: uses the official `Statistics_prediction` module with train-split
  station pretraining (`SAN_PRETRAIN_EPOCHS`, default `5`) and default
  `SAN_PERIOD_LEN=24`
- DishTS: uses the official `DishTS.py` normalization module on a raw-data
  path, matching the paper/repo motivation that preprocessing can hide
  distribution shift

This is a fairness-oriented unified protocol rather than each official repo's
full leaderboard recipe. SAN and DishTS are now stronger mechanism-faithful
adapters, but still not each repo's full hyperparameter-swept leaderboard run.
Smoothing controls are post-hoc baselines generated from the same raw prediction
files; `matched_sparse_smoothing` calibrates its trigger only on `split="val"`.

## Smoke

```bash
EXTRA_FLAGS="--smoke" EPOCHS=1 MAX_TRAIN_WINDOWS=128 MAX_EVAL_WINDOWS=32 \
  bash scripts/run_clean_claim_bigtable.sh
```

## Full Server Knobs

```bash
DEVICE=cuda \
EPOCHS=10 \
SAN_PERIOD_LEN=24 \
SAN_STATION_LR=0.0001 \
SAN_PRETRAIN_EPOCHS=5 \
MAX_TRAIN_WINDOWS=8192 \
MAX_EVAL_WINDOWS=1024 \
FETCH_DATA=1 \
FETCH_PLUGIN_REPOS=0 \
OUTPUT_DIR=experiments/halluguard/results/lrbn_clean_claim_bigtable_v1 \
  bash scripts/run_clean_claim_bigtable.sh
```

To restrict the run:

```bash
DATASETS=ETTm1,ETTh1 \
BACKBONES=DLinear,PatchTST \
SEEDS=2026,2027,2028 \
FETCH_DATA=1 \
  bash scripts/run_clean_claim_bigtable.sh
```

To run the expanded default matrix:

```bash
FETCH_DATA=1 \
DEVICE=cuda \
EPOCHS=10 \
MAX_TRAIN_WINDOWS=8192 \
MAX_EVAL_WINDOWS=1024 \
OUTPUT_DIR=experiments/halluguard/results/lrbn_clean_claim_bigtable_v2_expanded \
  bash scripts/run_clean_claim_bigtable.sh
```

## Monitoring

The top-level runner prints one progress line per dataset/model/horizon/seed
config. Child process stdout/stderr is written to:

```text
<OUTPUT_DIR>/logs/
```

Useful server checks:

```bash
tail -f experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/logs/fetch_data.log
tail -f experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/logs/ETTm1_DLinear_96_seed2026_lrbn.log
tail -f experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/logs/ETTm1_DLinear_96_seed2026_adaptation.log
```

## Fairness Contract

- Same `(dataset, backbone, horizon, seed)` across methods.
- Same `seq_len=96`, train/val/test split, optimizer, batch size, learning rate,
  epoch budget, and train/eval window caps.
- Train split fits all model and normalization parameters.
- Test split is final evaluation only.
- `test_threshold_leakage=False` for all completed rows.

## Claim Interpretation

The clean claim is not that LRBN beats every possible official model. The claim
under this table is:

> HalluGuard-LRBN is a simple learnable reversible boundary-normalization layer
> that improves raw forecasters and is competitive with normalization/adaptation
> baselines under a shared evaluation contract.

Unsupported rows are blockers for framework coverage, not hidden failures.
