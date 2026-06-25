# HalluGuard

This repository publishes the current clean-claim HalluGuard implementation and
the latest expanded BigTable results.

Main method:

```text
HalluGuard-LRBN unified_revin_rdn_hybrid
```

HalluGuard-LRBN is a learnable reversible boundary normalization wrapper for
time-series forecasting backbones. It normalizes each input window with a
learned mixture of boundary/instance centers and robust/instance scales, runs the
forecasting backbone in the normalized space, and reverses the transform on the
output. The current claim is a clean forecasting robustness claim, not a
test-threshold-tuned correction claim.

## What Is Included

- `scripts/run_clean_claim_bigtable.sh`: one-command expanded BigTable runner.
- `scripts/run_lrbn_clean_claim_bigtable.py`: orchestration script for datasets,
  backbones, seeds, methods, and aggregate reports.
- `scripts/run_halluguard_lrbn.py`: LRBN implementation and training/evaluation
  harness.
- `external/halluguard_real_pipeline/`: unified prediction/export pipeline.
- `external/Time-Series-Library/`: public backbone implementations used for
  iTransformer, TimesNet, and TimeMixer integration.
- `external/plugin_baselines/`: public test-time/adaptation baselines used by
  the comparison table.
- `results/lrbn_clean_claim_bigtable_v2_expanded_latest/`: latest uploaded
  4620-row expanded BigTable result.

## Server Quick Start

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

If the environment already exists:

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

For a small smoke run:

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

## Default Expanded BigTable

- Datasets: `ETTm1`, `ETTm2`, `ETTh1`, `ETTh2`, `Weather`, `ECL`, `Traffic`
- Backbones: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `TimeMixer`
- Horizons: `96`, `192`, `336`, `720`
- Seeds: `2026`, `2027`, `2028`
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

The runner writes every requested row as `completed`, `failed`, or `blocked`.
The latest uploaded table completed all 4620 rows with
`test_threshold_leakage=False`.

## Latest Result Snapshot

See:

- `results/lrbn_clean_claim_bigtable_v2_expanded_latest/summary.md`
- `results/lrbn_clean_claim_bigtable_v2_expanded_latest/combined_metrics.csv`

Key aggregate from the latest uploaded table:

- HalluGuard-LRBN mean MSE delta vs raw: `-3.7952%`
- RevIN mean MSE delta vs raw: `-3.6921%`
- NST mean MSE delta vs raw: `-3.6155%`
- Completed rows: `4620 / 4620`
- Test threshold leakage: `False`

The current evidence supports HalluGuard-LRBN as a competitive normalization
wrapper. It does not support a universal-win claim; ECL remains the major weak
dataset and some baselines have large instability.

## Documentation

- `HALLUGUARD_LRBN.md`: method details and LRBN variant definitions.
- `CLEAN_CLAIM_BIGTABLE_SERVER.md`: server command and matrix details.
- `docs/BIG_TABLE_RESULTS.md`: latest result interpretation.
- `docs/PROVENANCE.md`: source and dependency provenance.
