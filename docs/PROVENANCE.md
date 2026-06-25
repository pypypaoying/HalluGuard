# Provenance

This repository is a cleaned publication package assembled from the active
HalluGuard-run development line.

```text
source commit: 948be2b Repair SAN and DishTS baseline adapters
```

The latest owner-supplied expanded BigTable CSV was imported and summarized
under:

```text
results/lrbn_clean_claim_bigtable_v2_expanded_latest/
```

## External Code

The repository includes public external code needed for one-command server
reproduction:

- `external/Time-Series-Library/`: backbone implementations for iTransformer,
  TimesNet, and TimeMixer integration. Root upstream README/LICENSE files are
  retained in that directory.
- `external/plugin_baselines/RevIN/`
- `external/plugin_baselines/Dish-TS/`
- `external/plugin_baselines/Nonstationary_Transformers/`
- `external/plugin_baselines/SAN/`
- `external/plugin_baselines/TAFAS/`

Nested `.git` directories were removed before publication so this repository can
be cloned as a normal standalone project. The external code is used through
adapter scripts and is not mixed into the LRBN implementation itself.

## Data

Large datasets are intentionally not committed. Use:

```bash
FETCH_DATA=1 bash scripts/run_clean_claim_bigtable.sh
```

to download or prepare the public datasets into:

```text
external/ETDataset/
external/Time-Series-Library/dataset/
```
