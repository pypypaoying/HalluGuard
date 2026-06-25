#!/usr/bin/env bash
set -euo pipefail

# One-command HalluGuard-LRBN clean-claim big table.
#
# Default full matrix:
#   datasets: ETTm1, ETTm2, ETTh1, ETTh2, Weather, ECL, Traffic
#   backbones: DLinear, PatchTST, iTransformer, TimesNet, TimeMixer
#   horizons: 96,192,336,720
#   seeds: 2026,2027,2028
#
# Current unified runner completes the default matrix through one shared
# lightweight training/evaluation contract. Any future unsupported row is still
# recorded as blocked with a reproducible blocker reason.

DATASETS="${DATASETS:-ETTm1,ETTm2,ETTh1,ETTh2,Weather,ECL,Traffic}"
BACKBONES="${BACKBONES:-DLinear,PatchTST,iTransformer,TimesNet,TimeMixer}"
HORIZONS="${HORIZONS:-96,192,336,720}"
SEEDS="${SEEDS:-2026,2027,2028}"
METHODS="${METHODS:-raw_no_correction,HalluGuard-LRBN,matched_sparse_smoothing,naive_smoothing,ema_smoothing,median_smoothing,RevIN,DishTS,SAN,NST,TAFAS}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-10}"
SAN_PERIOD_LEN="${SAN_PERIOD_LEN:-24}"
SAN_STATION_LR="${SAN_STATION_LR:-0.0001}"
SAN_PRETRAIN_EPOCHS="${SAN_PRETRAIN_EPOCHS:-5}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-8192}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-1024}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/halluguard/results/lrbn_clean_claim_bigtable_v1}"
FETCH_DATA="${FETCH_DATA:-0}"
FETCH_DATASETS="${FETCH_DATASETS:-${DATASETS}}"
FETCH_PLUGIN_REPOS="${FETCH_PLUGIN_REPOS:-0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

if [[ "${FETCH_PLUGIN_REPOS}" == "1" || "${FETCH_PLUGIN_REPOS}" == "true" || "${FETCH_PLUGIN_REPOS}" == "TRUE" ]]; then
  bash scripts/fetch_plugin_repos.sh
fi

FETCH_FLAGS=()
if [[ "${FETCH_DATA}" == "1" || "${FETCH_DATA}" == "true" || "${FETCH_DATA}" == "TRUE" ]]; then
  FETCH_FLAGS=(--fetch-data --fetch-datasets "${FETCH_DATASETS}")
fi

python scripts/run_lrbn_clean_claim_bigtable.py \
  --datasets "${DATASETS}" \
  --backbones "${BACKBONES}" \
  --horizons "${HORIZONS}" \
  --seeds "${SEEDS}" \
  --methods "${METHODS}" \
  --data-root external/ETDataset \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --san-period-len "${SAN_PERIOD_LEN}" \
  --san-station-lr "${SAN_STATION_LR}" \
  --san-pretrain-epochs "${SAN_PRETRAIN_EPOCHS}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-eval-windows "${MAX_EVAL_WINDOWS}" \
  --device "${DEVICE}" \
  "${FETCH_FLAGS[@]}" \
  ${EXTRA_FLAGS}

echo "== BigTable outputs =="
echo "${OUTPUT_DIR}/combined_metrics.csv"
echo "${OUTPUT_DIR}/combined_metrics.json"
echo "${OUTPUT_DIR}/summary.md"
