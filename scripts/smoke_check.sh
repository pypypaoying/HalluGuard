#!/usr/bin/env bash
set -euo pipefail

python experiments/halluguard/run_mvp.py \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --quick

python external/halluguard_real_pipeline/export_predictions.py \
  --dataset ETTm1 \
  --model DLinear \
  --horizon 96 \
  --data-root external/ETDataset \
  --output experiments/halluguard/results/smoke_predictions/ETTm1_DLinear_96.jsonl \
  --epochs 1 \
  --max-train-windows 512 \
  --max-eval-windows 64 \
  --batch-size 128 \
  --device auto

python experiments/halluguard/evaluate_predictions.py \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --input experiments/halluguard/results/smoke_predictions/ETTm1_DLinear_96.jsonl \
  --calibration-split val \
  --split test \
  --output-dir experiments/halluguard/results/smoke_eval/ETTm1_DLinear_96
