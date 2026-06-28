#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/halluguard/results/lrbn_sra_bp_stage5_repro}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"

"${PYTHON_BIN}" experiments/halluguard/run_stage5_sra_bp.py \
  --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv \
  --stage3-dir experiments/halluguard/results/lrbn_bp_stage3 \
  --stage4-dir experiments/halluguard/results/lrbn_bp_stage4 \
  --stage45-dir experiments/halluguard/results/lrbn_bp_attribution_stage45 \
  --output-dir "${OUTPUT_DIR}" \
  --n-bootstrap "${N_BOOTSTRAP}"
