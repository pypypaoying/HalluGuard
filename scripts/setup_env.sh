#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-halluguard-run}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env ${ENV_NAME} already exists."
  else
    conda env create -f environment.yml
  fi
  echo
  echo "Activate with:"
  echo "  conda activate ${ENV_NAME}"
  echo
  echo "Then run:"
  echo "  bash scripts/smoke_check.sh"
else
  echo "conda not found; falling back to venv at .venv"
  "${PYTHON_BIN}" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  echo
  echo "Activated .venv for this shell. Run:"
  echo "  bash scripts/smoke_check.sh"
fi
