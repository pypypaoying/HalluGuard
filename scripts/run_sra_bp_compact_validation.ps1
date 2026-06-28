$ErrorActionPreference = "Stop"

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "experiments/halluguard/results/lrbn_sra_bp_stage5_repro" }
$NBootstrap = if ($env:N_BOOTSTRAP) { $env:N_BOOTSTRAP } else { "2000" }

& $PythonBin experiments/halluguard/run_stage5_sra_bp.py `
  --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv `
  --stage3-dir experiments/halluguard/results/lrbn_bp_stage3 `
  --stage4-dir experiments/halluguard/results/lrbn_bp_stage4 `
  --stage45-dir experiments/halluguard/results/lrbn_bp_attribution_stage45 `
  --output-dir $OutputDir `
  --n-bootstrap $NBootstrap
