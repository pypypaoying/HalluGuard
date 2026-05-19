# HalluGuard

[English](README.md) | [中文](README.zh-CN.md)

HalluGuard is a lightweight test-time correction layer for time-series forecasts. It is designed for cases where a forecasting model produces predictions that look plausible but violate local dynamics, such as boundary jumps, trend changes, or unsupported high-frequency behavior.

The current version is a research prototype. It does not retrain the forecasting model. Instead, it takes the model forecast and recent context, estimates whether the forecast is dynamically suspicious, and applies a small correction only when validation-calibrated rules say the correction is likely to help.

## Motivation

Modern time-series forecasters can be accurate on average while still producing locally inconsistent forecasts. Common failure modes include:

- smoothing over abrupt level shifts;
- missing slope or trend changes;
- introducing high-frequency artifacts not supported by the recent context;
- over-correcting already stable forecasts.

HalluGuard studies whether these failures can be reduced by a model-agnostic post-processing module that runs at inference time.

## Current Approach

The strongest current line combines three ideas:

- **Boundary-aware repair**: detect and repair local discontinuities near forecast boundaries.
- **Selective residual smoothing**: smooth only local residual spikes instead of smoothing the full prediction horizon.
- **Validation-calibrated routing**: choose between no correction, boundary repair, smoothing, and selective repair using validation-only thresholds, while checking against random-action and matched-smoothing controls.

This design aims to improve forecast quality without collapsing into a generic smoothing baseline.

## Preliminary Experiments

Initial experiments were run on standard time-series forecasting fixtures using DLinear and PatchTST across multiple horizons. The evaluation currently includes:

- a clean benchmark table;
- synthetic stress tests for boundary changes, trend drift, slope breaks, delayed level shifts, high-frequency perturbations, and variance shifts;
- a compact external fixture used mainly as a harm diagnostic.

The main metric is MSE delta percentage against the uncorrected forecast. Negative values are better.

| Method | Role | Clean MSE delta | Clean PatchTST delta | Stress MSE delta | External PatchTST delta | Notes |
|---|---|---:|---:|---:|---:|---|
| Adaptive router baseline | Previous baseline | -1.289% | -0.298% | -1.391% | +0.004% | Strong internal baseline, weak PatchTST external harm profile |
| Boundary-selective adaptive router | Best completed clean/stress result | -2.164% | -0.553% | -2.488% | -0.046% | Strongest fully completed clean/stress candidate |
| Smoothing-cap selective router | Clean/stress leader | -2.193% | -0.617% | -2.509% | -0.065% | Best clean and stress result so far, but external PatchTST harm is only partially improved |
| Stable smoothing-cap guard | External-harm guard | -2.135% | -0.571% | -2.463% | -0.366% | Best PatchTST harm reduction on the external fixture, with 0/8 harmed PatchTST configurations |
| Conditional stable-cap guard | Compromise candidate | -2.181% | -0.609% | -2.505% | -0.171% | Preserves most clean/stress gains while improving external behavior, but does not fully remove PatchTST harm |

See [preliminary results](docs/preliminary_results.md), [research narrative and architecture notes](docs/research_narrative.zh-CN.md), and [CSV table](results/preliminary_results.csv) for the current snapshot.

## Current Takeaway

The early evidence suggests that HalluGuard is useful as a post-processing module for time-series forecasts, especially under clean and stress-test settings. The most promising mechanism is not full-horizon smoothing, but local boundary repair plus selective residual smoothing with validation-only routing and confidence-capped smoothing deployment.

External generalization is not yet a settled claim. The current external fixture is useful for detecting harm, especially on PatchTST-like forecasts. A stable-forecast guard can remove the observed PatchTST harm in this fixture, but it trades off some clean/stress performance, so larger and more diverse external benchmarks are still needed.

## Next Steps

- Freeze the current clean/stress leader as the main research snapshot.
- Expand evaluation to more datasets, models, and horizons.
- Package the correction module behind a simple API that can accept forecasts from external frameworks.
- Add reproducible scripts for benchmark generation, correction, and result aggregation.
- Study complementary routing modules for cases where correction may harm already stable forecasts, especially external PatchTST-like cases.

## Repository Status

This repository currently contains an initial public-facing research snapshot: project motivation, method summary, and preliminary result tables. Code and reproducibility scripts will be organized here as the prototype stabilizes.
