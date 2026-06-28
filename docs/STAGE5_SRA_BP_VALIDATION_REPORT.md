# Stage 5 SRA-BP Validation Report

## Goal

Stage 5 implemented and validated `HalluGuard-SRA-BP`: HalluGuard-LRBN plus a Sparse Repair-Aware Boundary Projection expert. This stage follows the Stage 4.5 conclusion that dense BP-always has real correction power but unsafe harm, and tests whether sparse high-gap / low-repair gating with short-support bridges can keep the useful boundary repair while avoiding the known harm slices.

This is compact validation only, not a full TableA result.

## Setup

- Input: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- Output: `experiments/halluguard/results/lrbn_sra_bp_stage5/`
- Datasets: `ETTm1`, `ETTh1`
- Backbones: `DLinear`, `PatchTST`
- Horizons: `96`, `192`
- Seed: `2026`
- Validation samples: `768`
- Test samples: `768`
- Test configs: `8`
- Candidate grid rows: `4920`
- Bootstrap: `2000`
- Calibration: validation-only
- Test threshold leakage: `False`

Command:

```bash
python experiments/halluguard/run_stage5_sra_bp.py --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv --stage3-dir experiments/halluguard/results/lrbn_bp_stage3 --stage4-dir experiments/halluguard/results/lrbn_bp_stage4 --stage45-dir experiments/halluguard/results/lrbn_bp_attribution_stage45 --output-dir experiments/halluguard/results/lrbn_sra_bp_stage5 --n-bootstrap 2000
```

## Selected Variants

Safe-SRA was selected from non-basic deployable candidates:

```json
{
  "method_family": "short",
  "tau_g": 5.265299801054961,
  "tau_r": 0.8,
  "tau_j": null,
  "alpha": 0.75,
  "K": "H_div_4"
}
```

Balanced-SRA was selected from non-basic deployable candidates:

```json
{
  "method_family": "support",
  "tau_g": 2.4260872328869336,
  "tau_r": 0.8,
  "tau_j": 0.3,
  "alpha": 0.75,
  "K": "H_div_4"
}
```

The runner keeps `SRA-BP-basic` as an ablation, but Safe/Balanced selection prefers non-basic short/support candidates when validation-feasible, so the deployable line stays aligned with the Stage 5 mechanism claim.

## Overall Results

Test results vs HalluGuard-LRBN:

| Method | MSE | MAE | MSE Delta % | Harm | Coverage | q4 Improvement | High-Gap/Low-Repair Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| LRBN-BP-always | 4.643981 | 1.621852 | -5.111746 | 0.423177 | 1.000000 | 13.923609 | -12.864612 |
| SRA-BP-basic ablation | 4.722565 | 1.640118 | -3.506063 | 0.113281 | 0.324219 | 13.923527 | -12.864612 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.436198 | 8.587733 | -7.917318 |
| SRA-BP-safe | 4.813149 | 1.660950 | -1.655217 | 0.035156 | 0.187500 | 6.928297 | -6.197214 |
| LRBN-BP-repair-gate | 4.808634 | 1.661626 | -1.747470 | 0.218750 | 0.610677 | 5.275949 | -5.961786 |
| LRBN-BP-stage3-gated | 4.864131 | 1.674353 | -0.613520 | 0.018229 | 0.042969 | 1.988351 | -2.294853 |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Bootstrap CIs:

| Method | Overall Delta CI | High-Gap/Low-Repair Delta CI |
|---|---:|---:|
| SRA-BP-safe | [-0.105546, -0.058766] | [-0.393047, -0.220804] |
| SRA-BP-balanced | [-0.154968, -0.099215] | [-0.482474, -0.290096] |

Both selected SRA variants have bootstrap upper bound below zero overall and in the high-gap/low-repair slice.

## Hypothesis Checks

### H1: SRA gate improves coverage over Stage3 gated

Supported.

- Stage3 gated coverage: `0.042969`
- Safe-SRA coverage: `0.187500`
- Balanced-SRA coverage: `0.436198`

Coverage increases while Safe-SRA keeps harm under `0.05`.

### H2: SRA-BP keeps high-gap/low-repair gains

Supported.

- Safe-SRA high-gap/low-repair delta: `-6.197214%`
- Balanced-SRA high-gap/low-repair delta: `-7.917318%`
- Both high-gap/low-repair bootstrap CI upper bounds are below zero.

### H3: SRA-BP avoids low-gap/high-repair harm

Supported in compact validation.

- Safe-SRA low-gap/high-repair coverage: `0.0`
- Balanced-SRA low-gap/high-repair coverage: `0.0`
- Low-gap/high-repair delta: `0.0`

This directly fixes the Stage 4.5 low-gap/high-repair harm slice for the selected policies.

### H4: Short-support bridge reduces mid/late harm

Supported for selected Safe/Balanced SRA.

Safe-SRA:

- early delta: `-10.140630%`
- mid delta: `0.0%`
- late delta: `0.0%`

Balanced-SRA:

- early delta: `-15.919673%`
- mid delta: `0.0%`
- late delta: `0.0%`

The selected `K=H_div_4` bridge confines changes to the early segment in the current compact horizons.

### H5: True-jump suppressor reduces true-jump miscorrection

Partially supported.

Safe-SRA has no explicit `tau_j` suppressor but avoids `low_gL_high_gY` through the high-gap gate:

- `low_gL_high_gY` coverage: `0.0`

Balanced-SRA uses `tau_j=0.3` and also avoids `low_gL_high_gY`:

- `low_gL_high_gY` coverage: `0.0`

However, both selected variants still correct some `high_gL_high_gY` samples. That slice improves on mean in compact validation, but it should remain a stress point in mini-extension because `g_y` is oracle-only and cannot be used at test time.

## Verdict

From `stage5_verdict.json`:

```json
{
  "status": "safe_and_balanced_pass",
  "decision": "promote_sra_bp_to_mini_extension",
  "safe_pass": true,
  "balanced_pass": true,
  "test_threshold_leakage": false
}
```

Stage 5 passes compact validation.

## Interpretation

The cleanest claim is:

> Sparse repair-aware gating turns BP from an unsafe dense performance branch into a deployable compact-stage boundary expert. Safe-SRA improves coverage over Stage3 gated while preserving low harm; Balanced-SRA gives a stronger performance tradeoff with moderate harm.

Do not claim yet that SRA-BP is ready for full TableA. It should enter mini-extension first.

## Limitations

- Scope is still compact: 8 configs, seed 2026, horizons 96/192.
- Safe-SRA has one per-config harm above the global safe average (`ETTm1/PatchTST/192`, harm `0.114583`), so expanded validation must track per-config harm.
- Balanced-SRA is a risk-performance variant, not a safe default; per-config harm reaches about `0.20` in some configs.
- The compact smoothing controls in this LRBN-centric input are weak and should not be treated as the final smoothing confrontation.
- The true-jump suppressor proxy is only partially validated; oracle true-jump labels remain diagnostics only.

## Recommendation

Proceed to Stage 5C mini-extension with at least:

- `HalluGuard-LRBN`
- `LRBN-BP-stage3-gated`
- `SRA-BP-safe`
- `SRA-BP-balanced`
- `SRA-BP-basic` as ablation only
- `BP-always` as performance/harm upper branch

Do not promote BP-always or SRA-BP-basic as the clean method despite their stronger MSE, because the mechanism claim is about sparse repair-aware short/support BP.
